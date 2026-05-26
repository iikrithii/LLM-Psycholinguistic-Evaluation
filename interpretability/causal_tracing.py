"""
interpretability/causal_tracing.py

Experiment: Causal Tracing / Activation Patching
=================================================
The most mechanistically decisive experiment in this suite.

Question
--------
Are the model components (attention heads + MLP layers) responsible for
phonological-semantic associations the same components responsible for
factual recall of the bouba-kiki effect?

  HIGH overlap → the model retrieves a memorised fact when it "exhibits" the effect.
  LOW overlap  → a mechanistically distinct circuit exists for phonological-semantic
                 processing, supporting genuine encoding rather than recall.

Method (following Meng et al. 2022, ROME)
------------------------------------------
1. Clean run: model responds correctly to a novel Tier 3 phonological pair.
2. Corrupted run: the two words are swapped (round-phonology ↔ sharp-phonology).
3. For each (layer, component) pair, patch the activations from the clean run
   into the corrupted run and measure the Indirect Effect (IE):
       IE = P(correct | patched) − P(correct | corrupted)
4. Normalise by the total effect: NIE = IE / (P_clean − P_corrupt).
5. High NIE identifies causally important components.

The experiment is run under two conditions:
  A. Phonological-semantic task  (Tier 3 novel pairs)       → component set P
  B. Factual recall task         (Tier 1 famous pairs)      → component set F

Overlap coefficient = |P ∩ F| / min(|P|, |F|) where sets are the top-K
components by normalised IE.

Runtime note
------------
Full causal tracing for Gemma-2 9B requires approximately
  n_layers × 2 components × 2 conditions × N_stimuli forward passes.
For N=20 stimuli this is ~3,360 forward passes ≈ 2-3 hours on A100.
Use --n_stimuli 10 for a faster run (~1.5 hours).

Run
---
    python -m interpretability.causal_tracing --model gemma2_9b
    python -m interpretability.causal_tracing --model gemma2_9b --n_stimuli 10
    python -m interpretability.causal_tracing --model both

Output
------
    results/interpretability/causal_tracing/<model_key>/results.jsonl
    results/interpretability/causal_tracing/<model_key>/causal_summary.json
    results/interpretability/causal_tracing/<model_key>/fig_causal_heatmap.png
    results/interpretability/causal_tracing/<model_key>/fig_overlap.png
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import (
    load_model, load_stimuli, get_theory_stimuli,
    make_fc_prompt,
    save_result, save_json, RESULTS_DIR, print_vram, clear_cache,
)


# ---------------------------------------------------------------------------
# Core causal tracing primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_logprob_correct(model, prompt: str, correct_token: str = "A") -> float:
    """Return the log-probability of correct_token at the final sequence position."""
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    lp     = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    tok_id = model.to_tokens(correct_token, prepend_bos=False)[0, 0]
    return lp[tok_id].item()


@torch.no_grad()
def get_all_activations(model, prompt: str) -> dict:
    """
    Cache all intermediate activations for a given prompt.

    Captures attention output, MLP output, and residual stream at every layer.
    Used to build the "clean cache" for activation patching.
    """
    n_layers = model.cfg.n_layers
    cache    = {}

    def make_hook(key):
        def hook_fn(value, hook):
            cache[key] = value.clone()
        return hook_fn

    hooks = []
    for i in range(n_layers):
        hooks.append((f"blocks.{i}.hook_attn_out",   make_hook(f"attn_out_{i}")))
        hooks.append((f"blocks.{i}.hook_mlp_out",    make_hook(f"mlp_out_{i}")))
        hooks.append((f"blocks.{i}.hook_resid_post", make_hook(f"resid_post_{i}")))

    model.run_with_hooks(model.to_tokens(prompt), fwd_hooks=hooks)
    return cache


@torch.no_grad()
def patch_single_component(model, corrupted_prompt: str, clean_cache: dict,
                             component: str, layer: int,
                             correct_token: str = "A") -> float:
    """
    Run the corrupted prompt with one component replaced by its clean value.

    Args:
        model:            HookedTransformer instance.
        corrupted_prompt: The swapped-word prompt.
        clean_cache:      Activation cache from the clean prompt.
        component:        "attn_out" or "mlp_out".
        layer:            Layer index to patch.
        correct_token:    Token to score (default "A").

    Returns:
        Log-probability of correct_token after patching.
    """
    cache_key = f"{component}_{layer}"
    clean_act = clean_cache.get(cache_key)
    if clean_act is None:
        return float("-inf")

    hook_name = f"blocks.{layer}.hook_{component}"

    def patch_hook(value, hook):
        return clean_act.clone()

    tokens = model.to_tokens(corrupted_prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, patch_hook)])
    lp     = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
    tok_id = model.to_tokens(correct_token, prepend_bos=False)[0, 0]
    return lp[tok_id].item()


# ---------------------------------------------------------------------------
# Prompt builders for the two conditions
# ---------------------------------------------------------------------------

def build_phonological_prompts(word_round: str, word_sharp: str) -> tuple:
    """
    Clean: round-phonology word presented as option A (expected answer A).
    Corrupted: sharp-phonology word presented as option A.
    """
    clean     = make_fc_prompt(word_round, word_sharp, "round and soft")
    corrupted = make_fc_prompt(word_sharp, word_round, "round and soft")
    return clean, corrupted


def build_factual_recall_prompts(word_round: str, word_sharp: str) -> tuple:
    """
    Clean: famous pair in the correct order (round first).
    Corrupted: famous pair swapped (sharp first).

    Mirrors a factual recall task where the model knows which is which from training.
    """
    clean = (
        f"In the bouba-kiki effect experiment, two made-up words are used: "
        f"'{word_round}' and '{word_sharp}'. "
        f"Which word do people typically associate with a rounded shape? "
        f"Answer A (for '{word_round}') or B (for '{word_sharp}')."
    )
    corrupted = (
        f"In the bouba-kiki effect experiment, two made-up words are used: "
        f"'{word_sharp}' and '{word_round}'. "
        f"Which word do people typically associate with a rounded shape? "
        f"Answer A (for '{word_sharp}') or B (for '{word_round}')."
    )
    return clean, corrupted


# ---------------------------------------------------------------------------
# Full causal tracing over a set of pairs
# ---------------------------------------------------------------------------

def run_causal_tracing_for_pairs(model, pairs: list, prompt_builder,
                                   condition_name: str, model_key: str,
                                   correct_token: str = "A") -> dict:
    """
    Run causal tracing for a list of (word_a, word_b) pairs.

    For each pair:
      1. Compute clean and corrupted baselines.
      2. For each (layer, component), patch and measure Normalised IE.

    Returns:
        Dict {layer: {component: mean_NIE}} aggregated over all pairs.
    """
    n_layers      = model.cfg.n_layers
    ie_accumulator: dict = defaultdict(lambda: defaultdict(list))
    COMPONENTS    = ["attn_out", "mlp_out"]

    for word_a, word_b in tqdm(pairs, desc=f"  causal_tracing_{condition_name}"):
        clean_prompt, corrupted_prompt = prompt_builder(word_a, word_b)

        try:
            clean_cache = get_all_activations(model, clean_prompt)
            P_clean     = get_logprob_correct(model, clean_prompt,     correct_token)
            P_corrupt   = get_logprob_correct(model, corrupted_prompt, correct_token)
        except Exception as e:
            print(f"    Warning: baseline failed for {word_a}/{word_b}: {e}")
            continue

        total_effect = P_clean - P_corrupt
        if abs(total_effect) < 0.01:
            continue

        for layer in range(n_layers):
            for comp in COMPONENTS:
                try:
                    P_patched = patch_single_component(
                        model, corrupted_prompt, clean_cache, comp, layer, correct_token
                    )
                    nie = (P_patched - P_corrupt) / total_effect
                    ie_accumulator[layer][comp].append(nie)
                except Exception:
                    pass

        save_result("causal_tracing", model_key, {
            "model": model_key, "condition": condition_name,
            "word_a": word_a, "word_b": word_b,
            "P_clean": float(P_clean), "P_corrupt": float(P_corrupt),
            "total_effect": float(total_effect),
        })

    return {
        layer: {
            comp: float(np.mean(vals)) if vals else 0.0
            for comp, vals in comp_dict.items()
        }
        for layer, comp_dict in ie_accumulator.items()
    }


# ---------------------------------------------------------------------------
# Overlap analysis
# ---------------------------------------------------------------------------

def compute_overlap(ie_phonological: dict, ie_factual: dict, top_k: int = 10) -> dict:
    """
    Compare the top-K components by NIE across the two conditions.

    Overlap coefficient = |P ∩ F| / min(|P|, |F|).
    A value above 0.5 is interpreted as evidence of a shared memorisation circuit.
    """
    def get_top_components(ie_dict, k):
        all_comps = [
            ((layer, comp), val)
            for layer, comp_dict in ie_dict.items()
            for comp, val in comp_dict.items()
        ]
        all_comps.sort(key=lambda x: x[1], reverse=True)
        return set(x[0] for x in all_comps[:k])

    top_p = get_top_components(ie_phonological, top_k)
    top_f = get_top_components(ie_factual,      top_k)
    intersection = top_p & top_f
    denom = min(len(top_p), len(top_f))
    overlap = len(intersection) / denom if denom > 0 else 0

    return {
        "overlap_coefficient": float(overlap),
        "intersection_size": len(intersection),
        "top_k": top_k,
        "top_phonological": [list(x) for x in top_p],
        "top_factual":      [list(x) for x in top_f],
        "shared_components": [list(x) for x in intersection],
        "interpretation": (
            "HIGH OVERLAP: contamination/memorisation may drive phonological task"
            if overlap > 0.5 else
            "LOW OVERLAP: distinct circuits — supports genuine phonological encoding"
        ),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_causal_heatmap(ie_phonological: dict, ie_factual: dict,
                         model_key: str, n_layers: int):
    """Side-by-side heatmaps of normalised IE per component and condition."""
    fig, axes = plt.subplots(1, 2, figsize=(16, max(n_layers * 0.3, 6)))
    COMPONENTS = ["attn_out", "mlp_out"]

    for ax, (title, ie_dict) in zip(axes, [
        ("Phonological-Semantic Task (T3)", ie_phonological),
        ("Factual Recall Task (T1 Famous)", ie_factual),
    ]):
        matrix = np.zeros((n_layers, len(COMPONENTS)))
        for layer in range(n_layers):
            layer_data = ie_dict.get(layer, {})
            for j, comp in enumerate(COMPONENTS):
                matrix[layer, j] = layer_data.get(comp, 0.0)

        im = ax.imshow(matrix, aspect="auto", cmap="hot",
                       vmin=0, vmax=matrix.max() if matrix.max() > 0 else 1)
        ax.set_xticks(range(len(COMPONENTS)))
        ax.set_xticklabels(["Attention", "MLP"], fontsize=10)
        ax.set_ylabel("Layer" if ax == axes[0] else "")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(i) for i in range(n_layers)], fontsize=7)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, label="Normalised IE")

    fig.suptitle(f"Causal Tracing — {model_key}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = RESULTS_DIR / "causal_tracing" / model_key / "fig_causal_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_overlap(overlap_results: dict, ie_p: dict, ie_f: dict,
                  model_key: str, n_layers: int):
    """
    Scatter plot with each component positioned by its NIE under each condition.

    Components that appear in the top-K intersection are highlighted in red.
    """
    all_comps = {(layer, comp)
                 for layer in range(n_layers)
                 for comp in ["attn_out", "mlp_out"]}
    shared = {tuple(x) for x in overlap_results.get("shared_components", [])}

    x_vals, y_vals, colors = [], [], []
    for (layer, comp) in all_comps:
        x_vals.append(ie_p.get(layer, {}).get(comp, 0.0))
        y_vals.append(ie_f.get(layer, {}).get(comp, 0.0))
        colors.append("red" if (layer, comp) in shared else "steelblue")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=40)
    ax.set_xlabel("Normalised IE — Phonological Task", fontsize=11)
    ax.set_ylabel("Normalised IE — Factual Recall Task", fontsize=11)
    oc = overlap_results["overlap_coefficient"]
    ax.set_title(
        f"Component Overlap — {model_key}\n"
        f"Overlap coeff = {oc:.3f} "
        f"(shared={overlap_results['intersection_size']}/{overlap_results['top_k']})",
        fontsize=10,
    )
    lim = max(max(x_vals, default=0.1), max(y_vals, default=0.1)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="red",       label="Shared (top-K both)"),
        Patch(color="steelblue", label="Not shared"),
    ], fontsize=9)

    out = RESULTS_DIR / "causal_tracing" / model_key / "fig_overlap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_causal_tracing(model_key: str, n_stimuli: int = 20):
    """Run the full causal tracing experiment for a given model."""
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = load_model(model_key, device)
    n_layers = model.cfg.n_layers
    print_vram(device)

    stimuli = load_stimuli()

    # Condition A: novel phonological pairs (Tier 3)
    t3_ss    = get_theory_stimuli(stimuli, "sound_symbolism", tier=3)[:n_stimuli]
    t3_pairs = [(s["word_a"], s["word_b"]) for s in t3_ss]
    print(f"\nCondition A: Phonological-semantic (Tier 3, n={len(t3_pairs)})")
    ie_phonological = run_causal_tracing_for_pairs(
        model, t3_pairs, build_phonological_prompts,
        "phonological_semantic_t3", model_key,
    )

    # Condition B: famous pairs as factual recall (Tier 1)
    t1_ss    = get_theory_stimuli(stimuli, "sound_symbolism", tier=1)[:n_stimuli]
    t1_pairs = [(s["word_a"], s["word_b"]) for s in t1_ss]
    print(f"\nCondition B: Factual recall (Tier 1, n={len(t1_pairs)})")
    ie_factual = run_causal_tracing_for_pairs(
        model, t1_pairs, build_factual_recall_prompts,
        "factual_recall_t1", model_key,
    )

    print("\nComputing overlap...")
    overlap = compute_overlap(ie_phonological, ie_factual, top_k=10)
    print(f"  Overlap coefficient: {overlap['overlap_coefficient']:.3f}")
    print(f"  Shared components:   {overlap['intersection_size']}/10")
    print(f"  {overlap['interpretation']}")

    summary = {
        "model": model_key,
        "n_stimuli_per_condition": n_stimuli,
        "overlap": overlap,
        "ie_phonological": {str(l): d for l, d in ie_phonological.items()},
        "ie_factual":      {str(l): d for l, d in ie_factual.items()},
    }
    save_json("causal_tracing", model_key, "causal_summary.json", summary)

    plot_causal_heatmap(ie_phonological, ie_factual, model_key, n_layers)
    plot_overlap(overlap, ie_phonological, ie_factual, model_key, n_layers)

    del model
    clear_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal tracing experiment to compare phonological and factual circuits."
    )
    parser.add_argument("--model", default="gemma2_9b",
                        choices=["gemma2_9b", "pythia_1.4b", "both"])
    parser.add_argument("--n_stimuli", type=int, default=20,
                        help="Number of stimuli per condition. "
                             "Use 10 for a faster run, 20 for the full experiment.")
    args = parser.parse_args()

    if args.model == "both":
        for mk in ["gemma2_9b", "pythia_1.4b"]:
            run_causal_tracing(mk, args.n_stimuli)
    else:
        run_causal_tracing(args.model, args.n_stimuli)
