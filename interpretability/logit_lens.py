"""
interpretability/logit_lens.py

Experiment: Logit Lens
======================
At each transformer layer, project the residual stream back to vocabulary space
using the model's unembedding matrix. This reveals the model's "working hypothesis"
about a word's meaning as it propagates through the layers.

For Tier 3 novel stimuli (most diagnostic — the model cannot have memorised these):
  - Do intermediate vocabulary projections include words from the theory-consistent
    semantic cluster?
  - For round-phonology nonce words, do middle layers show "round", "soft", "smooth"
    in their top-K predictions?
  - For gl- phonestheme nonce words, do intermediate layers show "glow", "gleam"?

Key metric
----------
"Semantic consistency score" = proportion of the top-K predicted tokens at each
layer that belong to the theory-consistent semantic category (checked by
case-insensitive prefix matching against a curated word set).

Run
---
    python -m interpretability.logit_lens --model gemma2_9b
    python -m interpretability.logit_lens --model pythia_1.4b --topk 30

Output
------
    results/interpretability/logit_lens/<model_key>/results.jsonl
    results/interpretability/logit_lens/<model_key>/logit_lens_summary.json
    results/interpretability/logit_lens/<model_key>/fig_logit_lens.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import (
    load_model, load_stimuli, get_theory_stimuli,
    get_word_token_positions,
    save_result, save_json, RESULTS_DIR, print_vram, clear_cache,
)


# ---------------------------------------------------------------------------
# Semantic category word sets
# ---------------------------------------------------------------------------

SEMANTIC_CLUSTERS = {
    "sound_symbolism_round": {
        "round", "soft", "smooth", "gentle", "blob", "oval", "circle",
        "fluffy", "bubbly", "plump", "curved", "warm", "cozy", "mellow",
        "mild", "tender", "bouncy", "rotund", "spherical", "globe", "ball",
        "dome", "curve", "bend", "arc",
    },
    "sound_symbolism_sharp": {
        "sharp", "angular", "spiky", "jagged", "pointed", "hard", "rigid",
        "knife", "edge", "spike", "needle", "thorn", "claw", "prick",
        "piercing", "acute", "fierce", "harsh", "rough", "abrupt",
        "brittle", "crisp", "snag", "stab",
    },
    "vowel_size_small": {
        "small", "tiny", "little", "mini", "slim", "thin", "narrow", "light",
        "slight", "fine", "petite", "compact", "minuscule", "brief", "short",
        "frail", "slender", "micro", "quick", "swift", "bright", "high",
    },
    "vowel_size_large": {
        "large", "big", "huge", "heavy", "wide", "broad", "massive", "great",
        "vast", "deep", "thick", "dark", "slow", "dull", "dense", "fat",
        "solid", "strong", "loud", "grand", "giant", "bulk", "weighty",
    },
    "phonesthesia_gl": {
        "glow", "gleam", "glitter", "glint", "glimmer", "glare", "light",
        "shine", "bright", "luminous", "radiant", "flash", "sparkle",
        "shimmer", "twinkle", "glisten", "gloss", "illuminate", "beam", "ray",
    },
    "phonesthesia_sl": {
        "slime", "sludge", "dirty", "gross", "nasty", "unpleasant", "ugly",
        "bad", "awful", "horrible", "disgusting", "repulsive", "slimy",
        "slobber", "slovenly", "sluggish", "sleazy",
    },
    "phonesthesia_sn": {
        "sniff", "snort", "nose", "breath", "inhale", "exhale", "nasal",
        "snore", "sneeze", "smell", "aroma", "snicker", "snivel", "snuffle",
        "nostril",
    },
    "phonesthesia_fl": {
        "flash", "flicker", "flutter", "fly", "flit", "fast", "quick",
        "rapid", "swift", "speed", "dash", "rush", "movement", "motion",
        "flap", "flee", "flow",
    },
}


# ---------------------------------------------------------------------------
# Logit lens computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_logit_lens(model, prompt: str, word: str, top_k: int = 20) -> dict:
    """
    Project the residual stream at word's last token position back to vocabulary
    at every transformer layer.

    Args:
        model:   HookedTransformer instance.
        prompt:  Input prompt string.
        word:    The word whose token position is the probe target.
        top_k:   Number of top vocabulary items to retain per layer.

    Returns:
        Dict mapping layer_idx to {"top_tokens": [...], "top_logprobs": [...]}.
    """
    positions = get_word_token_positions(model, prompt, word)
    if not positions:
        tokens = model.to_tokens(prompt)
        pos = tokens.shape[1] - 1
    else:
        pos = positions[-1]

    n_layers   = model.cfg.n_layers
    layer_acts = {}

    def make_hook(layer_idx):
        def hook_fn(value, hook):
            layer_acts[layer_idx] = value[0, pos, :].float()
        return hook_fn

    hooks = [(f"blocks.{i}.hook_resid_post", make_hook(i)) for i in range(n_layers)]

    def embed_hook(value, hook):
        layer_acts[-1] = value[0, pos, :].float()
    hooks.append(("hook_embed", embed_hook))

    tokens = model.to_tokens(prompt)
    model.run_with_hooks(tokens, fwd_hooks=hooks)

    W_U_f = model.W_U.float()
    b_U_f = model.b_U.float() if model.b_U is not None else None
    results = {}

    for layer_idx in sorted(layer_acts.keys()):
        act    = layer_acts[layer_idx]
        logits = act @ W_U_f
        if b_U_f is not None:
            logits = logits + b_U_f
        log_probs = torch.log_softmax(logits, dim=-1)
        top_lp, top_idx = torch.topk(log_probs, top_k)
        top_tokens   = [model.tokenizer.decode([i.item()]).strip() for i in top_idx]
        top_logprobs = top_lp.cpu().tolist()
        results[layer_idx] = {"top_tokens": top_tokens, "top_logprobs": top_logprobs}

    return results


def compute_semantic_consistency(top_tokens: list, category_words: set) -> float:
    """
    Proportion of top_tokens that belong to a semantic category.

    Matching is case-insensitive and uses a 4-character prefix to handle
    common morphological variants (e.g., "glowing" matches "glow").
    """
    count = 0
    for tok in top_tokens:
        tok_lower = tok.lower().strip()
        if any(tok_lower == cw or tok_lower.startswith(cw[:4])
               for cw in category_words if len(cw) >= 4):
            count += 1
    return count / len(top_tokens) if top_tokens else 0.0


# ---------------------------------------------------------------------------
# Per-theory logit lens runner
# ---------------------------------------------------------------------------

def run_logit_lens_theory(model, stimuli: list, theory: str,
                            cluster_correct: str, cluster_wrong: str,
                            word_key: str, top_k: int, model_key: str) -> tuple:
    """
    Compute layer-wise semantic consistency scores for a single theory.

    Runs logit lens on each stimulus's correct-category word and (if present)
    its incorrect-category word, accumulating mean scores per layer.

    Returns:
        Tuple of (mean_correct_by_layer, mean_wrong_by_layer) dicts.
    """
    n_layers = model.cfg.n_layers
    correct_scores = {i: [] for i in range(-1, n_layers)}
    wrong_scores   = {i: [] for i in range(-1, n_layers)}

    cat_correct = SEMANTIC_CLUSTERS.get(cluster_correct, set())
    cat_wrong   = SEMANTIC_CLUSTERS.get(cluster_wrong, set())

    for s in tqdm(stimuli, desc=f"  logit_lens {theory}"):
        word_a = s.get("word_a", s.get("word", ""))
        word_b = s.get("word_b", "")

        if word_a:
            prompt = f"What does the invented word '{word_a}' bring to mind?"
            try:
                ll_a = compute_logit_lens(model, prompt, word_a, top_k)
            except Exception as e:
                print(f"    Warning: {e}")
                continue

            for layer_idx, ld in ll_a.items():
                correct_scores[layer_idx].append(
                    compute_semantic_consistency(ld["top_tokens"], cat_correct)
                )

            save_result("logit_lens", model_key, {
                "model": model_key, "theory": theory,
                "tier": s.get("tier"), "word": word_a,
                "label": "correct_category",
                "layer_semantic_scores": {
                    str(k): {
                        "semantic_consistency_correct": compute_semantic_consistency(
                            v["top_tokens"], cat_correct),
                        "top5_tokens": v["top_tokens"][:5],
                    }
                    for k, v in ll_a.items()
                },
            })

        if word_b:
            prompt_b = f"What does the invented word '{word_b}' bring to mind?"
            try:
                ll_b = compute_logit_lens(model, prompt_b, word_b, top_k)
            except Exception:
                continue
            for layer_idx, ld in ll_b.items():
                wrong_scores[layer_idx].append(
                    compute_semantic_consistency(ld["top_tokens"], cat_correct)
                )

    mean_correct = {i: np.mean(v) if v else 0.0
                    for i, v in correct_scores.items() if i >= 0}
    mean_wrong   = {i: np.mean(v) if v else 0.0
                    for i, v in wrong_scores.items() if i >= 0}
    return mean_correct, mean_wrong


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_logit_lens(all_results: dict, model_key: str):
    """Layer-wise semantic consistency curves for each theory."""
    theories = list(all_results.keys())
    n = len(theories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, theory in zip(axes, theories):
        data = all_results[theory]
        mc   = data["correct"]
        mw   = data["wrong"]
        lc   = sorted(int(k) for k in mc.keys())
        lw   = sorted(int(k) for k in mw.keys())

        ax.plot(lc, [mc[str(k)] for k in lc], color="#2196F3",
                linewidth=2, label="Correct-category word")
        ax.plot(lw, [mw[str(k)] for k in lw], color="#F44336",
                linewidth=2, linestyle="--", label="Incorrect-category word")

        ax.set_title(theory.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylim(0, 0.25)
        if ax == axes[0]:
            ax.set_ylabel("Semantic Consistency Score", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(f"Logit Lens: Semantic Consistency — {model_key}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = RESULTS_DIR / "logit_lens" / model_key / "fig_logit_lens.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_logit_lens(model_key: str, top_k: int = 20):
    """Run the logit lens experiment for a given model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(model_key, device)
    print_vram(device)

    stimuli = load_stimuli()

    CONFIGS = [
        ("sound_symbolism", 3, "sound_symbolism_round", "sound_symbolism_sharp", "word_a"),
        ("vowel_size",      3, "vowel_size_small",      "vowel_size_large",      "word_a"),
        ("phonesthesia",    3, "phonesthesia_gl",       "phonesthesia_sl",       "word_a"),
    ]

    all_results = {}

    for theory, tier, cluster_c, cluster_w, wk in CONFIGS:
        print(f"\n{'='*55}\nLOGIT LENS: {theory} Tier {tier} | model: {model_key}\n{'='*55}")

        stims = get_theory_stimuli(stimuli, theory, tier=tier)
        if not stims:
            print("  No stimuli, skipping.")
            continue

        if theory == "phonesthesia":
            all_ph_correct: dict = {}
            all_ph_wrong: dict   = {}
            cluster_map = {
                "gl": ("phonesthesia_gl", "phonesthesia_sl"),
                "sl": ("phonesthesia_sl", "phonesthesia_gl"),
                "fl": ("phonesthesia_fl", "phonesthesia_sl"),
                "sn": ("phonesthesia_sn", "phonesthesia_sl"),
            }
            for cluster, (cc, cw) in cluster_map.items():
                cl_stims = [s for s in stims if s.get("phonestheme") == cluster]
                if not cl_stims:
                    continue
                mc, mw = run_logit_lens_theory(
                    model, cl_stims, f"{theory}_{cluster}", cc, cw, wk, top_k, model_key
                )
                for k in mc:
                    all_ph_correct.setdefault(k, []).append(mc[k])
                    all_ph_wrong.setdefault(k, []).append(mw[k])

            mean_c = {k: float(np.mean(v)) for k, v in all_ph_correct.items()}
            mean_w = {k: float(np.mean(v)) for k, v in all_ph_wrong.items()}
        else:
            mean_c, mean_w = run_logit_lens_theory(
                model, stims, theory, cluster_c, cluster_w, wk, top_k, model_key
            )

        all_results[theory] = {
            "correct": {str(k): float(v) for k, v in mean_c.items()},
            "wrong":   {str(k): float(v) for k, v in mean_w.items()},
        }

        layers = sorted(int(k) for k in mean_c.keys())
        scores = [mean_c[k] for k in layers]
        peak   = layers[int(np.argmax(scores))]
        print(f"  Peak semantic consistency: layer {peak}, score={max(scores):.4f}")

    save_json("logit_lens", model_key, "logit_lens_summary.json", all_results)
    plot_logit_lens(all_results, model_key)

    del model
    clear_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logit lens semantic consistency experiment."
    )
    parser.add_argument("--model", default="gemma2_9b",
                        choices=["gemma2_9b", "pythia_1.4b", "both"])
    parser.add_argument("--topk", type=int, default=20,
                        help="Number of top vocabulary tokens to inspect per layer.")
    args = parser.parse_args()

    if args.model == "both":
        for mk in ["gemma2_9b", "pythia_1.4b"]:
            run_logit_lens(mk, args.topk)
    else:
        run_logit_lens(args.model, args.topk)
