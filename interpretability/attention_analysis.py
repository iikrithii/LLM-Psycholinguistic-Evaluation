"""
interpretability/attention_analysis.py

Experiment: Attention Pattern Analysis (Phonesthesia)
=====================================================
Tests whether the model preferentially attends to phonesthemic onset clusters
(e.g., "gl-" in "glenan") compared to matched neutral-onset controls
(e.g., "bl-" in "blenan").

Key metric
----------
"Onset attention score" = mean attention weight directed to the onset tokens
of the target word, averaged over all query positions and all heads.

Comparisons
-----------
  - Phonestheme-carrying nonce words vs matched neutral-onset controls
  - Per-layer divergence identifies when phonesthemic sensitivity emerges
  - Per-head analysis identifies "phonestheme detector heads"

Run
---
    python -m interpretability.attention_analysis --model gemma2_9b
    python -m interpretability.attention_analysis --model both

Output
------
    results/interpretability/attention/<model_key>/results.jsonl
    results/interpretability/attention/<model_key>/layer_results.json
    results/interpretability/attention/<model_key>/head_diff_matrix.json
    results/interpretability/attention/<model_key>/fig_attention.png
    results/interpretability/attention/<model_key>/fig_head_heatmap.png
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
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import (
    load_model, load_stimuli, get_theory_stimuli,
    make_fc_prompt, get_word_token_positions,
    save_result, save_json, RESULTS_DIR, print_vram, clear_cache,
)


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_to_onset(model, prompt: str, word: str,
                                 onset_len: int = 2) -> dict:
    """
    Extract the mean attention directed to the onset tokens of a word.

    For each (layer, head) pair, computes the mean attention weight from
    all query positions to the first onset_len tokens of word.

    Args:
        model:      HookedTransformer instance.
        prompt:     Input prompt string.
        word:       Target word whose onset tokens are the key positions.
        onset_len:  Number of onset tokens to use as the key set.

    Returns:
        Dict mapping layer_idx → {head_idx: mean_onset_attention_score}.
    """
    positions = get_word_token_positions(model, prompt, word)
    onset_positions = positions[:onset_len] if len(positions) >= onset_len else positions

    if not onset_positions:
        return {}

    n_layers    = model.cfg.n_layers
    attn_scores = {}

    def make_attn_hook(layer_idx):
        def hook_fn(value, hook):
            # value: (batch, n_heads, seq_len, seq_len)
            attn = value[0]  # (n_heads, seq_len, seq_len)
            attn_to_onset = attn[:, :, onset_positions].sum(dim=-1)  # (n_heads, seq_len)
            mean_onset_attn = attn_to_onset.mean(dim=-1)  # (n_heads,)
            attn_scores[layer_idx] = mean_onset_attn.float().cpu().numpy()
        return hook_fn

    hooks = [(f"blocks.{i}.attn.hook_pattern", make_attn_hook(i))
             for i in range(n_layers)]
    model.run_with_hooks(model.to_tokens(prompt), fwd_hooks=hooks)

    return {
        layer_idx: {head: float(scores[head]) for head in range(len(scores))}
        for layer_idx, scores in attn_scores.items()
    }


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def test_onset_preference(phonestheme_scores: list, control_scores: list) -> dict:
    """
    Wilcoxon signed-rank test: are phonestheme onset scores significantly
    higher than control onset scores (one-tailed, greater)?
    """
    if len(phonestheme_scores) < 5:
        return {"p_value": None, "statistic": None, "n": len(phonestheme_scores)}
    try:
        stat, p = wilcoxon(phonestheme_scores, control_scores, alternative="greater")
        return {
            "p_value": float(p), "statistic": float(stat),
            "n": len(phonestheme_scores),
            "mean_phonestheme": float(np.mean(phonestheme_scores)),
            "mean_control":     float(np.mean(control_scores)),
            "effect_size":      float(np.mean(phonestheme_scores) - np.mean(control_scores)),
        }
    except Exception as e:
        return {"p_value": None, "error": str(e), "n": len(phonestheme_scores)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_curves(layer_results: dict, model_key: str):
    """Layer-wise onset attention score for phonestheme vs control words."""
    clusters = list(layer_results.keys())
    n = len(clusters)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        data      = layer_results[cluster]
        layers_ph = sorted(data["phonestheme"].keys())
        layers_ct = sorted(data["control"].keys())

        ax.plot(layers_ph, [data["phonestheme"][l] for l in layers_ph],
                color="#2196F3", linewidth=2, label=f"{cluster}- (phonestheme)")
        ax.plot(layers_ct, [data["control"][l] for l in layers_ct],
                color="#F44336", linewidth=2, linestyle="--",
                label="neutral onset (control)")

        for sl in data.get("significant_layers_fdr", []):
            ax.axvline(sl, color="green", alpha=0.3, linewidth=1)

        ax.set_title(f"{cluster}- phonestheme", fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Mean Onset Attention Score", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(f"Phonestheme Onset Attention — {model_key}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = RESULTS_DIR / "attention" / model_key / "fig_attention.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_head_heatmap(head_results: dict, model_key: str):
    """
    Heatmap of (layer, head) mean onset attention difference.

    Positive values indicate that the head attends more to the phonesthemic
    onset than to the matched neutral onset — potential phonestheme detector.
    """
    if not head_results:
        return

    n_layers = max(int(l) for l in head_results.keys()) + 1
    first_layer = head_results[list(head_results.keys())[0]]
    n_heads = len(first_layer)

    matrix = np.zeros((n_layers, n_heads))
    for layer_str, head_scores in head_results.items():
        layer = int(layer_str)
        for head_str, score in head_scores.items():
            head = int(head_str)
            if layer < n_layers and head < n_heads:
                matrix[layer, head] = score

    fig, ax = plt.subplots(figsize=(min(n_heads * 0.5 + 2, 20), 8))
    sns.heatmap(matrix, ax=ax, cmap="RdBu_r", center=0,
                xticklabels=[str(h) for h in range(n_heads)],
                yticklabels=[str(l) for l in range(n_layers)],
                cbar_kws={"label": "Attention diff (phonestheme − control)"})
    ax.set_xlabel("Attention Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(f"Phonestheme Detector Heads — {model_key}", fontsize=12)

    out = RESULTS_DIR / "attention" / model_key / "fig_head_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_attention_analysis(model_key: str):
    """Run the phonestheme onset attention experiment for a given model."""
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = load_model(model_key, device)
    n_layers = model.cfg.n_layers
    print_vram(device)

    stimuli  = load_stimuli()
    ph_stims = get_theory_stimuli(stimuli, "phonesthesia", tier=3)
    ph_stims = [s for s in ph_stims if s.get("word_control")]

    if not ph_stims:
        print("No phonesthesia Tier 3 stimuli with control words found.")
        return

    print(f"Phonesthesia Tier 3 stimuli: {len(ph_stims)}")

    by_cluster: dict = defaultdict(list)
    for s in ph_stims:
        by_cluster[s.get("phonestheme", "?")].append(s)

    layer_results    = {}
    head_diff_matrix: dict = defaultdict(dict)

    for cluster, stims in by_cluster.items():
        print(f"\n  Cluster: {cluster}- ({len(stims)} stimuli)")

        ph_scores_by_layer: dict  = defaultdict(list)
        ct_scores_by_layer: dict  = defaultdict(list)
        head_ph_by_layer: dict    = defaultdict(lambda: defaultdict(list))
        head_ct_by_layer: dict    = defaultdict(lambda: defaultdict(list))

        for s in tqdm(stims, desc=f"    {cluster}"):
            word_ph = s["word_a"]
            word_ct = s["word_control"]
            prompt  = make_fc_prompt(word_ph, word_ct, "more related to visual/light")

            try:
                attn_ph = extract_attention_to_onset(model, prompt, word_ph)
                attn_ct = extract_attention_to_onset(model, prompt, word_ct)
            except Exception as e:
                print(f"      Warning: {e}")
                continue

            for layer_idx in attn_ph:
                ph_mean = float(np.mean(list(attn_ph[layer_idx].values())))
                ct_mean = float(np.mean(list(attn_ct[layer_idx].values())))
                ph_scores_by_layer[layer_idx].append(ph_mean)
                ct_scores_by_layer[layer_idx].append(ct_mean)

                for head, score in attn_ph[layer_idx].items():
                    head_ph_by_layer[layer_idx][head].append(score)
                for head, score in attn_ct[layer_idx].items():
                    head_ct_by_layer[layer_idx][head].append(score)

                save_result("attention", model_key, {
                    "model": model_key, "cluster": cluster,
                    "word_phonestheme": word_ph, "word_control": word_ct,
                    "tier": s["tier"], "layer": layer_idx,
                    "mean_attn_phonestheme": ph_mean,
                    "mean_attn_control": ct_mean,
                    "diff": ph_mean - ct_mean,
                })

        layers   = sorted(ph_scores_by_layer.keys())
        ph_means = {l: float(np.mean(ph_scores_by_layer[l])) for l in layers}
        ct_means = {l: float(np.mean(ct_scores_by_layer[l])) for l in layers}

        p_values = []
        for l in layers:
            test = test_onset_preference(ph_scores_by_layer[l], ct_scores_by_layer[l])
            p_values.append(test.get("p_value") or 1.0)

        _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
        sig_fdr         = [layers[i] for i, p in enumerate(p_fdr) if p < 0.05]
        sig_uncorrected = [layers[i] for i, p in enumerate(p_values) if p < 0.05]

        layer_results[cluster] = {
            "phonestheme": ph_means,
            "control": ct_means,
            "significant_layers_fdr": sig_fdr,
            "significant_layers_uncorrected": sig_uncorrected,
            "p_values_raw": {str(layers[i]): float(p_values[i])
                             for i in range(len(layers))},
        }

        for l in layers:
            for head in head_ph_by_layer[l]:
                ph_h = float(np.mean(head_ph_by_layer[l][head]))
                ct_h = float(np.mean(head_ct_by_layer[l].get(head, [0.0])))
                head_diff_matrix[str(l)].setdefault(str(head), []).append(ph_h - ct_h)

        print(f"    Sig layers (FDR p<0.05): {sig_fdr}")
        if ph_means:
            peak_l = max(ph_means, key=ph_means.get)
            print(f"    Peak onset attention: layer {peak_l}, "
                  f"score={ph_means[peak_l]:.4f} vs control {ct_means[peak_l]:.4f}")

    head_mean_diff = {
        l: {h: float(np.mean(diffs)) for h, diffs in heads.items()}
        for l, heads in head_diff_matrix.items()
    }

    save_json("attention", model_key, "layer_results.json",   layer_results)
    save_json("attention", model_key, "head_diff_matrix.json", head_mean_diff)

    plot_attention_curves(layer_results, model_key)
    plot_head_heatmap(head_mean_diff, model_key)

    del model
    clear_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phonestheme onset attention analysis."
    )
    parser.add_argument("--model", default="gemma2_9b",
                        choices=["gemma2_9b", "pythia_1.4b", "both"])
    args = parser.parse_args()

    if args.model == "both":
        for mk in ["gemma2_9b", "pythia_1.4b"]:
            run_attention_analysis(mk)
    else:
        run_attention_analysis(args.model)
