"""
interpretability/probing.py

Experiment: Linear Probing Classifiers
=======================================
For each psycholinguistic theory and stimulus tier, extract residual stream
activations at the critical token position at every transformer layer, then
train a logistic regression probe to classify the theory-consistent semantic
category (e.g., round vs angular, small vs large).

Key questions
-------------
1. At which layer does probe accuracy peak? (Early layers suggest phonological
   encoding; late layers suggest semantic or contextual encoding.)
2. Does a probe trained on Tier 1 (famous) stimuli generalize to Tier 3
   (novel) stimuli? Generalization supports a shared representation;
   failure supports a distinct circuit for memorised associations.
3. Does the peak-layer depth differ between Tier 1 and Tier 3?

Run
---
    python -m interpretability.probing --model gemma2_9b
    python -m interpretability.probing --model pythia_1.4b
    python -m interpretability.probing --model both

Output
------
    results/interpretability/probing/<model_key>/results.jsonl
    results/interpretability/probing/<model_key>/probe_summary.json
    results/interpretability/probing/<model_key>/fig_probing_curves.png
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import (
    load_model, load_stimuli, get_theory_stimuli,
    make_fc_prompt, get_word_token_positions,
    save_result, save_json, RESULTS_DIR, print_vram, clear_cache,
)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_residual_stream(model, prompt: str, critical_word: str) -> np.ndarray:
    """
    Extract residual stream activations at every layer for the last token
    of critical_word in prompt.

    Returns an array of shape (n_layers + 1, d_model) where row 0 is the
    embedding layer and row k is the output of transformer block k − 1.

    We target the final token of the critical word because that position
    has integrated the full phonological form through the attention mechanism.
    """
    positions = get_word_token_positions(model, prompt, critical_word)
    if not positions:
        tokens = model.to_tokens(prompt)
        pos = tokens.shape[1] - 1
    else:
        pos = positions[-1]

    n_layers    = model.cfg.n_layers
    activations = {}

    def make_hook(layer_idx):
        def hook_fn(value, hook):
            activations[layer_idx] = value[0, pos, :].float().cpu().numpy()
        return hook_fn

    hooks = [(f"blocks.{i}.hook_resid_post", make_hook(i)) for i in range(n_layers)]

    def embed_hook(value, hook):
        activations[-1] = value[0, pos, :].float().cpu().numpy()
    hooks.append(("hook_embed", embed_hook))

    tokens = model.to_tokens(prompt)
    model.run_with_hooks(tokens, fwd_hooks=hooks)

    embed_act = activations.get(-1, activations.get(0, np.zeros(model.cfg.d_model)))
    return np.stack([embed_act] + [activations[i] for i in range(n_layers)], axis=0)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_activation_dataset(model, stimuli_list: list, dimension: str,
                               prompt_fn, word_key_a: str = "word_a",
                               word_key_b: str = "word_b",
                               label_key: str = "expected") -> tuple:
    """
    Build a dataset of difference activation vectors for probing.

    For each stimulus pair, compute:
        diff = act(correct_word) − act(incorrect_word)   → label 1
       −diff = act(incorrect_word) − act(correct_word)   → label 0

    Using difference vectors eliminates the position artifact that arises
    because word_a always appears at a fixed position P_a in the prompt.
    The signed difference retains only phonological contrast information.

    Returns:
        X     — (n_pairs * 2, n_layers + 1, d_model) difference vectors
        y     — (n_pairs * 2,) binary labels
        meta  — list of metadata dicts
        n_pairs — number of original stimulus pairs
    """
    X_list, y_list, meta_list = [], [], []

    for s in tqdm(stimuli_list, desc="  extracting activations"):
        wa = s.get(word_key_a, "")
        wb = s.get(word_key_b, "")
        if not wa or not wb:
            continue
        expected = s.get(label_key, "A")
        prompt   = prompt_fn(wa, wb, dimension)

        try:
            act_a = extract_residual_stream(model, prompt, wa)
            act_b = extract_residual_stream(model, prompt, wb)
        except Exception as e:
            print(f"    Warning: activation extraction failed for {wa}/{wb}: {e}")
            continue

        diff = (act_a - act_b) if expected == "A" else (act_b - act_a)
        X_list.append(diff);   y_list.append(1)
        X_list.append(-diff);  y_list.append(0)

        base_meta = {
            "word_correct":   wa if expected == "A" else wb,
            "word_incorrect": wb if expected == "A" else wa,
            "tier": s.get("tier"), "theory": s.get("theory"),
        }
        meta_list.append({**base_meta, "orientation": "positive"})
        meta_list.append({**base_meta, "orientation": "negative"})

    if not X_list:
        return None, None, None, 0

    return np.stack(X_list, axis=0), np.array(y_list), meta_list, len(X_list) // 2


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def make_grouped_masks(n_samples: int, n_pairs: int,
                        train_frac: float = 0.8, seed: int = 42) -> tuple:
    """
    Create pair-grouped train/test masks to prevent data leakage.

    Each pair contributes two samples (indices 2i and 2i+1). Splitting at
    the pair level ensures that neither orientation of a pair appears in
    both train and test, which would inflate probe accuracy.
    """
    rng = np.random.default_rng(seed)
    pair_perm     = rng.permutation(n_pairs)
    n_train_pairs = max(2, int(train_frac * n_pairs))
    train_pairs   = set(pair_perm[:n_train_pairs])
    test_pairs    = set(pair_perm[n_train_pairs:])

    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask  = np.zeros(n_samples, dtype=bool)

    for p in train_pairs:
        if 2 * p + 1 < n_samples:
            train_mask[2 * p] = train_mask[2 * p + 1] = True
    for p in test_pairs:
        if 2 * p + 1 < n_samples:
            test_mask[2 * p] = test_mask[2 * p + 1] = True

    return train_mask, test_mask


# ---------------------------------------------------------------------------
# Layer-wise probing
# ---------------------------------------------------------------------------

def train_layer_probes(X: np.ndarray, y: np.ndarray, n_layers: int,
                        train_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """
    Train an L2-regularised logistic regression probe at each layer.

    PCA is applied first to reduce dimensionality (max 30 components).
    Masks must be pair-grouped to prevent leakage between orientations.

    Returns:
        Dict mapping layer_idx to {"accuracy", "n_train", "n_test", ...}.
    """
    from sklearn.decomposition import PCA

    results = {}
    scaler  = StandardScaler()
    n_train = int(train_mask.sum())
    n_components = min(30, n_train - 1, X.shape[2] - 1)

    if n_components < 2:
        for layer_idx in range(X.shape[1]):
            results[layer_idx] = {"accuracy": 0.5,
                                   "n_train": n_train,
                                   "n_test": int(test_mask.sum())}
        return results

    for layer_idx in range(X.shape[1]):
        X_layer  = X[:, layer_idx, :]
        X_scaled = scaler.fit_transform(X_layer)
        pca      = PCA(n_components=n_components, random_state=42)
        X_pca    = pca.fit_transform(X_scaled)

        X_train, X_test = X_pca[train_mask], X_pca[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            results[layer_idx] = {"accuracy": 0.5, "n_train": n_train,
                                   "n_test": int(test_mask.sum())}
            continue

        clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                  random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[layer_idx] = {
            "accuracy": float(acc),
            "n_train": n_train,
            "n_test": int(test_mask.sum()),
            "pca_components": n_components,
            "pca_var_explained": float(pca.explained_variance_ratio_.sum()),
        }

    return results


def cross_tier_probe(X_t1: np.ndarray, y_t1: np.ndarray,
                      X_t3: np.ndarray, y_t3: np.ndarray,
                      n_layers: int) -> dict:
    """
    Train probe on all Tier 1 samples, evaluate on all Tier 3 samples.

    PCA is fitted on Tier 1 only and applied to Tier 3 without refitting,
    so the probe must generalize the same direction across stimulus sets.

    Accuracy > 0.5 indicates that the phonological direction learned from
    famous stimuli transfers to novel stimuli — evidence of genuine encoding.
    """
    from sklearn.decomposition import PCA

    results   = {}
    scaler    = StandardScaler()
    n_train   = len(y_t1)
    n_components = min(30, n_train - 1, X_t1.shape[2] - 1)

    if n_components < 2:
        for layer_idx in range(X_t1.shape[1]):
            results[layer_idx] = {"accuracy": 0.5}
        return results

    for layer_idx in range(X_t1.shape[1]):
        X1 = X_t1[:, layer_idx, :]
        X3 = X_t3[:, layer_idx, :]
        X1_scaled = scaler.fit_transform(X1)
        pca = PCA(n_components=n_components, random_state=42)
        X1_pca = pca.fit_transform(X1_scaled)
        X3_pca = pca.transform(scaler.transform(X3))

        if len(np.unique(y_t1)) < 2:
            results[layer_idx] = {"accuracy": 0.5}
            continue

        clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs",
                                  random_state=42)
        clf.fit(X1_pca, y_t1)
        results[layer_idx] = {"accuracy": float(accuracy_score(y_t3, clf.predict(X3_pca)))}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probing_curves(all_results: dict, model_key: str):
    """Layer-wise probe accuracy curves for each theory and tier."""
    theories = list(all_results.keys())
    n = len(theories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    colors = {"tier1": "#2196F3", "tier2": "#FF9800",
              "tier3": "#F44336", "cross_tier": "#4CAF50"}

    for ax, theory in zip(axes, theories):
        t_data = all_results[theory]
        for key, color in colors.items():
            if key not in t_data:
                continue
            ld     = t_data[key]
            layers = sorted(int(k) for k in ld.keys())
            accs   = [ld[str(l)]["accuracy"] for l in layers]
            label  = {"tier1": "T1 (Famous)", "tier2": "T2 (Obscure)",
                      "tier3": "T3 (Novel)", "cross_tier": "T1→T3 transfer"}[key]
            ax.plot(layers, accs, color=color, linewidth=2, label=label,
                    linestyle="--" if key == "cross_tier" else "-")

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(theory.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylim(0.4, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("Probe Accuracy", fontsize=10)
            ax.legend(fontsize=8)

    fig.suptitle(f"Layer-wise Probing: {model_key}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = RESULTS_DIR / "probing" / model_key / "fig_probing_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_probing(model_key: str):
    """Run the full linear probing experiment for a given model."""
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    model    = load_model(model_key, device)
    n_layers = model.cfg.n_layers
    print_vram(device)

    stimuli = load_stimuli()

    THEORY_CONFIGS = [
        ("sound_symbolism", "word_a", "word_b", "round/soft"),
        ("vowel_size",      "word_a", "word_b", "small/light"),
    ]

    all_summary = {}

    for theory, wk_a, wk_b, dim in THEORY_CONFIGS:
        print(f"\n{'='*55}\nPROBING: {theory} | model: {model_key}\n{'='*55}")
        theory_results = {}
        datasets = {}

        for tier in [1, 2, 3]:
            stims = get_theory_stimuli(stimuli, theory, tier=tier)
            if not stims:
                continue
            print(f"\n  Building Tier {tier} dataset ({len(stims)} stimuli)...")
            result = build_activation_dataset(model, stims, dim, make_fc_prompt, wk_a, wk_b)
            if result[0] is None:
                continue
            X, y, meta, n_pairs = result
            datasets[tier] = (X, y, meta, n_pairs)
            print(f"    {n_pairs} pairs → {len(y)} samples")

        for tier, (X, y, meta, n_pairs) in datasets.items():
            print(f"\n  Within-tier probe (T{tier})...")
            train_mask, test_mask = make_grouped_masks(len(y), n_pairs)
            layer_results = train_layer_probes(X, y, n_layers, train_mask, test_mask)
            theory_results[f"tier{tier}"] = {str(k): v for k, v in layer_results.items()}

            peak_layer = max(layer_results, key=lambda k: layer_results[k]["accuracy"])
            peak_acc   = layer_results[peak_layer]["accuracy"]
            print(f"    Peak layer: {peak_layer} | Peak acc: {peak_acc:.3f}")

            for m in meta:
                save_result("probing", model_key, {
                    "model": model_key, "theory": theory, "tier": tier,
                    "word_correct": m.get("word_correct", ""),
                    "word_incorrect": m.get("word_incorrect", ""),
                    "peak_layer_probe_acc": peak_acc,
                })

        if 1 in datasets and 3 in datasets:
            print(f"\n  Cross-tier generalization (T1 train → T3 test)...")
            X_t1, y_t1, _, _ = datasets[1]
            X_t3, y_t3, _, _ = datasets[3]
            cross_results = cross_tier_probe(X_t1, y_t1, X_t3, y_t3, n_layers)
            theory_results["cross_tier"] = {str(k): v for k, v in cross_results.items()}
            peak = max(cross_results, key=lambda k: cross_results[k]["accuracy"])
            print(f"    Cross-tier peak: layer {peak}, acc={cross_results[peak]['accuracy']:.3f}")

        all_summary[theory] = theory_results

    # Phonesthesia: pair gl- words against sl- words at each tier
    print(f"\n{'='*55}\nPROBING: phonesthesia | model: {model_key}\n{'='*55}")
    ph_all_stims = get_theory_stimuli(stimuli, "phonesthesia")
    ph_summary   = {}

    for tier in [1, 2]:
        gl_stims = [s for s in ph_all_stims if s.get("tier") == tier and s.get("phonestheme") == "gl"]
        sl_stims = [s for s in ph_all_stims if s.get("tier") == tier and s.get("phonestheme") == "sl"]
        n_pairs_ph = min(len(gl_stims), len(sl_stims), 5)
        if n_pairs_ph < 3:
            continue

        X_list, y_list = [], []
        for i in range(n_pairs_ph):
            w_gl = gl_stims[i].get("word_a", "")
            w_sl = sl_stims[i].get("word_a", "")
            if not w_gl or not w_sl:
                continue
            prompt_gl = f"The word '{w_gl}' relates to a specific sensory domain."
            prompt_sl = f"The word '{w_sl}' relates to a specific sensory domain."
            try:
                act_gl = extract_residual_stream(model, prompt_gl, w_gl)
                act_sl = extract_residual_stream(model, prompt_sl, w_sl)
                diff = act_gl - act_sl
                X_list.append(diff);   y_list.append(1)
                X_list.append(-diff);  y_list.append(0)
            except Exception as e:
                print(f"    Warning: {e}")

        if len(X_list) < 4:
            continue
        X = np.stack(X_list)
        y = np.array(y_list)
        n_p = len(X_list) // 2
        train_mask, test_mask = make_grouped_masks(len(y), n_p)
        layer_results = train_layer_probes(X, y, n_layers, train_mask, test_mask)
        ph_summary[f"tier{tier}"] = {str(k): v for k, v in layer_results.items()}
        peak = max(layer_results, key=lambda k: layer_results[k]["accuracy"])
        print(f"  T{tier}: peak_layer={peak}  "
              f"peak_acc={layer_results[peak]['accuracy']:.3f}  (n_pairs={n_p})")

    # T3 phonesthesia: phonestheme nonce vs neutral-onset control
    t3_ph = [s for s in ph_all_stims if s.get("tier") == 3 and s.get("word_control")]
    if t3_ph:
        X_list, y_list = [], []
        for s in tqdm(t3_ph, desc="  ph T3"):
            w_ph = s.get("word_a", "")
            w_ct = s.get("word_control", "")
            if not w_ph or not w_ct:
                continue
            prompt = make_fc_prompt(w_ph, w_ct, "related to light/shine")
            try:
                act_ph = extract_residual_stream(model, prompt, w_ph)
                act_ct = extract_residual_stream(model, prompt, w_ct)
                diff = act_ph - act_ct
                X_list.append(diff);   y_list.append(1)
                X_list.append(-diff);  y_list.append(0)
            except Exception as e:
                print(f"    Warning: {e}")

        if len(X_list) >= 4:
            X = np.stack(X_list)
            y = np.array(y_list)
            n_p = len(X_list) // 2
            train_mask, test_mask = make_grouped_masks(len(y), n_p)
            layer_results = train_layer_probes(X, y, n_layers, train_mask, test_mask)
            ph_summary["tier3"] = {str(k): v for k, v in layer_results.items()}
            peak = max(layer_results, key=lambda k: layer_results[k]["accuracy"])
            print(f"  T3: peak_layer={peak}  acc={layer_results[peak]['accuracy']:.3f}")

    all_summary["phonesthesia"] = ph_summary
    save_json("probing", model_key, "probe_summary.json", all_summary)
    plot_probing_curves(all_summary, model_key)

    # Print summary table
    print(f"\n{'='*55}\nPROBING SUMMARY — {model_key}\n{'='*55}")
    for theory, t_data in all_summary.items():
        print(f"\n{theory}:")
        for key in ["tier1", "tier2", "tier3", "cross_tier"]:
            if key not in t_data:
                continue
            ld     = t_data[key]
            layers = sorted(int(k) for k in ld.keys())
            accs   = [ld[str(l)]["accuracy"] for l in layers]
            peak   = layers[int(np.argmax(accs))]
            print(f"  {key:12s}: peak_layer={peak:2d}  peak_acc={max(accs):.3f}  "
                  f"final_acc={accs[-1]:.3f}")

    del model
    clear_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probing experiment for phonological-semantic representations."
    )
    parser.add_argument("--model", default="gemma2_9b",
                        choices=["gemma2_9b", "pythia_1.4b", "both"])
    args = parser.parse_args()

    if args.model == "both":
        for mk in ["gemma2_9b", "pythia_1.4b"]:
            run_probing(mk)
    else:
        run_probing(args.model)
