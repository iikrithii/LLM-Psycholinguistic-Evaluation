"""
interpretability/contamination_trajectory.py

Experiment: Pythia Contamination Trajectory (Pythia-only)
==========================================================
Pythia's pretraining corpus (The Pile) is fully documented and searchable
via the EleutherAI infini-gram API. This allows us to directly estimate
training-data contamination for each stimulus by searching the Pile for
the exact stimulus string.

This analysis is only possible for Pythia — GPT-OSS and Gemma have
undocumented corpora.

Analysis
--------
1. For each stimulus, query The Pile for its exact occurrence count.
2. Correlate contamination frequency with:
   a. Behavioural FC accuracy (using Llama as a proxy, as Pile overlap
      patterns between Pythia and Llama are broadly similar).
   b. Peak-layer probe accuracy from the linear probing experiment.
3. Positive Spearman correlations across both measures provide direct
   evidence that training-data exposure drives model performance.

The Pile search uses the infini-gram API (https://infini-gram.io/).
No API key is required, but the API is rate-limited — a 0.3 s delay is
inserted between requests, and responses are cached to disk.

Run
---
    python -m interpretability.contamination_trajectory

Output
------
    results/interpretability/contamination_trajectory/pile_counts_<index>.json
    results/interpretability/contamination_trajectory/pile_counts_all.json
    results/interpretability/contamination_trajectory/correlation_analysis.json
    results/interpretability/contamination_trajectory/fig_contamination_<theory>.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import (
    load_stimuli, save_json, RESULTS_DIR,
)

TRAJ_DIR = RESULTS_DIR / "contamination_trajectory"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)

INFIGRAM_URL = "https://api.infini-gram.io/"
PILE_INDEX   = "v4_piletrain_llama"


# ---------------------------------------------------------------------------
# Pile search
# ---------------------------------------------------------------------------

def query_pile_count(text: str, index: str = PILE_INDEX, retries: int = 3) -> int:
    """
    Query the infini-gram API for the exact string count in The Pile.

    Args:
        text:    The string to search for.
        index:   Infini-gram index identifier.
        retries: Number of retry attempts on failure.

    Returns:
        Occurrence count, or -1 on error.
    """
    payload = {"index": index, "query_type": "count", "query": text}
    for attempt in range(retries):
        try:
            resp = requests.post(INFIGRAM_URL, json=payload, timeout=15)
            if resp.status_code == 200:
                return int(resp.json().get("count", 0))
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                time.sleep(2)
        except Exception as e:
            print(f"Infini-gram request failed: {e}")
            time.sleep(2)
    return -1


def query_pile_count_with_context(word: str, theory: str) -> dict:
    """
    Query the word alone and in a theory-descriptive context phrase.

    Co-occurrence with theory-relevant vocabulary is a stronger
    contamination signal than bare word occurrence.

    Returns:
        Dict with "count_direct", "count_context", and "contamination_score".
    """
    count_direct = query_pile_count(word)
    time.sleep(0.3)

    context_query_map = {
        "sound_symbolism": f"{word} sound symbolism",
        "phonesthesia":    f"{word} phonestheme",
        "vowel_size":      f"{word} vowel size symbolism",
    }
    context_query = context_query_map.get(theory, word)
    count_context = query_pile_count(context_query)
    time.sleep(0.3)

    return {
        "word": word,
        "count_direct": count_direct,
        "count_context": count_context,
        "contamination_score": (
            count_direct + count_context * 10
            if (count_direct >= 0 and count_context >= 0) else -1
        ),
    }


# ---------------------------------------------------------------------------
# Load existing results
# ---------------------------------------------------------------------------

def load_behavioral_accuracy(theory: str) -> dict:
    """
    Load Llama behavioural FC accuracy results and return {word: accuracy}.

    Llama is used as a proxy for Pythia since both models were trained on
    corpora with similar contamination characteristics for famous stimuli.
    """
    results_path = (RESULTS_DIR.parents[0] / "behavioral" / "llama"
                    / theory / "results.jsonl")
    word_acc = {}
    if not results_path.exists():
        return word_acc
    with open(results_path) as f:
        for line in f:
            try:
                r   = json.loads(line)
                fc  = r.get("fc", {})
                acc = fc.get("accuracy") if isinstance(fc, dict) else r.get("fc_accuracy")
                for wk in ["word_a", "word", "word_b"]:
                    w = r.get(wk, "")
                    if w and acc is not None:
                        word_acc[w] = float(acc)
            except Exception:
                pass
    return word_acc


def load_probe_accuracy(theory: str, model_key: str = "pythia_1.4b") -> dict:
    """
    Load peak-layer probe accuracy per word from the probing experiment.

    Returns:
        Dict mapping word string to peak-layer probe accuracy.
    """
    results_path = RESULTS_DIR / "probing" / model_key / "results.jsonl"
    word_probe = {}
    if not results_path.exists():
        return word_probe
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("theory") == theory:
                    w   = r.get("word", "")
                    acc = r.get("peak_layer_probe_acc")
                    if w and acc is not None:
                        word_probe[w] = float(acc)
            except Exception:
                pass
    return word_probe


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def correlate_contamination_with_performance(pile_counts: dict,
                                              word_acc: dict,
                                              word_probe: dict,
                                              theory: str) -> dict:
    """
    Compute Spearman correlations between contamination score and both
    behavioural accuracy and peak-layer probe accuracy.

    Returns:
        Dict with nested "behavioral_correlation" and "probe_correlation" entries.
    """
    results = {"theory": theory}

    def valid_words(ref_dict):
        return [w for w in pile_counts
                if w in ref_dict
                and pile_counts[w]["contamination_score"] >= 0]

    beh_words = valid_words(word_acc)
    if len(beh_words) >= 5:
        cont = [pile_counts[w]["contamination_score"] for w in beh_words]
        beh  = [word_acc[w] for w in beh_words]
        rho, p = spearmanr(cont, beh)
        results["behavioral_correlation"] = {
            "spearman_rho": float(rho), "p_value": float(p), "n": len(beh_words)
        }

    probe_words = valid_words(word_probe)
    if len(probe_words) >= 5:
        cont  = [pile_counts[w]["contamination_score"] for w in probe_words]
        probe = [word_probe[w] for w in probe_words]
        rho, p = spearmanr(cont, probe)
        results["probe_correlation"] = {
            "spearman_rho": float(rho), "p_value": float(p), "n": len(probe_words)
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_contamination_correlation(pile_counts: dict, word_acc: dict,
                                    theory: str, stimuli: dict):
    """Scatter plot: log contamination score vs behavioural accuracy by tier."""
    shared = [w for w in pile_counts
              if w in word_acc and pile_counts[w]["contamination_score"] >= 0]
    if len(shared) < 3:
        return

    x = [np.log1p(pile_counts[w]["contamination_score"]) for w in shared]
    y = [word_acc[w] for w in shared]

    stims_by_word = {}
    for s in stimuli.get(theory, []):
        for wk in ["word_a", "word"]:
            if s.get(wk):
                stims_by_word[s[wk]] = s.get("tier", 0)

    tier_colors = {1: "#2196F3", 2: "#FF9800", 3: "#F44336"}
    colors = [tier_colors.get(stims_by_word.get(w, 0), "gray") for w in shared]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=colors, alpha=0.7, s=60)
    for w, xi, yi in zip(shared[:5], x[:5], y[:5]):
        ax.annotate(w, (xi, yi), fontsize=7, xytext=(5, 5),
                    textcoords="offset points")

    ax.set_xlabel("log(1 + Pile contamination score)", fontsize=11)
    ax.set_ylabel("Behavioural FC Accuracy", fontsize=11)
    ax.set_title(f"Contamination vs Performance — {theory}", fontsize=12)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f"Tier {t}")
                       for t, c in tier_colors.items()], fontsize=9)

    out = TRAJ_DIR / f"fig_contamination_{theory}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_contamination_trajectory():
    """Query The Pile and correlate contamination scores with model performance."""
    print("Loading stimuli...")
    stimuli = load_stimuli()

    THEORIES = ["sound_symbolism", "phonesthesia", "vowel_size"]

    all_pile_counts  = {}
    all_correlations = {}

    pile_cache_path = TRAJ_DIR / f"pile_counts_{PILE_INDEX}.json"
    pile_cache: dict = {}
    if pile_cache_path.exists():
        with open(pile_cache_path) as f:
            pile_cache = json.load(f)
        print(f"Loaded {len(pile_cache)} cached Pile counts.")

    for theory in THEORIES:
        print(f"\n{'='*55}\nCONTAMINATION TRAJECTORY: {theory}\n{'='*55}")

        all_stims = stimuli.get(theory, [])
        words = set()
        for s in all_stims:
            for wk in ["word_a", "word"]:
                if s.get(wk):
                    words.add((s[wk], theory, s.get("tier", 0)))

        print(f"  Querying Pile for {len(words)} words...")
        pile_counts = {}

        for word, theory_, tier in tqdm(sorted(words), desc=f"  pile_search_{theory}"):
            cache_key = f"{theory_}|{word}"
            if cache_key in pile_cache:
                pile_counts[word] = pile_cache[cache_key]
                continue
            result = query_pile_count_with_context(word, theory_)
            pile_counts[word] = result
            pile_cache[cache_key] = result
            with open(pile_cache_path, "w") as f:
                json.dump(pile_cache, f, indent=2)

        all_pile_counts[theory] = pile_counts

        for tier in [1, 2, 3]:
            tier_words = [s.get("word_a") or s.get("word")
                          for s in all_stims if s.get("tier") == tier]
            tier_counts = [pile_counts.get(w, {}).get("contamination_score", -1)
                           for w in tier_words if w]
            valid = [c for c in tier_counts if c >= 0]
            if valid:
                print(f"  Tier {tier}: mean={np.mean(valid):.1f} "
                      f"max={max(valid)} n={len(valid)}")

        word_acc   = load_behavioral_accuracy(theory)
        word_probe = load_probe_accuracy(theory)
        corr = correlate_contamination_with_performance(
            pile_counts, word_acc, word_probe, theory
        )
        all_correlations[theory] = corr

        if "behavioral_correlation" in corr:
            bc = corr["behavioral_correlation"]
            print(f"\n  Contamination ↔ Behavioral accuracy: "
                  f"ρ={bc['spearman_rho']:.3f}, p={bc['p_value']:.4f}, n={bc['n']}")
        if "probe_correlation" in corr:
            pc = corr["probe_correlation"]
            print(f"  Contamination ↔ Probe accuracy: "
                  f"ρ={pc['spearman_rho']:.3f}, p={pc['p_value']:.4f}, n={pc['n']}")

        plot_contamination_correlation(pile_counts, word_acc, theory, stimuli)

    save_json("contamination_trajectory", "", "correlation_analysis.json", all_correlations)
    save_json("contamination_trajectory", "", "pile_counts_all.json",      all_pile_counts)

    print(f"\nContamination trajectory complete. Results in {TRAJ_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_contamination_trajectory()
