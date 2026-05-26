"""
interpretability/analyze_interpretability.py

Aggregate all interpretability experiment outputs into paper-ready figures
and LaTeX tables.

Run after all five interpretability experiments have completed:
    python -m interpretability.analyze_interpretability

Output
------
    results/interpretability/analysis/
      fig_combined_interpretability.png   (main paper figure, 4 panels)
      table_probing.tex
      table_causal_overlap.tex
      interpretability_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interpretability.utils import RESULTS_DIR

ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma2_9b", "pythia_1.4b"]
MODEL_DISPLAY = {"gemma2_9b": "Gemma-2 9B", "pythia_1.4b": "Pythia 1.4B"}

THEORIES = ["sound_symbolism", "vowel_size", "phonesthesia"]
THEORY_DISPLAY = {
    "sound_symbolism": "Sound Symbolism",
    "vowel_size":      "Vowel-Size Sym.",
    "phonesthesia":    "Phonesthesia",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    """Load a JSON file, returning an empty dict if the file does not exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_jsonl(path: Path) -> list:
    """Load all records from a JSONL file."""
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    return rows


# ---------------------------------------------------------------------------
# Combined 4-panel figure
# ---------------------------------------------------------------------------

def make_combined_figure():
    """
    Four-panel figure summarising all interpretability experiments:
      (A) Layer-wise probing accuracy — Gemma-2 9B
      (B) Logit lens semantic consistency — Gemma-2 9B
      (C) Phonestheme onset attention — Gemma-2 9B
      (D) Causal tracing component overlap scatter — Gemma-2 9B
    """
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    theory_colors = {
        "sound_symbolism": "#2196F3",
        "vowel_size":      "#4CAF50",
        "phonesthesia":    "#F44336",
    }

    # ── Panel A: Linear probing ───────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    probe_data = load_json(RESULTS_DIR / "probing" / "gemma2_9b" / "probe_summary.json")
    line_styles = {"tier1": "-", "tier3": "--", "cross_tier": ":"}

    for theory in THEORIES:
        if theory not in probe_data:
            continue
        t_data = probe_data[theory]
        color  = theory_colors[theory]
        for tier_key, ls in line_styles.items():
            if tier_key not in t_data:
                continue
            ld     = t_data[tier_key]
            layers = sorted(int(k) for k in ld.keys())
            accs   = [ld[str(l)]["accuracy"] for l in layers]
            label  = (f"{THEORY_DISPLAY[theory]} ({tier_key.replace('_', ' ')})"
                      if tier_key == "tier3" else None)
            ax_a.plot(layers, accs, color=color, linestyle=ls,
                      linewidth=1.5, label=label, alpha=0.8)

    ax_a.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax_a.set_xlabel("Layer", fontsize=11)
    ax_a.set_ylabel("Probe Accuracy", fontsize=11)
    ax_a.set_title("(A) Linear Probing — Gemma-2 9B", fontsize=12, fontweight="bold")
    ax_a.set_ylim(0.35, 1.05)
    ax_a.legend(fontsize=8, loc="lower right")

    # ── Panel B: Logit lens ───────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ll_data = load_json(RESULTS_DIR / "logit_lens" / "gemma2_9b" / "logit_lens_summary.json")

    for theory in THEORIES:
        if theory not in ll_data:
            continue
        t_data  = ll_data[theory]
        color   = theory_colors[theory]
        correct = t_data.get("correct", {})
        wrong   = t_data.get("wrong",   {})
        lc = sorted(int(k) for k in correct.keys())
        lw = sorted(int(k) for k in wrong.keys())
        ax_b.plot(lc, [correct[str(l)] for l in lc], color=color,
                  linewidth=2, label=f"{THEORY_DISPLAY[theory]} (correct)")
        ax_b.plot(lw, [wrong[str(l)] for l in lw], color=color,
                  linewidth=1.5, linestyle="--", alpha=0.5)

    ax_b.set_xlabel("Layer", fontsize=11)
    ax_b.set_ylabel("Semantic Consistency Score", fontsize=11)
    ax_b.set_title("(B) Logit Lens — Gemma-2 9B", fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=8)

    # ── Panel C: Attention analysis ───────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    attn_data = load_json(RESULTS_DIR / "attention" / "gemma2_9b" / "layer_results.json")
    cluster_colors = {"gl": "#2196F3", "sl": "#F44336", "fl": "#FF9800"}

    for cluster, t_data in attn_data.items():
        if cluster not in cluster_colors:
            continue
        color = cluster_colors[cluster]
        ph    = t_data.get("phonestheme", {})
        ct    = t_data.get("control",     {})
        lph   = sorted(int(k) for k in ph.keys())
        ax_c.plot(lph, [ph[str(l)] for l in lph], color=color,
                  linewidth=2, label=f"{cluster}- (onset)")
        ax_c.plot(lph, [ct.get(str(l), 0) for l in lph], color=color,
                  linewidth=1.5, linestyle="--", alpha=0.5)

    ax_c.set_xlabel("Layer", fontsize=11)
    ax_c.set_ylabel("Mean Onset Attention Score", fontsize=11)
    ax_c.set_title("(C) Phonestheme Onset Attention — Gemma-2 9B",
                   fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=8)

    # ── Panel D: Causal tracing overlap ──────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ct_summary = load_json(
        RESULTS_DIR / "causal_tracing" / "gemma2_9b" / "causal_summary.json"
    )

    if ct_summary:
        ie_p    = ct_summary.get("ie_phonological", {})
        ie_f    = ct_summary.get("ie_factual",      {})
        overlap = ct_summary.get("overlap",         {})
        shared  = {tuple(x) for x in overlap.get("shared_components", [])}

        x_vals, y_vals, point_colors = [], [], []
        for l_str, comp_dict in ie_p.items():
            l = int(l_str)
            for comp, val_p in comp_dict.items():
                val_f = ie_f.get(l_str, {}).get(comp, 0.0)
                x_vals.append(val_p)
                y_vals.append(val_f)
                point_colors.append("red" if (l, comp) in shared else "steelblue")

        ax_d.scatter(x_vals, y_vals, c=point_colors, alpha=0.6, s=30)
        lim = max(max(x_vals, default=0.1), max(y_vals, default=0.1)) * 1.15
        ax_d.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)
        ax_d.set_xlim(0, lim); ax_d.set_ylim(0, lim)
        ax_d.set_xlabel("Normalised IE — Phonological Task", fontsize=11)
        ax_d.set_ylabel("Normalised IE — Factual Recall", fontsize=11)
        oc = overlap.get("overlap_coefficient", 0)
        ax_d.set_title(
            f"(D) Causal Tracing: Component Overlap — Gemma-2 9B\n"
            f"Overlap coeff = {oc:.3f}",
            fontsize=12, fontweight="bold",
        )
        from matplotlib.patches import Patch
        ax_d.legend(handles=[
            Patch(color="red",       label="Shared top-10"),
            Patch(color="steelblue", label="Not shared"),
        ], fontsize=9)
    else:
        ax_d.text(0.5, 0.5, "Causal tracing\nnot yet complete",
                  ha="center", va="center", transform=ax_d.transAxes,
                  fontsize=12, color="gray")

    fig.suptitle(
        "Mechanistic Interpretability: Phonological-Semantic Encoding in LLMs",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out = ANALYSIS_DIR / "fig_combined_interpretability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def make_probing_table():
    """
    Table: peak layer-wise probe accuracy and peak-layer depth per
    (model, theory, tier/condition).
    """
    rows = []
    for model_key in MODELS:
        probe_data = load_json(RESULTS_DIR / "probing" / model_key / "probe_summary.json")
        for theory in THEORIES:
            if theory not in probe_data:
                continue
            t_data = probe_data[theory]
            row = {"Model": MODEL_DISPLAY[model_key], "Theory": THEORY_DISPLAY[theory]}
            for tier_key in ["tier1", "tier3", "cross_tier"]:
                if tier_key not in t_data:
                    row[tier_key] = "--"
                    continue
                ld     = t_data[tier_key]
                layers = sorted(int(k) for k in ld.keys())
                accs   = [ld[str(l)]["accuracy"] for l in layers]
                peak_l = layers[int(np.argmax(accs))]
                row[tier_key] = f"{max(accs):.3f} (L{peak_l})"
            rows.append(row)

    if not rows:
        print("  No probing data available yet.")
        return

    cols       = ["Model", "Theory", "tier1", "tier3", "cross_tier"]
    col_labels = ["Model", "Theory", "T1 Peak (layer)", "T3 Peak (layer)", "T1→T3 Transfer"]

    latex = [
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Linear probe peak accuracy and peak layer by model, theory, and tier. "
        "T1→T3 Transfer trains on Tier 1, tests on Tier 3.}",
        "\\label{tab:probing}",
        "\\begin{tabular}{llccc}", "\\toprule",
        " & ".join(col_labels) + " \\\\", "\\midrule",
    ]
    for row in rows:
        latex.append(" & ".join(str(row.get(c, "--")) for c in cols) + " \\\\")
    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = ANALYSIS_DIR / "table_probing.tex"
    with open(out, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out}")


def make_causal_table():
    """Table: causal tracing overlap coefficient per model."""
    rows = []
    for model_key in MODELS:
        ct_summary = load_json(
            RESULTS_DIR / "causal_tracing" / model_key / "causal_summary.json"
        )
        if not ct_summary:
            continue
        overlap = ct_summary.get("overlap", {})
        rows.append({
            "Model":            MODEL_DISPLAY[model_key],
            "Overlap coeff":    f"{overlap.get('overlap_coefficient', 0):.3f}",
            "Shared (top-10)":  str(overlap.get("intersection_size", 0)),
            "Interpretation":   overlap.get("interpretation", "")[:60],
        })

    if not rows:
        print("  No causal tracing data available yet.")
        return

    latex = [
        "\\begin{table}[ht]", "\\centering",
        "\\caption{Causal tracing overlap between phonological-semantic task components "
        "and factual recall task components.}",
        "\\label{tab:causal}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Model & Overlap Coeff & Shared (top-10) & Interpretation \\\\", "\\midrule",
    ]
    for row in rows:
        latex.append(
            f"{row['Model']} & {row['Overlap coeff']} & "
            f"{row['Shared (top-10)']} & {row['Interpretation']} \\\\"
        )
    latex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out = ANALYSIS_DIR / "table_causal_overlap.tex"
    with open(out, "w") as f:
        f.write("\n".join(latex))
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary():
    """Print a consolidated summary of all interpretability experiment results."""
    print("\n" + "=" * 65)
    print("INTERPRETABILITY EXPERIMENTS SUMMARY")
    print("=" * 65)

    for model_key in MODELS:
        print(f"\n--- {MODEL_DISPLAY[model_key]} ---")

        probe = load_json(RESULTS_DIR / "probing" / model_key / "probe_summary.json")
        if probe:
            print("  Probing (T3 peak / cross-tier transfer):")
            for theory in THEORIES:
                if theory not in probe:
                    continue
                t = probe[theory]
                for tier_key, label in [("tier3", "T3"), ("cross_tier", "CT")]:
                    if tier_key not in t:
                        continue
                    ld     = t[tier_key]
                    layers = sorted(int(k) for k in ld.keys())
                    accs   = [ld[str(l)]["accuracy"] for l in layers]
                    pk     = layers[int(np.argmax(accs))]
                    print(f"    {THEORY_DISPLAY[theory]:22s} {label}: "
                          f"{max(accs):.3f} (L{pk})", end="")
                print()

        ct = load_json(RESULTS_DIR / "causal_tracing" / model_key / "causal_summary.json")
        if ct:
            ov = ct.get("overlap", {})
            print(f"  Causal overlap: {ov.get('overlap_coefficient', 0):.3f} "
                  f"({ov.get('intersection_size', 0)}/10 shared)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating interpretability analysis outputs...")
    make_combined_figure()
    make_probing_table()
    make_causal_table()
    print_summary()
    print(f"\nAll outputs saved to {ANALYSIS_DIR}/")
