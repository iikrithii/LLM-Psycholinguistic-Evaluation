"""
analyze_results.py

Aggregates all experiment results, runs statistical tests, and generates:
  - LaTeX-ready tables for inclusion in a paper
  - Publication-quality figures (PNG, 300 DPI)
  - Cross-tier degradation analysis (primary contamination result)
  - Cross-linguistic consistency analysis
  - Effect-size estimates with bootstrap confidence intervals

Usage
-----
    python analyze_results.py
    python analyze_results.py --output_format csv
"""

import json
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

RESULTS_DIR  = Path("results")
ANALYSIS_DIR = Path("results/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "gpt":   "openai/gpt-oss-120b",
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}

MODEL_DISPLAY = {
    "openai/gpt-oss-120b":     "GPT-OSS-120B",
    "llama-3.3-70b-versatile": "Llama-3.3-70B",
    "qwen/qwen3-32b":          "Qwen3-32B",
}

THEORY_DISPLAY = {
    "sound_symbolism":            "Sound Symbolism",
    "phonesthesia":               "Phonesthesia",
    "vowel_size":                 "Vowel-Size Sym.",
    "semantic_prosody":           "Semantic Prosody",
    "ideophone_compositionality": "Ideophone Comp.",
}

LANG_DISPLAY = {
    "en": "English", "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi",   "de": "German",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_behavioral_results() -> pd.DataFrame:
    """Load all behavioral JSONL files into a single DataFrame."""
    rows = []
    behavioral_dir = RESULTS_DIR / "behavioral"
    if not behavioral_dir.exists():
        return pd.DataFrame()

    for model_dir in behavioral_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_short = model_dir.name
        for theory_dir in model_dir.iterdir():
            if not theory_dir.is_dir():
                continue
            theory = theory_dir.name
            results_file = theory_dir / "results.jsonl"
            if not results_file.exists():
                continue
            with open(results_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        rows.append({
                            "model_short": model_short,
                            "model": r.get("model", ""),
                            "theory": theory,
                            "tier": r.get("tier"),
                            "language": r.get("language", "en"),
                            "fc_accuracy": (
                                r.get("fc", {}).get("accuracy")
                                if "fc" in r else r.get("fc_accuracy")
                            ),
                            "rating_diff": (
                                r.get("rating", {}).get("diff_b_minus_a")
                                if "rating" in r else r.get("rating_diff")
                            ),
                            "expected": r.get("expected", "A"),
                        })
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(rows)


def load_multilingual_results() -> pd.DataFrame:
    """Load cross-linguistic sound symbolism results into a DataFrame."""
    rows = []
    ml_dir = RESULTS_DIR / "multilingual"
    if not ml_dir.exists():
        return pd.DataFrame()

    for lang_file in ml_dir.glob("*_results.jsonl"):
        lang = lang_file.stem.replace("_results", "")
        with open(lang_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("theory") == "sound_symbolism":
                        rows.append({
                            "model": r.get("model", ""),
                            "language": lang,
                            "tier": r.get("tier"),
                            "fc_accuracy": r.get("fc_accuracy"),
                            "rating_diff": r.get("rating_diff"),
                        })
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def load_contamination_results() -> pd.DataFrame:
    """Load contamination probe records into a DataFrame."""
    rows = []
    cont_dir = RESULTS_DIR / "contamination"
    if not cont_dir.exists():
        return pd.DataFrame()

    for f in cont_dir.glob("*_contamination.jsonl"):
        theory = f.stem.replace("_contamination", "")
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    rows.append({
                        "theory": theory,
                        "model": r.get("model", ""),
                        "probe_type": r.get("probe_type", ""),
                        "tier": r.get("tier", None),
                        "response": r.get("response", ""),
                        "recognized": r.get("recognized", None),
                        "hedge_count": r.get("hedge_count", None),
                    })
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def cohen_d(x1: list, x2: list) -> float:
    """Cohen's d for two independent groups using pooled standard deviation."""
    if not x1 or not x2:
        return float("nan")
    n1, n2 = len(x1), len(x2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(x1) ** 2 + (n2 - 1) * np.std(x2) ** 2) / (n1 + n2 - 2)
    )
    return float("nan") if pooled_std == 0 else (np.mean(x1) - np.mean(x2)) / pooled_std


def bootstrap_ci(data: list, stat_fn, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Bootstrap percentile confidence interval for a given statistic function."""
    if not data:
        return (float("nan"), float("nan"))
    boot_stats = [stat_fn(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def compute_tier_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean FC accuracy per (theory, model, tier) with bootstrap CIs
    and binomial tests against 50% chance.

    Returns a DataFrame with one row per (theory, model, tier) combination.
    """
    rows = []
    for (theory, model), grp in df.groupby(["theory", "model"]):
        for tier in [1, 2, 3]:
            tier_grp = grp[grp["tier"] == tier]["fc_accuracy"].dropna()
            if len(tier_grp) == 0:
                continue
            vals = tier_grp.tolist()
            mean_acc = np.mean(vals)
            ci_lo, ci_hi = bootstrap_ci(vals, np.mean)
            n_correct = int(round(mean_acc * len(vals)))
            binom = binomtest(n_correct, len(vals), p=0.5, alternative="greater")
            rows.append({
                "theory": theory,
                "model": model,
                "tier": tier,
                "mean_accuracy": float(mean_acc),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n": len(vals),
                "p_vs_chance": float(binom.pvalue),
                "above_chance": binom.pvalue < 0.05,
            })
    return pd.DataFrame(rows)


def compute_contamination_score(cont_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate contamination evidence per (theory, model) from passage recognition
    and hedging language in confidence comparison probes.
    """
    if cont_df.empty:
        return pd.DataFrame()

    rows = []
    for (theory, model), grp in cont_df.groupby(["theory", "model"]):
        recognized = grp[grp["probe_type"] == "passage_recognition"]["recognized"]
        tier1_recog = (
            recognized[grp["tier"] == 1].mean() if len(recognized) > 0 else float("nan")
        )
        hedges = grp[grp["probe_type"] == "confidence_comparison"]["hedge_count"]
        t1_hedge = hedges[grp["tier"] == 1].mean() if len(hedges) > 0 else float("nan")
        t3_hedge = hedges[grp["tier"] == 3].mean() if len(hedges) > 0 else float("nan")

        rows.append({
            "theory": theory,
            "model": model,
            "tier1_recognition": float(tier1_recog) if not np.isnan(tier1_recog) else None,
            "t1_hedge_count": float(t1_hedge) if not np.isnan(t1_hedge) else None,
            "t3_hedge_count": float(t3_hedge) if not np.isnan(t3_hedge) else None,
            "hedge_delta_t3_minus_t1": (
                float(t3_hedge - t1_hedge)
                if not (np.isnan(t1_hedge) or np.isnan(t3_hedge)) else None
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_tier_degradation(tier_df: pd.DataFrame):
    """
    Figure 1: FC accuracy by tier, theory, and model.

    Three-panel plot (one panel per model) with one line per theory.
    Shaded regions show 95% bootstrap confidence intervals.
    """
    if tier_df.empty:
        print("  No tier degradation data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = sns.color_palette("husl", n_colors=5)
    theories = list(THEORY_DISPLAY.keys())

    for ax_idx, (short, full_model) in enumerate(MODEL_NAMES.items()):
        ax = axes[ax_idx]
        model_df = tier_df[tier_df["model"] == full_model]

        for t_idx, theory in enumerate(theories):
            t_df = model_df[model_df["theory"] == theory].sort_values("tier")
            if t_df.empty:
                continue
            tiers = t_df["tier"].values
            accs  = t_df["mean_accuracy"].values
            ci_lo = t_df["ci_lo"].values
            ci_hi = t_df["ci_hi"].values
            ax.plot(tiers, accs, marker="o", color=colors[t_idx],
                    label=THEORY_DISPLAY.get(theory, theory), linewidth=2)
            ax.fill_between(tiers, ci_lo, ci_hi, alpha=0.15, color=colors[t_idx])

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                   label="Chance (0.5)")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["T1\n(Famous)", "T2\n(Obscure)", "T3\n(Novel)"])
        ax.set_ylim(0.3, 1.05)
        ax.set_xlabel("Stimulus Tier", fontsize=12)
        ax.set_title(MODEL_DISPLAY.get(full_model, full_model), fontsize=13)
        if ax_idx == 0:
            ax.set_ylabel("Forced Choice Accuracy", fontsize=12)
            ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("FC Accuracy by Tier: Contamination Degradation Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = ANALYSIS_DIR / "fig1_tier_degradation.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_cross_lingual(ml_df: pd.DataFrame):
    """Figure 2: Cross-linguistic FC accuracy per language and tier."""
    if ml_df.empty:
        print("  No multilingual data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, tier in enumerate([1, 3]):
        ax = axes[ax_idx]
        tier_df = ml_df[ml_df["tier"] == tier]
        if tier_df.empty:
            continue

        lang_model = (tier_df
                      .groupby(["language", "model"])["fc_accuracy"]
                      .mean()
                      .reset_index())
        langs  = [l for l in ["en", "ja", "ko", "hi", "de"]
                  if l in lang_model["language"].values]
        x      = np.arange(len(langs))
        models = lang_model["model"].unique()
        w      = 0.25

        for m_idx, model in enumerate(models):
            m_df = lang_model[lang_model["model"] == model]
            accs = [m_df[m_df["language"] == l]["fc_accuracy"].values[0]
                    if l in m_df["language"].values else float("nan")
                    for l in langs]
            ax.bar(x + m_idx * w, accs, width=w,
                   label=MODEL_DISPLAY.get(model, model), alpha=0.85)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x + w)
        ax.set_xticklabels([LANG_DISPLAY.get(l, l) for l in langs])
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("FC Accuracy" if ax_idx == 0 else "")
        ax.set_title(f"Tier {tier} — {'Famous' if tier == 1 else 'Novel'} Stimuli")
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Cross-Linguistic Sound Symbolism: FC Accuracy by Language and Tier",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = ANALYSIS_DIR / "fig2_cross_lingual.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_theory_heatmap(tier_df: pd.DataFrame):
    """
    Figure 3: Heatmap of Tier 1 and Tier 3 accuracy and their gap
    (contamination effect) per theory × model.
    """
    if tier_df.empty:
        return

    pivot_t1 = (tier_df[tier_df["tier"] == 1]
                .pivot_table(values="mean_accuracy",
                             index="theory", columns="model", aggfunc="mean"))
    pivot_t3 = (tier_df[tier_df["tier"] == 3]
                .pivot_table(values="mean_accuracy",
                             index="theory", columns="model", aggfunc="mean"))
    gap = pivot_t1 - pivot_t3

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dfs    = [pivot_t1, pivot_t3, gap]
    titles = ["Tier 1 Accuracy (Famous)", "Tier 3 Accuracy (Novel)",
              "T1−T3 Gap (Contamination Effect)"]
    cmaps  = ["YlGn", "YlOrRd", "RdYlGn_r"]

    for ax, df, title, cmap in zip(axes, dfs, titles, cmaps):
        if df.empty:
            continue
        df = df.rename(columns=MODEL_DISPLAY, index=THEORY_DISPLAY)
        sns.heatmap(df, ax=ax, cmap=cmap, vmin=0.4, vmax=1.0,
                    annot=True, fmt=".2f", linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    fig.suptitle("Accuracy and Contamination Gap by Theory and Model",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = ANALYSIS_DIR / "fig3_theory_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def generate_main_results_table(tier_df: pd.DataFrame, output_format: str = "latex"):
    """
    Table 1: FC accuracy per (theory, model, tier) with Bonferroni-corrected
    binomial test significance stars and 95% bootstrap CIs.
    """
    if tier_df.empty:
        print("  No data for main results table.")
        return

    tier_df = tier_df.dropna(subset=["p_vs_chance"]).copy()
    if tier_df.empty:
        return

    _, pvals_corrected, _, _ = multipletests(tier_df["p_vs_chance"].values,
                                              method="bonferroni")
    tier_df["p_corrected"] = pvals_corrected

    def sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    tier_df["acc_str"] = tier_df.apply(
        lambda r: (f"{r['mean_accuracy']:.2f}{sig_star(r['p_corrected'])} "
                   f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"),
        axis=1,
    )
    pivot = tier_df.pivot_table(
        values="acc_str", index=["theory", "model"], columns="tier", aggfunc="first"
    )

    if output_format == "latex":
        latex = pivot.to_latex(
            escape=False,
            caption=(
                "Forced-choice accuracy by theory, model, and stimulus tier. "
                "Values are mean accuracy [95\\% CI]. "
                "Significance against chance (50\\%) after Bonferroni correction: "
                "*p<.05, **p<.01, ***p<.001."
            ),
            label="tab:main_results",
        )
        out = ANALYSIS_DIR / "table1_main_results.tex"
        with open(out, "w") as f:
            f.write(latex)
    else:
        out = ANALYSIS_DIR / "table1_main_results.csv"
        pivot.to_csv(out)
    print(f"  Saved {out}")


def generate_degradation_table(tier_df: pd.DataFrame, output_format: str = "latex"):
    """
    Table 2: Tier 1 vs Tier 3 accuracy gap per (theory, model) with
    a qualitative contamination label (High / Medium / Low).
    """
    if tier_df.empty:
        return

    rows = []
    for (theory, model), grp in tier_df.groupby(["theory", "model"]):
        t1 = grp[grp["tier"] == 1]["mean_accuracy"].values
        t3 = grp[grp["tier"] == 3]["mean_accuracy"].values
        if len(t1) == 0 or len(t3) == 0:
            continue
        gap = float(t1[0] - t3[0])
        rows.append({
            "Theory": THEORY_DISPLAY.get(theory, theory),
            "Model": MODEL_DISPLAY.get(model, model),
            "T1": f"{t1[0]:.2f}",
            "T3": f"{t3[0]:.2f}",
            "Gap (T1-T3)": f"{gap:.2f}",
            "Contamination": "High" if gap > 0.15 else ("Medium" if gap > 0.05 else "Low"),
        })

    df = pd.DataFrame(rows)
    if output_format == "latex":
        latex = df.to_latex(
            index=False, escape=True,
            caption=(
                "Tier 1 vs Tier 3 accuracy gap per theory and model. "
                "Large gaps indicate contamination-driven performance."
            ),
            label="tab:degradation",
        )
        out = ANALYSIS_DIR / "table2_degradation.tex"
        with open(out, "w") as f:
            f.write(latex)
    else:
        out = ANALYSIS_DIR / "table2_degradation.csv"
        df.to_csv(out, index=False)
    print(f"  Saved {out}")


def generate_multilingual_table(ml_df: pd.DataFrame, output_format: str = "latex"):
    """Table 3: Cross-linguistic FC accuracy per (language, tier, model)."""
    if ml_df.empty:
        print("  No multilingual data for table.")
        return

    summary = (ml_df.groupby(["language", "tier", "model"])["fc_accuracy"]
               .agg(["mean", "count"])
               .reset_index())
    summary.columns = ["language", "tier", "model", "mean_acc", "n"]
    summary["display"] = summary.apply(
        lambda r: f"{r['mean_acc']:.2f} (n={r['n']})", axis=1
    )
    pivot = summary.pivot_table(
        values="display", index="language", columns=["tier", "model"],
        aggfunc="first",
    )

    if output_format == "latex":
        out = ANALYSIS_DIR / "table3_multilingual.tex"
        pivot.to_latex(
            out, escape=False,
            caption=(
                "Cross-linguistic forced-choice accuracy for sound symbolism "
                "by language and stimulus tier."
            ),
            label="tab:multilingual",
        )
    else:
        pivot.to_csv(ANALYSIS_DIR / "table3_multilingual.csv")
    print("  Saved multilingual table.")


# ---------------------------------------------------------------------------
# Contamination correlation
# ---------------------------------------------------------------------------

def analyze_contamination_correlation(tier_df: pd.DataFrame,
                                       cont_df: pd.DataFrame) -> dict:
    """
    Spearman correlation between contamination evidence scores and
    (a) Tier 1 accuracy and (b) the Tier 1 minus Tier 3 accuracy gap.
    """
    if tier_df.empty or cont_df.empty:
        return {}

    cont_scores = compute_contamination_score(cont_df)
    if cont_scores.empty:
        return {}

    t1_acc = tier_df[tier_df["tier"] == 1][["theory", "model", "mean_accuracy"]]
    t3_acc = (tier_df[tier_df["tier"] == 3][["theory", "model", "mean_accuracy"]]
              .rename(columns={"mean_accuracy": "acc_t3"}))
    merged = t1_acc.merge(t3_acc, on=["theory", "model"], how="inner")
    merged["gap"] = merged["mean_accuracy"] - merged["acc_t3"]
    merged = merged.merge(cont_scores, on=["theory", "model"], how="inner")

    results = {}
    for target_col, result_key in [
        ("mean_accuracy", "recognition_vs_t1_accuracy"),
        ("gap",           "recognition_vs_gap"),
    ]:
        valid = merged.dropna(subset=["tier1_recognition", target_col])
        if len(valid) >= 5:
            rho, pval = spearmanr(valid["tier1_recognition"], valid[target_col])
            results[result_key] = {
                "spearman_rho": float(rho), "p": float(pval), "n": len(valid)
            }

    out = ANALYSIS_DIR / "contamination_correlation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved contamination correlation to {out}")
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def generate_summary_stats(tier_df: pd.DataFrame, ml_df: pd.DataFrame):
    """Write a JSON summary of aggregate statistics across all experiments."""
    summary = {
        "n_models": len(MODEL_NAMES),
        "n_theories": 5,
        "n_tiers": 3,
        "n_languages": 5,
    }
    if not tier_df.empty:
        t1_mean = tier_df[tier_df["tier"] == 1]["mean_accuracy"].mean()
        t3_mean = tier_df[tier_df["tier"] == 3]["mean_accuracy"].mean()
        summary["overall_t1_mean"] = float(t1_mean)
        summary["overall_t3_mean"] = float(t3_mean)
        summary["overall_gap"] = float(t1_mean - t3_mean)

        theory_summary = {}
        for theory in tier_df["theory"].unique():
            t_df = tier_df[tier_df["theory"] == theory]
            theory_summary[theory] = {
                "t1_mean": float(t_df[t_df["tier"] == 1]["mean_accuracy"].mean()),
                "t3_mean": float(t_df[t_df["tier"] == 3]["mean_accuracy"].mean()),
            }
        summary["by_theory"] = theory_summary

    out = ANALYSIS_DIR / "summary_stats.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary stats to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results and generate figures and tables."
    )
    parser.add_argument("--output_format", default="latex",
                        choices=["latex", "csv"],
                        help="Output format for tables.")
    args = parser.parse_args()

    print("Loading results...")
    behavioral_df    = load_behavioral_results()
    multilingual_df  = load_multilingual_results()
    contamination_df = load_contamination_results()

    print(f"  Behavioral records:    {len(behavioral_df)}")
    print(f"  Multilingual records:  {len(multilingual_df)}")
    print(f"  Contamination records: {len(contamination_df)}")

    if behavioral_df.empty:
        print("\nNo behavioral results found. Run run_behavioral.py first.")
        return

    print("\nComputing tier degradation...")
    tier_df = compute_tier_degradation(behavioral_df)

    print("\nGenerating figures...")
    plot_tier_degradation(tier_df)
    plot_cross_lingual(multilingual_df)
    plot_theory_heatmap(tier_df)

    print("\nGenerating tables...")
    generate_main_results_table(tier_df, args.output_format)
    generate_degradation_table(tier_df, args.output_format)
    generate_multilingual_table(multilingual_df, args.output_format)

    print("\nContamination correlation analysis...")
    corr_results = analyze_contamination_correlation(tier_df, contamination_df)
    for k, v in corr_results.items():
        print(f"  {k}: ρ={v['spearman_rho']:.3f}, p={v['p']:.3f}")

    print("\nSummary statistics...")
    generate_summary_stats(tier_df, multilingual_df)

    print(f"\nAll outputs saved to {ANALYSIS_DIR}/")
    for f in sorted(ANALYSIS_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
