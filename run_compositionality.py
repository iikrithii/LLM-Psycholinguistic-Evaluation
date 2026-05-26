"""
run_compositionality.py

Full ideophone compositionality experiment.

Tests whether language models encode COMPOSITIONAL phonological-to-semantic
mapping rules rather than retrieving holistically memorised pairings.

Experimental design
-------------------
A 2×2×2×2 factorial over binary phonological features:
  voiced × geminated × reduplicated × back_vowel

16 cells × 3 novel Tier 3 stimuli per cell = 48 items in total.

After rating collection a linear regression model is fitted to predict each
semantic dimension rating from the four binary features. Compositionality
predicts that (a) main effects are significant, and (b) interaction terms
add little variance beyond main effects (additivity).

An additional cross-language transfer test administers the same rating task
to Korean Tier 1 ideophones to assess whether Japanese-learned feature-to-
meaning mappings transfer to a typologically related ideophonic system.

Usage
-----
    python run_compositionality.py --models all
    python run_compositionality.py --models gpt --skip_ratings
    python run_compositionality.py --analyze_only
"""

import json
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from api_client import call_model, parallel_call, parse_rating_response, parse_ab_response
from prompts.templates import IDEOPHONE_RATING_TEMPLATES, IDEOPHONE_FC_TEMPLATES

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results/compositionality")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "gpt":   "openai/gpt-oss-120b",
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}

DIMENSIONS = [
    "heavy_vs_light",
    "dark_vs_bright",
    "rough_vs_smooth",
    "sudden_vs_gradual",
    "continuous_vs_sudden",
]

DIM_LABELS = {
    "heavy_vs_light":       ("light", "heavy"),
    "dark_vs_bright":       ("bright", "dark"),
    "rough_vs_smooth":      ("smooth", "rough"),
    "sudden_vs_gradual":    ("gradual", "sudden"),
    "continuous_vs_sudden": ("sudden", "continuous"),
}

# Four templates for the compositionality experiment to improve reliability.
N_TEMPLATES = 4


def load_stimuli() -> list:
    with open(DATA_DIR / "stimuli.json", encoding="utf-8") as f:
        data = json.load(f)
    return data["stimuli"]["ideophone_compositionality"]


def save_result(record: dict):
    out_file = RESULTS_DIR / "compositionality_results.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_analysis(analysis: dict):
    out_file = RESULTS_DIR / "compositionality_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)


# ---------------------------------------------------------------------------
# Collect ratings for all Tier 3 novel ideophones
# ---------------------------------------------------------------------------

def run_compositionality_ratings(stimuli: list, model: str, model_short: str):
    """
    Rate all 48 novel Tier 3 ideophones on five semantic dimensions.

    Also rates Tier 1 famous gitaigo as a calibration set to verify that the
    model's ratings align with known human associations for those items.
    """
    tier3 = [s for s in stimuli if s["tier"] == 3 and s["language"] == "ja"]
    print(f"\n[{model_short}] Compositionality ratings: {len(tier3)} novel ideophones")

    for s in tqdm(tier3, desc="compositionality"):
        word = s["word"]
        cell_label = (
            f"v{int(s['voiced'])}g{int(s['geminated'])}"
            f"r{int(s['reduplicated'])}bv{int(s['back_vowel'])}"
        )

        specs = [
            {"model": model,
             "prompt": IDEOPHONE_RATING_TEMPLATES[t % len(IDEOPHONE_RATING_TEMPLATES)](
                 word, "Japanese", DIM_LABELS[dim][0], DIM_LABELS[dim][1]),
             "max_tokens": 5, "temperature": 0.0}
            for dim in DIMENSIONS
            for t in range(N_TEMPLATES)
        ]
        res_all = parallel_call(specs)
        ratings = {}
        idx = 0
        for dim in DIMENSIONS:
            dim_values = []
            for _ in range(N_TEMPLATES):
                r = parse_rating_response(res_all[idx]["response_text"])
                if r is not None:
                    dim_values.append(r)
                idx += 1
            ratings[dim] = {
                "mean": float(np.mean(dim_values)) if dim_values else None,
                "std":  float(np.std(dim_values))  if dim_values else None,
                "values": dim_values,
                "n": len(dim_values),
            }

        record = {
            "theory": "ideophone_compositionality",
            "tier": 3, "language": "ja", "word": word, "cell": cell_label,
            "voiced": s["voiced"], "geminated": s["geminated"],
            "reduplicated": s["reduplicated"], "back_vowel": s["back_vowel"],
            "model": model, "ratings": ratings, "notes": s.get("notes", ""),
        }
        save_result(record)

    # Rate Tier 1 gitaigo for calibration
    tier1_ja = [s for s in stimuli if s["tier"] == 1 and s["language"] == "ja"]
    print(f"  Calibration: rating {len(tier1_ja)} Tier 1 gitaigo...")

    for s in tqdm(tier1_ja, desc="tier1_gitaigo"):
        word = s["word"]
        specs = [
            {"model": model,
             "prompt": IDEOPHONE_RATING_TEMPLATES[t % len(IDEOPHONE_RATING_TEMPLATES)](
                 word, "Japanese", DIM_LABELS[dim][0], DIM_LABELS[dim][1]),
             "max_tokens": 5, "temperature": 0.0}
            for dim in DIMENSIONS
            for t in range(N_TEMPLATES)
        ]
        res_all = parallel_call(specs)
        ratings = {}
        idx = 0
        for dim in DIMENSIONS:
            dim_values = []
            for _ in range(N_TEMPLATES):
                r = parse_rating_response(res_all[idx]["response_text"])
                if r is not None:
                    dim_values.append(r)
                idx += 1
            ratings[dim] = {
                "mean": float(np.mean(dim_values)) if dim_values else None,
                "values": dim_values,
            }
        record = {
            "theory": "ideophone_compositionality",
            "tier": 1, "language": "ja", "word": word,
            "meaning": s.get("meaning", ""),
            "voiced": s["voiced"], "geminated": s["geminated"],
            "reduplicated": s["reduplicated"], "back_vowel": s["back_vowel"],
            "model": model, "ratings": ratings,
        }
        save_result(record)


# ---------------------------------------------------------------------------
# Feature-pair FC tests (minimal pairs differing on one feature)
# ---------------------------------------------------------------------------

def run_feature_fc_tests(stimuli: list, model: str, model_short: str):
    """
    For each phonological feature, find Tier 3 stimulus pairs that differ
    only on that feature. Administer an FC task asking which of the two items
    better exemplifies the expected semantic property (e.g., heavier, darker).

    Significant above-chance performance on each feature provides evidence that
    the model has learned feature-level phonological-to-semantic correspondences.
    """
    tier3 = [s for s in stimuli if s["tier"] == 3 and s["language"] == "ja"]

    cell_to_stimuli = {}
    for s in tier3:
        cell = (s["voiced"], s["geminated"], s["reduplicated"], s["back_vowel"])
        cell_to_stimuli.setdefault(cell, []).append(s)

    feature_tests = [
        ("voiced",        "heavy_vs_light",     "heavy",
         (True,  False, False, False), (False, False, False, False)),
        ("geminated",     "sudden_vs_gradual",   "sudden",
         (False, True,  False, False), (False, False, False, False)),
        ("reduplicated",  "continuous_vs_sudden", "continuous",
         (False, False, True,  False), (False, False, False, False)),
        ("back_vowel",    "dark_vs_bright",      "dark",
         (False, False, False, True),  (False, False, False, False)),
        ("voiced+back",   "heavy_vs_light",      "heavy",
         (True,  False, False, True),  (False, False, False, False)),
        ("geminated+voiced", "sudden_vs_gradual", "sudden",
         (True,  True,  False, False), (True,  False, False, False)),
    ]

    print(f"\n[{model_short}] Feature FC tests...")

    results = []
    for feat_name, dim, expected_direction, cell_hi, cell_lo in feature_tests:
        stims_hi = cell_to_stimuli.get(cell_hi, [])
        stims_lo = cell_to_stimuli.get(cell_lo, [])
        if not stims_hi or not stims_lo:
            continue

        dim_low, dim_high = DIM_LABELS[dim]
        pairs = list(zip(stims_hi[:3], stims_lo[:3]))
        for s_hi, s_lo in tqdm(pairs, desc=f"  feature_{feat_name}"):
            specs = [
                {"model": model,
                 "prompt": IDEOPHONE_FC_TEMPLATES[t % len(IDEOPHONE_FC_TEMPLATES)](
                     s_hi["word"], s_lo["word"], "Japanese", expected_direction),
                 "max_tokens": 5}
                for t in range(4)
            ]
            res_all = parallel_call(specs)
            fc_responses = [
                {
                    "parsed": parse_ab_response(r["response_text"]),
                    "correct": (parse_ab_response(r["response_text"]) == "A")
                               if parse_ab_response(r["response_text"]) else None,
                    "raw": r["response_text"],
                }
                for r in res_all
            ]

            valid = [r["correct"] for r in fc_responses if r["correct"] is not None]
            accuracy = sum(valid) / len(valid) if valid else None

            record = {
                "test_type": "feature_fc",
                "feature": feat_name,
                "dimension": dim,
                "expected_direction": expected_direction,
                "word_feature_present": s_hi["word"],
                "word_feature_absent":  s_lo["word"],
                "cell_hi": str(cell_hi), "cell_lo": str(cell_lo),
                "model": model,
                "fc_accuracy": accuracy,
                "fc_responses": fc_responses,
            }
            save_result(record)
            results.append(record)

    return results


# ---------------------------------------------------------------------------
# Korean transfer test
# ---------------------------------------------------------------------------

def run_korean_transfer(stimuli: list, model: str, model_short: str):
    """
    Rate Korean Tier 1 ideophones on the same semantic dimensions as Japanese.

    If feature-to-meaning coefficients for Korean match those derived from
    Japanese Tier 3 novel items, this supports cross-linguistic compositionality
    rather than language-specific memorisation.
    """
    tier1_ko = [s for s in stimuli if s["tier"] == 1 and s["language"] == "ko"]
    print(f"\n[{model_short}] Korean transfer test: {len(tier1_ko)} stimuli")

    for s in tqdm(tier1_ko, desc="korean_transfer"):
        word = s["word"]
        specs = [
            {"model": model,
             "prompt": IDEOPHONE_RATING_TEMPLATES[t % len(IDEOPHONE_RATING_TEMPLATES)](
                 word, "Korean", DIM_LABELS[dim][0], DIM_LABELS[dim][1]),
             "max_tokens": 5, "temperature": 0.0}
            for dim in DIMENSIONS
            for t in range(N_TEMPLATES)
        ]
        res_all = parallel_call(specs)
        ratings = {}
        idx = 0
        for dim in DIMENSIONS:
            dim_values = []
            for _ in range(N_TEMPLATES):
                r = parse_rating_response(res_all[idx]["response_text"])
                if r is not None:
                    dim_values.append(r)
                idx += 1
            ratings[dim] = {
                "mean": float(np.mean(dim_values)) if dim_values else None,
                "values": dim_values,
            }

        record = {
            "theory": "korean_transfer",
            "tier": 1, "language": "ko", "word": word,
            "meaning": s.get("meaning", ""),
            "voiced": s["voiced"], "geminated": s["geminated"],
            "reduplicated": s["reduplicated"], "back_vowel": s["back_vowel"],
            "model": model, "ratings": ratings,
        }
        save_result(record)


# ---------------------------------------------------------------------------
# Linear model analysis
# ---------------------------------------------------------------------------

def analyze_compositionality(model_short: str):
    """
    Fit a linear regression predicting each semantic dimension rating from the
    four binary phonological features.

    Reports R² and per-feature coefficients for a main-effects-only model,
    and delta-R² when 2-way interaction terms are added. Compositionality
    predicts small interaction contributions (additivity).
    """
    try:
        from sklearn.linear_model import LinearRegression
        import statsmodels.api as sm
    except ImportError:
        print("statsmodels not available, skipping regression analysis.")
        return {}

    results_file = RESULTS_DIR / "compositionality_results.jsonl"
    if not results_file.exists():
        return {}

    records = []
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            if r.get("tier") == 3 and r.get("model", "").startswith(model_short):
                records.append(r)

    if not records:
        return {}

    X_rows = {dim: [] for dim in DIMENSIONS}
    y_rows = {dim: [] for dim in DIMENSIONS}
    for r in records:
        features = [
            int(r["voiced"]),
            int(r["geminated"]),
            int(r["reduplicated"]),
            int(r["back_vowel"]),
        ]
        for dim in DIMENSIONS:
            rating = r["ratings"].get(dim, {}).get("mean")
            if rating is not None:
                X_rows[dim].append(features)
                y_rows[dim].append(rating)

    analysis = {}
    for dim in DIMENSIONS:
        if len(X_rows[dim]) < 10:
            continue
        X = np.array(X_rows[dim])
        y = np.array(y_rows[dim])

        X_sm = sm.add_constant(X)
        model_me = sm.OLS(y, X_sm).fit()

        from itertools import combinations
        interaction_cols = [X[:, i] * X[:, j] for i, j in combinations(range(4), 2)]
        X_inter = np.column_stack([X, np.array(interaction_cols).T])
        X_inter_sm = sm.add_constant(X_inter)
        model_inter = sm.OLS(y, X_inter_sm).fit()

        analysis[dim] = {
            "main_effects": {
                "r_squared": float(model_me.rsquared),
                "coefs": {
                    "voiced":       float(model_me.params[1]),
                    "geminated":    float(model_me.params[2]),
                    "reduplicated": float(model_me.params[3]),
                    "back_vowel":   float(model_me.params[4]),
                },
                "pvalues": {
                    "voiced":       float(model_me.pvalues[1]),
                    "geminated":    float(model_me.pvalues[2]),
                    "reduplicated": float(model_me.pvalues[3]),
                    "back_vowel":   float(model_me.pvalues[4]),
                },
                "aic": float(model_me.aic),
            },
            "with_interactions": {
                "r_squared": float(model_inter.rsquared),
                "aic": float(model_inter.aic),
                "delta_r2": float(model_inter.rsquared - model_me.rsquared),
            },
            "n": len(y),
        }

    save_analysis(analysis)
    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ideophone compositionality experiment."
    )
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Model shorthand(s): gpt, llama, qwen, or all.")
    parser.add_argument("--skip_ratings", action="store_true",
                        help="Skip the rating collection phase (use existing data).")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Run only the linear model analysis on existing results.")
    args = parser.parse_args()

    if args.analyze_only:
        for short_name in MODEL_NAMES:
            print(f"\nAnalyzing compositionality for {short_name}...")
            analysis = analyze_compositionality(short_name)
            if analysis:
                print(f"  Dimensions analyzed: {list(analysis.keys())}")
        return

    if "all" in args.models:
        models_to_run = list(MODEL_NAMES.items())
    else:
        models_to_run = [(k, MODEL_NAMES[k]) for k in args.models if k in MODEL_NAMES]

    stimuli = load_stimuli()

    for short_name, model_id in models_to_run:
        print(f"\n{'='*60}\nCOMPOSITIONALITY: {model_id}\n{'='*60}")

        if not args.skip_ratings:
            run_compositionality_ratings(stimuli, model_id, short_name)

        run_feature_fc_tests(stimuli, model_id, short_name)
        run_korean_transfer(stimuli, model_id, short_name)

    print("\nRunning linear model analysis...")
    for short_name in MODEL_NAMES:
        analysis = analyze_compositionality(short_name)
        if analysis:
            for dim, result in analysis.items():
                print(f"  {short_name} | {dim}: R²={result['main_effects']['r_squared']:.3f}")

    print(f"\nCompositionality experiments complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
