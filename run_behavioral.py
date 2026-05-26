"""
run_behavioral.py

Runs the full behavioral evaluation battery across all five psycholinguistic
theories, three stimulus tiers, and multiple task formats:
  - Forced Choice (FC): two-alternative selection
  - Rating: 7-point Likert-scale judgment
  - Generation (sound symbolism only): produce a phonologically matching word

All task formats are run with multiple prompt templates and repeated trials;
results are averaged to reduce sensitivity to any single prompt phrasing.
Parallel API dispatch is used to maximise throughput.

Results are written incrementally to JSONL files under results/behavioral/.
Completed stimuli are tracked so interrupted runs can be safely resumed.

Usage
-----
    python run_behavioral.py --models all
    python run_behavioral.py --models gpt qwen
    python run_behavioral.py --models llama --theory sound_symbolism --tier 3
"""

import json
import argparse
import random
import sys
from pathlib import Path
from tqdm import tqdm

from api_client import (
    call_model, parallel_call,
    parse_ab_response, parse_rating_response, parse_generation_response,
)
from prompts.templates import (
    SOUND_FC_TEMPLATES, SOUND_RATING_TEMPLATES, SOUND_GENERATION_TEMPLATES,
    PHONESTHESIA_FC_TEMPLATES, PHONESTHESIA_RATING_TEMPLATES,
    PROSODY_CLOZE_TEMPLATES, PROSODY_RATING_TEMPLATES,
    IDEOPHONE_RATING_TEMPLATES, IDEOPHONE_FC_TEMPLATES,
    LANG_NAMES,
)

random.seed(42)

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results/behavioral")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "gpt":   "openai/gpt-oss-120b",
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}

N_TEMPLATES = 3  # Number of prompt surface variants per stimulus
N_REPS      = 3  # FC repetitions per template


def load_stimuli() -> dict:
    with open(DATA_DIR / "stimuli.json", encoding="utf-8") as f:
        return json.load(f)["stimuli"]


def _has_tpd_error(record: dict) -> bool:
    return "All keys TPD-exhausted" in json.dumps(record)


def load_completed(model_short: str, theory: str, key_field: str) -> set:
    """
    Load already-completed stimulus keys from the JSONL results file.

    Also de-duplicates and removes any records that resulted from exhausted
    API quota (TPD errors), so those stimuli will be retried on the next run.
    """
    out_file = RESULTS_DIR / model_short / theory / "results.jsonl"
    if not out_file.exists():
        return set()
    seen: dict = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _has_tpd_error(rec):
            continue
        key = rec.get(key_field)
        if key and key not in seen:
            seen[key] = rec
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in seen.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return set(seen.keys())


def save_result(model_short: str, theory: str, record: dict):
    out_dir = RESULTS_DIR / model_short / theory
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Theories 1 & 3: Sound Symbolism + Vowel-Size Symbolism
# ---------------------------------------------------------------------------

def run_sound_symbolism(stimuli: list, model: str, model_short: str,
                        theory_key: str, tier_filter=None):
    """
    Run FC and rating tasks for sound symbolism or vowel-size stimuli.

    For each stimulus, N_TEMPLATES × N_REPS forced-choice calls and
    N_TEMPLATES rating calls for each word are fired in a single parallel batch.
    """
    print(f"\n[{model_short}] Running {theory_key}...")
    done = load_completed(model_short, theory_key, "word_a")

    for s in tqdm(stimuli, desc=theory_key):
        if tier_filter and s["tier"] != tier_filter:
            continue
        if s.get("language", "en") != "en":
            continue
        if s["word_a"] in done:
            continue

        w_a     = s["word_a"]
        w_b     = s["word_b"]
        dim_low  = s.get("dimension_low",  "round/soft")
        dim_high = s.get("dimension_high", "angular/sharp")
        dim_q    = dim_low.split("/")[0]

        fc_specs = [
            {"model": model,
             "prompt": SOUND_FC_TEMPLATES[t % len(SOUND_FC_TEMPLATES)](w_a, w_b, dim_q),
             "max_tokens": 5}
            for t in range(N_TEMPLATES)
            for _ in range(N_REPS)
        ]
        rating_specs = []
        for t in range(N_TEMPLATES):
            tmpl = SOUND_RATING_TEMPLATES[t % len(SOUND_RATING_TEMPLATES)]
            rating_specs.append({"model": model, "prompt": tmpl(w_a, dim_low, dim_high), "max_tokens": 5})
            rating_specs.append({"model": model, "prompt": tmpl(w_b, dim_low, dim_high), "max_tokens": 5})

        all_results = parallel_call(fc_specs + rating_specs)
        fc_res      = all_results[:len(fc_specs)]
        rating_res  = all_results[len(fc_specs):]

        fc_responses = []
        for i, r in enumerate(fc_res):
            parsed = parse_ab_response(r["response_text"])
            fc_responses.append({
                "template_idx": i // N_REPS,
                "raw": r["response_text"],
                "parsed": parsed,
                "correct": (parsed == s["expected"]) if parsed else None,
                "error": r["error"],
            })
        valid_fc = [x["correct"] for x in fc_responses if x["correct"] is not None]
        fc_accuracy = sum(valid_fc) / len(valid_fc) if valid_fc else None

        rating_a_responses, rating_b_responses = [], []
        for t in range(N_TEMPLATES):
            ra = rating_res[t * 2]
            rb = rating_res[t * 2 + 1]
            rating_a_responses.append({
                "template_idx": t, "raw": ra["response_text"],
                "rating": parse_rating_response(ra["response_text"]),
            })
            rating_b_responses.append({
                "template_idx": t, "raw": rb["response_text"],
                "rating": parse_rating_response(rb["response_text"]),
            })

        valid_a = [r["rating"] for r in rating_a_responses if r["rating"] is not None]
        valid_b = [r["rating"] for r in rating_b_responses if r["rating"] is not None]
        mean_a  = sum(valid_a) / len(valid_a) if valid_a else None
        mean_b  = sum(valid_b) / len(valid_b) if valid_b else None

        record = {
            "theory": theory_key, "tier": s["tier"],
            "word_a": w_a, "word_b": w_b, "expected": s["expected"], "model": model,
            "fc": {"responses": fc_responses, "accuracy": fc_accuracy, "n_valid": len(valid_fc)},
            "rating": {
                "responses_a": rating_a_responses,
                "responses_b": rating_b_responses,
                "mean_a": mean_a, "mean_b": mean_b,
                "diff_b_minus_a": (mean_b - mean_a) if (mean_a and mean_b) else None,
                "correct_direction": (mean_a < mean_b) if (mean_a and mean_b) else None,
            },
        }
        save_result(model_short, theory_key, record)


# ---------------------------------------------------------------------------
# Theory 2: Phonesthesia
# ---------------------------------------------------------------------------

def run_phonesthesia(stimuli: list, model: str, model_short: str, tier_filter=None):
    """
    Run FC and rating tasks for phonesthesia stimuli.

    For Tier 3 stimuli, a matched neutral-onset control word is also tested
    via FC to provide a within-stimulus baseline for the phonesthemic effect.
    """
    print(f"\n[{model_short}] Running phonesthesia...")
    done = load_completed(model_short, "phonesthesia", "word")

    for s in tqdm(stimuli, desc="phonesthesia"):
        if tier_filter and s["tier"] != tier_filter:
            continue
        word = s["word_a"]
        if word in done:
            continue

        cat_match = s["category_match"]
        cat_other = s["category_other"]

        fc_specs = [
            {"model": model,
             "prompt": PHONESTHESIA_FC_TEMPLATES[t % len(PHONESTHESIA_FC_TEMPLATES)](
                 word, cat_match, cat_other),
             "max_tokens": 5}
            for t in range(N_TEMPLATES)
            for _ in range(N_REPS)
        ]
        rating_specs = []
        for t in range(N_TEMPLATES):
            tmpl = PHONESTHESIA_RATING_TEMPLATES[t % len(PHONESTHESIA_RATING_TEMPLATES)]
            rating_specs.append({"model": model, "prompt": tmpl(word, cat_match), "max_tokens": 5})
            rating_specs.append({"model": model, "prompt": tmpl(word, cat_other), "max_tokens": 5})

        ctrl_specs = []
        if s["tier"] == 3 and "word_control" in s:
            ctrl_word = s["word_control"]
            ctrl_specs = [
                {"model": model,
                 "prompt": PHONESTHESIA_FC_TEMPLATES[t % len(PHONESTHESIA_FC_TEMPLATES)](
                     ctrl_word, cat_match, cat_other),
                 "max_tokens": 5}
                for t in range(N_TEMPLATES)
            ]

        all_results = parallel_call(fc_specs + rating_specs + ctrl_specs)
        fc_res      = all_results[:len(fc_specs)]
        rating_res  = all_results[len(fc_specs):len(fc_specs)+len(rating_specs)]
        ctrl_res    = all_results[len(fc_specs)+len(rating_specs):]

        fc_responses = []
        for i, r in enumerate(fc_res):
            parsed = parse_ab_response(r["response_text"])
            fc_responses.append({
                "template_idx": i // N_REPS, "raw": r["response_text"],
                "parsed": parsed,
                "correct": (parsed == "A") if parsed else None,
                "error": r["error"],
            })
        valid_fc = [x["correct"] for x in fc_responses if x["correct"] is not None]

        rating_responses = []
        for t in range(N_TEMPLATES):
            rm = rating_res[t * 2]
            ro = rating_res[t * 2 + 1]
            rating_responses.append({
                "template_idx": t,
                "rating_match": parse_rating_response(rm["response_text"]),
                "raw_match": rm["response_text"],
                "rating_other": parse_rating_response(ro["response_text"]),
                "raw_other": ro["response_text"],
            })
        valid_m = [r["rating_match"] for r in rating_responses if r["rating_match"]]
        valid_o = [r["rating_other"] for r in rating_responses if r["rating_other"]]

        ctrl_accuracy = None
        if ctrl_res:
            ctrl_parsed = [parse_ab_response(r["response_text"]) for r in ctrl_res]
            valid_ctrl  = [p == "A" for p in ctrl_parsed if p]
            ctrl_accuracy = sum(valid_ctrl) / len(valid_ctrl) if valid_ctrl else None

        record = {
            "theory": "phonesthesia", "tier": s["tier"],
            "word": word, "phonestheme": s["phonestheme"],
            "category_match": cat_match, "category_other": cat_other,
            "expected": "A", "model": model,
            "fc_accuracy": sum(valid_fc) / len(valid_fc) if valid_fc else None,
            "fc_responses": fc_responses,
            "rating_mean_match": sum(valid_m) / len(valid_m) if valid_m else None,
            "rating_mean_other": sum(valid_o) / len(valid_o) if valid_o else None,
            "rating_diff": (
                (sum(valid_m)/len(valid_m)) - (sum(valid_o)/len(valid_o))
                if (valid_m and valid_o) else None
            ),
            "control_word": s.get("word_control"),
            "control_fc_accuracy": ctrl_accuracy,
        }
        save_result(model_short, "phonesthesia", record)


# ---------------------------------------------------------------------------
# Theory 4: Semantic Prosody
# ---------------------------------------------------------------------------

def run_semantic_prosody(stimuli: list, model: str, model_short: str, tier_filter=None):
    """
    Run cloze (FC) and naturalness rating tasks for semantic prosody stimuli.

    Each sentence frame is presented with a prosodically marked verb (option A)
    and a neutral near-synonym (option B). Rating tasks rate the naturalness of
    each completed sentence independently.
    """
    print(f"\n[{model_short}] Running semantic_prosody...")
    done = load_completed(model_short, "semantic_prosody", "sentence_frame")

    for s in tqdm(stimuli, desc="semantic_prosody"):
        if tier_filter and s["tier"] != tier_filter:
            continue
        if s["sentence_frame"] in done:
            continue

        sent_p = s["sentence_frame"].replace("___", s["option_a"])
        sent_n = s["sentence_frame"].replace("___", s["option_b"])

        fc_specs = [
            {"model": model,
             "prompt": PROSODY_CLOZE_TEMPLATES[t % len(PROSODY_CLOZE_TEMPLATES)](
                 s["sentence_frame"], s["option_a"], s["option_b"]),
             "max_tokens": 5}
            for t in range(N_TEMPLATES)
            for _ in range(N_REPS)
        ]
        rating_specs = []
        for t in range(N_TEMPLATES):
            tmpl = PROSODY_RATING_TEMPLATES[t % len(PROSODY_RATING_TEMPLATES)]
            rating_specs.append({"model": model, "prompt": tmpl(sent_p), "max_tokens": 5})
            rating_specs.append({"model": model, "prompt": tmpl(sent_n), "max_tokens": 5})

        all_results = parallel_call(fc_specs + rating_specs)
        fc_res      = all_results[:len(fc_specs)]
        rating_res  = all_results[len(fc_specs):]

        fc_responses = []
        for i, r in enumerate(fc_res):
            parsed = parse_ab_response(r["response_text"])
            fc_responses.append({
                "template_idx": i // N_REPS, "raw": r["response_text"],
                "parsed": parsed,
                "correct": (parsed == "A") if parsed else None,
                "error": r["error"],
            })
        valid_fc = [x["correct"] for x in fc_responses if x["correct"] is not None]

        rating_p, rating_n = [], []
        for t in range(N_TEMPLATES):
            rp = parse_rating_response(rating_res[t * 2]["response_text"])
            rn = parse_rating_response(rating_res[t * 2 + 1]["response_text"])
            if rp: rating_p.append(rp)
            if rn: rating_n.append(rn)

        record = {
            "theory": "semantic_prosody", "tier": s["tier"],
            "sentence_frame": s["sentence_frame"],
            "option_a": s["option_a"], "option_b": s["option_b"],
            "expected": "A", "prosody_type": s["prosody_type"], "model": model,
            "fc_accuracy": sum(valid_fc) / len(valid_fc) if valid_fc else None,
            "fc_responses": fc_responses,
            "rating_prosodic": sum(rating_p) / len(rating_p) if rating_p else None,
            "rating_neutral":  sum(rating_n) / len(rating_n) if rating_n else None,
            "rating_diff": (
                (sum(rating_p)/len(rating_p)) - (sum(rating_n)/len(rating_n))
                if (rating_p and rating_n) else None
            ),
        }
        save_result(model_short, "semantic_prosody", record)


# ---------------------------------------------------------------------------
# Theory 5: Ideophone Compositionality (basic rating)
# ---------------------------------------------------------------------------

def run_ideophones_basic(stimuli: list, model: str, model_short: str, tier_filter=None):
    """
    Run rating tasks for ideophone compositionality stimuli.

    Each ideophone is rated on four semantic dimensions (heavy/light,
    dark/bright, rough/smooth, sudden/gradual) using N_TEMPLATES rating prompts.
    All dimension × template calls are batched into a single parallel dispatch.
    """
    print(f"\n[{model_short}] Running ideophones (basic)...")
    done = load_completed(model_short, "ideophone_compositionality", "word")

    dims = [
        ("heavy_vs_light",    "light", "heavy"),
        ("dark_vs_bright",    "bright", "dark"),
        ("rough_vs_smooth",   "smooth", "rough"),
        ("sudden_vs_gradual", "gradual", "sudden"),
    ]

    for s in tqdm(stimuli, desc="ideophones"):
        if tier_filter and s["tier"] != tier_filter:
            continue
        word = s["word"]
        if word in done:
            continue
        lang_name = LANG_NAMES.get(s["language"], s["language"])

        specs = [
            {"model": model,
             "prompt": IDEOPHONE_RATING_TEMPLATES[t % len(IDEOPHONE_RATING_TEMPLATES)](
                 word, lang_name, dim_low, dim_high),
             "max_tokens": 5}
            for _, dim_low, dim_high in dims
            for t in range(N_TEMPLATES)
        ]
        results = parallel_call(specs)

        ratings = {}
        idx = 0
        for dim_key, dim_low, dim_high in dims:
            vals = []
            for _ in range(N_TEMPLATES):
                r = parse_rating_response(results[idx]["response_text"])
                if r is not None:
                    vals.append(r)
                idx += 1
            ratings[dim_key] = {"mean": sum(vals) / len(vals) if vals else None, "values": vals}

        record = {
            "theory": "ideophone_compositionality", "tier": s["tier"],
            "language": s["language"], "word": word,
            "voiced": s["voiced"], "geminated": s["geminated"],
            "reduplicated": s["reduplicated"], "back_vowel": s["back_vowel"],
            "model": model, "ratings": ratings,
        }
        save_result(model_short, "ideophone_compositionality", record)


# ---------------------------------------------------------------------------
# Generation task (sound symbolism)
# ---------------------------------------------------------------------------

def run_generation_task(model: str, model_short: str):
    """
    Prompt the model to generate novel words for four phonosemantic conditions.

    Generated words are scored by the proportion of consonants and vowels
    drawn from the round-phonology pool (m, n, l, r, b, w; a, o, u) vs the
    sharp-phonology pool (k, t, p, s, f; i, e).
    """
    print(f"\n[{model_short}] Generation task...")
    conditions = [
        ("round_soft_large",    "round, soft, and large"),
        ("angular_sharp_small", "angular, sharp, and small"),
        ("small_light_fast",    "small, light, and fast"),
        ("large_heavy_slow",    "large, heavy, and slow"),
    ]
    out_dir  = RESULTS_DIR / model_short / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generation_results.jsonl"

    pool_r = set('mnlrbwaou')
    pool_s = set('ktpsfxie')

    for condition_key, description in conditions:
        specs = [
            {"model": model,
             "prompt": SOUND_GENERATION_TEMPLATES[t % len(SOUND_GENERATION_TEMPLATES)](description),
             "max_tokens": 15, "temperature": 0.8, "use_cache": False}
            for t in range(len(SOUND_GENERATION_TEMPLATES))
            for _ in range(10)
        ]
        results = parallel_call(specs)

        generated_words = []
        for i, r in enumerate(results):
            word = parse_generation_response(r["response_text"])
            if word:
                chars = word.lower()
                rc = sum(1 for c in chars if c in pool_r)
                sc = sum(1 for c in chars if c in pool_s)
                total = rc + sc
                generated_words.append({
                    "word": word, "raw": r["response_text"],
                    "template_idx": i // 10, "rep": i % 10,
                    "roundness_score": (rc / total) if total > 0 else 0.5,
                    "r_count": rc, "s_count": sc,
                })

        record = {
            "task": "generation", "condition": condition_key,
            "description": description, "model": model,
            "n_generated": len(generated_words),
            "mean_roundness": (
                sum(w["roundness_score"] for w in generated_words) / len(generated_words)
                if generated_words else None
            ),
            "words": generated_words,
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Generation results saved to {out_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run behavioral evaluation battery across models and theories."
    )
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Model shorthand(s): gpt, llama, qwen, or all.")
    parser.add_argument("--theory", nargs="+", default=["all"],
                        help="Theory key(s): sound_symbolism, vowel_size, phonesthesia, "
                             "semantic_prosody, ideophones, or all.")
    parser.add_argument("--tier", type=int, default=None,
                        help="Restrict to a specific tier (1, 2, or 3).")
    args = parser.parse_args()

    if "all" in args.models:
        models_to_run = list(MODEL_NAMES.items())
    else:
        models_to_run = [(k, MODEL_NAMES[k]) for k in args.models if k in MODEL_NAMES]

    print(f"Running models: {[m for _, m in models_to_run]}")
    all_stimuli = load_stimuli()

    for short_name, model_id in models_to_run:
        print(f"\n{'='*60}\nMODEL: {model_id}\n{'='*60}")
        run_all = "all" in args.theory

        if run_all or "sound_symbolism" in args.theory:
            run_sound_symbolism(all_stimuli["sound_symbolism"],
                                model_id, short_name, "sound_symbolism", args.tier)

        if run_all or "vowel_size" in args.theory:
            run_sound_symbolism(all_stimuli["vowel_size"],
                                model_id, short_name, "vowel_size", args.tier)

        if run_all or "phonesthesia" in args.theory:
            run_phonesthesia(all_stimuli["phonesthesia"],
                             model_id, short_name, args.tier)

        if run_all or "semantic_prosody" in args.theory:
            run_semantic_prosody(all_stimuli["semantic_prosody"],
                                 model_id, short_name, args.tier)

        if run_all or "ideophones" in args.theory:
            run_ideophones_basic(all_stimuli["ideophone_compositionality"],
                                 model_id, short_name, args.tier)

        if run_all or "sound_symbolism" in args.theory:
            run_generation_task(model_id, short_name)

    print(f"\nBehavioral experiments complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
