"""
run_multilingual.py

Runs cross-linguistic sound symbolism experiments (Tier 1 and Tier 3) in
Japanese, Korean, Hindi, and German across all configured models.

In addition to the main FC and rating tasks, this module also:
  - Administers ideophone rating tasks for Japanese (gitaigo) and Korean (uitate),
    allowing comparison with the compositionality experiment results.
  - Runs language-matched contamination probes by querying the model about the
    bouba-kiki effect in each target language.

Usage
-----
    python run_multilingual.py --models all --languages ja ko hi de
    python run_multilingual.py --models llama --languages ja ko
"""

import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from api_client import call_model, parallel_call, parse_ab_response, parse_rating_response
from prompts.templates import (
    multilingual_fc, multilingual_rating,
    LANG_INSTRUCTIONS, LANG_NAMES,
    IDEOPHONE_RATING_TEMPLATES,
)

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results/multilingual")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "gpt":   "openai/gpt-oss-120b",
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}

N_TEMPLATES = 3
N_REPS      = 3


def load_stimuli() -> dict:
    with open(DATA_DIR / "stimuli.json", encoding="utf-8") as f:
        return json.load(f)["stimuli"]


def save_result(lang: str, record: dict):
    out_file = RESULTS_DIR / f"{lang}_results.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Cross-lingual sound symbolism
# ---------------------------------------------------------------------------

def run_lang_sound_symbolism(stimuli: list, model: str, model_short: str,
                              target_langs: list):
    """
    Run FC and rating tasks for multilingual sound symbolism stimuli.

    Each prompt is prefaced with a language-context instruction asking the model
    to consider phonological intuitions in the target language.
    """
    ml_stimuli = [s for s in stimuli if s.get("language") in target_langs]
    print(f"\n[{model_short}] Multilingual sound symbolism: "
          f"{len(ml_stimuli)} stimuli across {target_langs}")

    for s in tqdm(ml_stimuli, desc="multilingual_SS"):
        lang      = s["language"]
        lang_name = LANG_NAMES.get(lang, lang)
        lang_instr = LANG_INSTRUCTIONS.get(lang, "")

        w_a      = s["word_a"]
        w_b      = s["word_b"]
        dim_low  = s.get("dimension_low", "round/soft")
        dim_high = s.get("dimension_high", "angular/sharp")
        dim_query = dim_low.split("/")[0]

        fc_specs = [
            {"model": model,
             "prompt": multilingual_fc(w_a, w_b, dim_query, lang_name, lang_instr),
             "max_tokens": 5}
            for _ in range(N_TEMPLATES)
            for _ in range(N_REPS)
        ]
        rating_specs = []
        for _ in range(N_TEMPLATES):
            rating_specs.append({
                "model": model,
                "prompt": multilingual_rating(w_a, dim_low, dim_high, lang_name, lang_instr),
                "max_tokens": 5,
            })
            rating_specs.append({
                "model": model,
                "prompt": multilingual_rating(w_b, dim_low, dim_high, lang_name, lang_instr),
                "max_tokens": 5,
            })

        all_res    = parallel_call(fc_specs + rating_specs)
        fc_raw     = all_res[:len(fc_specs)]
        rating_raw = all_res[len(fc_specs):]

        fc_responses = []
        for i, r in enumerate(fc_raw):
            parsed = parse_ab_response(r["response_text"])
            fc_responses.append({
                "template_idx": i // N_REPS,
                "raw": r["response_text"],
                "parsed": parsed,
                "correct": parsed == s["expected"] if parsed else None,
            })
        valid = [r["correct"] for r in fc_responses if r["correct"] is not None]
        fc_accuracy = sum(valid) / len(valid) if valid else None

        rating_responses = []
        for t in range(N_TEMPLATES):
            r_a = parse_rating_response(rating_raw[t * 2]["response_text"])
            r_b = parse_rating_response(rating_raw[t * 2 + 1]["response_text"])
            rating_responses.append({"rating_a": r_a, "rating_b": r_b})

        valid_a = [r["rating_a"] for r in rating_responses if r["rating_a"]]
        valid_b = [r["rating_b"] for r in rating_responses if r["rating_b"]]
        mean_a = sum(valid_a) / len(valid_a) if valid_a else None
        mean_b = sum(valid_b) / len(valid_b) if valid_b else None

        record = {
            "theory": "sound_symbolism",
            "tier": s["tier"],
            "language": lang,
            "word_a": w_a, "word_b": w_b,
            "expected": s["expected"],
            "model": model,
            "fc_accuracy": fc_accuracy,
            "fc_responses": fc_responses,
            "rating_mean_a": mean_a, "rating_mean_b": mean_b,
            "rating_diff": (mean_b - mean_a) if (mean_a and mean_b) else None,
        }
        save_result(lang, record)


# ---------------------------------------------------------------------------
# Multilingual ideophone tests (Japanese + Korean)
# ---------------------------------------------------------------------------

def run_multilingual_ideophones(stimuli: list, model: str, model_short: str):
    """
    Run ideophone semantic rating tasks for Japanese (Tier 1 gitaigo) and
    Korean (Tier 1 uitate) using the same four-dimension rating protocol
    as the compositionality experiment, enabling direct comparison.
    """
    ideo_stimuli = [
        s for s in stimuli
        if s.get("language") in ("ja", "ko") and s["tier"] == 1
    ]
    print(f"\n[{model_short}] Multilingual ideophones: {len(ideo_stimuli)} stimuli")

    dims = [
        ("heavy_vs_light",    "light", "heavy"),
        ("dark_vs_bright",    "bright", "dark"),
        ("rough_vs_smooth",   "smooth", "rough"),
        ("sudden_vs_gradual", "gradual", "sudden"),
    ]

    for s in tqdm(ideo_stimuli, desc="ml_ideophones"):
        lang      = s["language"]
        lang_name = LANG_NAMES.get(lang, lang)
        word      = s["word"]

        specs = [
            {"model": model,
             "prompt": IDEOPHONE_RATING_TEMPLATES[t % len(IDEOPHONE_RATING_TEMPLATES)](
                 word, lang_name, dim_low, dim_high),
             "max_tokens": 5}
            for dim_key, dim_low, dim_high in dims
            for t in range(N_TEMPLATES)
        ]
        res_all = parallel_call(specs)
        ratings = {}
        idx = 0
        for dim_key, dim_low, dim_high in dims:
            dim_ratings = []
            for _ in range(N_TEMPLATES):
                r = parse_rating_response(res_all[idx]["response_text"])
                if r:
                    dim_ratings.append(r)
                idx += 1
            ratings[dim_key] = {
                "mean": sum(dim_ratings) / len(dim_ratings) if dim_ratings else None,
                "values": dim_ratings,
            }

        record = {
            "theory": "ideophone_compositionality",
            "tier": s["tier"],
            "language": lang,
            "word": word,
            "meaning": s.get("meaning", ""),
            "voiced": s["voiced"], "geminated": s["geminated"],
            "reduplicated": s["reduplicated"], "back_vowel": s["back_vowel"],
            "model": model, "ratings": ratings,
        }
        save_result(lang, record)


# ---------------------------------------------------------------------------
# Multilingual contamination probes
# ---------------------------------------------------------------------------

def run_multilingual_contamination(model: str, model_short: str, target_langs: list):
    """
    Query the model about the bouba-kiki effect and sound symbolism in each
    target language. Fluent, accurate responses indicate multilingual exposure
    to the relevant literature in that language.
    """
    print(f"\n[{model_short}] Multilingual contamination probes...")

    lang_queries = {
        "ja": [
            "ブーバ・キキ効果とは何ですか？",
            "音象徴（おとしょうちょう）について説明してください。",
        ],
        "ko": [
            "부바키키 효과란 무엇인가요?",
            "음성 상징에 대해 설명해 주세요.",
        ],
        "hi": [
            "बूबा-किकी प्रभाव क्या है?",
            "ध्वनि प्रतीकवाद के बारे में बताइए।",
        ],
        "de": [
            "Was ist der Bouba-Kiki-Effekt?",
            "Erklären Sie Lautsymbolik.",
        ],
    }

    system = "Answer in the language of the question."

    for lang in target_langs:
        queries = lang_queries.get(lang, [])
        for query in tqdm(queries, desc=f"  contamination_{lang}"):
            result = call_model(model, query, system=system,
                                max_tokens=200, temperature=0.0)
            record = {
                "probe_type": "multilingual_contamination",
                "language": lang,
                "query": query,
                "model": model,
                "response": result["response_text"],
                "error": result["error"],
            }
            save_result(lang, record)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run cross-linguistic behavioral experiments."
    )
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Model shorthand(s): gpt, llama, qwen, or all.")
    parser.add_argument("--languages", nargs="+", default=["ja", "ko", "hi", "de"],
                        help="ISO 639-1 language codes.")
    args = parser.parse_args()

    if "all" in args.models:
        models_to_run = list(MODEL_NAMES.items())
    else:
        models_to_run = [(k, MODEL_NAMES[k]) for k in args.models if k in MODEL_NAMES]

    stimuli      = load_stimuli()
    ml_stimuli   = stimuli.get("multilingual", [])
    ideo_stimuli = stimuli.get("ideophone_compositionality", [])

    for short_name, model_id in models_to_run:
        print(f"\n{'='*60}\nMULTILINGUAL: {model_id}\n{'='*60}")
        run_lang_sound_symbolism(ml_stimuli, model_id, short_name, args.languages)
        run_multilingual_ideophones(ideo_stimuli, model_id, short_name)
        run_multilingual_contamination(model_id, short_name, args.languages)

    print(f"\nMultilingual experiments complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
