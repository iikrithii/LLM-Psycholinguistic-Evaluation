"""
run_contamination.py

Runs training data contamination probes for all five psycholinguistic theories
across all configured models. Four types of probe are administered:

  1. Theory knowledge probes   — Does the model know each theory by name?
  2. Word recognition probes   — Does the model recognise famous experimental stimuli?
  3. Confidence comparison     — Is the model more certain on Tier 1 (famous) than
                                 Tier 3 (novel) stimuli? Operationalised via hedging
                                 language detection in open-ended FC responses.
  4. Passage recognition       — Can the model identify paraphrases of landmark
                                 paper abstracts vs. constructed novel passages?

High passage recognition and low hedging on Tier 1 but not Tier 3 constitute
positive evidence of training data contamination driving Tier 1 performance.

Usage
-----
    python run_contamination.py --models all
    python run_contamination.py --models llama
"""

import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from api_client import call_model, call_all_models, parallel_call
from prompts.templates import CONTAMINATION_DIRECT, CONTAMINATION_WORD_PROBE

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results/contamination")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = {
    "gpt":   "openai/gpt-oss-120b",
    "llama": "llama-3.3-70b-versatile",
    "qwen":  "qwen/qwen3-32b",
}


def load_stimuli() -> dict:
    with open(DATA_DIR / "stimuli.json", encoding="utf-8") as f:
        return json.load(f)["stimuli"]


def save_result(theory: str, record: dict):
    out_file = RESULTS_DIR / f"{theory}_contamination.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Probe 1: Direct theory knowledge
# ---------------------------------------------------------------------------

def run_theory_knowledge_probes(model: str, model_short: str):
    """
    Ask the model to describe each theory by name using several prompt variants.

    Responses are recorded verbatim; keyword presence is later used to score
    how accurately the model characterises each theory.
    """
    print(f"\n[{model_short}] Theory knowledge probes...")

    theory_queries = {
        "sound_symbolism": [
            "bouba-kiki effect",
            "maluma-takete effect",
            "sound symbolism in language",
        ],
        "phonesthesia": [
            "phonesthesia in English",
            "phonesthemes and their meaning",
        ],
        "vowel_size": [
            "frequency code hypothesis Ohala",
            "vowel size symbolism",
        ],
        "semantic_prosody": [
            "semantic prosody Louw Sinclair",
            "collocational prosody corpus linguistics",
        ],
        "ideophone_compositionality": [
            "ideophones Japanese gitaigo",
            "ideophone compositionality Dingemanse",
        ],
    }

    for theory, queries in theory_queries.items():
        for query in tqdm(queries, desc=f"  {theory}"):
            specs = [
                {"model": model,
                 "prompt": CONTAMINATION_DIRECT[t % len(CONTAMINATION_DIRECT)](query),
                 "max_tokens": 200, "temperature": 0.0}
                for t in range(3)
            ]
            results = parallel_call(specs)
            for t_idx, result in enumerate(results):
                record = {
                    "probe_type": "theory_knowledge",
                    "theory": theory,
                    "query": query,
                    "template_idx": t_idx,
                    "model": model,
                    "response": result["response_text"],
                    "error": result["error"],
                    "keywords": _theory_keywords(theory),
                }
                save_result(theory, record)


def _theory_keywords(theory: str) -> list:
    """Return a list of keywords expected in an accurate description of each theory."""
    keywords = {
        "sound_symbolism": [
            "bouba", "kiki", "round", "angular", "sound symbolism",
            "ramachandran", "köhler", "maluma", "takete",
        ],
        "phonesthesia": [
            "gl-", "sl-", "phonestheme", "bergen", "hutchins",
            "onset cluster", "semantic",
        ],
        "vowel_size": [
            "ohala", "frequency code", "small", "large", "vowel",
            "front vowel", "back vowel", "i vs a",
        ],
        "semantic_prosody": [
            "sinclair", "louw", "set in", "cause", "collocate",
            "negative", "prosody", "evaluative",
        ],
        "ideophone_compositionality": [
            "ideophone", "gitaigo", "dingemanse",
            "voiced", "geminated", "reduplicated", "japanese", "sound symbolic",
        ],
    }
    return keywords.get(theory, [])


# ---------------------------------------------------------------------------
# Probe 2: Famous word recognition
# ---------------------------------------------------------------------------

def run_word_recognition_probes(model: str, model_short: str):
    """
    Ask whether the model recognises famous experimental stimuli from the literature.

    Recognition of Tier 1 stimuli but not Tier 3 novel stimuli would be
    consistent with training data contamination.
    """
    print(f"\n[{model_short}] Word recognition probes...")

    famous_words = {
        "sound_symbolism": ["bouba", "kiki", "maluma", "takete"],
        "phonesthesia": ["gl- cluster", "sl- cluster", "phonesthesia"],
        "vowel_size": ["mil vs mol", "teeni vs toona"],
        "semantic_prosody": ["set in", "cause damage", "commit"],
        "ideophone_compositionality": ["pika-pika", "kira-kira", "zawa-zawa", "fura-fura"],
    }

    for theory, words in famous_words.items():
        for word in tqdm(words, desc=f"  {theory}"):
            specs = [
                {"model": model,
                 "prompt": CONTAMINATION_WORD_PROBE[t % len(CONTAMINATION_WORD_PROBE)](word),
                 "max_tokens": 150, "temperature": 0.0}
                for t in range(3)
            ]
            results = parallel_call(specs)
            for t_idx, result in enumerate(results):
                record = {
                    "probe_type": "word_recognition",
                    "theory": theory,
                    "word": word,
                    "template_idx": t_idx,
                    "model": model,
                    "response": result["response_text"],
                    "error": result["error"],
                }
                save_result(theory, record)


# ---------------------------------------------------------------------------
# Probe 3: Tier 1 vs Tier 3 confidence comparison
# ---------------------------------------------------------------------------

def run_confidence_comparison(stimuli: dict, model: str, model_short: str):
    """
    Compare hedging language in open-ended FC responses for Tier 1 vs Tier 3 stimuli.

    A prompt explicitly states that the words are novel and have not appeared
    in any study, then asks for a free-response FC answer. More hedging on
    Tier 1 than Tier 3 is unlikely; less hedging on Tier 1 than Tier 3
    (i.e., more confidence on famous stimuli) is consistent with prior exposure.
    """
    print(f"\n[{model_short}] Confidence comparison (Tier1 vs Tier3)...")

    UNCERTAINTY_PROMPT = (
        "I will show you two made-up words. These do NOT exist in any language "
        "and have NOT been used in any scientific study. Please choose which seems "
        "more round/soft based purely on how the letters look. "
        "Word A: {w_a}  Word B: {w_b}\n"
        "Before answering, note: these are completely novel words you have never seen. "
        "Answer with A or B and briefly explain your reasoning."
    )

    for theory in ["sound_symbolism", "vowel_size"]:
        theory_stimuli = stimuli.get(theory, [])
        tier1 = [s for s in theory_stimuli if s["tier"] == 1 and s.get("language", "en") == "en"]
        tier3 = [s for s in theory_stimuli if s["tier"] == 3 and s.get("language", "en") == "en"]

        for tier_name, tier_items in [("tier1", tier1[:5]), ("tier3", tier3[:5])]:
            for s in tqdm(tier_items, desc=f"  {theory} {tier_name}"):
                prompt = UNCERTAINTY_PROMPT.format(w_a=s["word_a"], w_b=s["word_b"])
                result = call_model(model, prompt, max_tokens=100, temperature=0.0)

                text = result["response_text"].lower()
                hedging_words = [
                    "not sure", "unsure", "difficult", "hard to say",
                    "guess", "uncertain", "arbitrary", "random",
                    "no idea", "can't tell", "cannot determine",
                ]
                hedge_count = sum(1 for h in hedging_words if h in text)

                record = {
                    "probe_type": "confidence_comparison",
                    "theory": theory,
                    "tier": s["tier"],
                    "word_a": s["word_a"],
                    "word_b": s["word_b"],
                    "model": model,
                    "response": result["response_text"],
                    "hedge_count": hedge_count,
                    "response_length": len(text.split()),
                    "error": result["error"],
                }
                save_result(theory, record)


# ---------------------------------------------------------------------------
# Probe 4: Passage recognition
# ---------------------------------------------------------------------------

def run_passage_recognition(model: str, model_short: str):
    """
    Present the model with passages paraphrasing landmark papers (Tier 1) or
    describing novel constructed stimuli (Tier 3) and ask whether it recognises
    the source.

    High recognition of Tier 1 paraphrases indicates that the model has been
    exposed to the relevant primary literature during training.
    """
    print(f"\n[{model_short}] Passage recognition probes...")

    passages = [
        {
            "tier": 1, "theory": "sound_symbolism",
            "passage": (
                "When people are shown an amoeba-like rounded shape and a jagged, "
                "spiky shape, and asked which one is called 'bouba' and which is "
                "called 'kiki', nearly all respondents across different language "
                "backgrounds assign 'bouba' to the rounded shape."
            ),
        },
        {
            "tier": 1, "theory": "phonesthesia",
            "passage": (
                "English words beginning with gl- show a disproportionate tendency "
                "to refer to visual phenomena involving light, such as gleam, glitter, "
                "glow, and glare. This onset cluster is an example of a phonestheme."
            ),
        },
        {
            "tier": 1, "theory": "semantic_prosody",
            "passage": (
                "The phrasal verb 'set in' almost invariably appears in contexts "
                "describing the onset of something unpleasant or undesirable, such as "
                "decay, disease, or despair, giving the phrase a negative evaluative "
                "semantic prosody."
            ),
        },
        {
            "tier": 3, "theory": "sound_symbolism",
            "passage": (
                "Novel pseudoword pairs were constructed by sampling consonants and "
                "vowels from two phonological pools differing in sonority and place "
                "of articulation. Participants were asked to assign shape labels to "
                "these words without any prior exposure to sound symbolism research."
            ),
        },
        {
            "tier": 3, "theory": "phonesthesia",
            "passage": (
                "Nonce words were constructed by appending the onset clusters gl-, "
                "sl-, sn-, and fl- to random syllable strings with no prior lexical "
                "status. These were used to test whether the phonesthemic association "
                "transfers to novel, unprecedented word forms."
            ),
        },
    ]

    for p in tqdm(passages, desc="  passage_recognition"):
        prompt = (
            f"Does the following passage come from or closely paraphrase a specific "
            f"published scientific paper or textbook you are aware of? "
            f"If yes, name the source. If no, say 'not recognized'.\n\n"
            f"Passage: \"{p['passage']}\"\n\nAnswer:"
        )
        result = call_model(model, prompt, max_tokens=100, temperature=0.0)
        recognized = "not recognized" not in result["response_text"].lower()
        record = {
            "probe_type": "passage_recognition",
            "theory": p["theory"],
            "tier": p["tier"],
            "passage": p["passage"],
            "model": model,
            "response": result["response_text"],
            "recognized": recognized,
            "error": result["error"],
        }
        save_result(p["theory"], record)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run contamination probes for all theories and models."
    )
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Model shorthand(s): gpt, llama, qwen, or all.")
    args = parser.parse_args()

    if "all" in args.models:
        models_to_run = list(MODEL_NAMES.items())
    else:
        models_to_run = [(k, MODEL_NAMES[k]) for k in args.models if k in MODEL_NAMES]

    stimuli = load_stimuli()

    for short_name, model_id in models_to_run:
        print(f"\n{'='*60}\nCONTAMINATION PROBES: {model_id}\n{'='*60}")
        run_theory_knowledge_probes(model_id, short_name)
        run_word_recognition_probes(model_id, short_name)
        run_confidence_comparison(stimuli, model_id, short_name)
        run_passage_recognition(model_id, short_name)

    print(f"\nContamination probes complete. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
