"""
interpretability/utils.py

Shared utilities for all mechanistic interpretability experiments.
Handles model loading via TransformerLens, stimulus loading, tokenization
helpers, activation extraction utilities, and result persistence.

Models
------
  - Gemma-2 9B  (google/gemma-2-9b-it)   — primary interpretability model
  - Pythia 1.4B (EleutherAI/pythia-1.4b) — secondary model with a documented
    training corpus (The Pile), enabling direct contamination analysis

Both models are loaded through TransformerLens for hook-based access to
residual stream activations, attention patterns, and MLP outputs.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "interpretability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemma2_9b":   "google/gemma-2-9b-it",
    "pythia_1.4b": "EleutherAI/pythia-1.4b",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_key: str, device: str = "cuda"):
    """
    Load a model via TransformerLens and return the HookedTransformer instance.

    TransformerLens wraps HuggingFace models and exposes residual stream
    activations at every layer through named forward hooks.

    Args:
        model_key: Key into the MODELS dict (e.g., "gemma2_9b").
        device:    PyTorch device string ("cuda" or "cpu").

    Returns:
        A TransformerLens HookedTransformer in eval mode.
    """
    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise ImportError(
            "TransformerLens not found. Install it with:\n"
            "  pip install transformer_lens"
        )

    model_id = MODELS[model_key]
    print(f"Loading {model_key} ({model_id}) on {device}...")

    model = HookedTransformer.from_pretrained(
        model_id,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
        dtype=torch.bfloat16,
        device=device,
    )
    model.eval()

    print(f"  Layers:  {model.cfg.n_layers}")
    print(f"  d_model: {model.cfg.d_model}")
    print(f"  n_heads: {model.cfg.n_heads}")
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated(device) / 1e9
        print(f"  VRAM:    {vram:.1f} GB")

    return model


# ---------------------------------------------------------------------------
# Stimulus loading
# ---------------------------------------------------------------------------

def load_stimuli() -> dict:
    """Load the full stimulus battery from data/stimuli.json."""
    path = DATA_DIR / "stimuli.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["stimuli"]


def get_theory_stimuli(stimuli: dict, theory: str,
                        tier: Optional[int] = None,
                        language: str = "en") -> list:
    """
    Filter stimuli for a given theory, optionally by tier and language.

    Args:
        stimuli:  Full stimuli dict from load_stimuli().
        theory:   Theory key (e.g., "sound_symbolism").
        tier:     If provided, keep only stimuli at this tier level.
        language: ISO 639-1 language code; defaults to English.

    Returns:
        Filtered list of stimulus dicts.
    """
    items = stimuli.get(theory, [])
    if tier is not None:
        items = [s for s in items if s.get("tier") == tier]
    items = [s for s in items if s.get("language", "en") == language]
    return items


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_word(model, word: str) -> list:
    """
    Tokenize a single word without special tokens or leading space.

    Returns:
        List of integer token IDs.
    """
    tokens = model.to_tokens(word, prepend_bos=False)[0].tolist()
    return tokens


def get_word_token_positions(model, prompt: str, word: str) -> list:
    """
    Find the token positions of a word within a tokenized prompt.

    Performs a sliding-window search to locate the first occurrence of
    the word's token sequence within the prompt's token sequence.

    Returns:
        List of 0-indexed positions (including the BOS token offset).
        Empty list if the word is not found.
    """
    prompt_tokens = model.to_tokens(prompt)[0].tolist()
    word_tokens   = tokenize_word(model, word)

    for i in range(len(prompt_tokens) - len(word_tokens) + 1):
        if prompt_tokens[i:i + len(word_tokens)] == word_tokens:
            return list(range(i, i + len(word_tokens)))
    return []


def get_last_token_position(model, prompt: str) -> int:
    """Return the index of the last token in a tokenized prompt."""
    tokens = model.to_tokens(prompt)[0]
    return len(tokens) - 1


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def make_fc_prompt(word_a: str, word_b: str, dimension: str) -> str:
    """
    Canonical forced-choice prompt used across all interpretability experiments.

    Using a single template ensures that differences in activation patterns
    reflect the stimulus words rather than prompt phrasing.
    """
    return (
        f"Two made-up words: '{word_a}' and '{word_b}'. "
        f"Which seems more {dimension}? Answer with just A or B."
    )


def make_rating_prompt(word: str, dim_low: str, dim_high: str) -> str:
    """Canonical rating prompt for a single word on a bipolar scale."""
    return (
        f"Rate the invented word '{word}' on a scale: "
        f"1 = very {dim_low}, 7 = very {dim_high}. "
        f"Reply with a single number."
    )


# ---------------------------------------------------------------------------
# Token probability extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_token_logprobs(model, prompt: str,
                        target_tokens: list) -> dict:
    """
    Run a forward pass and return log-probabilities for each target token
    at the final sequence position.

    Used for forced-choice tasks: compare logprob("A") vs logprob("B").

    Args:
        model:         HookedTransformer instance.
        prompt:        Input prompt string.
        target_tokens: List of token strings to score (e.g., ["A", "B"]).

    Returns:
        Dict mapping each target token string to its log-probability.
    """
    tokens    = model.to_tokens(prompt)
    logits    = model(tokens)[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    result = {}
    for tok_str in target_tokens:
        tok_ids = model.to_tokens(tok_str, prepend_bos=False)[0]
        if len(tok_ids) > 0:
            result[tok_str] = log_probs[tok_ids[0]].item()
        else:
            result[tok_str] = float("-inf")
    return result


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(experiment: str, model_key: str, record: dict):
    """Append a single result record to the experiment's JSONL file."""
    out_dir = RESULTS_DIR / experiment / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(experiment: str, model_key: str, filename: str, data: dict):
    """Write a dict as a formatted JSON file under the experiment directory."""
    out_dir = RESULTS_DIR / experiment / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# VRAM utilities
# ---------------------------------------------------------------------------

def print_vram(device: str = "cuda"):
    """Print current VRAM allocation and reservation."""
    if not torch.cuda.is_available():
        print("  VRAM: CUDA not available")
        return
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved  = torch.cuda.memory_reserved(device) / 1e9
    print(f"  VRAM: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved")


def clear_cache():
    """Free CUDA cache and run Python garbage collection."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()
