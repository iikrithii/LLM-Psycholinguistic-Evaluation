"""
prompts/templates.py

Prompt templates for all five psycholinguistic theories and five task formats.

Each theory has templates for:
  - Forced-choice (FC): two-alternative selection between word A and word B.
  - Rating: Likert-scale (1–7) judgment on a semantic dimension.
  - Generation (sound symbolism only): produce a word matching a description.
  - Contamination probes: assess the model's prior knowledge of each theory.

Each template group contains multiple surface-form variants (indexed 0–4) so
that results can be averaged across prompt phrasings, reducing sensitivity to
any single wording.

Multilingual prompt functions are provided for Japanese, Korean, Hindi, and German.
"""

# ---------------------------------------------------------------------------
# Theory 1 & 3: Sound Symbolism + Vowel-Size Symbolism (shared task format)
# ---------------------------------------------------------------------------

SOUND_FC_TEMPLATES = [
    # 0
    lambda w1, w2, dim: (
        f"I will show you two made-up words that do not exist in any language. "
        f"Based purely on how they sound/look, which one seems more likely to refer to "
        f"something {dim}?\nWord A: {w1}\nWord B: {w2}\n"
        f"Answer with only the letter A or B."
    ),
    # 1
    lambda w1, w2, dim: (
        f"These are two invented, nonsense words: {w1} and {w2}. "
        f"Neither is a real word. If you had to guess which one names something {dim}, "
        f"which would it be? Reply with just A (for {w1}) or B (for {w2})."
    ),
    # 2
    lambda w1, w2, dim: (
        f"Imagine an alien language you have never seen. It has two words: '{w1}' and '{w2}'. "
        f"One of them means something {dim}. Which one? Answer: A or B only."
    ),
    # 3
    lambda w1, w2, dim: (
        f"Two nonsense words: A={w1}, B={w2}. "
        f"Which feels more like a word for something {dim}? "
        f"Trust your instinct. Reply A or B."
    ),
    # 4
    lambda w1, w2, dim: (
        f"You are playing a word-feel game. The words below are made up and meaningless. "
        f"Which word FEELS like it should mean '{dim}'?\n"
        f"Option A: {w1}\nOption B: {w2}\n"
        f"Reply with A or B only. No explanation."
    ),
]

SOUND_RATING_TEMPLATES = [
    # 0
    lambda word, dim_low, dim_high: (
        f"Here is a made-up word that does not exist in any language: {word}\n"
        f"On a scale from 1 to 7, where 1 = very {dim_low} and 7 = very {dim_high}, "
        f"how would you rate what this word seems to refer to?\n"
        f"Reply with a single number between 1 and 7. Nothing else."
    ),
    # 1
    lambda word, dim_low, dim_high: (
        f"Nonsense word: {word} (this is not a real word in any language).\n"
        f"Rate it on this scale: 1 ({dim_low}) to 7 ({dim_high}).\n"
        f"Just give the number."
    ),
    # 2
    lambda word, dim_low, dim_high: (
        f"I invented the word '{word}'. It doesn't mean anything yet. "
        f"Intuitively, if it had to refer to something, would that thing be more {dim_low} or {dim_high}? "
        f"Rate from 1 (totally {dim_low}) to 7 (totally {dim_high}). Number only."
    ),
    # 3
    lambda word, dim_low, dim_high: (
        f"Rate the 'feel' of this nonsense word: {word}\n"
        f"Scale: 1 = extremely {dim_low} ... 7 = extremely {dim_high}\n"
        f"Respond with one integer only."
    ),
    # 4
    lambda word, dim_low, dim_high: (
        f"Word: {word} (invented, not real).\n"
        f"Dimension: {dim_low} ←→ {dim_high} (1 to 7).\n"
        f"Your rating (just the number):"
    ),
]

SOUND_GENERATION_TEMPLATES = [
    # 0
    lambda desc: (
        f"Invent a new word for an imaginary language. The word should FEEL like "
        f"it names something that is {desc}. Do not use any real word from any language. "
        f"Give only the invented word, nothing else."
    ),
    # 1
    lambda desc: (
        f"Make up a single nonsense word that intuitively sounds like it could mean "
        f"'{desc}'. It must not be a real word. Output only the word."
    ),
    # 2
    lambda desc: (
        f"You are a linguist designing words for a new language. "
        f"Create one word whose sound matches the concept: {desc}. "
        f"No real words allowed. Just the invented word."
    ),
    # 3
    lambda desc: (
        f"Coin a new word that phonetically 'feels' right for: {desc}. "
        f"Must be a non-word in all languages you know. One word only."
    ),
    # 4
    lambda desc: (
        f"If you had to invent a word from scratch that sounds like '{desc}', "
        f"what would it be? Not a real word. Just the word."
    ),
]


# ---------------------------------------------------------------------------
# Theory 2: Phonesthesia
# ---------------------------------------------------------------------------

PHONESTHESIA_FC_TEMPLATES = [
    # 0
    lambda word, cat_a, cat_b: (
        f"The following is a made-up word: {word}\n"
        f"Which of these two categories does it fit better?\n"
        f"A: {cat_a}\nB: {cat_b}\n"
        f"Answer A or B only."
    ),
    # 1
    lambda word, cat_a, cat_b: (
        f"Nonsense word: {word}. If it had to belong to one category, which?\n"
        f"A = {cat_a}\nB = {cat_b}\n"
        f"One letter answer only."
    ),
    # 2
    lambda word, cat_a, cat_b: (
        f"Imagine you encountered the invented word '{word}' in a text. "
        f"Based purely on how it looks/sounds, is it more related to:\n"
        f"(A) {cat_a}  or  (B) {cat_b}?\n"
        f"Reply: A or B."
    ),
    # 3
    lambda word, cat_a, cat_b: (
        f"Made-up word: {word}\nWhich semantic field does it belong to by feel?\n"
        f"A: {cat_a}\nB: {cat_b}\nAnswer:"
    ),
    # 4
    lambda word, cat_a, cat_b: (
        f"'{word}' is an invented word. "
        f"Does it sound more like something from category A ({cat_a}) "
        f"or category B ({cat_b})? Just A or B."
    ),
]

PHONESTHESIA_RATING_TEMPLATES = [
    # 0
    lambda word, dim: (
        f"Rate the invented word '{word}' on the following dimension:\n"
        f"{dim}\nScale: 1 (not at all) to 7 (very strongly).\n"
        f"Single number only."
    ),
    # 1
    lambda word, dim: (
        f"Nonsense word: {word}.\nHow strongly does it evoke: {dim}?\n"
        f"1 = not at all, 7 = very strongly. Number only."
    ),
    # 2
    lambda word, dim: (
        f"'{word}' is a made-up, meaningless word. "
        f"If you had to assign it a meaning, how much would it relate to {dim}? "
        f"Rate 1-7. Just the number."
    ),
    # 3
    lambda word, dim: (
        f"Word (invented): {word}\nProperty: {dim}\n"
        f"Rating 1 (none) to 7 (strong):"
    ),
    # 4
    lambda word, dim: (
        f"On a 1-7 scale, does the nonsense word '{word}' feel connected to '{dim}'? "
        f"(1=no, 7=yes). One number."
    ),
]


# ---------------------------------------------------------------------------
# Theory 4: Semantic Prosody (cloze / completion task)
# ---------------------------------------------------------------------------

PROSODY_CLOZE_TEMPLATES = [
    # 0
    lambda sent, opt_a, opt_b: (
        f"Complete the sentence by choosing the more natural-sounding option.\n"
        f"Sentence: \"{sent}\"\n"
        f"Option A: {opt_a}\nOption B: {opt_b}\n"
        f"Which fits better? Answer A or B only."
    ),
    # 1
    lambda sent, opt_a, opt_b: (
        f"Which word fits more naturally in the blank?\n"
        f"\"{sent}\"\nA: {opt_a}  B: {opt_b}\nA or B?"
    ),
    # 2
    lambda sent, opt_a, opt_b: (
        f"A native English speaker would most naturally say:\n"
        f"\"{sent}\" — using (A) {opt_a} or (B) {opt_b}?\n"
        f"Reply with A or B."
    ),
    # 3
    lambda sent, opt_a, opt_b: (
        f"Sentence with blank: {sent}\n"
        f"Fill the blank: A={opt_a} or B={opt_b}. Which sounds more idiomatic? A or B."
    ),
    # 4
    lambda sent, opt_a, opt_b: (
        f"Natural language preference: in the sentence '{sent}', "
        f"is '{opt_a}' (A) or '{opt_b}' (B) the more typical word? A or B."
    ),
]

PROSODY_RATING_TEMPLATES = [
    # 0
    lambda sent: (
        f"Rate the naturalness of this sentence on a scale from 1 (very unnatural) "
        f"to 7 (completely natural for a native English speaker):\n"
        f"\"{sent}\"\nJust the number."
    ),
    # 1
    lambda sent: (
        f"How natural does this sound in English? (1=odd, 7=perfectly natural)\n"
        f"\"{sent}\"\nNumber only."
    ),
    # 2
    lambda sent: (
        f"A native speaker rating: 1-7 for how idiomatic this sentence is.\n"
        f"\"{sent}\"\nRating:"
    ),
    # 3
    lambda sent: (
        f"Idiomaticity score (1=unnatural, 7=natural) for:\n\"{sent}\"\nScore:"
    ),
    # 4
    lambda sent: (
        f"Does this sentence sound like something a native English speaker "
        f"would say? 1 (no) to 7 (yes).\n\"{sent}\"\nNumber:"
    ),
]


# ---------------------------------------------------------------------------
# Theory 5: Ideophone Compositionality
# ---------------------------------------------------------------------------

IDEOPHONE_RATING_TEMPLATES = [
    # 0
    lambda word, lang, dim_low, dim_high: (
        f"The following is an invented word written in the style of {lang} sound-symbolic "
        f"words (words whose sound reflects their meaning): {word}\n"
        f"Rate what this word seems to depict on a scale:\n"
        f"1 = very {dim_low}  ...  7 = very {dim_high}\n"
        f"One number only."
    ),
    # 1
    lambda word, lang, dim_low, dim_high: (
        f"In {lang}, some words iconically reflect what they describe. "
        f"Here is an invented word in that style: {word}\n"
        f"Does it feel more {dim_low} (1) or {dim_high} (7)? Just the number."
    ),
    # 2
    lambda word, lang, dim_low, dim_high: (
        f"Imagine this is a {lang} expressive/sound-symbolic word: {word}\n"
        f"On the scale 1 ({dim_low}) to 7 ({dim_high}), rate what it depicts."
    ),
    # 3
    lambda word, lang, dim_low, dim_high: (
        f"{lang} sound-symbolic invented word: {word}\n"
        f"Dimension: {dim_low} ↔ {dim_high}\nRating (1-7):"
    ),
    # 4
    lambda word, lang, dim_low, dim_high: (
        f"You are evaluating {lang}-style iconic words. Word: {word}\n"
        f"Rate how {dim_high} vs {dim_low} it feels (1={dim_low}, 7={dim_high}). Number only."
    ),
]

IDEOPHONE_FC_TEMPLATES = [
    # 0
    lambda w1, w2, lang, dim: (
        f"Both words below are invented in the style of {lang} sound-symbolic words. "
        f"Which one feels more {dim}?\nA: {w1}\nB: {w2}\nA or B only."
    ),
    # 1
    lambda w1, w2, lang, dim: (
        f"{lang} style invented words: {w1} vs {w2}. "
        f"Which is more {dim}? A ({w1}) or B ({w2})?"
    ),
    # 2
    lambda w1, w2, lang, dim: (
        f"Compare these invented {lang}-style expressive words: "
        f"A={w1}, B={w2}. Which better depicts '{dim}'? A or B."
    ),
    # 3
    lambda w1, w2, lang, dim: (
        f"Invented {lang} sound-words:\nA: {w1}\nB: {w2}\n"
        f"Which depicts '{dim}' more strongly? A or B."
    ),
    # 4
    lambda w1, w2, lang, dim: (
        f"Pick the {lang}-style invented word that sounds more {dim}:\n"
        f"A: {w1}  B: {w2}\nAnswer:"
    ),
]


# ---------------------------------------------------------------------------
# Contamination probe templates
# ---------------------------------------------------------------------------

CONTAMINATION_DIRECT = [
    lambda theory: f"What is the {theory}? Describe it briefly in 2-3 sentences.",
    lambda theory: f"Have you heard of '{theory}'? What does it involve?",
    lambda theory: f"Explain the {theory} as if to a psychology student.",
    lambda theory: f"What does research on the {theory} show?",
    lambda theory: f"In linguistics or psychology, what is the '{theory}'?",
]

CONTAMINATION_WORD_PROBE = [
    lambda word: (
        f"Is the word or pseudoword '{word}' associated with any psychological or "
        f"linguistic phenomenon you know of? Yes or no, then briefly explain."
    ),
    lambda word: (
        f"Do you recognize '{word}' from any scientific study or experiment? "
        f"What do you know about it?"
    ),
    lambda word: f"Has the word '{word}' been used in any famous experiment? If yes, which one?",
    lambda word: (
        f"'{word}' — does this appear in any psychology or linguistics research "
        f"you are aware of?"
    ),
    lambda word: (
        f"What, if anything, do you associate with the word '{word}' "
        f"from an academic context?"
    ),
]


# ---------------------------------------------------------------------------
# Multilingual prompt functions
# ---------------------------------------------------------------------------

def multilingual_fc(w1, w2, dim, lang_name, lang_instruction):
    """Forced-choice template with language-context preamble."""
    return (
        f"The following task is about phonological intuitions in {lang_name}.\n"
        f"{lang_instruction}\n\n"
        f"Two invented words (not real words in any language): {w1} and {w2}.\n"
        f"Which one sounds more like a {lang_name} word that means something {dim}?\n"
        f"Answer A (for {w1}) or B (for {w2}) only."
    )


def multilingual_rating(word, dim_low, dim_high, lang_name, lang_instruction):
    """Rating template with language-context preamble."""
    return (
        f"The following task concerns phonological intuitions in {lang_name}.\n"
        f"{lang_instruction}\n\n"
        f"Invented word (not real): {word}\n"
        f"If this were a {lang_name} word, rate what it would likely mean:\n"
        f"1 = very {dim_low}  ...  7 = very {dim_high}\n"
        f"One number only."
    )


LANG_INSTRUCTIONS = {
    "ja": "Think about Japanese sound-symbolic (擬態語/擬音語) intuitions when answering.",
    "ko": "Think about Korean sound-symbolic (의태어/의성어) intuitions when answering.",
    "hi": "Think about Hindi phonological and sound-symbolic patterns when answering.",
    "de": "Think about German phonological intuitions when answering.",
    "en": "Think about English phonological intuitions when answering.",
}

LANG_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "de": "German",
    "en": "English",
}
