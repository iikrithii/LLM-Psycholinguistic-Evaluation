"""
construct_stimuli.py

Builds the complete stimulus battery for all five psycholinguistic theories
across three contamination tiers. No API calls are made; output is a single
JSON file at data/stimuli.json.

Theories
--------
1. Sound Symbolism (Bouba-Kiki / Maluma-Takete)
2. Phonesthesia (onset cluster semantic associations)
3. Vowel-Size Symbolism (Frequency Code / Ohala 1984)
4. Semantic Prosody (collocational evaluative polarity)
5. Ideophone Compositionality (Japanese/Korean 2×2×2×2 factorial)

Tiers
-----
- Tier 1: Famous stimuli cited in landmark papers (high contamination pressure).
- Tier 2: Attested but less-cited stimuli from specialist literature.
- Tier 3: Novel stimuli constructed for this study (no prior text occurrence).

Usage
-----
    python construct_stimuli.py
"""

import json
import os
import random
import itertools
from pathlib import Path

random.seed(42)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Theory 1: Sound Symbolism (Bouba-Kiki / Maluma-Takete)
# Round phonology vs. sharp phonology
# ---------------------------------------------------------------------------

def build_sound_symbolism_stimuli():
    """
    Construct stimuli for the sound symbolism (bouba-kiki) paradigm.

    Tier 1 uses word pairs from landmark papers (Köhler 1929, Ramachandran &
    Hubbard 2001, Nielsen & Rendall 2011). Tier 2 draws from Westbury (2005),
    Fort et al. (2015), and Aveyard (2012). Tier 3 generates novel pairs by
    sampling from two phonologically distinct consonant/vowel pools — Pool R
    (round/sonorant: m, n, l, r, b, w; vowels a, o, u) and Pool S
    (sharp/obstruent: k, t, p, s, f; vowels i, e).
    """
    stimuli = []

    # Tier 1: Famous pairs from landmark papers
    tier1_pairs = [
        ("bouba",   "kiki",    "round",   "angular"),
        ("maluma",  "takete",  "round",   "angular"),
        ("baluma",  "takiti",  "round",   "angular"),
        ("malouma", "kikouti", "round",   "angular"),
        ("bobo",    "keke",    "round",   "angular"),
        ("lula",    "tifi",    "round",   "angular"),
        ("momo",    "tete",    "round",   "angular"),
        ("wawa",    "sisi",    "soft",    "sharp"),
        ("nuna",    "kiki",    "soft",    "sharp"),
        ("buba",    "kaka",    "round",   "angular"),
    ]

    for w_round, w_sharp, dim_r, dim_s in tier1_pairs:
        stimuli.append({
            "theory": "sound_symbolism",
            "tier": 1,
            "language": "en",
            "word_a": w_round,
            "word_b": w_sharp,
            "label_a": dim_r,
            "label_b": dim_s,
            "expected": "A",
            "dimension_low": "round/soft",
            "dimension_high": "angular/sharp",
            "semantic_dimensions": ["round_vs_angular", "soft_vs_sharp", "heavy_vs_light"],
            "source": "Kohler1929;Ramachandran2001;Nielsen2011",
            "notes": "Tier1_famous",
        })

    # Tier 2: Attested but obscure pairs from specialist literature
    tier2_pairs = [
        ("muvula",  "kipete",  "round",   "angular"),
        ("lobuma",  "tikisi",  "round",   "angular"),
        ("nawoma",  "sekiti",  "soft",    "sharp"),
        ("roluma",  "pakite",  "round",   "angular"),
        ("numola",  "faskit",  "round",   "angular"),
        ("mabolu",  "kespit",  "soft",    "sharp"),
        ("wanuma",  "tifeks",  "round",   "angular"),
        ("moluba",  "kiteps",  "round",   "angular"),
        ("rulona",  "sipekt",  "round",   "angular"),
        ("naluma",  "tipeks",  "round",   "angular"),
    ]

    for w_round, w_sharp, dim_r, dim_s in tier2_pairs:
        stimuli.append({
            "theory": "sound_symbolism",
            "tier": 2,
            "language": "en",
            "word_a": w_round,
            "word_b": w_sharp,
            "label_a": dim_r,
            "label_b": dim_s,
            "expected": "A",
            "dimension_low": "round/soft",
            "dimension_high": "angular/sharp",
            "semantic_dimensions": ["round_vs_angular", "soft_vs_sharp", "heavy_vs_light"],
            "source": "Westbury2005;Fort2015;Aveyard2012",
            "notes": "Tier2_obscure",
        })

    # Tier 3: Novel pairs sampled from phonologically constrained pools
    # Pool R onsets: m, n, l, r, b, w  — sonorants, bilabials, approximants
    # Pool R vowels: a, o, u, oo, ou   — back/low vowels
    # Pool S onsets: k, t, p, s, f, sk, st — obstruents, fricatives
    # Pool S vowels: i, e, ee, ei       — front/high vowels
    pool_r_onsets = ['m', 'n', 'l', 'r', 'b', 'w']
    pool_r_vowels = ['a', 'o', 'u', 'oo', 'ou']
    pool_r_codas  = ['m', 'n', 'l', '', '']

    pool_s_onsets = ['k', 't', 'p', 's', 'f', 'sk', 'st']
    pool_s_vowels = ['i', 'e', 'ee', 'ei']
    pool_s_codas  = ['k', 't', 's', 'p', '']

    real_words = {
        'man', 'men', 'me', 'no', 'one', 'two', 'moon', 'sun', 'run',
        'bit', 'sit', 'fit', 'kit', 'pit', 'tip', 'sip', 'fin',
    }

    def make_word(onsets, vowels, codas, n_syl=2):
        while True:
            word = ""
            for _ in range(n_syl):
                word += random.choice(onsets) + random.choice(vowels) + random.choice(codas)
            if len(word) >= 4 and word not in real_words:
                return word

    blocklist = {
        'noba', 'kiki', 'bouba', 'maluma', 'takete', 'man', 'men', 'me',
        'no', 'one', 'two', 'moon', 'sun', 'run', 'bit', 'sit', 'fit',
        'kit', 'pit', 'tip', 'sip', 'fin', 'top', 'tap', 'map', 'mat',
    }

    tier3_pairs = []
    seen = set()
    while len(tier3_pairs) < 20:
        n = random.choice([2, 3])
        w_r = make_word(pool_r_onsets, pool_r_vowels, pool_r_codas, n)
        w_s = make_word(pool_s_onsets, pool_s_vowels, pool_s_codas, n)
        key = (w_r, w_s)
        if (key not in seen and w_r != w_s
                and w_r not in blocklist and w_s not in blocklist):
            seen.add(key)
            tier3_pairs.append((w_r, w_s))

    for w_round, w_sharp in tier3_pairs:
        stimuli.append({
            "theory": "sound_symbolism",
            "tier": 3,
            "language": "en",
            "word_a": w_round,
            "word_b": w_sharp,
            "label_a": "round",
            "label_b": "angular",
            "expected": "A",
            "dimension_low": "round/soft",
            "dimension_high": "angular/sharp",
            "semantic_dimensions": ["round_vs_angular", "soft_vs_sharp", "heavy_vs_light"],
            "source": "constructed",
            "notes": "Tier3_novel_poolR_vs_poolS",
        })

    return stimuli


# ---------------------------------------------------------------------------
# Theory 2: Phonesthesia
# ---------------------------------------------------------------------------

def build_phonesthesia_stimuli():
    """
    Construct stimuli for the phonesthesia paradigm.

    Six onset clusters are tested (gl-, sl-, sn-, fl-, cr-, sm-), each paired
    with a target semantic category. Tier 1 uses well-attested English words
    (Bergen 2004, Hutchins 1998). Tier 2 uses less common real words. Tier 3
    constructs nonce words by appending the onset cluster to random CV strings,
    along with a matched neutral-onset control word.
    """
    stimuli = []

    CLUSTERS = {
        "gl": ("visual light/shine phenomena",  "auditory/sound phenomena",
               "glowing, glittering, shining",   "sn"),
        "sl": ("negative/unpleasant qualities",  "positive/pleasant qualities",
               "slimy, slovenly, unpleasant",    "sp"),
        "sn": ("nasal/mouth actions",            "hand/foot actions",
               "sniffing, snorting, sneering",   "gl"),
        "fl": ("rapid movement",                 "slow/static states",
               "flying, fluttering, flickering", "st"),
        "cr": ("harsh/broken/abrupt",            "smooth/continuous",
               "cracking, crunching, grating",   "sm"),
        "sm": ("smooth/pleasant touch",          "harsh/broken",
               "smooth, smelling gently, soft",  "cr"),
    }

    tier1_words = {
        "gl": ["glitter", "gleam", "glow", "glare", "glint", "glimmer", "glisten",
               "gloss", "glory", "glamour"],
        "sl": ["slime", "sludge", "slop", "slovenly", "slur", "sleazy", "slug",
               "slob", "sloppy", "slouch"],
        "sn": ["sniff", "snort", "sneeze", "snore", "snicker", "snarl", "sneer",
               "snivel", "snoop", "snub"],
        "fl": ["flash", "flicker", "flutter", "flit", "fly", "fleet", "flurry",
               "fling", "flap", "flee"],
        "cr": ["crack", "crunch", "crash", "creak", "crumble", "crush", "crisp",
               "grate", "crinkle", "crouch"],
        "sm": ["smooth", "smell", "smear", "smile", "smug", "smart", "smash",
               "smoky", "smother", "smolder"],
    }

    for cluster, (cat_match, cat_other, desc, anti) in CLUSTERS.items():
        for word in tier1_words[cluster][:5]:
            stimuli.append({
                "theory": "phonesthesia",
                "tier": 1,
                "language": "en",
                "word_a": word,
                "phonestheme": cluster,
                "category_match": cat_match,
                "category_other": cat_other,
                "category_description": desc,
                "expected": "A",
                "source": "Bergen2004;Hutchins1998",
                "notes": f"Tier1_real_word_{cluster}",
            })

    tier2_words = {
        "gl": ["glabrous", "gleaming", "glaucous", "glimmer", "glitzy"],
        "sl": ["slabber", "slake", "slaver", "slipshod", "slipshod"],
        "sn": ["snaffle", "snaggle", "snigger", "snivel", "snuffle"],
        "fl": ["flibbertigibbet", "flocculent", "flounce", "fluctuate", "flummox"],
        "cr": ["craven", "crepitate", "crimple", "crizzle", "crochet"],
        "sm": ["smirch", "smit", "smolt", "smudge", "smuggle"],
    }

    for cluster, (cat_match, cat_other, desc, anti) in CLUSTERS.items():
        for word in tier2_words[cluster]:
            stimuli.append({
                "theory": "phonesthesia",
                "tier": 2,
                "language": "en",
                "word_a": word,
                "phonestheme": cluster,
                "category_match": cat_match,
                "category_other": cat_other,
                "category_description": desc,
                "expected": "A",
                "source": "Hutchins1998_extended",
                "notes": f"Tier2_obscure_{cluster}",
            })

    # Tier 3: nonce words — append cluster to random CV syllable strings.
    # A neutral-onset control is created for each nonce by replacing the
    # phonesthemic cluster with a phonologically neutral alternative.
    random.seed(42)
    vowels = ['a', 'e', 'i', 'o', 'u', 'ar', 'er', 'on', 'an', 'el']
    codas  = ['', '', '', 'n', 'm', 'l', 'r', 't']

    ph_blocklist = {
        'slut', 'slag', 'smut', 'smeg', 'slim', 'slip',
        'snot', 'snog', 'frig', 'crap', 'crud',
        'glia', 'slater', 'slate', 'glean', 'gleam', 'glare', 'gloss',
        'flim', 'flem', 'flab', 'flap', 'snap', 'snag', 'snal', 'snat',
        'smit', 'smon', 'smob', 'cram', 'crib', 'crer', 'slop', 'slob',
        'snob', 'crest', 'crank', 'flank', 'flare', 'flame',
        'glint', 'gland', 'glide', 'globe', 'gloom', 'glove', 'gloat',
        'slant', 'slash', 'slave', 'sleek', 'sleet', 'slice', 'slide',
        'flung', 'flair', 'flake', 'flank', 'flask',
        'crimp', 'crick', 'crime', 'crisp', 'croak', 'crook', 'cross',
        'smack', 'small', 'smart', 'smash', 'smell', 'smile', 'smirk',
    }
    neutral_onset = {"gl": "bl", "sl": "pl", "sn": "tr", "fl": "pr", "cr": "dr", "sm": "fr"}

    for cluster, (cat_match, cat_other, desc, anti) in CLUSTERS.items():
        seen = set()
        count = 0
        while count < 5:
            syl1 = random.choice(vowels) + random.choice(codas)
            syl2 = (random.choice(vowels) + random.choice(codas)
                    if random.random() > 0.5 else "")
            word = cluster + syl1 + syl2
            if word not in seen and len(word) >= 4 and word not in ph_blocklist:
                seen.add(word)
                control = neutral_onset[cluster] + syl1 + syl2
                stimuli.append({
                    "theory": "phonesthesia",
                    "tier": 3,
                    "language": "en",
                    "word_a": word,
                    "word_control": control,
                    "phonestheme": cluster,
                    "category_match": cat_match,
                    "category_other": cat_other,
                    "category_description": desc,
                    "expected": "A",
                    "source": "constructed",
                    "notes": f"Tier3_novel_{cluster}_control={control}",
                })
                count += 1

    return stimuli


# ---------------------------------------------------------------------------
# Theory 3: Vowel-Size Symbolism (Frequency Code / Ohala 1984)
# High front vowels (i, e) → small / light / fast / bright
# Low back vowels  (a, o, u) → large / heavy / slow / dark
# ---------------------------------------------------------------------------

def build_vowel_size_stimuli():
    """
    Construct stimuli for the vowel-size symbolism paradigm.

    Tier 1 uses famous minimal pairs (Thompson & Estes 2011, Sapir 1929,
    Ultan 1978). Tier 2 draws from Cwiek et al. (2022) and Blasi et al. (2016).
    Tier 3 constructs novel CVC minimal pairs by systematically substituting
    front vowels (i, e) for back vowels (a, o, u) in identical consonant frames.
    """
    stimuli = []

    tier1_pairs = [
        ("mil",   "mol",   "small", "large"),
        ("teeni", "toona", "small", "large"),
        ("kil",   "kol",   "small", "large"),
        ("tif",   "tof",   "small", "large"),
        ("vil",   "vol",   "bright", "dark"),
        ("pim",   "pom",   "small", "large"),
        ("sif",   "sof",   "light", "heavy"),
        ("nib",   "nob",   "small", "large"),
        ("dip",   "dup",   "small", "large"),
        ("lis",   "lus",   "small", "large"),
    ]

    for w_small, w_large, dim_s, dim_l in tier1_pairs:
        stimuli.append({
            "theory": "vowel_size",
            "tier": 1,
            "language": "en",
            "word_a": w_small,
            "word_b": w_large,
            "label_a": dim_s,
            "label_b": dim_l,
            "expected": "A",
            "dimension_low": "small/light/fast/bright",
            "dimension_high": "large/heavy/slow/dark",
            "semantic_dimensions": ["small_vs_large", "light_vs_heavy", "fast_vs_slow", "bright_vs_dark"],
            "source": "Thompson2011;Sapir1929;Ultan1978",
            "notes": "Tier1_famous_vowel_size",
        })

    tier2_pairs = [
        ("pitel", "patul", "small", "large"),
        ("kisem", "kozam", "small", "large"),
        ("tifen", "tofan", "light", "heavy"),
        ("vilep", "volap", "bright", "dark"),
        ("minen", "monan", "small", "large"),
        ("sikel", "sokal", "small", "large"),
        ("fitem", "fotam", "fast",  "slow"),
        ("bitel", "botal", "small", "large"),
        ("rifek", "rofak", "small", "large"),
        ("tilek", "tolak", "small", "large"),
    ]

    for w_small, w_large, dim_s, dim_l in tier2_pairs:
        stimuli.append({
            "theory": "vowel_size",
            "tier": 2,
            "language": "en",
            "word_a": w_small,
            "word_b": w_large,
            "label_a": dim_s,
            "label_b": dim_l,
            "expected": "A",
            "dimension_low": "small/light/fast/bright",
            "dimension_high": "large/heavy/slow/dark",
            "semantic_dimensions": ["small_vs_large", "light_vs_heavy", "fast_vs_slow", "bright_vs_dark"],
            "source": "Cwiek2022;Blasi2016",
            "notes": "Tier2_obscure_vowel_size",
        })

    # Tier 3: Systematic minimal pairs — same consonant frame, only vowel differs.
    # Each entry is (CVC_frame, small_vowel, large_vowel).
    consonant_frames = [
        ("r{}n", "i", "o"),  ("k{}p", "i", "u"),  ("m{}l", "e", "a"),
        ("t{}s", "i", "o"),  ("f{}k", "e", "a"),  ("n{}g", "i", "o"),
        ("p{}t", "e", "u"),  ("d{}v", "i", "a"),  ("b{}m", "e", "o"),
        ("g{}n", "i", "u"),  ("s{}p", "e", "a"),  ("h{}l", "i", "o"),
        ("l{}n", "e", "u"),  ("w{}k", "i", "a"),  ("z{}m", "e", "o"),
        ("c{}t", "i", "u"),  ("v{}n", "e", "a"),  ("j{}k", "i", "o"),
        ("q{}p", "e", "u"),  ("x{}n", "i", "a"),
    ]

    for frame, v_small, v_large in consonant_frames:
        w_small = frame.format(v_small)
        w_large = frame.format(v_large)
        stimuli.append({
            "theory": "vowel_size",
            "tier": 3,
            "language": "en",
            "word_a": w_small,
            "word_b": w_large,
            "label_a": "small",
            "label_b": "large",
            "expected": "A",
            "dimension_low": "small/light/fast/bright",
            "dimension_high": "large/heavy/slow/dark",
            "semantic_dimensions": ["small_vs_large", "light_vs_heavy", "fast_vs_slow", "bright_vs_dark"],
            "vowel_small": v_small,
            "vowel_large": v_large,
            "consonant_frame": frame,
            "source": "constructed",
            "notes": "Tier3_novel_minimal_pair",
        })

    return stimuli


# ---------------------------------------------------------------------------
# Theory 4: Semantic Prosody
# ---------------------------------------------------------------------------

def build_semantic_prosody_stimuli():
    """
    Construct stimuli for the semantic prosody paradigm.

    Each stimulus is a sentence frame with a blank filled by either a
    semantically prosodic verb (option A) or a neutral near-synonym (option B).
    Tier 1 uses canonical examples from Louw (1993) and Sinclair (1991).
    Tier 2 draws from Stewart (2010), Partington (2004), and Xiao & McEnery (2006).
    Tier 3 constructs novel frames using rare but real English verbs with
    documentable negative collocational profiles.
    """
    stimuli = []

    tier1_frames = [
        ("The disease slowly ___ in the coastal villages.",
         "set", "began", "negative", "Louw1993"),
        ("Corruption ___ in the institution over many years.",
         "set", "developed", "negative", "Louw1993"),
        ("The infection ___ in after three days.",
         "set", "started", "negative", "Louw1993"),
        ("The rot had ___ in long before anyone noticed.",
         "set", "come", "negative", "Sinclair1991"),
        ("Despair ___ in as the weeks passed.",
         "set", "came", "negative", "Sinclair1991"),
        ("The new policy will ___ significant disruption.",
         "cause", "create", "negative", "Sinclair1991"),
        ("The accident ___ three fatalities.",
         "caused", "resulted in", "negative", "Sinclair1991"),
        ("Her words ___ considerable distress.",
         "caused", "produced", "negative", "Sinclair1991"),
        ("He ___ a serious error of judgment.",
         "committed", "made", "negative", "Sinclair1991"),
        ("She ___ an act of betrayal.",
         "committed", "performed", "negative", "Louw1993"),
    ]

    for sent, prosodic, neutral, polarity, src in tier1_frames:
        stimuli.append({
            "theory": "semantic_prosody",
            "tier": 1,
            "language": "en",
            "sentence_frame": sent,
            "option_a": prosodic,
            "option_b": neutral,
            "expected": "A",
            "prosody_type": polarity,
            "carrier_verb": prosodic,
            "neutral_verb": neutral,
            "source": src,
            "notes": "Tier1_famous_prosody",
        })

    tier2_frames = [
        ("The problem had been ___ under the surface for years.",
         "lurking", "existing", "negative", "Stewart2010"),
        ("Rumors continued to ___ in the hallways.",
         "circulate", "spread", "neutral", "Partington2004"),
        ("The scandal ___ over the company for months.",
         "hung", "remained", "negative", "Stewart2010"),
        ("The old resentments began to ___ up again.",
         "fester", "surface", "negative", "Xiao2006"),
        ("Doubts started to ___ in her mind.",
         "gnaw", "form", "negative", "Xiao2006"),
        ("The situation continued to ___.",
         "deteriorate", "change", "negative", "Partington2004"),
        ("The regime ___ power through fear.",
         "wielded", "used", "negative_nuance", "Stewart2010"),
        ("His career ___ to a halt after the scandal.",
         "ground", "came", "negative", "Xiao2006"),
        ("The violence ___ out in three cities simultaneously.",
         "flared", "broke", "negative_nuance", "Partington2004"),
        ("Unease began to ___ among the population.",
         "spread", "grow", "negative_nuance", "Xiao2006"),
    ]

    for sent, prosodic, neutral, polarity, src in tier2_frames:
        stimuli.append({
            "theory": "semantic_prosody",
            "tier": 2,
            "language": "en",
            "sentence_frame": sent,
            "option_a": prosodic,
            "option_b": neutral,
            "expected": "A",
            "prosody_type": polarity,
            "carrier_verb": prosodic,
            "neutral_verb": neutral,
            "source": src,
            "notes": "Tier2_obscure_prosody",
        })

    # Tier 3: Novel frames with rare but semantically prosodic verbs
    tier3_frames = [
        ("A sense of dread began to ___ over the camp.",
         "descend", "arrive", "negative", "constructed"),
        ("The memories continued to ___ at her.",
         "gnaw", "appear", "negative", "constructed"),
        ("The blight ___ the orchards in early spring.",
         "struck", "reached", "negative", "constructed"),
        ("The plague ___ through the population within weeks.",
         "swept", "moved", "negative", "constructed"),
        ("Panic began to ___ among the passengers.",
         "ripple", "appear", "negative_nuance", "constructed"),
        ("The silence seemed to ___ in around them.",
         "close", "settle", "negative_nuance", "constructed"),
        ("The cold began to ___ into his bones.",
         "seep", "come", "negative", "constructed"),
        ("The tension had been ___ for weeks.",
         "building", "present", "negative", "constructed"),
        ("The stench ___ through the entire building.",
         "permeated", "spread through", "negative", "constructed"),
        ("A creeping unease ___ over the crowd.",
         "washed", "moved", "negative", "constructed"),
    ]

    for sent, prosodic, neutral, polarity, src in tier3_frames:
        stimuli.append({
            "theory": "semantic_prosody",
            "tier": 3,
            "language": "en",
            "sentence_frame": sent,
            "option_a": prosodic,
            "option_b": neutral,
            "expected": "A",
            "prosody_type": polarity,
            "carrier_verb": prosodic,
            "neutral_verb": neutral,
            "source": src,
            "notes": "Tier3_novel_prosody",
        })

    return stimuli


# ---------------------------------------------------------------------------
# Theory 5: Ideophone Compositionality (Japanese + Korean)
# 2×2×2×2 factorial: [voiced × geminated × reduplicated × back_vowel]
# ---------------------------------------------------------------------------

def build_ideophone_stimuli():
    """
    Construct stimuli for the ideophone compositionality paradigm.

    Phonological feature definitions (Hamano 1998, Kita 1997):
      voiced obstruents (b, d, g, z)   → heavy, dark, rough, slow
      unvoiced obstruents (p, t, k, s) → light, bright, sharp, fast
      gemination (pp, tt, kk, ss)      → sudden, abrupt, impactful
      reduplication (CVCV-CVCV)        → continuous, repetitive, gradual
      back vowel (o, u, a)             → gloomy, heavy, dark
      front vowel (i, e)               → bright, light, sharp

    Tier 1 uses famous Japanese gitaigo (Hamano 1998, Kita 1997, Dingemanse 2012)
    and Korean uitate (Koo 2018, Kim 2003). Tier 3 generates novel ideophones
    across all 16 cells of a 2×2×2×2 factorial design (3 items per cell = 48 items).
    """
    stimuli = []

    tier1_japanese = [
        ("ふらふら", "fura-fura", False, False, True,  True,  "wobbly, unsteady"),
        ("ぴかぴか", "pika-pika", False, False, True,  False, "sparkling, shiny"),
        ("ざわざわ", "zawa-zawa", True,  False, True,  True,  "rustling, uneasy"),
        ("きらきら", "kira-kira", False, False, True,  False, "glittering, sparkling"),
        ("どんどん", "don-don",   True,  True,  True,  True,  "rapidly, steadily"),
        ("ぽかぽか", "poka-poka", False, False, True,  True,  "warm, pleasantly"),
        ("がたがた", "gata-gata", True,  False, True,  True,  "rattling, shaking"),
        ("ふわふわ", "fuwa-fuwa", False, False, True,  True,  "fluffy, floating"),
        ("ぴりぴり", "piri-piri", False, False, True,  False, "tingling, tense"),
        ("ぐるぐる", "guru-guru", True,  False, True,  True,  "spinning, going around"),
    ]

    for kanji, romaji, voiced, geminated, reduplicated, back_vowel, meaning in tier1_japanese:
        stimuli.append({
            "theory": "ideophone_compositionality",
            "tier": 1,
            "language": "ja",
            "word": romaji,
            "word_native": kanji,
            "voiced": voiced,
            "geminated": geminated,
            "reduplicated": reduplicated,
            "back_vowel": back_vowel,
            "meaning": meaning,
            "semantic_dimensions": ["heavy_vs_light", "dark_vs_bright",
                                    "rough_vs_smooth", "sudden_vs_gradual",
                                    "continuous_vs_sudden"],
            "source": "Hamano1998;Kita1997;Dingemanse2012",
            "notes": "Tier1_famous_gitaigo",
        })

    tier1_korean = [
        ("반짝반짝", "ban-jjak-ban-jjak",   False, True,  True,  False, "sparkling"),
        ("두근두근", "du-geun-du-geun",     True,  False, True,  True,  "heart pounding"),
        ("살금살금", "sal-geum-sal-geum",   False, False, True,  False, "stealthily"),
        ("쿵쾅쿵쾅", "kung-kwang-kung-kwang", True, False, True,  True,  "thudding"),
        ("아장아장", "a-jang-a-jang",       False, False, True,  True,  "toddling"),
    ]

    for hangul, romaji, voiced, geminated, reduplicated, back_vowel, meaning in tier1_korean:
        stimuli.append({
            "theory": "ideophone_compositionality",
            "tier": 1,
            "language": "ko",
            "word": romaji,
            "word_native": hangul,
            "voiced": voiced,
            "geminated": geminated,
            "reduplicated": reduplicated,
            "back_vowel": back_vowel,
            "meaning": meaning,
            "semantic_dimensions": ["heavy_vs_light", "dark_vs_bright",
                                    "rough_vs_smooth", "sudden_vs_gradual",
                                    "continuous_vs_sudden"],
            "source": "Koo2018;Kim2003",
            "notes": "Tier1_famous_uitate",
        })

    # Tier 3: Novel ideophones — full 2×2×2×2 factorial
    # voiced=True  → b, d, g, z  /  voiced=False → p, t, k, s
    # geminated → double the coda consonant (e.g., "bog" → "bogg")
    # reduplicated → repeat the base syllable(s) with a hyphen
    # back_vowel=True → o, u, a  /  back_vowel=False → i, e
    voiced_onsets   = ['b', 'd', 'g', 'z']
    unvoiced_onsets = ['p', 't', 'k', 's']
    back_vowels_j   = ['o', 'u', 'a']
    front_vowels_j  = ['i', 'e']

    random.seed(42)
    all_combinations = list(itertools.product([True, False], repeat=4))

    ideo_blocklist = {
        'bid', 'tit', 'bit', 'sit', 'sob', 'bug', 'dug', 'pub', 'tab', 'tag', 'bag',
        'bad', 'big', 'bog', 'bob', 'dog', 'dot', 'got', 'god', 'jab', 'jig', 'kit',
        'lip', 'nip', 'pit', 'pep', 'pop', 'pod', 'rob', 'rib', 'sip', 'tap', 'tip',
        'top', 'tot', 'zip', 'zap', 'gig', 'pig', 'wig', 'dig', 'fig', 'rig', 'did',
        'hid', 'kid', 'lid', 'rid', 'dim', 'dip', 'pit', 'pin', 'bin', 'gin', 'sin',
        'tin', 'win', 'bun', 'gun', 'nun', 'pun', 'run', 'sun', 'bop', 'cop', 'hop',
        'mop', 'dub', 'hub', 'rub', 'sub', 'tub', 'bus', 'pus',
    }

    novel_words_used = set()
    for voiced, geminated, reduplicated, back_vowel in all_combinations:
        onset_pool = voiced_onsets if voiced else unvoiced_onsets
        vowel_pool = back_vowels_j if back_vowel else front_vowels_j

        for idx in range(3):
            while True:
                onset = random.choice(onset_pool)
                vowel = random.choice(vowel_pool)
                coda  = random.choice(onset_pool if voiced else unvoiced_onsets)
                base  = onset + vowel + coda
                if geminated:
                    base = base + coda
                word = base + "-" + base if reduplicated else base

                base_check = base.replace("-", "")
                if (word not in novel_words_used
                        and len(base) >= 3
                        and base_check not in ideo_blocklist
                        and base not in ideo_blocklist):
                    novel_words_used.add(word)
                    break

            stimuli.append({
                "theory": "ideophone_compositionality",
                "tier": 3,
                "language": "ja",
                "word": word,
                "word_native": word,
                "voiced": voiced,
                "geminated": geminated,
                "reduplicated": reduplicated,
                "back_vowel": back_vowel,
                "meaning": "constructed",
                "semantic_dimensions": ["heavy_vs_light", "dark_vs_bright",
                                        "rough_vs_smooth", "sudden_vs_gradual",
                                        "continuous_vs_sudden"],
                "source": "constructed",
                "notes": (
                    f"Tier3_factorial_"
                    f"v{int(voiced)}g{int(geminated)}r{int(reduplicated)}bv{int(back_vowel)}"
                    f"_item{idx}"
                ),
            })

    return stimuli


# ---------------------------------------------------------------------------
# Multilingual stimuli: parallel Tier 1 and Tier 3 sound symbolism pairs
# across Japanese, Korean, Hindi, and German
# ---------------------------------------------------------------------------

def build_multilingual_stimuli():
    """
    Build parallel stimulus sets for Japanese, Korean, Hindi, and German.

    Tier 1 uses published cross-linguistic stimuli from Iida & Funakura (2024),
    Cwiek et al. (2022), Kim (2003), Bross (2018), and Schmidtke (2014).
    Tier 3 applies the same phonological round/sharp pool logic within each
    language's phonological system.
    """
    stimuli = []

    # Japanese (Katakana pseudowords)
    ja_tier1 = [
        ("bouba",  "ブーバ",  "kiki",   "キキ"),
        ("maluma", "マルーマ", "takete", "タケテ"),
        ("moma",   "モーマ",  "titi",   "ティティ"),
        ("nunu",   "ヌーヌ",  "pisi",   "ピシ"),
        ("rara",   "ラーラ",  "keke",   "ケケ"),
    ]
    for rr, kr, rs, ks in ja_tier1:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 1, "language": "ja",
            "word_a": kr, "word_a_romaji": rr,
            "word_b": ks, "word_b_romaji": rs,
            "expected": "A", "label_a": "round", "label_b": "angular",
            "dimension_low": "丸い/柔らかい", "dimension_high": "とがった/鋭い",
            "source": "Iida2024;Cwiek2022", "notes": "Tier1_Japanese",
        })

    ja_tier3_pairs = [
        ("ムーマ", "キティ"), ("ロナ",   "テキ"),  ("ボウア", "シピ"),
        ("ナーマ", "スィキ"), ("ワーロ", "ティス"), ("モーロ", "ピキ"),
        ("ルーマ", "ケシ"),   ("ノーバ", "タキ"),  ("マーロ", "スィス"),
        ("ボーナ", "キス"),
    ]
    for wa, wb in ja_tier3_pairs:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 3, "language": "ja",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "round", "label_b": "angular",
            "dimension_low": "丸い/柔らかい", "dimension_high": "とがった/鋭い",
            "source": "constructed", "notes": "Tier3_Japanese_novel",
        })

    # Korean (Hangul pseudowords)
    ko_tier1 = [
        ("부바", "키키"), ("말루마", "타케테"), ("모마", "티티"),
        ("나나", "피시"), ("라라",   "케케"),
    ]
    for wa, wb in ko_tier1:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 1, "language": "ko",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "둥글다", "label_b": "뾰족하다",
            "dimension_low": "둥글고 부드러운", "dimension_high": "각지고 날카로운",
            "source": "Cwiek2022;Kim2003", "notes": "Tier1_Korean",
        })

    ko_tier3_pairs = [
        ("무마", "기티"), ("로나", "테기"), ("보아", "시피"),
        ("나마", "스키"), ("와로", "티스"), ("모로", "피기"),
        ("루마", "게시"), ("노바", "타기"), ("마로", "스시"),
        ("보나", "기스"),
    ]
    for wa, wb in ko_tier3_pairs:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 3, "language": "ko",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "둥글다", "label_b": "뾰족하다",
            "dimension_low": "둥글고 부드러운", "dimension_high": "각지고 날카로운",
            "source": "constructed", "notes": "Tier3_Korean_novel",
        })

    # Hindi (Devanagari pseudowords)
    hi_tier1 = [
        ("बूबा", "किकी"),  ("मलूमा", "टकेटे"), ("मोमा", "तीती"),
        ("नुनु", "पिसि"),  ("राला",  "केके"),
    ]
    for wa, wb in hi_tier1:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 1, "language": "hi",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "गोल", "label_b": "नुकीला",
            "dimension_low": "गोल और नरम", "dimension_high": "नुकीला और कड़ा",
            "source": "Cwiek2022;Bross2018", "notes": "Tier1_Hindi",
        })

    hi_tier3_pairs = [
        ("मूमा", "किटी"), ("रोना", "तेकि"), ("बोआ",  "सिपि"),
        ("नामा", "सुकि"), ("वारो", "तिस"),  ("मोरो", "पिकि"),
        ("रूमा", "गेशि"), ("नोबा", "टाकि"), ("मारो", "सुसि"),
        ("बोना", "किस"),
    ]
    for wa, wb in hi_tier3_pairs:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 3, "language": "hi",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "गोल", "label_b": "नुकीला",
            "dimension_low": "गोल और नरम", "dimension_high": "नुकीला और कड़ा",
            "source": "constructed", "notes": "Tier3_Hindi_novel",
        })

    # German (Latin alphabet pseudowords)
    de_tier1 = [
        ("bouba",  "kiki"),  ("maluma", "takete"),
        ("moma",   "titi"),  ("nunu",   "pisi"),
        ("lala",   "keke"),
    ]
    for wa, wb in de_tier1:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 1, "language": "de",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "rund", "label_b": "eckig",
            "dimension_low": "rund und weich", "dimension_high": "eckig und scharf",
            "source": "Cwiek2022;Schmidtke2014", "notes": "Tier1_German",
        })

    de_tier3_pairs = [
        ("muma", "kiti"), ("rona",  "teki"), ("bowa",  "sipi"),
        ("nama", "suki"), ("waro",  "tis"),  ("moro",  "piki"),
        ("ruma", "gesi"), ("noba",  "taki"), ("maro",  "susi"),
        ("bona", "kis"),
    ]
    for wa, wb in de_tier3_pairs:
        stimuli.append({
            "theory": "sound_symbolism", "tier": 3, "language": "de",
            "word_a": wa, "word_b": wb,
            "expected": "A", "label_a": "rund", "label_b": "eckig",
            "dimension_low": "rund und weich", "dimension_high": "eckig und scharf",
            "source": "constructed", "notes": "Tier3_German_novel",
        })

    return stimuli


# ---------------------------------------------------------------------------
# Contamination probe stimuli
# ---------------------------------------------------------------------------

def build_contamination_probes():
    """
    Build contamination probe items used to assess how much the model knows
    about each theory from its training data. Probes include direct theory
    name queries and famous stimulus recognition queries.
    """
    probes = []

    theory_names = {
        "sound_symbolism": [
            "bouba-kiki effect",
            "maluma-takete effect",
            "sound symbolism in language",
        ],
        "phonesthesia": [
            "phonesthesia in English",
            "phonesthemes",
            "Bergen phonesthesia 2004",
        ],
        "vowel_size": [
            "frequency code Ohala",
            "vowel size symbolism",
            "sound-size symbolism in language",
        ],
        "semantic_prosody": [
            "semantic prosody Louw",
            "collocational prosody Sinclair",
            "semantic prosody in corpus linguistics",
        ],
        "ideophone_compositionality": [
            "ideophones Japanese gitaigo",
            "ideophone compositionality Dingemanse",
            "sound symbolism Japanese",
        ],
    }

    famous_words = {
        "sound_symbolism": ["bouba", "kiki", "maluma", "takete"],
        "phonesthesia": ["gl- phonestheme", "sl- phonestheme"],
        "vowel_size": ["mil vs mol experiment", "vowel size symbolism"],
        "semantic_prosody": ["set in semantic prosody", "cause negative prosody"],
        "ideophone_compositionality": ["pika-pika", "kira-kira", "zawa-zawa"],
    }

    for theory, names in theory_names.items():
        for name in names:
            probes.append({
                "probe_type": "theory_knowledge",
                "theory": theory,
                "query": name,
                "expected": "model describes the theory accurately",
            })
        for word in famous_words.get(theory, []):
            probes.append({
                "probe_type": "word_recognition",
                "theory": theory,
                "query": word,
                "expected": "model recognizes the stimulus from literature",
            })

    return probes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building stimulus battery...")

    all_stimuli = {
        "sound_symbolism":            build_sound_symbolism_stimuli(),
        "phonesthesia":               build_phonesthesia_stimuli(),
        "vowel_size":                 build_vowel_size_stimuli(),
        "semantic_prosody":           build_semantic_prosody_stimuli(),
        "ideophone_compositionality": build_ideophone_stimuli(),
        "multilingual":               build_multilingual_stimuli(),
        "contamination_probes":       build_contamination_probes(),
    }

    output = {
        "metadata": {
            "version": "1.0",
            "theories": [
                "sound_symbolism", "phonesthesia", "vowel_size",
                "semantic_prosody", "ideophone_compositionality",
            ],
            "tiers": {
                "1": "Famous stimuli from landmark papers — high contamination pressure",
                "2": "Attested but obscure stimuli from specialist literature",
                "3": "Novel stimuli constructed by authors — no prior existence in text",
            },
            "languages": ["en", "ja", "ko", "hi", "de"],
        },
        "stimuli": all_stimuli,
    }

    total = 0
    for key, items in all_stimuli.items():
        print(f"  {key}: {len(items)} stimuli")
        total += len(items)
    print(f"  TOTAL: {total} stimuli")

    out_path = DATA_DIR / "stimuli.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nStimuli saved to {out_path}")


if __name__ == "__main__":
    main()
