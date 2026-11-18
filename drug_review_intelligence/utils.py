"""
utils.py - Text preprocessing, negation handling and medical lexicons
=====================================================================
Drug Review Intelligence: Side-Effect Extraction and Patient Experience Mining

Contents
--------
* NLP preprocessing pipeline (Slide 5): cleaning, lowercasing, tokenization,
  stop-word removal, lemmatization.
* NEGATION PRESERVATION: words such as "not", "no", "never" are never removed,
  and every content word in their scope also yields a "neg_<word>" feature so
  "did NOT help" keeps its negative polarity in the TF-IDF space.
* Medical lexicons: side effects, perceived benefits, treatment/cause cues.
* Small helpers shared by model.py and app.py.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# 1. Negation handling
# ---------------------------------------------------------------------------
NEGATIONS = {
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "without", "cannot", "nobody", "nowhere",
}

# Words that open a new clause: negation scope never crosses them.
CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "whereas", "except"}

_PUNCT_BREAK = {".", "!", "?", ";"}
_PUNCT_ALL = _PUNCT_BREAK | {","}

# Stop-words: a deliberately SMALL list. Negations and contrast words
# ("not", "no", "never", "but", "very", "only", ...) are NOT in it.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "i", "me", "my", "mine", "myself", "we", "our", "you", "your", "he", "she",
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "as", "so",
    "and", "or", "if", "then", "than", "there", "here", "also", "just",
    "will", "would", "could", "should", "may", "might", "have", "has", "had",
    "do", "does", "did", "get", "got", "mg",
}

_CONTRACTIONS = [
    (r"\bcan't\b", "can not"),
    (r"\bcannot\b", "can not"),
    (r"\bwon't\b", "will not"),
    (r"\bain't\b", "is not"),
    (r"n't\b", " not"),
    (r"'m\b", " am"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'d\b", " would"),
    (r"'s\b", ""),
]

# ---------------------------------------------------------------------------
# 2. Cleaning, tokenization, lemmatization
# ---------------------------------------------------------------------------

def clean_text(text) -> str:
    """Lowercase, unescape HTML, expand contractions (n't -> not), strip noise.

    Sentence punctuation (. ! ? ; ,) is kept on purpose: it is needed to limit
    the scope of negations and to split reviews into clauses.
    """
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = t.replace("\u2019", "'").replace("`", "'").lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    for pattern, repl in _CONTRACTIONS:
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[^a-z0-9.!?;,'\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> list[str]:
    """Alphabetic word tokenizer (hyphens become spaces: light-headed)."""
    return re.findall(r"[a-z]+", text.replace("-", " "))


_SPLIT_RE = re.compile(r"[.!?;]+|\b(?:but|however|although|though|yet|whereas|except)\b")


def split_clauses(text: str) -> list[str]:
    """Split a review into clauses at sentence punctuation and contrast words."""
    cleaned = clean_text(text)
    out = []
    for part in _SPLIT_RE.split(cleaned):
        part = part.strip(" ,-")
        if tokenize(part):
            out.append(part)
    return out


_IRREGULAR = {
    "was": "be", "were": "be", "been": "be", "is": "be", "are": "be",
    "had": "have", "has": "have", "did": "do", "does": "do", "done": "do",
    "got": "get", "gotten": "get", "felt": "feel", "took": "take", "taken": "take",
    "slept": "sleep", "lost": "lose", "gained": "gain", "gave": "give", "given": "give",
    "went": "go", "gone": "go", "made": "make", "woke": "wake", "ate": "eat",
}


def _fallback_lemma(w: str) -> str:
    """Tiny rule-based lemmatizer used only if NLTK WordNet is unavailable."""
    if w in NEGATIONS or w in _IRREGULAR:
        return _IRREGULAR.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]

    def _stem(stem: str) -> str:
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
            return stem[:-1]
        if stem[-1] in "cvz" or stem.endswith(("ir", "ur", "at", "iz", "us")):
            return stem + "e"
        return stem

    if len(w) > 5 and w.endswith("ing"):
        return _stem(w[:-3])
    if len(w) > 4 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ed"):
        return _stem(w[:-2])
    return w


_LEMMATIZER = None
_LEMMATIZER_READY = False
USING_WORDNET = False


def _init_lemmatizer() -> None:
    global _LEMMATIZER, _LEMMATIZER_READY, USING_WORDNET
    _LEMMATIZER_READY = True
    try:
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        try:
            lem.lemmatize("tests")
        except LookupError:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lem.lemmatize("tests")
        _LEMMATIZER = lem
        USING_WORDNET = True
    except Exception:  # offline / NLTK not available -> rule-based fallback
        _LEMMATIZER = None
        USING_WORDNET = False


@lru_cache(maxsize=50000)
def lemmatize_word(w: str) -> str:
    """Lemmatize with NLTK WordNet (verb then noun); never touches negations."""
    if w in NEGATIONS:
        return w
    if not _LEMMATIZER_READY:
        _init_lemmatizer()
    if _LEMMATIZER is None:
        return _fallback_lemma(w)
    try:
        v = _LEMMATIZER.lemmatize(w, "v")
        return v if v != w else _LEMMATIZER.lemmatize(w, "n")
    except Exception:
        return _fallback_lemma(w)


_TOKEN_RE = re.compile(r"[a-z]+|[.!?;,]")


def preprocess(text, mark_negation: bool = True, lemmatize: bool = True,
               negation_scope: int = 3) -> str:
    """Full preprocessing used for TF-IDF.

    1. clean + lowercase + expand contractions   (didn't -> did not)
    2. tokenize
    3. remove stop-words  (NEGATIONS and contrast words are always kept)
    4. lemmatize
    5. negation marking: the next `negation_scope` content words after a
       negation word additionally emit a "neg_<word>" feature, e.g.
           "did not help my headache" -> "not neg_help help neg_headache headache"
       The scope ends at . ! ? ; or at a contrast word (but, however, ...).
    """
    cleaned = clean_text(text).replace("-", " ")
    out: list[str] = []
    scope = 0
    for tok in _TOKEN_RE.findall(cleaned):
        if tok in _PUNCT_BREAK:
            scope = 0
            continue
        if tok in _PUNCT_ALL:
            continue
        if tok in NEGATIONS:
            out.append(tok)
            scope = negation_scope
            continue
        if tok in CONTRAST_WORDS:
            out.append(tok)
            scope = 0
            continue
        if tok in STOPWORDS or len(tok) < 2:
            continue
        lem = lemmatize_word(tok) if lemmatize else tok
        out.append(lem)
        if mark_negation and scope > 0:
            out.append("neg_" + lem)
            scope -= 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# 3. Light normalisation used for lexicon matching
# ---------------------------------------------------------------------------

def norm_token(w: str) -> str:
    """Strip simple plurals so lexicon lookups are robust (headaches -> headache)."""
    if w in NEGATIONS or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def nset(words) -> set[str]:
    return {norm_token(w) for w in words}


# ---------------------------------------------------------------------------
# 4. Medical lexicons
# ---------------------------------------------------------------------------
SIDE_EFFECT_LEXICON: dict[str, list[str]] = {
    "nausea": ["nausea", "nauseous", "nauseated", "queasy", "queasiness",
               "feeling sick", "felt sick", "feel sick", "sick to my stomach"],
    "vomiting": ["vomiting", "vomit", "vomited", "throwing up", "threw up", "throw up"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light headed",
                  "lightheadedness", "vertigo", "woozy"],
    "headache": ["headache", "headaches", "head ache", "head pain"],
    "fatigue": ["fatigue", "fatigued", "tired", "tiredness", "exhausted",
                "exhaustion", "lethargy", "lethargic", "no energy", "low energy",
                "worn out"],
    "drowsiness": ["drowsiness", "drowsy", "sleepy", "sleepiness", "sedated",
                   "sedation", "groggy", "zombie"],
    "insomnia": ["insomnia", "can not sleep", "trouble sleeping",
                 "difficulty sleeping", "sleeplessness", "unable to sleep",
                 "restless nights"],
    "rash": ["rash", "rashes", "hives", "itching", "itchy", "itchiness",
             "skin irritation"],
    "weight gain": ["weight gain", "gained weight", "gaining weight", "gain weight",
                    "put on weight", "putting on weight"],
    "weight loss": ["weight loss", "lost weight", "losing weight"],
    "anxiety": ["anxiety", "anxious", "nervousness", "nervous", "jittery",
                "jitters", "panic", "panic attacks", "restlessness",
                "agitation", "agitated"],
    "dry mouth": ["dry mouth", "cottonmouth", "cotton mouth"],
    "constipation": ["constipation", "constipated"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stools", "loose stool"],
    "stomach pain": ["stomach pain", "stomach ache", "stomachache", "upset stomach",
                     "stomach upset", "abdominal pain", "stomach cramps",
                     "cramps", "cramping", "bloating", "bloated", "indigestion"],
    "loss of appetite": ["loss of appetite", "lost my appetite", "no appetite",
                         "decreased appetite", "reduced appetite", "appetite loss"],
    "mood swings": ["mood swings", "mood swing", "irritability", "irritable", "moody"],
    "blurred vision": ["blurred vision", "blurry vision", "vision problems", "blurry"],
    "palpitations": ["palpitations", "palpitation", "racing heart", "heart racing",
                     "rapid heartbeat", "heart palpitations"],
    "sweating": ["sweating", "sweats", "night sweats", "excessive sweating", "sweaty"],
    "muscle pain": ["muscle pain", "muscle aches", "muscle ache", "muscle cramps",
                    "joint pain", "body aches", "myalgia"],
    "hair loss": ["hair loss", "hair falling out", "losing hair", "hair thinning"],
    "tremors": ["tremors", "tremor", "shaking", "shaky", "trembling"],
}

BENEFIT_LEXICON: dict[str, list[str]] = {
    "Symptom relief": ["relief", "relieved", "relieve", "relieves", "helped", "helps",
                       "help", "helpful", "worked", "works", "eased", "ease"],
    "Effective": ["effective", "very effective", "wonders", "life saver", "lifesaver",
                  "game changer", "miracle"],
    "Condition improved": ["improved", "improvement", "improve", "better", "much better",
                           "cured", "gone", "went away", "under control", "controlled"],
    "Fast-acting": ["quickly", "fast", "right away", "immediately", "within hours",
                    "within a few days", "quick"],
    "Better sleep": ["sleep better", "slept better", "sleeping better", "better sleep",
                     "sleeping well", "sleep well"],
    "Improved mood": ["mood improved", "happier", "less anxious", "calmer", "feel calm"],
    "More energy": ["more energy", "energized", "energetic"],
    "Pain reduction": ["pain gone", "less pain", "pain went away", "no more pain",
                       "pain free"],
    "Well tolerated": ["no side effects", "without side effects", "no adverse effects",
                       "well tolerated", "tolerated well", "side effect free",
                       "no problems"],
    "Affordable": ["cheap", "affordable", "inexpensive", "low cost"],
    "Easy to use": ["easy to take", "once a day", "easy to use", "convenient"],
}


def compile_lexicon(lexicon: dict[str, list[str]]):
    """-> list of (canonical, tuple_of_normalised_tokens), longest phrases first."""
    items = []
    for canon, phrases in lexicon.items():
        for p in phrases:
            toks = tuple(norm_token(t) for t in tokenize(p))
            if toks:
                items.append((canon, toks))
    items.sort(key=lambda x: -len(x[1]))
    return items


COMPILED_SIDE_EFFECTS = compile_lexicon(SIDE_EFFECT_LEXICON)
COMPILED_BENEFITS = compile_lexicon(BENEFIT_LEXICON)


def find_phrase_matches(norm_tokens: list[str], compiled) -> list[tuple[str, int, int]]:
    """Greedy longest-first phrase matching -> [(canonical, start, end)] sorted by start."""
    matches, used = [], set()
    for canon, phrase in compiled:
        n = len(phrase)
        for i in range(len(norm_tokens) - n + 1):
            if tuple(norm_tokens[i:i + n]) == phrase and not any(
                    j in used for j in range(i, i + n)):
                matches.append((canon, i, i + n))
                used.update(range(i, i + n))
    return sorted(matches, key=lambda m: m[1])


def is_negated(norm_tokens: list[str], start: int, window: int = 5) -> bool:
    """True if a negation word occurs within `window` tokens before `start`."""
    return any(t in NEGATIONS for t in norm_tokens[max(0, start - window):start])


# ---------------------------------------------------------------------------
# 5. Context cues
# ---------------------------------------------------------------------------
# "helped my HEADACHE" -> headache is the condition being treated, not a side effect
TREAT_CUES = nset("""help helped helps helping helpful relieve relieved relieves relieving
relief cure cured cures treat treated treats treating reduce reduced reduces ease eased
eases stop stopped stops fix fixed fixes improve improved improves improvement manage
managed manages control controlled controls for against gone heal healed prevent
prevented prevents eliminate eliminated work worked works""".split())

# "experienced NAUSEA", "caused DIZZINESS" -> side effect
CAUSE_CUES = nset("""cause caused causes causing gave give gives giving experience experienced
experiences experiencing had have has got get gets getting develop developed developing
suffer suffered suffering made make makes felt feel feels feeling include included including
lead led trigger triggered started start noticed notice due because bring brought brings
produce produced""".split())

SEVERE_WORDS = nset("""severe terrible unbearable horrible awful extreme extremely intense
debilitating bad worst serious constant crippling brutal""".split())
MILD_WORDS = nset("""mild slight slightly minor minimal little occasional bit small light
tiny barely hardly""".split())

# ---------------------------------------------------------------------------
# 6. Rendering helper
# ---------------------------------------------------------------------------

def highlight_html(text: str, items: list[tuple[list[str], str]]) -> str:
    """Return HTML where each token-sequence in `items` is wrapped in <mark class=...>.

    items = [(["dry", "mouth"], "hl-se"), (["helped"], "hl-benefit"), ...]
    """
    esc = html.escape(text)
    spans = []
    for tokens, cls in items:
        if not tokens:
            continue
        pat = r"(?<![A-Za-z])" + r"[\W_]+".join(re.escape(t) for t in tokens) + r"(?![A-Za-z])"
        for m in re.finditer(pat, esc, flags=re.I):
            spans.append((m.start(), m.end(), cls))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, pos = [], 0
    for s, e, cls in spans:
        if s < pos:
            continue
        out.append(esc[pos:s])
        out.append(f'<mark class="{cls}">{esc[s:e]}</mark>')
        pos = e
    out.append(esc[pos:])
    return "".join(out)
