# Rename this file to your zID, e.g. z1234567.py

from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


# -------------------------------------------------
# Optional: Load training data
# -------------------------------------------------


def load_training_data(path="data.csv"):
    """
    Loads the provided training dataset.

    Expected columns:
    - text
    - cefr_level
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)

    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")

    return df


# -------------------------------------------------
# Optional: Initialise resources globally
# -------------------------------------------------

# Map CEFR levels to numeric scores for averaging.
CEFR_LEVEL_TO_NUM = {level: i + 1 for i, level in enumerate(CEFR_LEVELS)}
NUM_TO_CEFR_LEVEL = {v: k for k, v in CEFR_LEVEL_TO_NUM.items()}

# The following helper functions perform lightweight preprocessing and
# lexical normalization. They are designed to work without requiring
# extra dependencies, but will use NLTK's WordNet lemmatizer if it is
# available in the environment.

try:
    from nltk.stem import WordNetLemmatizer  # type: ignore

    _WN_LEMMATIZER = WordNetLemmatizer()
except Exception:
    _WN_LEMMATIZER = None


def clean_text(text: str) -> str:
    """Lowercase and remove non-word characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep letters and apostrophes; replace everything else with space.
    text = re.sub(r"[^a-z'\s]", " ", text)
    # Normalize whitespace.
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def lemmatize_word(word: str) -> str:
    """Return a normalized form of the word.

    If NLTK WordNet lemmatizer is available, use it; otherwise, apply
    a small set of heuristic rules to reduce common inflections.
    """
    if not word:
        return word

    if _WN_LEMMATIZER is not None:
        try:
            # Try verb first, then noun.
            lemma = _WN_LEMMATIZER.lemmatize(word, pos="v")
            lemma = _WN_LEMMATIZER.lemmatize(lemma, pos="n")
            if lemma:
                return lemma
        except Exception:
            pass

    # Fallback heuristics
    if word.endswith("ies") and len(word) > 3:
        return word[: -3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[: -2]
    if word.endswith("s") and len(word) > 3:
        return word[: -1]
    if word.endswith("ed") and len(word) > 3:
        return word[: -2]
    if word.endswith("ing") and len(word) > 4:
        return word[: -3]

    return word


def sentence_to_lemmas(sentence: str) -> List[str]:
    """Convert a sentence into a list of normalized lemmas."""
    tokens = tokenize(sentence)
    return [lemmatize_word(t) for t in tokens if t]


def build_cefr_vocab(df: pd.DataFrame) -> Dict[str, float]:
    """Build a word -> average CEFR score mapping from the training data."""
    scores = defaultdict(list)

    for _, row in df.iterrows():
        level = row.get("cefr_level")
        if level not in CEFR_LEVEL_TO_NUM:
            continue
        score = CEFR_LEVEL_TO_NUM[level]

        sentence = row.get("text", "")
        for word in sentence_to_lemmas(sentence):
            if len(word) < 2:
                continue
            scores[word].append(score)

    # average scores
    return {w: sum(v) / len(v) for w, v in scores.items()}


def load_cefr_vocab(path: str = "data.csv") -> Dict[str, float]:
    """Load data.csv and return a word -> CEFR score mapping."""
    df = load_training_data(path)
    return build_cefr_vocab(df)


# Cached vocab (lazy-loaded)
_CEFR_VOCAB: Optional[Dict[str, float]] = None


def get_cefr_vocab(path: str = "data.csv") -> Dict[str, float]:
    """Get (and cache) the CEFR word score dictionary."""
    global _CEFR_VOCAB
    if _CEFR_VOCAB is None:
        _CEFR_VOCAB = load_cefr_vocab(path)
    return _CEFR_VOCAB


# -------------------------------------------------
# Sentence analysis (spaCy)
# -------------------------------------------------

# spaCy is optional; the code will attempt to use it if available.
# If spaCy is not installed, a minimal fallback will be used.

try:
    import spacy  # type: ignore

    _SPACY_NLP = None

    def get_spacy_nlp():
        """Lazily load a spaCy English model."""
        global _SPACY_NLP
        if _SPACY_NLP is not None:
            return _SPACY_NLP

        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            # Try to download model if missing.
            try:
                spacy.cli.download("en_core_web_sm")
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = None
        return _SPACY_NLP

    def analyze_sentence_spacy(sentence: str) -> List[Dict[str, str]]:
        """Analyze sentence using spaCy (token, lemma, POS, and meta info)."""
        nlp = get_spacy_nlp()
        if nlp is None:
            raise RuntimeError("spaCy model is not available")

        doc = nlp(sentence)
        return [
            {
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "is_digit": token.like_num,
                "is_propn": token.pos_ == "PROPN",
                "ws": token.whitespace_,
            }
            for token in doc
            if not token.is_space
        ]

except Exception:
    # SpaCy not available �C provide a fallback.

    _FALLBACK_STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "at",
        "by",
        "for",
        "in",
        "of",
        "on",
        "to",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "as",
        "that",
        "this",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "my",
        "your",
        "our",
        "we",
        "you",
        "me",
        "him",
        "her",
    }

    def analyze_sentence_spacy(sentence: str) -> List[Dict[str, str]]:
        """Fallback analysis: tokenize + lemma, POS set to 'X'."""
        items = []
        pattern = re.compile(r"\S+\s*")
        for m in pattern.finditer(sentence):
            chunk = m.group(0)
            token = chunk.rstrip()
            ws = chunk[len(token) :]

            is_punct = bool(re.fullmatch(r"[^\w\s]+", token))
            items.append(
                {
                    "token": token,
                    "lemma": lemmatize_word(token.lower()) if not is_punct else token,
                    "pos": "X",
                    "tag": "",
                    "is_stop": token.lower() in _FALLBACK_STOPWORDS,
                    "is_punct": is_punct,
                    "is_digit": token.isdigit(),
                    "is_propn": token.istitle(),
                    "ws": ws,
                }
            )
        return items


# -------------------------------------------------
# Replacement candidate identification
# -------------------------------------------------

_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def is_content_token(tok: Dict[str, str]) -> bool:
    """Decide whether a token is a content word worth considering."""
    if tok.get("is_punct") or tok.get("is_digit") or tok.get("is_stop"):
        return False

    # Proper nouns and determiners are not considered for replacement here.
    if tok.get("is_propn"):
        return False

    pos = tok.get("pos", "").upper()
    if pos in _CONTENT_POS:
        return True

    # If POS is unknown (fallback), treat as content if it is not clearly a function word.
    return pos == "X"


def get_cefr_score_for_lemma(lemma: str, vocab: Dict[str, float]) -> Optional[float]:
    """Return the CEFR score for a normalized lemma, if known."""
    if not lemma:
        return None
    return vocab.get(lemma.lower())


def identify_replacement_candidates(
    sentence: str,
    source_level: str,
    target_level: str,
    max_candidates: int = 2,
) -> List[Dict[str, object]]:
    """Identify which tokens are most in need of replacement.

    Returns a list of candidate dicts containing token metadata and CEFR score.
    """
    source_num = CEFR_LEVEL_TO_NUM.get(source_level)
    target_num = CEFR_LEVEL_TO_NUM.get(target_level)
    if source_num is None or target_num is None:
        return []

    direction = "down" if target_num < source_num else "up" if target_num > source_num else None
    if direction is None:
        return []

    vocab = get_cefr_vocab()
    tokens = analyze_sentence_spacy(sentence)

    candidates = []
    for idx, tok in enumerate(tokens):
        if not is_content_token(tok):
            continue

        score = get_cefr_score_for_lemma(tok.get("lemma", ""), vocab)
        if score is None:
            continue

        if direction == "down" and score <= target_num:
            continue
        if direction == "up" and score >= target_num:
            continue

        candidates.append(
            {
                "idx": idx,
                "token": tok.get("token"),
                "lemma": tok.get("lemma"),
                "pos": tok.get("pos"),
                "score": score,
                "distance": abs(score - target_num),
            }
        )

    # Prefer candidates that deviate most from target level (bigger distance).
    candidates.sort(key=lambda c: c["distance"], reverse=True)
    return candidates[:max_candidates]


# -------------------------------------------------
# Language Model + Candidate Scoring
# -------------------------------------------------

import math
from collections import Counter


class NGramLanguageModel:
    """Simple n-gram language model with add-TestResults smoothing."""

    def __init__(self, n: int = 3, k: float = 1.0):
        self.n = n
        self.k = k
        self.counts = {i: Counter() for i in range(1, n + 1)}
        self.vocab = set()

    def update(self, tokens: List[str]):
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        self.vocab.update(tokens)
        for i in range(1, self.n + 1):
            for j in range(len(tokens) - i + 1):
                ngram = tuple(tokens[j : j + i])
                self.counts[i][ngram] += 1

    def score_sentence(self, tokens: List[str]) -> float:
        """Return sentence log-probability (natural log) using interpolated n-grams."""
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0.0
        V = len(self.vocab)

        for i in range(self.n - 1, len(tokens)):
            trigram = tuple(tokens[i - 2 : i + 1]) if self.n >= 3 else None
            bigram = tuple(tokens[i - 1 : i + 1]) if self.n >= 2 else None
            unigram = (tokens[i],)

            # Interpolation weights (simple fixed weights)
            lam1, lam2, lam3 = 0.1, 0.3, 0.6

            p1 = (
                (self.counts[1][unigram] + self.k)
                / (sum(self.counts[1].values()) + self.k * V)
            )
            p2 = 0.0
            p3 = 0.0

            if bigram is not None:
                denom = self.counts[1][(bigram[0],)] + self.k * V
                p2 = (self.counts[2][bigram] + self.k) / denom if denom > 0 else p1

            if trigram is not None:
                denom = self.counts[2][trigram[:2]] + self.k * V
                p3 = (self.counts[3][trigram] + self.k) / denom if denom > 0 else p2

            prob = lam1 * p1 + lam2 * p2 + lam3 * p3
            log_prob += math.log(prob)

        return log_prob


_LM: Optional[NGramLanguageModel] = None


def get_language_model(path: str = "data.csv") -> NGramLanguageModel:
    """Build or return cached trigram language model from data.csv."""
    global _LM
    if _LM is not None:
        return _LM

    df = load_training_data(path)
    lm = NGramLanguageModel(n=3, k=1.0)

    for text in df["text"].dropna():
        tokens = tokenize(text)
        if tokens:
            lm.update(tokens)

    _LM = lm
    return lm


def context_score(sentence_tokens: List[str], lm: NGramLanguageModel) -> float:
    """Compute a normalized context score for a token sequence."""
    if not sentence_tokens:
        return 0.0

    log_prob = lm.score_sentence(sentence_tokens)
    avg_log_prob = log_prob / max(1, len(sentence_tokens))

    # Map to [0, TestResults] using a sigmoid centered around a typical log-prob range.
    return 1.0 / (1.0 + math.exp(-(avg_log_prob + 4.0)))


def semantic_similarity(word1: str, word2: str) -> float:
    """Return semantic similarity between two words (0..TestResults).

    Uses spaCy vectors if available, else falls back to equality.
    """
    w1 = word1.lower().strip()
    w2 = word2.lower().strip()
    if not w1 or not w2:
        return 0.0

    try:
        nlp = get_spacy_nlp()
        if nlp is not None:
            v1 = nlp.vocab[w1].vector
            v2 = nlp.vocab[w2].vector
            if v1 is not None and v2 is not None and v1.any() and v2.any():
                dot = float((v1 * v2).sum())
                mag = float((v1**2).sum() ** 0.5) * float((v2**2).sum() ** 0.5)
                if mag > 0:
                    return max(0.0, min(1.0, dot / mag))
    except Exception:
        pass

    # Fall back to NLTK WordNet path similarity if available.
    try:
        from nltk.corpus import wordnet as wn  # type: ignore

        syns1 = wn.synsets(w1)
        syns2 = wn.synsets(w2)
        if syns1 and syns2:
            best = 0.0
            for s1 in syns1:
                for s2 in syns2:
                    sim = s1.path_similarity(s2)
                    if sim is not None and sim > best:
                        best = sim
            if best is not None:
                return max(0.0, min(1.0, best))
    except Exception:
        pass

    # Fallback: exact match only.
    return 1.0 if w1 == w2 else 0.0


def score_candidate(
    sentence: str,
    candidate_idx: int,
    candidate_token: str,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> float:
    """Score a single replacement candidate.

    final_score = alpha * semantic_similarity + beta * context_score.
    """

    tokens = [t["token"] for t in analyze_sentence_spacy(sentence)]
    if candidate_idx < 0 or candidate_idx >= len(tokens):
        return 0.0

    original = tokens[candidate_idx]
    tokens[candidate_idx] = candidate_token

    sem = semantic_similarity(original, candidate_token)
    lm = get_language_model()
    ctx = context_score(tokens, lm)

    return alpha * sem + beta * ctx


# -------------------------------------------------
# Inflection recovery + sentence reconstruction
# -------------------------------------------------

try:
    import pyinflect  # type: ignore  # noqa: F401

    _HAS_PYINFLECT = True
except Exception:
    _HAS_PYINFLECT = False


_IRREGULAR_VERB_FORMS = {
    "be": {"VBD": "was", "VBN": "been", "VBZ": "is", "VBG": "being", "VBP": "are"},
    "have": {"VBD": "had", "VBN": "had", "VBZ": "has", "VBG": "having"},
    "do": {"VBD": "did", "VBN": "done", "VBZ": "does", "VBG": "doing"},
    "go": {"VBD": "went", "VBN": "gone", "VBZ": "goes", "VBG": "going"},
    "give": {"VBD": "gave", "VBN": "given", "VBZ": "gives", "VBG": "giving"},
    "make": {"VBD": "made", "VBN": "made", "VBZ": "makes", "VBG": "making"},
    "take": {"VBD": "took", "VBN": "taken", "VBZ": "takes", "VBG": "taking"},
    "get": {"VBD": "got", "VBN": "got", "VBZ": "gets", "VBG": "getting"},
}


def _match_case(word: str, template: str) -> str:
    """Match candidate casing to the original token's casing style."""
    if template.isupper():
        return word.upper()
    if template.istitle():
        return word.title()
    return word.lower()


def _heuristic_pluralize(noun: str) -> str:
    """Very small pluralization helper for fallback mode."""
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "sh", "ch", "x", "z")):
        return noun + "es"
    return noun + "s"


def _heuristic_verb_inflect(lemma: str, tag: str) -> str:
    """Inflect verb lemma by tag with simple rules + irregular map."""
    tag = (tag or "").upper()
    base = lemma.lower()

    if base in _IRREGULAR_VERB_FORMS and tag in _IRREGULAR_VERB_FORMS[base]:
        return _IRREGULAR_VERB_FORMS[base][tag]

    if tag == "VBD" or tag == "VBN":
        if base.endswith("e"):
            return base + "d"
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return base[:-1] + "ied"
        return base + "ed"

    if tag == "VBG":
        if base.endswith("e") and not base.endswith("ee"):
            return base[:-1] + "ing"
        return base + "ing"

    if tag == "VBZ":
        if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
            return base[:-1] + "ies"
        if base.endswith(("s", "sh", "ch", "x", "z", "o")):
            return base + "es"
        return base + "s"

    return base


def restore_word_form(
    candidate_lemma: str,
    original_token: str,
    pos: str,
    tag: str = "",
) -> str:
    """Recover surface form for candidate lemma based on original token form.

    Priority:
    TestResults) Use pyinflect with POS tag when available.
    2) Fallback to simple POS-aware heuristic rules.
    3) Match original casing style.
    """
    candidate_lemma = (candidate_lemma or "").strip()
    if not candidate_lemma:
        return original_token

    pos = (pos or "").upper()
    tag = (tag or "").upper()

    # TestResults) pyinflect path
    if _HAS_PYINFLECT:
        try:
            nlp = get_spacy_nlp()
            if nlp is not None and tag:
                cand_doc = nlp(candidate_lemma)
                inflected = cand_doc[0]._.inflect(tag)  # type: ignore[attr-defined]
                if inflected:
                    return _match_case(inflected, original_token)
        except Exception:
            pass

    # 2) heuristic path
    out = candidate_lemma.lower()
    if pos == "VERB":
        inferred_tag = tag
        if not inferred_tag:
            original_lower = original_token.lower()
            if original_lower.endswith("ing"):
                inferred_tag = "VBG"
            elif original_lower.endswith("ed"):
                inferred_tag = "VBD"
            elif original_lower.endswith("s"):
                inferred_tag = "VBZ"
            else:
                inferred_tag = "VB"
        out = _heuristic_verb_inflect(out, inferred_tag)
    elif pos == "NOUN":
        original_lower = original_token.lower()
        is_plural = tag in {"NNS", "NNPS"} or (
            original_lower.endswith("s") and not original_lower.endswith("ss")
        )
        if is_plural:
            out = _heuristic_pluralize(out)
    elif pos == "ADJ":
        # keep base adjective form for now
        out = out
    elif pos == "ADV":
        # keep base adverb form for now
        out = out

    return _match_case(out, original_token)


def reconstruct_sentence_from_tokens(tokens: List[Dict[str, object]]) -> str:
    """Reconstruct sentence from token dicts preserving whitespace when possible."""
    parts = []
    for token_info in tokens:
        tok = str(token_info.get("token", ""))
        ws = token_info.get("ws")
        if ws is None:
            ws = " "
        parts.append(tok + str(ws))
    return "".join(parts).rstrip()


def apply_lemma_replacement(
    sentence: str,
    token_idx: int,
    candidate_lemma: str,
) -> str:
    """Apply lemma replacement at token_idx with proper inflection and rebuild sentence."""
    analyzed = analyze_sentence_spacy(sentence)
    if token_idx < 0 or token_idx >= len(analyzed):
        return sentence

    target = analyzed[token_idx]
    original_token = str(target.get("token", ""))
    pos = str(target.get("pos", ""))
    tag = str(target.get("tag", ""))

    surface = restore_word_form(
        candidate_lemma=candidate_lemma,
        original_token=original_token,
        pos=pos,
        tag=tag,
    )
    target["token"] = surface

    return reconstruct_sentence_from_tokens(analyzed)


def generate_candidate_lemmas(
    source_lemma: str,
    source_pos: str,
    source_score_num: float,
    target_level_num: int,
    direction: str,
    vocab: Dict[str, float],
    top_k: int = 20,
) -> List[str]:
    """Generate replacement lemma candidates and filter by CEFR direction.

    direction:
        - "down": candidate CEFR score should be <= target level
        - "up": candidate CEFR score should be >= target level
    """
    source_lemma = (source_lemma or "").lower().strip()
    if not source_lemma:
        return []

    wn_pos_map = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",
    }

    raw_candidates = set()
    raw_candidates.add(source_lemma)

    # WordNet-based candidate generation
    try:
        from nltk.corpus import wordnet as wn  # type: ignore

        wn_pos = wn_pos_map.get((source_pos or "").upper())
        synsets = wn.synsets(source_lemma, pos=wn_pos) if wn_pos else wn.synsets(source_lemma)
        for syn in synsets:
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ").lower().strip()
                if not w:
                    continue
                # keep single-token candidates for stable replacement
                if " " in w:
                    continue
                if not re.fullmatch(r"[a-z]+", w):
                    continue
                raw_candidates.add(w)
    except Exception:
        pass

    filtered = []
    source_distance = abs(source_score_num - target_level_num)
    for cand in raw_candidates:
        if cand == source_lemma:
            continue
        score = vocab.get(cand)
        if score is None:
            continue
        # Keep candidates that move difficulty toward the target.
        if direction == "down" and score >= source_score_num:
            continue
        if direction == "up" and score <= source_score_num:
            continue
        cand_distance = abs(score - target_level_num)
        if cand_distance >= source_distance:
            continue
        filtered.append((cand, cand_distance))

    filtered.sort(key=lambda x: x[1])
    return [w for w, _ in filtered[:top_k]]


def choose_best_replacement(
    sentence: str,
    token_idx: int,
    source_lemma: str,
    source_pos: str,
    source_score_num: float,
    target_level_num: int,
    direction: str,
    vocab: Dict[str, float],
    alpha: float = 0.6,
    beta: float = 0.4,
) -> Optional[str]:
    """Choose best replacement lemma using semantic + context combined scoring."""
    candidates = generate_candidate_lemmas(
        source_lemma=source_lemma,
        source_pos=source_pos,
        source_score_num=source_score_num,
        target_level_num=target_level_num,
        direction=direction,
        vocab=vocab,
    )
    if not candidates:
        return None

    best_lemma = None
    best_score = float("-inf")

    original_analysis = analyze_sentence_spacy(sentence)
    if token_idx < 0 or token_idx >= len(original_analysis):
        return None
    original_surface = str(original_analysis[token_idx].get("token", ""))

    lm = get_language_model()
    for cand_lemma in candidates:
        replaced_sentence = apply_lemma_replacement(sentence, token_idx, cand_lemma)
        sem = semantic_similarity(source_lemma, cand_lemma)
        ctx = context_score(tokenize(replaced_sentence), lm)
        # Small penalty when surface form does not change, to avoid no-op picks.
        replaced_surface = str(analyze_sentence_spacy(replaced_sentence)[token_idx].get("token", ""))
        no_op_penalty = 0.05 if replaced_surface.lower() == original_surface.lower() else 0.0

        final = alpha * sem + beta * ctx - no_op_penalty
        if final > best_score:
            best_score = final
            best_lemma = cand_lemma

    return best_lemma



# -------------------------------------------------
# Required Function
# -------------------------------------------------


# -------------------------------------------------
# Required Function
# -------------------------------------------------

# Students may:
# - Load and preprocess data here
# - Train a lightweight model
# - Build lookup tables
# - Create rule-based mappings


# -------------------------------------------------
# Required Function
# -------------------------------------------------


def transform_sentence(sentence, source_level, target_level):
    """
    Transform a sentence from source CEFR level to target CEFR level.

    Parameters:
        sentence (str): Input sentence.
        source_level (str): CEFR level of the input sentence.
        target_level (str): Target CEFR level.

    Returns:
        str: Transformed sentence.
    """

    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")

    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    # No change needed when source and target are the same.
    if source_level == target_level:
        return sentence

    source_num = CEFR_LEVEL_TO_NUM[source_level]
    target_num = CEFR_LEVEL_TO_NUM[target_level]
    direction = "down" if target_num < source_num else "up"

    vocab = get_cefr_vocab()

    # Identify likely mismatch words and keep replacement count small.
    candidates = identify_replacement_candidates(
        sentence=sentence,
        source_level=source_level,
        target_level=target_level,
        max_candidates=2,
    )

    if not candidates:
        return sentence

    updated_sentence = sentence
    replacements_done = 0
    max_replacements = 2

    for cand in candidates:
        if replacements_done >= max_replacements:
            break

        token_idx = int(cand["idx"])

        # Re-analyze current sentence so token info stays aligned after prior edits.
        current_analysis = analyze_sentence_spacy(updated_sentence)
        if token_idx < 0 or token_idx >= len(current_analysis):
            continue

        source_lemma = str(current_analysis[token_idx].get("lemma", ""))
        source_pos = str(current_analysis[token_idx].get("pos", ""))
        source_score = vocab.get(source_lemma.lower())
        if source_score is None:
            continue

        best = choose_best_replacement(
            sentence=updated_sentence,
            token_idx=token_idx,
            source_lemma=source_lemma,
            source_pos=source_pos,
            source_score_num=source_score,
            target_level_num=target_num,
            direction=direction,
            vocab=vocab,
            alpha=0.6,
            beta=0.4,
        )

        if not best:
            continue

        new_sentence = apply_lemma_replacement(updated_sentence, token_idx, best)
        if new_sentence != updated_sentence:
            updated_sentence = new_sentence
            replacements_done += 1

    return updated_sentence
