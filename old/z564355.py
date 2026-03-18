# Rename this file to your zID, e.g. z1234567.py
# python main.py z5643559
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


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
        file_path = Path(__file__).parent / path
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)

    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")

    return df


def _tokenize_text(text):
    """Extract alphabetic words from text (lowercase) for vocabulary building."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# CEFR order for fallback: "words acceptable at level L" can include lower levels.
CEFR_ORDER = {L: i for i, L in enumerate(CEFR_LEVELS)}


def build_level_vocab(df, max_words_per_level=15000):
    """
    Build a set of words per CEFR level from the dataset.
    Returns (level_vocab, level_counts) for replacement scoring.
    """
    level_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        level = str(row["cefr_level"]).strip()
        if level not in CEFR_LEVELS:
            continue
        for w in _tokenize_text(row["text"]):
            if len(w) > 1:
                level_counts[level][w] += 1

    level_vocab = {}
    for level in CEFR_LEVELS:
        if level not in level_counts:
            level_vocab[level] = set()
            continue
        sorted_words = sorted(
            level_counts[level].items(), key=lambda x: -x[1]
        )[:max_words_per_level]
        level_vocab[level] = set(w for w, _ in sorted_words)
    return level_vocab, level_counts


# -------------------------------------------------
# Locally trained n-gram language model (Laplace smoothing)
# -------------------------------------------------

_SENT_START = "<s>"
_SENT_END = "</s>"


class BigramLM:
    """
    Bigram language model trained on data.csv with Laplace (add-TestResults) smoothing.
    P(w_i | w_{i-TestResults}) = (count(w_{i-TestResults}, w_i) + TestResults) / (count(w_{i-TestResults}) + V)
    """

    def __init__(self, k=1.0):
        """
        k: smoothing parameter (Laplace = TestResults, add-k smoothing).
        """
        self.k = k
        self.bigram_counts = defaultdict(lambda: defaultdict(float))
        self.unigram_counts = defaultdict(float)
        self.vocab_size = 0

    def train(self, token_sequences):
        """Train on a list of token sequences (each is a list of lowercase words)."""
        self.bigram_counts.clear()
        self.unigram_counts.clear()
        n_seq = len(token_sequences)
        print(f"[BigramLM] Training on {n_seq} sequences ...")
        for i, seq in enumerate(token_sequences):
            if not seq:
                continue
            if (i + 1) % 2000 == 0 or i == 0:
                print(f"  processed {i + 1}/{n_seq} sequences")
            self.unigram_counts[_SENT_START] += 1
            prev = _SENT_START
            for w in seq:
                if len(w) > 1:
                    self.bigram_counts[prev][w] += 1
                    self.unigram_counts[w] += 1
                    prev = w
            self.bigram_counts[prev][_SENT_END] += 1
        self.vocab_size = len(
            set(self.unigram_counts.keys()) - {_SENT_START, _SENT_END}
        )
        if self.vocab_size < 1:
            self.vocab_size = 1
        n_bigrams = sum(len(v) for v in self.bigram_counts.values())
        print(f"[BigramLM] Done. Vocab size = {self.vocab_size}, bigram types = {n_bigrams}")

    def log_prob(self, word, prev_word=None):
        """
        Log P(word | prev_word) with Laplace smoothing.
        If prev_word is None or unseen, use unigram P(word).
        """
        if prev_word is None or prev_word not in self.bigram_counts:
            count = self.unigram_counts.get(word, 0) + self.k
            total = sum(self.unigram_counts.values()) + self.k * (
                self.vocab_size + 1
            )
        else:
            count = self.bigram_counts[prev_word].get(word, 0) + self.k
            total = sum(self.bigram_counts[prev_word].values()) + self.k * (
                self.vocab_size + 1
            )
        if total < 1e-12:
            return -20.0
        import math
        return math.log(max(count / total, 1e-12))


# -------------------------------------------------
# Optional: Initialise resources globally
# -------------------------------------------------

_level_vocab = None
_level_counts = None
_nlp = None
_lm = None
_data_path = "../data.csv"

# Replacement quality: min semantic similarity to even consider a candidate
_MIN_SIMILARITY = 0.45
# If the best candidate overall score is below this, we skip replacement
_MIN_OVERALL_SCORE = 0.40
# Weight: similarity vs LM vs frequency
_W_SIM, _W_LM, _W_FREQ = 0.55, 0.30, 0.15


def _get_lm():
    """Lazy-load and train bigram LM on data.csv."""
    global _lm
    if _lm is None:
        print("[LM] Loading data.csv ...")
        df = load_training_data(_data_path)
        print(f"[LM] Loaded {len(df)} rows.")
        sequences = []
        for _, row in df.iterrows():
            toks = _tokenize_text(row["text"])
            if toks:
                sequences.append([w for w in toks if len(w) > 1])
        print(f"[LM] Built {len(sequences)} token sequences. Starting bigram training ...")
        _lm = BigramLM(k=1.0)
        _lm.train(sequences)
        print("[LM] Language model ready.")
    return _lm


def _get_level_vocab():
    """Lazy-load level vocab and level_counts from data.csv."""
    global _level_vocab, _level_counts
    if _level_vocab is None:
        df = load_training_data(_data_path)
        _level_vocab, _level_counts = build_level_vocab(df)
    return _level_vocab


def _get_nlp():
    """Lazy-load SpaCy model. Prefer md/lg (have word vectors) over sm (no vectors)."""
    global _nlp
    if _nlp is None:
        import spacy
        for name in ("en_core_web_md", "en_core_web_lg", "en_core_web_sm"):
            try:
                _nlp = spacy.load(name)
                break
            except OSError:
                continue
        if _nlp is None:
            _nlp = spacy.load("en")
    return _nlp


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _best_replacement_wordnet(original_word, target_vocab):
    """Fallback: pick a WordNet synonym (single word) that is in target_vocab."""
    try:
        from nltk.corpus import wordnet
    except Exception:
        return None
    orig_lower = original_word.lower()
    if orig_lower in target_vocab:
        return None
    seen = {orig_lower}
    for syn in wordnet.synsets(orig_lower):
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            if "_" in name:
                continue
            if name in seen or name not in target_vocab:
                continue
            if name.isalpha():
                return name
            seen.add(name)
    return None


def _inflect_to_match(replacement_word, orig_tok):
    """
    Try to inflect replacement_word so that its morphological form
    (tense/number, etc.) matches orig_tok as much as possible.
    Uses pyinflect when available.
    """
    if not replacement_word or not orig_tok:
        return replacement_word
    try:
        from pyinflect import getInflection
        tag = getattr(orig_tok, "tag_", None)
        if not tag:
            return replacement_word
        inflected = getInflection(replacement_word, tag=tag)
        if inflected and isinstance(inflected, (list, tuple)) and len(inflected) > 0:
            return inflected[0]
        if isinstance(inflected, str):
            return inflected
    except Exception:
        pass
    return replacement_word


def _best_replacement(
    original_word,
    target_vocab,
    nlp,
    lm=None,
    prev_word=None,
    target_level=None,
    orig_tok=None,
    level_counts=None,
    max_candidates=8000,
):
    """
    Find the best replacement: same POS, similarity + LM + frequency, min similarity.
    """
    if not target_vocab:
        return None
    orig_lower = original_word.lower()
    if orig_lower in target_vocab:
        return None

    try:
        tok = nlp(orig_lower)
        v_orig = tok.vector
        has_vectors = np.linalg.norm(v_orig) >= 1e-9
        orig_pos = tok.pos_ if hasattr(tok, "pos_") else None

        candidates = list(target_vocab)
        if orig_pos:
            pos_filtered = []
            for w in candidates:
                if w == orig_lower:
                    continue
                try:
                    w_tok = nlp(w)
                    if w_tok and len(w_tok) > 0 and w_tok[0].pos_ == orig_pos:
                        pos_filtered.append(w)
                except Exception:
                    pos_filtered.append(w)
            candidates = pos_filtered if pos_filtered else candidates
        if len(candidates) > max_candidates:
            import random
            candidates = random.Random(42).sample(candidates, max_candidates)

        # Frequency in target level (normalised 0..TestResults)
        def freq_score(w):
            if not level_counts or not target_level:
                return 0.5
            cnt = level_counts.get(target_level, {}).get(w, 0)
            if cnt <= 0:
                return 0.0
            return min(1.0, np.log1p(cnt) / 10.0)

        best_word = None
        best_score = -1e9
        for w in candidates:
            if w == orig_lower:
                continue
            sim = 0.0
            if has_vectors:
                try:
                    v_w = nlp.vocab[w].vector
                    if np.linalg.norm(v_w) >= 1e-9:
                        sim = _cosine_similarity(v_orig, v_w)
                except Exception:
                    pass
            if sim < _MIN_SIMILARITY and has_vectors:
                continue
            lm_score = 0.0
            if lm is not None:
                lm_score = lm.log_prob(w, prev_word)
            fsc = freq_score(w)
            score = (
                _W_SIM * sim
                + _W_LM * max(0.0, (lm_score / 10.0 + 0.5))
                + _W_FREQ * fsc
            )
            if score > best_score:
                best_score = score
                best_word = w
        # 加一层整体分数阈值：分数太低宁可不换
        if best_word is not None and best_score >= _MIN_OVERALL_SCORE:
            if orig_tok is not None:
                best_word = _inflect_to_match(best_word, orig_tok) or best_word
            return best_word
    except Exception:
        pass
    rep = _best_replacement_wordnet(original_word, target_vocab)
    if rep and orig_tok is not None:
        rep = _inflect_to_match(rep, orig_tok) or rep
    return rep


def _match_case(original_token_text, replacement_word):
    """Apply the casing of original token to the replacement word."""
    if not replacement_word:
        return replacement_word
    if original_token_text.isupper():
        return replacement_word.upper()
    if original_token_text and original_token_text[0].isupper():
        return replacement_word.capitalize()
    return replacement_word.lower()


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

    if source_level == target_level:
        return sentence

    level_vocab = _get_level_vocab()
    target_vocab = level_vocab.get(target_level, set())
    if not target_vocab:
        target_order = CEFR_ORDER.get(target_level, 0)
        target_vocab = set()
        for L in CEFR_LEVELS:
            if CEFR_ORDER[L] <= target_order and level_vocab.get(L):
                target_vocab |= level_vocab[L]
        if not target_vocab:
            for L in CEFR_LEVELS:
                target_vocab |= level_vocab.get(L, set())
    if not target_vocab:
        return sentence

    try:
        nlp = _get_nlp()
    except Exception:
        return sentence

    lm = None
    try:
        lm = _get_lm()
    except Exception:
        pass

    level_counts = _level_counts
    doc = nlp(sentence)
    rebuilt = []
    last_word = None
    for tok in doc:
        # Only consider replacing content words, and avoid named entities.
        if (
            tok.is_alpha
            and tok.lower_ not in target_vocab
            and tok.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
            and not tok.ent_type_
        ):
            rep = _best_replacement(
                tok.text,
                target_vocab,
                nlp,
                lm=lm,
                prev_word=last_word,
                target_level=target_level,
                orig_tok=tok,
                level_counts=level_counts,
            )
            text = _match_case(tok.text, rep) if rep else tok.text
        else:
            text = tok.text
        rebuilt.append(text)
        rebuilt.append(tok.whitespace_)
        if tok.is_alpha and text:
            last_word = text.lower()
    return "".join(rebuilt)


# Eager-load vocab and LM when module is imported, so training output
# appears before main.py prints "Test TestResults", "Test 2", ...
if __name__ != "__main__":
    try:
        print("Initialising (training runs first, then tests below).\n")
        _get_level_vocab()
        _get_lm()
        print("")
    except Exception:
        pass
