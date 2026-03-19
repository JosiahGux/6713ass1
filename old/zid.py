from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional
import math
import re

import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_TO_NUM = {lvl: i + 1 for i, lvl in enumerate(CEFR_LEVELS)}

# -------------------------------------------------
# Tunable parameters
# -------------------------------------------------

MAX_REPLACEMENTS = 1
MAX_POSITIONS_TO_CHECK = 4
MAX_WORDNET_CANDIDATES = 12

MIN_DOC_FREQ = 2
MIN_GLOBAL_CAND_FREQ = 5
MIN_SEMANTIC_SIM = 0.24
MIN_TARGET_GAIN = 0.10
MIN_FINAL_SCORE = 0.28

ALPHA_TARGET = 0.42
ALPHA_SEM = 0.36
ALPHA_CONTEXT = 0.16
ALPHA_FREQ = 0.06

# -------------------------------------------------
# Optional resources
# -------------------------------------------------

_NLTK_READY = False
_wordnet = None
_WN_LEMMATIZER = None
_STOPWORDS = None
_nltk = None

_SPACY_READY = False
_SPACY_NLP = None
_HAS_PYINFLECT = False


def _try_init_nltk():
    global _NLTK_READY, _wordnet, _WN_LEMMATIZER, _STOPWORDS, _nltk
    if _NLTK_READY:
        return

    try:
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        resources = [
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("corpora/stopwords", "stopwords"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ]

        for lookup_name, download_name in resources:
            try:
                nltk.data.find(lookup_name)
            except LookupError:
                try:
                    nltk.download(download_name, quiet=True)
                    nltk.data.find(lookup_name)
                except Exception:
                    pass

        _nltk = nltk

        try:
            wn.synsets("dog")
            _wordnet = wn
        except Exception:
            _wordnet = None

        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("dogs", "n")
            _WN_LEMMATIZER = lemmatizer
        except Exception:
            _WN_LEMMATIZER = None

        try:
            _STOPWORDS = set(stopwords.words("english"))
        except Exception:
            _STOPWORDS = set()

    except Exception:
        _nltk = None
        _wordnet = None
        _WN_LEMMATIZER = None
        _STOPWORDS = set()

    _NLTK_READY = True


def _try_init_spacy():
    global _SPACY_READY, _SPACY_NLP, _HAS_PYINFLECT
    if _SPACY_READY:
        return

    try:
        import spacy

        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            try:
                spacy.cli.download("en_core_web_sm")
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = None

        try:
            import pyinflect  # noqa: F401
            _HAS_PYINFLECT = True
        except Exception:
            _HAS_PYINFLECT = False

    except Exception:
        _SPACY_NLP = None
        _HAS_PYINFLECT = False

    _SPACY_READY = True


# -------------------------------------------------
# Training data loading
# -------------------------------------------------

def load_training_data(path: str = "data.csv") -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)
    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


# -------------------------------------------------
# Text processing
# -------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")


def clean_training_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\b(?:title|subject|date|memo)\s*:\s*", " ", text, flags=re.I)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\S+@\S+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text))


def is_word(tok: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok))


def is_stopword(word: str) -> bool:
    _try_init_nltk()
    return word.lower() in (_STOPWORDS or set())


def simple_lemma(word: str) -> str:
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("ied"):
        return w[:-3] + "y"
    if w.endswith("ed"):
        base = w[:-2]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("es"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def lemmatize_word(word: str, pos: str = "n") -> str:
    _try_init_nltk()
    if _WN_LEMMATIZER is not None:
        try:
            return _WN_LEMMATIZER.lemmatize(word.lower(), pos=pos)
        except Exception:
            pass
    return simple_lemma(word)


def sentence_to_lemmas(text: str) -> List[str]:
    out = []
    for tok in tokenize(clean_training_text(text)):
        if is_word(tok):
            out.append(lemmatize_word(tok))
    return out


# -------------------------------------------------
# Sentence analysis
# -------------------------------------------------

def analyze_sentence(sentence: str) -> List[Dict[str, object]]:
    _try_init_spacy()

    if _SPACY_NLP is not None:
        doc = _SPACY_NLP(sentence)
        out = []
        for token in doc:
            if token.is_space:
                continue
            lemma = token.lemma_.lower() if is_word(token.text) else token.text
            out.append(
                {
                    "token": token.text,
                    "lemma": lemma,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "is_stop": token.is_stop,
                    "is_punct": token.is_punct,
                    "is_digit": token.like_num,
                    "is_propn": token.pos_ == "PROPN",
                    "ws": token.whitespace_,
                }
            )
        return out

    parts = []
    pattern = re.compile(r"\S+\s*")
    for m in pattern.finditer(sentence):
        chunk = m.group(0)
        token = chunk.rstrip()
        ws = chunk[len(token):]
        punct = bool(re.fullmatch(r"[^\w\s]+", token))
        parts.append(
            {
                "token": token,
                "lemma": lemmatize_word(token.lower()) if not punct and is_word(token) else token,
                "pos": "X",
                "tag": "",
                "is_stop": token.lower() in (_STOPWORDS or set()),
                "is_punct": punct,
                "is_digit": token.isdigit(),
                "is_propn": token.istitle(),
                "ws": ws,
            }
        )
    return parts


# -------------------------------------------------
# CEFR lexicon + target-level language model
# -------------------------------------------------

class BigramLM:
    def __init__(self, k: float = 0.5):
        self.k = k
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()
        self.vocab = set()

    def update(self, tokens: List[str]):
        prev = "<s>"
        self.unigram[prev] += 1
        self.vocab.add(prev)
        for tok in tokens + ["</s>"]:
            self.bigram[prev][tok] += 1
            self.unigram[tok] += 1
            self.vocab.add(tok)
            prev = tok

    def score(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        prev = "<s>"
        logp = 0.0
        V = max(1, len(self.vocab))
        for tok in tokens + ["</s>"]:
            num = self.bigram[prev][tok] + self.k
            den = sum(self.bigram[prev].values()) + self.k * V
            logp += math.log(num / den)
            prev = tok
        return logp / max(1, len(tokens))


class ResourceBundle:
    def __init__(self, df: pd.DataFrame):
        self.word_level_counts = defaultdict(lambda: Counter())
        self.word_total_freq = Counter()
        self.level_freq = {lvl: Counter() for lvl in CEFR_LEVELS}
        self.main_pos = defaultdict(Counter)
        self.lms = {lvl: BigramLM(k=0.5) for lvl in CEFR_LEVELS}
        self.cefr_score = {}
        self.df = df
        self._build()

    def _build(self):
        for _, row in self.df.iterrows():
            text = clean_training_text(row.get("text", ""))
            level = str(row.get("cefr_level", "")).strip()
            if level not in CEFR_LEVELS:
                continue

            analysis = analyze_sentence(text)
            lemmas = []
            seen = set()

            for item in analysis:
                tok = str(item["token"])
                lemma = str(item["lemma"]).lower()
                pos = str(item["pos"]).upper()

                if not is_word(tok):
                    continue
                if len(lemma) < 2 or not re.fullmatch(r"[a-z]+", lemma):
                    continue
                if item.get("is_propn", False):
                    continue

                lemmas.append(lemma)
                self.word_total_freq[lemma] += 1
                self.level_freq[level][lemma] += 1
                self.main_pos[lemma][pos] += 1

                if lemma not in seen:
                    self.word_level_counts[lemma][level] += 1
                    seen.add(lemma)

            if lemmas:
                self.lms[level].update(lemmas)

        for lemma, counts in self.word_level_counts.items():
            if sum(counts.values()) < MIN_DOC_FREQ:
                continue
            alpha = 0.25
            total = 0.0
            denom = 0.0
            for i, lvl in enumerate(CEFR_LEVELS, start=1):
                c = counts.get(lvl, 0)
                total += i * (c + alpha)
                denom += c + alpha
            self.cefr_score[lemma] = total / max(denom, 1e-9)

    def lemma_score(self, lemma: str) -> Optional[float]:
        return self.cefr_score.get(lemma.lower())

    def dominant_pos(self, lemma: str) -> str:
        cnt = self.main_pos.get(lemma)
        if not cnt:
            return "X"
        return cnt.most_common(1)[0][0]

    def lm_score(self, level: str, tokens: List[str]) -> float:
        return self.lms[level].score(tokens)

    def target_freq_bonus(self, lemma: str, target_level: str) -> float:
        return math.log1p(self.level_freq[target_level].get(lemma, 0))

    def global_freq(self, lemma: str) -> int:
        return self.word_total_freq.get(lemma, 0)


_RESOURCES = None


def get_resources(path: str = "data.csv") -> ResourceBundle:
    global _RESOURCES
    if _RESOURCES is None:
        df = load_training_data(path)
        _RESOURCES = ResourceBundle(df)
    return _RESOURCES


# -------------------------------------------------
# Candidate identification
# -------------------------------------------------

ALLOWED_REPLACE_POS = {"VERB", "ADJ"}


def is_replaceable(item: Dict[str, object]) -> bool:
    if item.get("is_punct") or item.get("is_digit") or item.get("is_stop"):
        return False
    if item.get("is_propn"):
        return False
    pos = str(item.get("pos", "")).upper()
    if pos not in ALLOWED_REPLACE_POS:
        return False
    token = str(item.get("token", ""))
    if not is_word(token):
        return False
    if len(token) <= 2:
        return False
    return True


def identify_positions(sentence: str, source_level: str, target_level: str) -> List[Dict[str, object]]:
    bundle = get_resources()
    src_num = LEVEL_TO_NUM[source_level]
    tgt_num = LEVEL_TO_NUM[target_level]
    direction = "down" if tgt_num < src_num else "up"

    analysis = analyze_sentence(sentence)
    candidates = []

    for idx, item in enumerate(analysis):
        if not is_replaceable(item):
            continue

        lemma = str(item["lemma"]).lower()
        score = bundle.lemma_score(lemma)
        if score is None:
            continue

        if direction == "down" and score <= tgt_num:
            continue
        if direction == "up" and score >= tgt_num:
            continue

        candidates.append(
            {
                "idx": idx,
                "lemma": lemma,
                "token": item["token"],
                "pos": str(item["pos"]).upper(),
                "tag": str(item["tag"]),
                "score": score,
                "distance": abs(score - tgt_num),
            }
        )

    candidates.sort(key=lambda x: x["distance"], reverse=True)
    return candidates[:MAX_POSITIONS_TO_CHECK]


# -------------------------------------------------
# Semantic similarity + candidate generation
# -------------------------------------------------

def semantic_similarity(w1: str, w2: str) -> float:
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()
    if not w1 or not w2:
        return 0.0
    if w1 == w2:
        return 1.0

    _try_init_spacy()
    if _SPACY_NLP is not None:
        try:
            lex1 = _SPACY_NLP.vocab[w1]
            lex2 = _SPACY_NLP.vocab[w2]
            if lex1.has_vector and lex2.has_vector:
                dot = float((lex1.vector * lex2.vector).sum())
                mag = float((lex1.vector**2).sum() ** 0.5) * float((lex2.vector**2).sum() ** 0.5)
                if mag > 0:
                    return max(0.0, min(1.0, dot / mag))
        except Exception:
            pass

    _try_init_nltk()
    if _wordnet is not None:
        try:
            syns1 = _wordnet.synsets(w1)
            syns2 = _wordnet.synsets(w2)
            best = 0.0
            for s1 in syns1[:3]:
                for s2 in syns2[:3]:
                    sim = s1.path_similarity(s2)
                    if sim is not None and sim > best:
                        best = sim
            return best
        except Exception:
            pass

    return 0.0


def generate_candidates(
    lemma: str,
    pos: str,
    source_score: float,
    target_num: int,
    target_level: str,
) -> List[str]:
    bundle = get_resources()
    direction = "down" if target_num < source_score else "up"

    out = []
    seen = set()

    _try_init_nltk()
    if _wordnet is None:
        return []

    wn_pos_map = {
        "VERB": _wordnet.VERB,
        "ADJ": _wordnet.ADJ,
        "ADV": _wordnet.ADV,
        "NOUN": _wordnet.NOUN,
    }
    wn_pos = wn_pos_map.get(pos)

    try:
        synsets = _wordnet.synsets(lemma, pos=wn_pos) if wn_pos else _wordnet.synsets(lemma)
    except Exception:
        synsets = []

    if pos == "VERB":
        selected_synsets = synsets[:2]
        for syn in selected_synsets:
            for l in syn.lemmas():
                cand = l.name().replace("_", " ").lower().strip()
                if " " in cand or cand == lemma or cand in seen:
                    continue
                if not re.fullmatch(r"[a-z]+", cand):
                    continue
                if bundle.global_freq(cand) < MIN_GLOBAL_CAND_FREQ:
                    continue
                cand_score = bundle.lemma_score(cand)
                if cand_score is None:
                    continue
                if direction == "down" and cand_score >= source_score:
                    continue
                if direction == "up" and cand_score <= source_score:
                    continue
                if abs(cand_score - target_num) >= abs(source_score - target_num):
                    continue
                if bundle.dominant_pos(cand) not in {"VERB", "X"}:
                    continue
                seen.add(cand)
                out.append(cand)

    elif pos == "ADJ":
        selected_synsets = synsets[:4]
        for syn in selected_synsets:
            related = [syn]
            try:
                related.extend(syn.similar_tos()[:1])
            except Exception:
                pass
            for rs in related:
                for l in rs.lemmas():
                    cand = l.name().replace("_", " ").lower().strip()
                    if " " in cand or cand == lemma or cand in seen:
                        continue
                    if not re.fullmatch(r"[a-z]+", cand):
                        continue
                    if bundle.global_freq(cand) < MIN_GLOBAL_CAND_FREQ:
                        continue
                    cand_score = bundle.lemma_score(cand)
                    if cand_score is None:
                        continue
                    if direction == "down" and cand_score >= source_score:
                        continue
                    if direction == "up" and cand_score <= source_score:
                        continue
                    if abs(cand_score - target_num) >= abs(source_score - target_num):
                        continue
                    if bundle.dominant_pos(cand) != "ADJ":
                        continue
                    seen.add(cand)
                    out.append(cand)

    out.sort(key=lambda w: bundle.target_freq_bonus(w, target_level), reverse=True)
    return out[:MAX_WORDNET_CANDIDATES]


# -------------------------------------------------
# Inflection / reconstruction
# -------------------------------------------------

_IRREGULAR_FORMS = {
    "be": {"VBD": "was", "VBN": "been", "VBG": "being", "VBZ": "is", "VBP": "are"},
    "buy": {"VBD": "bought", "VBN": "bought", "VBG": "buying", "VBZ": "buys"},
    "build": {"VBD": "built", "VBN": "built", "VBG": "building", "VBZ": "builds"},
    "do": {"VBD": "did", "VBN": "done", "VBG": "doing", "VBZ": "does"},
    "go": {"VBD": "went", "VBN": "gone", "VBG": "going", "VBZ": "goes"},
    "have": {"VBD": "had", "VBN": "had", "VBG": "having", "VBZ": "has"},
    "see": {"VBD": "saw", "VBN": "seen", "VBG": "seeing", "VBZ": "sees"},
    "take": {"VBD": "took", "VBN": "taken", "VBG": "taking", "VBZ": "takes"},
    "try": {"VBD": "tried", "VBN": "tried", "VBG": "trying", "VBZ": "tries"},
}


def match_case(word: str, template: str) -> str:
    if template.isupper():
        return word.upper()
    if template.istitle():
        return word.title()
    return word.lower()


def heuristic_inflect_verb(lemma: str, tag: str) -> str:
    tag = (tag or "").upper()
    base = lemma.lower()
    if base in _IRREGULAR_FORMS and tag in _IRREGULAR_FORMS[base]:
        return _IRREGULAR_FORMS[base][tag]
    if tag in {"VBD", "VBN"}:
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


def restore_surface(candidate_lemma: str, original_token: str, pos: str, tag: str) -> str:
    _try_init_spacy()
    if _HAS_PYINFLECT and _SPACY_NLP is not None and tag:
        try:
            doc = _SPACY_NLP(candidate_lemma)
            infl = doc[0]._.inflect(tag)
            if infl:
                return match_case(infl, original_token)
        except Exception:
            pass

    out = candidate_lemma.lower()
    if pos == "VERB":
        out = heuristic_inflect_verb(out, tag)
    elif pos == "ADJ":
        out = candidate_lemma.lower()

    return match_case(out, original_token)


def reconstruct_sentence(items: List[Dict[str, object]]) -> str:
    parts = []
    for item in items:
        parts.append(str(item["token"]) + str(item.get("ws", "")))
    return "".join(parts).rstrip()


def apply_replacement(sentence: str, idx: int, candidate_lemma: str) -> str:
    items = analyze_sentence(sentence)
    if idx < 0 or idx >= len(items):
        return sentence

    item = items[idx]
    item["token"] = restore_surface(
        candidate_lemma,
        str(item["token"]),
        str(item["pos"]).upper(),
        str(item["tag"]),
    )
    return reconstruct_sentence(items)


# -------------------------------------------------
# Candidate scoring
# -------------------------------------------------

def score_candidate(
    sentence: str,
    idx: int,
    source_lemma: str,
    candidate_lemma: str,
    source_score: float,
    target_num: int,
    target_level: str,
) -> float:
    bundle = get_resources()

    candidate_score = bundle.lemma_score(candidate_lemma)
    if candidate_score is None:
        return -1.0

    target_gain = abs(source_score - target_num) - abs(candidate_score - target_num)
    if target_gain <= MIN_TARGET_GAIN:
        return -1.0

    sem = semantic_similarity(source_lemma, candidate_lemma)
    if sem < MIN_SEMANTIC_SIM:
        return -1.0

    replaced = apply_replacement(sentence, idx, candidate_lemma)
    lm_score = bundle.lm_score(target_level, sentence_to_lemmas(replaced))
    freq_bonus = bundle.target_freq_bonus(candidate_lemma, target_level)

    context = 1.0 / (1.0 + math.exp(-(lm_score + 4.0)))

    length_penalty = 0.0
    if len(candidate_lemma) > len(source_lemma) + 2:
        length_penalty = -0.08

    final = (
        ALPHA_TARGET * target_gain
        + ALPHA_SEM * sem
        + ALPHA_CONTEXT * context
        + ALPHA_FREQ * min(1.0, freq_bonus / 5.0)
        + length_penalty
    )
    return final


# -------------------------------------------------
# Required function
# -------------------------------------------------

def transform_sentence_impl(sentence, source_level, target_level):
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    if not sentence or source_level == target_level:
        return sentence

    target_num = LEVEL_TO_NUM[target_level]
    positions = identify_positions(sentence, source_level, target_level)
    if not positions:
        return sentence

    updated_sentence = sentence
    replacements_done = 0

    for pos_info in positions:
        if replacements_done >= MAX_REPLACEMENTS:
            break

        idx = int(pos_info["idx"])
        source_lemma = str(pos_info["lemma"])
        source_pos = str(pos_info["pos"]).upper()
        source_score = float(pos_info["score"])

        candidates = generate_candidates(
            lemma=source_lemma,
            pos=source_pos,
            source_score=source_score,
            target_num=target_num,
            target_level=target_level,
        )
        if not candidates:
            continue

        best_cand = None
        best_score = -1.0

        for cand in candidates:
            s = score_candidate(
                sentence=updated_sentence,
                idx=idx,
                source_lemma=source_lemma,
                candidate_lemma=cand,
                source_score=source_score,
                target_num=target_num,
                target_level=target_level,
            )
            if s > best_score:
                best_score = s
                best_cand = cand

        if best_cand is None or best_score < MIN_FINAL_SCORE:
            continue

        new_sentence = apply_replacement(updated_sentence, idx, best_cand)
        if new_sentence != updated_sentence:
            updated_sentence = new_sentence
            replacements_done += 1

    return updated_sentence


def transform_sentence(sentence, source_level, target_level):
    return transform_sentence_impl(sentence, source_level, target_level)