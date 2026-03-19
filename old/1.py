from pathlib import Path
from collections import Counter, defaultdict
import math
import re

import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_TO_IDX = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}

# -------------------------------------------------
# Tunable parameters
# -------------------------------------------------

DEBUG_TRACE = False

MIN_WORD_FREQ = 8
MAX_REPLACEMENTS = 1

DIFF_THRESHOLD = 1.00
TARGET_BAND = 0.90
MIN_REPLACE_MARGIN = 1.00 #fallback only #0.78 #TestResults.00

W_TARGET = 1.90
W_SEMANTIC = 2.20
W_CONTEXT = 1.20
W_LM = 0.55
W_FREQ = 0.45 #0.30
W_LENGTH = 0.15

CONTEXT_WINDOW = 2
MAX_DIST_CANDIDATES = 15
MAX_WN_CANDIDATES = 15
TOP_POSITIONS = 4

#Threshold
DIST_CTX_THRESHOLD = 0.22#0.18 #0.22
FINAL_SEM_THRESHOLD = 0.35#0.28 #0.35
POS_DIST_CTX_THRESHOLD = {
    "VERB": 0.14,
    "ADJ": 0.14,
    "ADV": 0.32,
    "NOUN": 0.30,
}

POS_SEM_THRESHOLD = {
    "VERB": 0.32, #0.24
    "ADJ": 0.20, #0.24
    "ADV": 0.40,
    "NOUN": 0.42,
}

POS_MIN_MARGIN = {
    "VERB":0.75 ,#0.60
    "ADJ": 0.35 , #0.58
    "ADV": 0.95,
    "NOUN": 1.15,
}

FORCE_ONE_REPLACEMENT = True
FALLBACK_SENTENCE_GAP = 1.20
FALLBACK_MIN_SCORE = 0.20
FALLBACK_MIN_SEM = 0.22
# -------------------------------------------------
# Optional external NLP resources with safe fallback
# -------------------------------------------------

_NLTK_READY = False
_nltk = None
_wordnet = None
_WN_LEMMATIZER = None
_STOPWORDS = None

def _try_init_nltk():
    global _NLTK_READY, _nltk, _wordnet, _WN_LEMMATIZER, _STOPWORDS
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

        # Check WordNet availability
        try:
            wn.synsets("dog")
            _wordnet = wn
        except Exception:
            _wordnet = None

        # Check WordNetLemmatizer availability
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("dogs", "n")
            _WN_LEMMATIZER = lemmatizer
        except Exception:
            _WN_LEMMATIZER = None

        # Check stopwords availability
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


# -------------------------------------------------
# Tokenisation / text utils
# -------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(str(text))


def detokenize(tokens):
    text = ""
    for tok in tokens:
        if not text:
            text = tok
        elif re.match(r"[.,!?;:%)\]\}]", tok):
            text += tok
        elif re.match(r"['’]", tok) and text and text[-1].isalpha():
            text += tok
        elif text[-1] in "([{" or tok in ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]:
            text += tok
        else:
            text += " " + tok

    text = re.sub(
        r"\b([Aa])\s+([aeiouAEIOU])",
        lambda m: ("An" if m.group(1) == "A" else "an") + " " + m.group(2),
        text,
    )
    text = re.sub(
        r"\b([Aa])n\s+([^aeiouAEIOU\W])",
        lambda m: ("A" if m.group(1) == "A" else "a") + " " + m.group(2),
        text,
    )
    return text


def is_word(tok):
    return bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", tok))


def is_content_pos(pos):
    return pos in {"NOUN", "VERB", "ADJ", "ADV"}


def is_stopword(word):
    _try_init_nltk()
    return word.lower() in _STOPWORDS


def valid_vocab_token(tok):
    if tok is None:
        return False
    if not isinstance(tok, str):
        return False
    if not re.fullmatch(r"[a-z]+", tok):
        return False
    if len(tok) < 3 or len(tok) > 18:
        return False
    return True


def is_clean_candidate(cand):
    if not isinstance(cand, str):
        return False
    if not re.fullmatch(r"[a-z]+", cand):
        return False
    if len(cand) < 3 or len(cand) > 15:
        return False
    return True


# -------------------------------------------------
# POS tagging / lemmatisation
# -------------------------------------------------

def nltk_to_coarse(tag):
    if tag.startswith("NN"):
        return "NOUN"
    if tag.startswith("VB"):
        return "VERB"
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("RB"):
        return "ADV"
    return "OTHER"


def heuristic_pos(tok, prev_tok=""):
    w = tok.lower()
    prev = prev_tok.lower() if prev_tok else ""

    if not is_word(tok):
        return "OTHER"
    if w.endswith("ly"):
        return "ADV"
    if prev == "to":
        return "VERB"
    if prev in {"a", "an", "the", "this", "that", "these", "those"}:
        if w.endswith(("ing", "ed")):
            return "ADJ"
        return "NOUN"
    if w.endswith(("ing", "ed")):
        return "VERB"
    if w.endswith(("ous", "ful", "able", "ible", "ive", "al", "ic", "ish", "less", "ant", "ent")):
        return "ADJ"
    return "NOUN"


def pos_tag_tokens(tokens):
    _try_init_nltk()

    words = [t for t in tokens if is_word(t)]
    if _nltk is not None and words:
        try:
            tagged_words = _nltk.pos_tag(words)
            word_iter = iter(tagged_words)
            coarse_tags = []
            raw_tags = []
            for tok in tokens:
                if is_word(tok):
                    _, raw = next(word_iter)
                    coarse_tags.append(nltk_to_coarse(raw))
                    raw_tags.append(raw)
                else:
                    coarse_tags.append("OTHER")
                    raw_tags.append("OTHER")
            return coarse_tags, raw_tags
        except Exception:
            pass

    coarse_tags = []
    raw_tags = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        coarse = heuristic_pos(tok, prev)
        coarse_tags.append(coarse)
        if coarse == "NOUN":
            raw_tags.append("NN")
        elif coarse == "VERB":
            raw_tags.append("VB")
        elif coarse == "ADJ":
            raw_tags.append("JJ")
        elif coarse == "ADV":
            raw_tags.append("RB")
        else:
            raw_tags.append("OTHER")
    return coarse_tags, raw_tags


def _wn_pos_for_coarse(pos):
    if _wordnet is None:
        return None
    return {
        "NOUN": _wordnet.NOUN,
        "VERB": _wordnet.VERB,
        "ADJ": _wordnet.ADJ,
        "ADV": _wordnet.ADV,
    }.get(pos)


def simple_lemma(word, pos):
    w = word.lower()

    if pos == "NOUN":
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ses") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            return w[:-1]

    if pos == "VERB":
        if w.endswith("ing") and len(w) > 5:
            base = w[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if w.endswith("ied") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ed") and len(w) > 4:
            base = w[:-2]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            return base
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]

    if pos == "ADJ":
        if w.endswith("er") and len(w) > 4:
            return w[:-2]
        if w.endswith("est") and len(w) > 5:
            return w[:-3]

    return w


def lemmatize(word, pos):
    _try_init_nltk()
    if _WN_LEMMATIZER is not None:
        try:
            wn_pos = _wn_pos_for_coarse(pos)
            if wn_pos is not None:
                return _WN_LEMMATIZER.lemmatize(word.lower(), pos=wn_pos)
            return _WN_LEMMATIZER.lemmatize(word.lower())
        except Exception:
            pass
    return simple_lemma(word, pos)


def preserve_case(src, tgt):
    if src.isupper():
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def _guess_ptb_tag(source_tok, source_pos, prev_tok="", next_tok=""):
    src = source_tok.lower()
    prev = prev_tok.lower() if prev_tok else ""

    if source_pos == "NOUN":
        if src.endswith("s") and not src.endswith("ss") and len(src) > 3:
            return "NNS"
        return "NN"

    if source_pos == "ADJ":
        if src.endswith("est"):
            return "JJS"
        if src.endswith("er"):
            return "JJR"
        return "JJ"

    if source_pos == "ADV":
        return "RB"

    if source_pos == "VERB":
        if src.endswith("ing"):
            return "VBG"
        if src.endswith("ed"):
            if prev in {"have", "has", "had"}:
                return "VBN"
            return "VBD"
        if src.endswith("s"):
            return "VBZ"
        return "VB"

    return "NN"


def _regular_plural(noun):
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _regular_past(verb):
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    if len(verb) >= 3 and verb[-1] not in "aeiouywx" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        return verb + verb[-1] + "ed"
    return verb + "ed"


def _regular_ing(verb):
    if verb.endswith("ie"):
        return verb[:-2] + "ying"
    if verb.endswith("e") and verb not in {"be", "see"}:
        return verb[:-1] + "ing"
    if len(verb) >= 3 and verb[-1] not in "aeiouywx" and verb[-2] in "aeiou" and verb[-3] not in "aeiou":
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _regular_3sg(verb):
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if verb.endswith(("s", "x", "z", "ch", "sh", "o")):
        return verb + "es"
    return verb + "s"


def inflect_like(candidate_lemma, source_tok, source_pos, prev_tok="", next_tok=""):
    cand = candidate_lemma.lower()
    src = source_tok.lower()

    try:
        import spacy
        import pyinflect  # noqa: F401

        if not hasattr(inflect_like, "_nlp"):
            try:
                inflect_like._nlp = spacy.load("en_core_web_sm")
            except Exception:
                inflect_like._nlp = spacy.blank("en")

        ptb_tag = _guess_ptb_tag(source_tok, source_pos, prev_tok, next_tok)
        doc = inflect_like._nlp(cand)
        if len(doc) > 0:
            infl = doc[0]._.inflect(ptb_tag)
            if infl:
                return preserve_case(source_tok, infl)
    except Exception:
        pass

    if source_pos in {"ADJ", "ADV"}:
        return preserve_case(source_tok, cand)

    if source_pos == "NOUN":
        if src.endswith("s") and len(src) > 3 and not src.endswith("ss"):
            cand = _regular_plural(cand)
        return preserve_case(source_tok, cand)

    if source_pos == "VERB":
        if src.endswith("ing"):
            cand = _regular_ing(cand)
        elif src.endswith("ed"):
            cand = _regular_past(cand)
        elif src.endswith("s"):
            cand = _regular_3sg(cand)
        return preserve_case(source_tok, cand)

    return preserve_case(source_tok, cand)


# -------------------------------------------------
# Training data
# -------------------------------------------------

def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")

    df = pd.read_csv(file_path)
    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


def clean_text(s):
    s = str(s)
    s = re.sub(r"\b(?:title|subject|date|memo)\s*:\s*", " ", s, flags=re.I)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\b\S+@\S+\b", " ", s)
    s = re.sub(r"\b[A-Z]{2,}\b", " ", s)
    s = re.sub(r"[_/\\|@#*+=~<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class LexicalModel:
    def __init__(self, df):
        self.df = df
        self.word_level_counts = defaultdict(lambda: Counter())
        self.word_freq = Counter()
        self.bigram_counts = {lvl: defaultdict(Counter) for lvl in CEFR_LEVELS}
        self.unigram_counts = {lvl: Counter() for lvl in CEFR_LEVELS}
        self.context_vectors = defaultdict(Counter)
        self.pos_counts = defaultdict(Counter)
        self.candidate_pool = []
        self._difficulty_cache = {}

        self._build()

    def _build(self):
        for _, row in self.df.iterrows():
            text = clean_text(row["text"])
            level = str(row["cefr_level"]).strip()
            if level not in CEFR_LEVELS:
                continue

            toks = tokenize(text)
            if not toks:
                continue

            coarse_tags, raw_tags = pos_tag_tokens(toks)

            lemmas = []
            clean_positions = []
            seen_lemmas = set()

            for i, tok in enumerate(toks):
                if not is_word(tok):
                    lemmas.append(None)
                    continue

                raw_tag = raw_tags[i]
                coarse_pos = coarse_tags[i]

                # Remove proper nouns from training vocabulary.
                if raw_tag in {"NNP", "NNPS"}:
                    lemmas.append(None)
                    continue

                lemma = lemmatize(tok, coarse_pos)

                if not valid_vocab_token(lemma):
                    lemmas.append(None)
                    continue

                lemmas.append(lemma)

                self.word_freq[lemma] += 1
                self.pos_counts[lemma][coarse_pos] += 1
                self.unigram_counts[level][lemma] += 1

                if lemma not in seen_lemmas:
                    self.word_level_counts[lemma][level] += 1
                    seen_lemmas.add(lemma)

                if is_content_pos(coarse_pos) and not is_stopword(lemma):
                    clean_positions.append(i)

            prev = "<s>"
            for lemma in lemmas:
                if lemma is None:
                    continue
                self.bigram_counts[level][prev][lemma] += 1
                prev = lemma
            self.bigram_counts[level][prev]["</s>"] += 1

            for i in clean_positions:
                src = lemmas[i]
                if src is None:
                    continue
                left = max(0, i - CONTEXT_WINDOW)
                right = min(len(lemmas), i + CONTEXT_WINDOW + 1)
                for j in range(left, right):
                    if i == j:
                        continue
                    nb = lemmas[j]
                    if nb is None or is_stopword(nb):
                        continue
                    self.context_vectors[src][nb] += 1.0 / (abs(i - j))

        self.candidate_pool = [
            lemma for lemma, freq in self.word_freq.items()
            if freq >= MIN_WORD_FREQ
            and valid_vocab_token(lemma)
            and not is_stopword(lemma)
        ]

    def main_pos(self, lemma):
        cnt = self.pos_counts.get(lemma)
        if not cnt:
            return "NOUN"
        return cnt.most_common(1)[0][0]

    def difficulty(self, lemma):
        if lemma in self._difficulty_cache:
            return self._difficulty_cache[lemma]

        counts = self.word_level_counts.get(lemma)
        if not counts:
            self._difficulty_cache[lemma] = 2.5
            return 2.5

        alpha = 0.25
        total = 0.0
        denom = 0.0
        for i, lvl in enumerate(CEFR_LEVELS):
            c = counts.get(lvl, 0)
            total += i * (c + alpha)
            denom += c + alpha

        score = total / max(denom, 1e-9)
        self._difficulty_cache[lemma] = score
        return score

    def confidence(self, lemma):
        return math.log1p(self.word_freq.get(lemma, 0))

    def level_freq(self, lemma, level):
        return self.unigram_counts[level].get(lemma, 0)

    def target_band_ok(self, lemma, target_idx):
        diff = self.difficulty(lemma)
        return abs(diff - target_idx) <= TARGET_BAND

    def local_lm_logprob(self, prev_lemma, lemma, level):
        prev_counter = self.bigram_counts[level].get(prev_lemma, Counter())
        vocab_size = max(len(self.candidate_pool), 1)
        numerator = prev_counter.get(lemma, 0) + 0.25
        denominator = sum(prev_counter.values()) + 0.25 * vocab_size
        return math.log(numerator / max(denominator, 1e-12))

    def context_cosine(self, a, b):
        if a == b:
            return 1.0

        va = self.context_vectors.get(a)
        vb = self.context_vectors.get(b)
        if not va or not vb:
            return 0.0

        keys = set(va.keys()) & set(vb.keys())
        if not keys:
            return 0.0

        dot = sum(va[k] * vb[k] for k in keys)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def distributional_candidates(self, lemma, pos, target_idx):
        if lemma not in self.context_vectors:
            return []

        scored = []
        src_diff = self.difficulty(lemma)

        for cand in self.candidate_pool:
            if cand == lemma:
                continue
            if self.main_pos(cand) != pos:
                continue
            if not self.target_band_ok(cand, target_idx):
                continue
            if not is_clean_candidate(cand):
                continue

            ctx = self.context_cosine(lemma, cand)
            ctx_threshold = POS_DIST_CTX_THRESHOLD.get(pos, DIST_CTX_THRESHOLD)
            if ctx < ctx_threshold:
                continue

            tgt_gain = abs(src_diff - target_idx) - abs(self.difficulty(cand) - target_idx)
            if tgt_gain < 0.05:
                continue

            scored.append((ctx + 0.25 * tgt_gain, cand))

        scored.sort(reverse=True)
        return [c for _, c in scored[:MAX_DIST_CANDIDATES]]

    def wordnet_candidates(self, lemma, pos):
        _try_init_nltk()
        if _wordnet is None:
            return []

        wn_pos = _wn_pos_for_coarse(pos)
        if wn_pos is None:
            return []

        out = []
        seen = set()

        try:
            synsets = _wordnet.synsets(lemma, pos=wn_pos)
        except Exception:
            synsets = []

        # Be very conservative for verbs:
        # only use direct lemma names from the first few synsets.
        if pos == "VERB":
            for syn in synsets[:3]:
                for l in syn.lemmas():
                    cand = l.name().replace("_", " ").lower()
                    if " " in cand:
                        continue
                    if cand == lemma or cand in seen:
                        continue
                    if not is_clean_candidate(cand):
                        continue
                    seen.add(cand)
                    out.append(cand)
            return out[:8]

        # For adjectives, allow slightly broader expansion
        if pos == "ADJ":
            for syn in synsets[:4]:
                related = [syn]
                try:
                    related.extend(syn.similar_tos()[:1])
                except Exception:
                    pass

                for rs in related:
                    for l in rs.lemmas():
                        cand = l.name().replace("_", " ").lower()
                        if " " in cand:
                            continue
                        if cand == lemma or cand in seen:
                            continue
                        if not is_clean_candidate(cand):
                            continue
                        seen.add(cand)
                        out.append(cand)
            return out[:12]
        
        return []

# -------------------------------------------------
# Global model init
# -------------------------------------------------

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        df = load_training_data("../data.csv")
        _MODEL = LexicalModel(df)
    return _MODEL


# -------------------------------------------------
# Candidate scoring
# -------------------------------------------------

def semantic_similarity(src_lemma, cand_lemma, pos, model):
    base = model.context_cosine(src_lemma, cand_lemma)

    _try_init_nltk()
    wn_bonus = 0.0
    if _wordnet is not None:
        wn_pos = _wn_pos_for_coarse(pos)
        if wn_pos is not None:
            try:
                src_syns = _wordnet.synsets(src_lemma, pos=wn_pos)[:4]
                cand_syns = _wordnet.synsets(cand_lemma, pos=wn_pos)[:4]
                best = 0.0
                for s1 in src_syns:
                    for s2 in cand_syns:
                        val = s1.wup_similarity(s2) or 0.0
                        if val > best:
                            best = val
                wn_bonus = best
            except Exception:
                wn_bonus = 0.0

    return max(base, 0.60 * base + 0.40 * wn_bonus)

# Prioritize adj adv verb

def token_is_replaceable(tok, pos, idx, tokens):
    if not is_word(tok):
        return False
    if pos not in {"VERB", "ADJ"}:
        return False
    if is_stopword(tok):
        return False
    if len(tok) <= 2:
        return False
    if tok[0].isupper() and idx != 0:
        return False
    return True

def score_candidate(
    source_lemma,
    cand_lemma,
    source_pos,
    prev_lemma,
    target_level,
    target_idx,
    model,
    direction,
):
    src_diff = model.difficulty(source_lemma)
    cand_diff = model.difficulty(cand_lemma)

    target_gain = abs(src_diff - target_idx) - abs(cand_diff - target_idx)
    sem = semantic_similarity(source_lemma, cand_lemma, source_pos, model)
    ctx = model.context_cosine(source_lemma, cand_lemma)

    lm_keep = model.local_lm_logprob(prev_lemma, source_lemma, target_level)
    lm_cand = model.local_lm_logprob(prev_lemma, cand_lemma, target_level)
    lm_gain = lm_cand - lm_keep

    freq_bonus = (
        math.log1p(model.level_freq(cand_lemma, target_level))
        - math.log1p(model.level_freq(source_lemma, target_level))
    )

    if direction < 0:
        length_bonus = (len(source_lemma) - len(cand_lemma)) / 8.0
    else:
        length_bonus = (len(cand_lemma) - len(source_lemma)) / 8.0

    #noun_penalty = -0.45 if source_pos == "NOUN" else 0.0

    total = (
            W_TARGET * target_gain
            + W_SEMANTIC * sem
            + W_CONTEXT * ctx
            + W_LM * lm_gain
            + W_FREQ * freq_bonus
            + W_LENGTH * length_bonus
            #+ noun_penalty
    )

    return {
        "target_gain": target_gain,
        "semantic": sem,
        "context": ctx,
        "lm_gain": lm_gain,
        "freq_bonus": freq_bonus,
        "length_bonus": length_bonus,
        "final_total": total,
        "cand_diff": cand_diff,
    }


# -------------------------------------------------
# Required function
# -------------------------------------------------

def transform_sentence(sentence, source_level, target_level):
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    if not sentence or source_level == target_level:
        return sentence

    model = get_model()
    target_idx = LEVEL_TO_IDX[target_level]
    source_idx = LEVEL_TO_IDX[source_level]
    direction = -1 if target_idx < source_idx else 1

    tokens = tokenize(sentence)
    if not tokens:
        return sentence

    pos_tags, raw_tags = pos_tag_tokens(tokens)
    lemmas = [lemmatize(tok, pos_tags[i]) if is_word(tok) else None for i, tok in enumerate(tokens)]

    sentence_max_gap = 0.0
    position_scores = []
    for i, tok in enumerate(tokens):
        pos = pos_tags[i]
        if not token_is_replaceable(tok, pos, i, tokens):
            continue

        lemma = lemmas[i]
        if lemma is None or not valid_vocab_token(lemma):
            continue

        diff = model.difficulty(lemma)
        gap = abs(diff - target_idx)
        conf = model.confidence(lemma)

        sentence_max_gap = max(sentence_max_gap, gap)

        if gap < DIFF_THRESHOLD:
            continue

        if direction < 0 and diff <= target_idx + 0.15:
            continue
        if direction > 0 and diff >= target_idx - 0.15:
            continue

        priority = gap + 0.10 * conf
        position_scores.append((priority, i))

        if DEBUG_TRACE:
            print(
                f"[TRACE] inspect token='{tok}' lemma='{lemma}' pos={pos} "
                f"difficulty={diff:.2f} target={target_idx} gap={gap:.2f} conf={conf:.2f}"
            )

    position_scores.sort(reverse=True)
    chosen_positions = [i for _, i in position_scores[:TOP_POSITIONS]]

    new_tokens = list(tokens)
    replacements_done = 0

    fallback_best = None
    fallback_best_feats = None
    fallback_best_pos = None
    fallback_best_tok = None
    fallback_best_pos_tag = None

    for i in chosen_positions:
        if replacements_done >= MAX_REPLACEMENTS:
            break

        tok = new_tokens[i]
        pos = pos_tags[i]
        lemma = lemmas[i]
        prev_lemma = lemmas[i - 1] if i > 0 and lemmas[i - 1] is not None else "<s>"

        candidates = set()

        if pos == "VERB":
        # verbs: only use conservative WordNet candidates
            for cand in model.wordnet_candidates(lemma, pos):
                candidates.add(cand)

        elif pos == "ADJ":
        # adjectives: WordNet + distributional candidates
            for cand in model.wordnet_candidates(lemma, pos):
                candidates.add(cand)

    for cand in model.distributional_candidates(lemma, pos, target_idx):
        candidates.add(cand)

        filtered = []
        src_diff = model.difficulty(lemma)

        for cand in candidates:
            if cand == lemma:
                continue
            if not is_clean_candidate(cand):
                continue
            if model.word_freq.get(cand, 0) < MIN_WORD_FREQ:
                continue
            if model.main_pos(cand) != pos:
                continue
            if not model.target_band_ok(cand, target_idx):
                continue

            cand_diff = model.difficulty(cand)
            if abs(cand_diff - target_idx) >= abs(src_diff - target_idx) - 0.01:
                continue
            
            sem = semantic_similarity(lemma, cand, pos, model)
            sem_threshold = POS_SEM_THRESHOLD.get(pos, FINAL_SEM_THRESHOLD)
            if sem < sem_threshold:
                continue
            filtered.append(cand)

        if DEBUG_TRACE:
            print(f"[TRACE] position={i} token='{tok}' lemma='{lemma}' pos={pos}")
            print(f"[TRACE] raw candidates={sorted(candidates)[:20]}")
            print(f"[TRACE] filtered={filtered[:15]}")

        best = None
        best_feats = None

        for cand in filtered:
            feats = score_candidate(
                lemma,
                cand,
                pos,
                prev_lemma,
                target_level,
                target_idx,
                model,
                direction,
            )

            # candidate must actually move toward target level
            if feats["target_gain"] <= 0.10:
                continue
        
            if DEBUG_TRACE:
                print(
                    f"[TRACE] cand='{cand}' diff={feats['cand_diff']:.2f} "
                    f"target_gain={feats['target_gain']:.2f} sem={feats['semantic']:.2f} "
                    f"ctx={feats['context']:.2f} lm={feats['lm_gain']:.2f} "
                    f"freq={feats['freq_bonus']:.2f} len={feats['length_bonus']:.2f} "
                    f"final={feats['final_total']:.2f}"
                )

            if best_feats is None or feats["final_total"] > best_feats["final_total"]:
                best = cand
                best_feats = feats

        if best is None:
            continue

        if best is not None and best_feats is not None:
            if fallback_best_feats is None or best_feats["final_total"] > fallback_best_feats["final_total"]:
                fallback_best = best
                fallback_best_feats = best_feats
                fallback_best_pos = i
                fallback_best_tok = tok
                fallback_best_pos_tag = pos

        # if best_feats["final_total"] < MIN_REPLACE_MARGIN:
        # continue
        min_margin = POS_MIN_MARGIN.get(pos, MIN_REPLACE_MARGIN)
        if best_feats["final_total"] < min_margin:
            continue

        replacement = inflect_like(
            best,
            tok,
            pos,
            prev_tok=new_tokens[i - 1] if i > 0 else "",
            next_tok=new_tokens[i + 1] if i + 1 < len(new_tokens) else "",
        )

        if replacement.lower() == tok.lower():
            continue
        if not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", replacement):
            continue

        if DEBUG_TRACE:
            print(f"[TRACE] replace '{tok}' -> '{replacement}' score={best_feats['final_total']:.2f}")

        new_tokens[i] = replacement
        lemmas[i] = lemmatize(replacement, pos)
        replacements_done += 1

    if (
            replacements_done == 0
            and FORCE_ONE_REPLACEMENT
            and fallback_best is not None
            and fallback_best_feats is not None
            and sentence_max_gap >= FALLBACK_SENTENCE_GAP
            and fallback_best_feats["final_total"] >= FALLBACK_MIN_SCORE
            and fallback_best_feats["semantic"] >= FALLBACK_MIN_SEM
    ):
        replacement = inflect_like(
            fallback_best,
            fallback_best_tok,
            fallback_best_pos_tag,
            prev_tok=new_tokens[fallback_best_pos - 1] if fallback_best_pos > 0 else "",
            next_tok=new_tokens[fallback_best_pos + 1] if fallback_best_pos + 1 < len(new_tokens) else "",
        )

        if replacement.lower() != fallback_best_tok.lower() and re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", replacement):
            new_tokens[fallback_best_pos] = replacement

    return detokenize(new_tokens)