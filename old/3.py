from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
import math
import re

import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_TO_INT = {level: i for i, level in enumerate(CEFR_LEVELS)}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]")
ALPHA_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "as", "is", "am", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "them", "my", "your", "his", "their", "our", "this", "that",
    "these", "those", "there", "here", "very", "too", "so", "than", "then", "not", "no",
    "yesterday", "today", "tomorrow", "after", "before", "near", "under", "over", "into", "out",
    "up", "down", "off", "about", "through", "because", "what", "which", "who", "whom", "whose",
}

# High-precision manual lexicon, with data-driven scoring used to decide when to apply it.
MANUAL_SUBS = {
    "purchase": ["buy"],
    "purchased": ["bought"],
    "magnificent": ["big", "great", "nice"],
    "extremely": ["very", "really"],
    "delighted": ["glad", "happy"],
    "performance": ["show"],
    "attempt": ["try"],
    "attempted": ["tried"],
    "resolve": ["solve", "fix"],
    "resolved": ["solved"],
    "complicated": ["hard", "difficult"],
    "professor": ["teacher"],
    "concept": ["idea"],
    "construct": ["build"],
    "constructed": ["built"],
    "realise": ["notice"],
    "realised": ["noticed"],
    "committee": ["group", "team"],
    "evaluate": ["check", "review"],
    "proposal": ["plan", "idea"],
    "exhausted": ["tired"],
    "completing": ["doing", "finishing"],
    "complete": ["do", "finish"],
    "assignment": ["work", "task"],
    "demonstrate": ["show"],
    "demonstrates": ["shows"],
    "significant": ["big", "clear", "good"],
    "improvement": ["progress", "change"],
    "scientist": ["researcher"],
    "conduct": ["do"],
    "conducted": ["did"],
    "experiment": ["test"],
}

MANUAL_COMPLEX = {
    "buy": ["purchase"],
    "bought": ["purchased"],
    "big": ["significant", "substantial"],
    "great": ["excellent", "remarkable"],
    "nice": ["pleasant"],
    "very": ["extremely"],
    "happy": ["delighted"],
    "show": ["demonstrate"],
    "shows": ["demonstrates"],
    "try": ["attempt"],
    "tried": ["attempted"],
    "solve": ["resolve"],
    "solved": ["resolved"],
    "hard": ["complicated"],
    "idea": ["concept", "proposal"],
    "teacher": ["professor"],
    "build": ["construct"],
    "built": ["constructed"],
    "check": ["evaluate"],
    "plan": ["proposal"],
    "tired": ["exhausted"],
    "work": ["assignment"],
    "test": ["experiment"],
    "do": ["conduct"],
    "did": ["conducted"],
}


def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")
    df = pd.read_csv(file_path)
    required_columns = {"text", "cefr_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


def _normalize_text(text: str) -> str:
    text = str(text).replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\b(?:title|subject|date)\s*:\s*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=1)
def _get_wordnet():
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synsets("dog")
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-TestResults.4", quiet=True)
                wn.synsets("dog")
            except Exception:
                return None
        return wn
    except Exception:
        return None


@lru_cache(maxsize=1)
def _build_resources():
    path = Path(__file__).with_name("data.csv")
    if not path.exists():
        path = Path("../data.csv")
    df = load_training_data(path)

    doc_freq = defaultdict(lambda: [0] * len(CEFR_LEVELS))
    unigrams = [Counter() for _ in CEFR_LEVELS]
    bigrams = [Counter() for _ in CEFR_LEVELS]
    vocab = set()

    for row in df.itertuples(index=False):
        level = row.cefr_level
        if level not in LEVEL_TO_INT:
            continue
        idx = LEVEL_TO_INT[level]
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", _normalize_text(row.text).lower())
        if not words:
            continue
        for w in set(words):
            doc_freq[w][idx] += 1
        seq = ["<s>"] + words + ["</s>"]
        for w in words:
            unigrams[idx][w] += 1
            vocab.add(w)
        for a, b in zip(seq, seq[1:]):
            bigrams[idx][(a, b)] += 1

    word_score = {}
    word_total = {}
    for w, counts in doc_freq.items():
        smooth = [c + 0.5 for c in counts]
        denom = sum(smooth)
        word_score[w] = sum(i * c for i, c in enumerate(smooth)) / denom
        word_total[w] = sum(counts)

    return {
        "doc_freq": doc_freq,
        "unigrams": unigrams,
        "bigrams": bigrams,
        "word_score": word_score,
        "word_total": word_total,
        "vocab_size": max(len(vocab), 1),
    }


def _tokenize(text: str):
    return WORD_RE.findall(text)


def _is_word(tok: str) -> bool:
    return bool(ALPHA_RE.fullmatch(tok or ""))


def _lemma(word: str) -> str:
    w = word.lower()
    irregular = {
        "bought": "buy",
        "built": "build",
        "did": "do",
        "done": "do",
        "ran": "run",
        "saw": "see",
        "seen": "see",
        "knew": "know",
        "known": "know",
        "was": "be",
        "were": "be",
        "children": "child",
    }
    if w in irregular:
        return irregular[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ing"):
        base = w[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(w) > 4 and w.endswith("ised"):
        return w[:-4] + "ise"
    if len(w) > 4 and w.endswith("ized"):
        return w[:-4] + "ize"
    if len(w) > 3 and w.endswith("ed"):
        base = w[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _inflect_like(candidate: str, source: str, prev_word: str = "") -> str:
    src = source.lower()
    cand = candidate.lower()
    prev = (prev_word or "").lower()

    irregular_past = {"buy": "bought", "build": "built", "do": "did", "run": "ran", "see": "saw", "know": "knew"}
    irregular_past_values = set(irregular_past.values())
    irregular_3sg = {"do": "does", "go": "goes", "have": "has"}
    be_words = {"am", "is", "are", "was", "were", "be", "been", "being"}
    adjectival_ed = {"delighted", "exhausted", "interested", "worried", "excited", "pleased", "surprised"}

    if src.endswith("ing") and len(src) > 4:
        if cand.endswith("ing"):
            pass
        elif cand.endswith("e") and cand not in {"be", "see"}:
            cand = cand[:-1] + "ing"
        else:
            cand = cand + "ing"
    elif src.endswith("ed") and len(src) > 3:
        if src in adjectival_ed or prev in be_words or cand.endswith("ed") or cand in irregular_past_values:
            pass
        elif cand in irregular_past:
            cand = irregular_past[cand]
        elif cand.endswith("e"):
            cand = cand + "d"
        else:
            cand = cand + "ed"
    elif src.endswith("ies") and len(src) > 4:
        cand = cand[:-1] + "ies" if cand.endswith("y") else cand + "s"
    elif src.endswith("es") and len(src) > 3:
        cand = cand + "es" if cand.endswith(("s", "x", "z", "ch", "sh", "o")) else cand + "s"
    elif src.endswith("s") and len(src) > 2 and not src.endswith("ss"):
        cand = irregular_3sg.get(cand, cand + "s")

    return _match_case(source, cand)


def _word_difficulty(word: str, resources) -> float:
    lw = word.lower()
    lem = _lemma(word)
    if lem in resources["word_score"]:
        return resources["word_score"][lem]
    if lw in resources["word_score"]:
        return resources["word_score"][lw]
    return 2.8


def _local_lm_score(prev_word: str, cand: str, next_word: str, target_idx: int, resources) -> float:
    prev_w = prev_word.lower() if _is_word(prev_word) else "<s>"
    next_w = next_word.lower() if _is_word(next_word) else "</s>"
    cand_w = cand.lower()
    unigrams = resources["unigrams"][target_idx]
    bigrams = resources["bigrams"][target_idx]
    v = resources["vocab_size"]

    def prob(a, b):
        if a == "<s>":
            denom = sum(count for (x, _), count in bigrams.items() if x == "<s>") + 0.5 * v
        else:
            denom = unigrams[a] + 0.5 * v
        return (bigrams[(a, b)] + 0.5) / max(denom, 1.0)

    return math.log(prob(prev_w, cand_w)) + math.log(prob(cand_w, next_w))


def _meaning_bonus(source_word: str, candidate: str) -> float:
    src_lemma = _lemma(source_word)
    cand_lemma = _lemma(candidate)
    if cand_lemma in MANUAL_SUBS.get(src_lemma, []) or cand_lemma in MANUAL_COMPLEX.get(src_lemma, []):
        return 1.2
    wn = _get_wordnet()
    if wn is None:
        return 0.0
    try:
        src_synsets = wn.synsets(src_lemma)
        cand_synsets = wn.synsets(cand_lemma)
        best = 0.0
        for s1 in src_synsets[:4]:
            for s2 in cand_synsets[:4]:
                sim = s1.wup_similarity(s2) or 0.0
                best = max(best, sim)
        return 0.8 * best
    except Exception:
        return 0.0


def _candidate_score(source_word: str, candidate: str, prev_word: str, next_word: str, target_idx: int, resources) -> float:
    cand_lemma = _lemma(candidate)
    target_alignment = -abs(_word_difficulty(cand_lemma, resources) - target_idx)
    freq_bonus = math.log(resources["word_total"].get(cand_lemma, 1) + 1)
    meaning = _meaning_bonus(source_word, candidate)
    lm = _local_lm_score(prev_word, candidate, next_word, target_idx, resources)
    return 1.8 * target_alignment + 0.5 * freq_bonus + 0.8 * meaning + 0.45 * lm


def _wordnet_candidates(word: str):
    wn = _get_wordnet()
    if wn is None:
        return []
    out = []
    seen = set()
    try:
        for syn in wn.synsets(_lemma(word)):
            for lem in syn.lemmas():
                name = lem.name().replace("_", " ").lower()
                if " " not in name and name.isalpha() and name not in seen and name != _lemma(word):
                    seen.add(name)
                    out.append(name)
    except Exception:
        return []
    return out[:20]


def _generate_candidates(word: str, target_idx: int, resources):
    lemma = _lemma(word)
    cands = []
    if target_idx <= 2:
        cands.extend(MANUAL_SUBS.get(word.lower(), []))
        cands.extend(MANUAL_SUBS.get(lemma, []))
    else:
        cands.extend(MANUAL_COMPLEX.get(word.lower(), []))
        cands.extend(MANUAL_COMPLEX.get(lemma, []))
    cands.extend(_wordnet_candidates(word))

    out = []
    seen = set()
    for cand in cands:
        cand = cand.strip().lower()
        if not cand or not cand.isalpha() or cand in seen:
            continue
        seen.add(cand)
        if resources["word_total"].get(_lemma(cand), 0) < 1 and cand not in MANUAL_SUBS.get(lemma, []) and cand not in MANUAL_COMPLEX.get(lemma, []):
            continue
        out.append(cand)
    return out[:12]


def _rebuild_text(tokens):
    text = ""
    for i, tok in enumerate(tokens):
        if i == 0:
            text += tok
        elif tok in ".,!?;:%)]}":
            text += tok
        elif text and text[-1] in "([{":
            text += tok
        else:
            text += " " + tok
    text = re.sub(r"\ba ([aeiouAEIOU])", r"an \1", text)
    text = re.sub(r"\ban ([^aeiouAEIOU\W])", r"a \1", text)
    return text


def transform_sentence(sentence, source_level, target_level):
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")

    if not isinstance(sentence, str):
        sentence = str(sentence)
    if not sentence.strip() or source_level == target_level:
        return sentence

    resources = _build_resources()
    source_idx = LEVEL_TO_INT[source_level]
    target_idx = LEVEL_TO_INT[target_level]
    simplify = target_idx < source_idx

    tokens = _tokenize(sentence)
    if not tokens:
        return sentence

    candidates = []
    for i, tok in enumerate(tokens):
        if not _is_word(tok):
            continue
        if tok.lower() in STOPWORDS or len(tok) <= 2:
            continue
        diff = _word_difficulty(tok, resources) - target_idx
        strength = diff if simplify else -diff
        if strength < 0.45:
            continue
        confidence = min(math.log(resources["word_total"].get(_lemma(tok), 1) + 1), 2.5)
        candidates.append((strength + 0.15 * confidence, i))

    if not candidates:
        return sentence

    candidates.sort(reverse=True)
    max_changes = 1 if abs(target_idx - source_idx) <= 1 else 2
    changed = 0

    for _, pos in candidates[: max_changes + 2]:
        original = tokens[pos]
        prev_word = tokens[pos - 1] if pos > 0 else "<s>"
        next_word = tokens[pos + 1] if pos + 1 < len(tokens) else "</s>"
        generated = _generate_candidates(original, target_idx, resources)
        if not generated:
            continue

        orig_score = _candidate_score(original, original.lower(), prev_word, next_word, target_idx, resources)
        best_word = None
        best_score = orig_score
        for cand in generated:
            inflected = _inflect_like(cand, original, prev_word)
            score = _candidate_score(original, inflected, prev_word, next_word, target_idx, resources)
            # Small boost to trusted manual replacements.
            if cand in MANUAL_SUBS.get(original.lower(), []) or cand in MANUAL_SUBS.get(_lemma(original), []):
                score += 0.6 if simplify else 0.0
            if cand in MANUAL_COMPLEX.get(original.lower(), []) or cand in MANUAL_COMPLEX.get(_lemma(original), []):
                score += 0.6 if not simplify else 0.0
            if score > best_score:
                best_score = score
                best_word = inflected

        if best_word and _lemma(best_word) != _lemma(original):
            tokens[pos] = best_word
            changed += 1
            if changed >= max_changes:
                break

    return _rebuild_text(tokens)
