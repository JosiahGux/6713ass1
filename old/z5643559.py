# Rename this file to your zID, e.g. z1234567.py

from pathlib import Path
import pandas as pd
import spacy
from nltk.corpus import wordnet
from collections import defaultdict, Counter
import math
import pyinflect

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX = {level: i for i, level in enumerate(CEFR_LEVELS)}

# -------------------------------------------------
# Load Spacy
# -------------------------------------------------

nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------
# Optional: Load training data
# -------------------------------------------------


def load_training_data(path="data.csv"):
    """
    Loads the provided training dataset.

    Primary expected columns:
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

training_df = load_training_data()

# -------------------------------------------------
# Optional: Initialise resources globally
# -------------------------------------------------

# Students may:
# - Load and preprocess data here
# - Train a lightweight model
# - Build lookup tables
# - Create rule-based mappings

# -------------------------------------------------
# Build CEFR vocabulary lookup
# -------------------------------------------------

word_level_dict = {}
level_word_counts = defaultdict(Counter)


def build_vocab():
    """
    Build a mapping from lemma -> most frequent CEFR level in the data,
    and per-level lemma frequencies for later scoring.
    Uses spaCy's lemma_ and nlp.pipe for efficiency.
    """
    level_words = defaultdict(list)

    texts = training_df["text"].tolist()
    levels = training_df["cefr_level"].tolist()

    for doc, level in zip(nlp.pipe(texts, batch_size=256), levels):
        for token in doc:
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower()
            level_words[lemma].append(level)
            level_word_counts[level][lemma] += 1

    for word, lvl_list in level_words.items():
        word_level_dict[word] = Counter(lvl_list).most_common(1)[0][0]


build_vocab()


# -------------------------------------------------
# Train trigram language model
# -------------------------------------------------

trigram_counts = defaultdict(Counter)
bigram_counts = Counter()


def train_language_model():
    texts = training_df["text"].tolist()

    for doc in nlp.pipe(texts, batch_size=256):
        words = ["<s>", "<s>"] + [t.text.lower() for t in doc] + ["</s>"]

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            trigram_counts[(w1, w2)][w3] += 1
            bigram_counts[(w1, w2)] += 1


train_language_model()


def trigram_prob(w1, w2, w3):

    vocab_size = len(word_level_dict)

    count_tri = trigram_counts[(w1, w2)][w3]
    count_bi = bigram_counts[(w1, w2)]

    # Laplace smoothing
    return (count_tri + 1) / (count_bi + vocab_size)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def lookup_level(word):

    return word_level_dict.get(word.lower(), None)


def get_synonyms(word):

    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():

            candidate = lemma.name().replace("_", " ").lower()

            if candidate != word:
                synonyms.add(candidate)

    return list(synonyms)


def semantic_similarity(word1, word2):
    try:
        token1 = nlp(word1)[0]
        token2 = nlp(word2)[0]
        return token1.similarity(token2)
    except Exception:
        return None


def choose_best_candidate(left1, left2, right, candidates, target_level=None):

    best = None
    best_score = -math.inf

    for candidate in candidates:

        prob = trigram_prob(left1, left2, candidate)

        score = math.log(prob)

        if right:
            score += math.log(trigram_prob(left2, candidate, right))

        # Small bonus for words that are more frequent at the target level
        if target_level is not None:
            counts = level_word_counts.get(target_level)
            if counts:
                freq = counts.get(candidate, 0)
                if freq > 0:
                    score += 0.1 * math.log(1 + freq)

        if score > best_score:
            best_score = score
            best = candidate

    return best


def inflect_word(token, candidate):

    try:

        tag = token.tag_
        inflected = nlp(candidate)[0]._.inflect(tag)

        if inflected:
            return inflected

    except:
        pass

    return candidate


def match_case(original, replacement):
    """
    Apply the casing pattern of the original token to the replacement.
    """
    if not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement.capitalize()
    return replacement.lower()

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

    doc = nlp(sentence)

    # Use original token texts as a starting point.
    new_tokens = [t.text for t in doc]

    source_idx = LEVEL_INDEX[source_level]
    target_idx = LEVEL_INDEX[target_level]
    going_easier = target_idx < source_idx

    max_replacements = 2
    replacements = 0

    for i, token in enumerate(doc):
        if not token.is_alpha:
            continue

        # Focus on content words and avoid named entities
        if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV"} or token.ent_type_:
            continue

        lemma = token.lemma_.lower()
        current_level = lookup_level(lemma)

        if current_level is None:
            continue

        cur_idx = LEVEL_INDEX.get(current_level, None)
        if cur_idx is None:
            continue

        # Decide whether this word is a good candidate to change
        if going_easier and cur_idx <= target_idx:
            continue
        if not going_easier and cur_idx >= target_idx:
            continue

        # Generate candidate synonyms biased towards the target level
        synonyms = get_synonyms(lemma)
        filtered = []

        # Strict filter: only words at or beyond the target level in the right direction
        for s in synonyms:
            level = lookup_level(s)
            if level is None:
                continue

            lvl_idx = LEVEL_INDEX.get(level, None)
            if lvl_idx is None:
                continue

            if going_easier and lvl_idx > target_idx:
                continue
            if not going_easier and lvl_idx < target_idx:
                continue

            sim = semantic_similarity(lemma, s)
            if sim is None or sim <= 0.4:
                continue

            # Require some minimum frequency at the target level, if available
            counts = level_word_counts.get(target_level)
            if counts is not None and counts.get(s, 0) < 3:
                continue

            filtered.append(s)

        # Relaxed pass: if nothing found, allow candidates within +/-1 level of target
        if not filtered:
            for s in synonyms:
                level = lookup_level(s)
                if level is None:
                    continue

                lvl_idx = LEVEL_INDEX.get(level, None)
                if lvl_idx is None:
                    continue

                # Still respect overall direction (don't go beyond source in wrong way)
                if going_easier and lvl_idx > source_idx:
                    continue
                if not going_easier and lvl_idx < source_idx:
                    continue

                if abs(lvl_idx - target_idx) > 1:
                    continue

                sim = semantic_similarity(lemma, s)
                if sim is None or sim <= 0.4:
                    continue

                counts = level_word_counts.get(target_level)
                if counts is not None and counts.get(s, 0) < 2:
                    continue

                filtered.append(s)

        if not filtered:
            continue

        # Use trigram language model to pick best in context
        left1 = doc[i - 2].text.lower() if i > 1 else "<s>"
        left2 = doc[i - 1].text.lower() if i > 0 else "<s>"
        right = doc[i + 1].text.lower() if i < len(doc) - 1 else "</s>"

        best = choose_best_candidate(left1, left2, right, filtered, target_level=target_level)

        if best:
            best = inflect_word(token, best)
            best = match_case(token.text, best)
            new_tokens[i] = best
            replacements += 1

            if replacements >= max_replacements:
                break

    # Reconstruct sentence with original whitespace to avoid extra spaces.
    out_pieces = []
    for tok, new_text in zip(doc, new_tokens):
        out_pieces.append(str(new_text) + tok.whitespace_)

    return "".join(out_pieces).strip()

