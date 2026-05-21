import re
from typing import List

from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_common_words(q1: str, q2: str) -> int:
    """Return number of common words between two tokenized strings."""
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1 & w2)


def test_total_words(q1: str, q2: str) -> int:
    """Return total unique words across both questions."""
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1) + len(w2)


def test_fetch_token_features(q1: str, q2: str) -> List[float]:
    SAFE_DIV = 1e-4

    STOP_WORDS = _load_pickle("stopwords.pkl")

    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1: str, q2: str) -> List[float]:
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    substrings = list(distance.lcsubstrings(q1, q2))
    longest = substrings[0] if substrings else ""
    length_features[2] = len(longest) / (min(len(q1), len(q2)) + 1)

    return length_features


def test_fetch_fuzzy_features(q1: str, q2: str) -> List[float]:
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


def preprocess(q: str) -> str:
    """Normalize and clean a question string for feature extraction."""
    q = str(q).lower().strip()

    replacements = {
        "%": " percent",
        "$": " dollar ",
        "₹": " rupee ",
        "€": " euro ",
        "@": " at ",
        "[math]": "",
        ",000,000,000 ": "b ",
        ",000,000 ": "m ",
        ",000 ": "k ",
    }

    for k, v in replacements.items():
        q = q.replace(k, v)

    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)

    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he's": "he is",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "she's": "she is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "we're": "we are",
        "won't": "will not",
        "you're": "you are",
    }

    q_decontracted = []
    for word in q.split():
        q_decontracted.append(contractions.get(word, word))

    q = " ".join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r"\W", " ", q).strip()

    return q


def query_point_creator(q1: str, q2: str) -> np.ndarray:
    """Create feature vector for a pair of questions suitable for model prediction."""
    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    total_words = test_total_words(q1, q2) or 1
    input_query.append(round(test_common_words(q1, q2) / total_words, 2))

    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    cv = _load_pickle("cv.pkl")
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    base_len = len(input_query)
    return np.hstack((np.array(input_query).reshape(1, base_len), q1_bow, q2_bow))