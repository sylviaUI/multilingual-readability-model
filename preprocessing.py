from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


LABEL_MAP = {
    "easy": "Easy to Understand",
    "simple": "Easy to Understand",
    "0": "Easy to Understand",
    "difficult": "Difficult to Understand",
    "hard": "Difficult to Understand",
    "complex": "Difficult to Understand",
    "1": "Difficult to Understand",
}


def ensure_nltk_resources() -> None:
    # Non-blocking initialization: preprocessing functions gracefully fall back
    # when NLTK resources are missing or corrupted.
    for path in [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
        "corpora/wordnet",
        "corpora/omw-1.4",
    ]:
        try:
            nltk.data.find(path)
        except Exception:
            continue


def normalize_label(value: object) -> str:
    raw = str(value).strip().lower()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    if raw in {"easy to understand", "difficult to understand"}:
        return raw.title() if "easy" in raw else "Difficult to Understand"
    raise ValueError(
        f"Unsupported label '{value}'. Supported labels include easy/simple/0 and difficult/hard/1."
    )


def generate_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    if n_samples < 1000:
        raise ValueError("n_samples must be at least 1000.")

    half = n_samples // 2

    easy_sentences = [
        "Please open the door.",
        "The boy is reading a book.",
        "We will meet after lunch.",
        "This lesson is short and clear.",
        "Drink clean water every day.",
        "The weather is warm today.",
        "She likes to play music.",
        "The bus arrives at eight.",
        "Write your name on the form.",
        "I need help with this task.",
    ]

    difficult_sentences = [
        "The committee recommended a comprehensive restructuring of the institutional governance framework.",
        "Notwithstanding the preliminary findings, longitudinal validation remains methodologically indispensable.",
        "Participants demonstrated heterogeneous responses to multifactorial intervention protocols.",
        "The jurisprudential implications of the ruling extend beyond the immediate contractual dispute.",
        "Operational inefficiencies were exacerbated by asynchronous interdepartmental communication channels.",
        "A nuanced interpretation of the dataset necessitates controlling for confounding demographic variables.",
        "The pharmacokinetic profile suggests diminished bioavailability under concurrent metabolic inhibition.",
        "Macroeconomic volatility precipitated a disproportionate contraction in export-dependent sectors.",
        "The manuscript synthesizes interdisciplinary perspectives to contextualize emergent sociotechnical risks.",
        "Policy implementation encountered resistance due to ambiguously articulated compliance requirements.",
    ]

    easy_df = pd.DataFrame(
        {
            "text": [easy_sentences[i % len(easy_sentences)] for i in range(half)],
            "label": ["Easy to Understand"] * half,
        }
    )
    difficult_df = pd.DataFrame(
        {
            "text": [
                difficult_sentences[i % len(difficult_sentences)]
                for i in range(n_samples - half)
            ],
            "label": ["Difficult to Understand"] * (n_samples - half),
        }
    )

    df = pd.concat([easy_df, difficult_df], ignore_index=True)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    out = df[["text", "label"]].dropna().copy()
    out["label"] = out["label"].map(normalize_label)
    return out


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    lowered = str(text).lower()
    try:
        tokens = word_tokenize(lowered)
    except Exception:
        # Fallback when punkt resources are unavailable.
        tokens = re.findall(r"[a-z]+", lowered)

    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    lemmas = []
    for token in tokens:
        try:
            lemmas.append(lemmatizer.lemmatize(token))
        except Exception:
            # If wordnet is unavailable, keep the token as-is.
            lemmas.append(token)
    return " ".join(lemmas)


def preprocess_texts(texts: List[str]) -> List[str]:
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = set(ENGLISH_STOP_WORDS)

    lemmatizer = WordNetLemmatizer()
    return [preprocess_text(text, stop_words, lemmatizer) for text in texts]


def load_or_generate_dataset(
    dataset_path: str | None,
    n_samples: int,
    random_state: int,
) -> Tuple[pd.DataFrame, str]:
    if dataset_path:
        path = Path(dataset_path)
        if path.exists():
            return load_dataset(path), "loaded"

    return generate_synthetic_dataset(n_samples=n_samples, random_state=random_state), "generated"
