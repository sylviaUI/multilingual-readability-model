from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def get_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        ),
        "Support Vector Machine (SVM)": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
                ("classifier", LinearSVC(random_state=random_state)),
            ]
        ),
    }
