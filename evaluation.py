from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


LABELS = ["Easy to Understand", "Difficult to Understand"]


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_train: list[str],
    y_train: list[str],
    x_test: list[str],
    y_test: list[str],
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_metrics = cross_validate(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        },
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    cm = confusion_matrix(y_test, predictions, labels=LABELS)

    return {
        "model": model_name,
        "trained_model": model,
        "predictions": predictions.tolist(),
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average="weighted")),
            "recall": float(recall_score(y_test, predictions, average="weighted")),
            "f1_score": float(f1_score(y_test, predictions, average="weighted")),
        },
        "cross_validation": {
            "folds": cv_folds,
            "accuracy_mean": float(np.mean(cv_metrics["test_accuracy"])),
            "accuracy_std": float(np.std(cv_metrics["test_accuracy"])),
            "precision_mean": float(np.mean(cv_metrics["test_precision"])),
            "recall_mean": float(np.mean(cv_metrics["test_recall"])),
            "f1_mean": float(np.mean(cv_metrics["test_f1"])),
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist(),
        },
    }


def print_model_results(result: dict[str, Any]) -> None:
    print(f"\nModel: {result['model']}")
    print("Test Metrics:")
    for metric, value in result["test_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print("Cross-Validation (5-Fold):")
    cv = result["cross_validation"]
    print(f"  accuracy_mean: {cv['accuracy_mean']:.4f} (+/- {cv['accuracy_std']:.4f})")
    print(f"  precision_mean: {cv['precision_mean']:.4f}")
    print(f"  recall_mean: {cv['recall_mean']:.4f}")
    print(f"  f1_mean: {cv['f1_mean']:.4f}")

    print("Confusion Matrix [Easy, Difficult]:")
    print(np.array(result["confusion_matrix"]["matrix"]))


def save_confusion_matrix_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, result in enumerate(results):
        ax = axes[idx]
        cm = np.array(result["confusion_matrix"]["matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=ax,
        )
        ax.set_title(f"{result['model']} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def print_sample_predictions(
    model: Pipeline,
    texts: list[str],
    true_labels: list[str],
    sample_size: int = 5,
) -> None:
    print("\nSample Predictions:")
    for idx in range(min(sample_size, len(texts))):
        prediction = model.predict([texts[idx]])[0]
        print(f"  Text: {texts[idx][:90]}...")
        print(f"  True: {true_labels[idx]} | Predicted: {prediction}\n")
