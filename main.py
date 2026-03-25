from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from evaluation import (
    evaluate_model,
    print_model_results,
    print_sample_predictions,
    save_confusion_matrix_plot,
)
from model import get_models
from preprocessing import ensure_nltk_resources, load_or_generate_dataset, preprocess_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised ML model for text complexity classification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Optional path to CSV with 'text' and 'label'. If omitted, a synthetic dataset is generated.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic samples to generate when dataset is not provided (>=1000).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default 0.2 for 80/20 split).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save results JSON and confusion matrix figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_nltk_resources()

    dataset, source = load_or_generate_dataset(
        dataset_path=args.dataset,
        n_samples=args.samples,
        random_state=args.random_state,
    )

    texts = preprocess_texts(dataset["text"].tolist())
    labels = dataset["label"].tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    models = get_models(random_state=args.random_state)

    print("\n=== Text Complexity Classification ===")
    print(f"Dataset source: {source}")
    print(f"Total samples: {len(dataset)}")
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

    results = []
    for model_name, model in models.items():
        result = evaluate_model(
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv_folds=5,
            random_state=args.random_state,
        )
        results.append(result)
        print_model_results(result)

    best_result = max(results, key=lambda item: item["test_metrics"]["f1_score"])
    print(
        f"Best model: {best_result['model']} "
        f"(F1-score: {best_result['test_metrics']['f1_score']:.4f})"
    )

    best_model = best_result["trained_model"]
    print_sample_predictions(best_model, x_test, y_test, sample_size=5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_results = []
    for result in results:
        serializable_result = {k: v for k, v in result.items() if k != "trained_model"}
        serializable_results.append(serializable_result)

    summary = {
        "dataset_source": source,
        "dataset_size": len(dataset),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "best_model": best_result["model"],
        "results": serializable_results,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cm_plot_path = output_dir / "confusion_matrices.png"
    save_confusion_matrix_plot(results, cm_plot_path)

    print(f"Saved metrics to: {results_path}")
    print(f"Saved confusion matrix plot to: {cm_plot_path}")


if __name__ == "__main__":
    main()
