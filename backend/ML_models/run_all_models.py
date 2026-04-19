from __future__ import annotations

import argparse

try:
    from backend.ML_models.train_ml_models import (
        FEATURE_CHOICES,
        TrainingConfig,
        resolve_project_path,
        train_and_evaluate_models,
    )
except ModuleNotFoundError:
    from train_ml_models import (
        FEATURE_CHOICES,
        TrainingConfig,
        resolve_project_path,
        train_and_evaluate_models,
    )


MODEL_RUNS = (
    ("naive_bayes", "Naive Bayes"),
    ("random_forest", "Random Forest"),
    ("svm", "SVM"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Naive Bayes, Random Forest, and SVM one after another."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory containing saved labels and feature matrices.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_CHOICES,
        default=None,
        help="Feature sets to train on. Defaults to every saved feature directory that exists.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=150,
        help="Number of trees to use in the random forest model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting model and metric artifacts for all model runs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    feature_types = tuple(args.features) if args.features else None
    results: list[tuple[str, dict[str, object]]] = []

    for model_type, label in MODEL_RUNS:
        config = TrainingConfig(
            artifacts_dir=resolve_project_path(args.artifacts_dir),
            feature_types=feature_types,
            model_types=(model_type,),
            rf_estimators=args.rf_estimators,
            random_state=args.random_state,
            overwrite=args.overwrite,
            output_suffix=model_type,
        )
        summary = train_and_evaluate_models(config)
        results.append((label, summary))
        print(f"{label} training complete.")
        print(f"Best run: {summary['best_run']}")

    print("All model runs complete.")
    print(f"Artifacts saved to: {args.artifacts_dir}")
    print("Models trained: Naive Bayes, Random Forest, SVM")


if __name__ == "__main__":
    main()
