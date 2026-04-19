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


MODEL_TYPE = "naive_bayes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Naive Bayes model from saved feature artifacts."
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
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the Naive Bayes model and its metric files.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = TrainingConfig(
        artifacts_dir=resolve_project_path(args.artifacts_dir),
        feature_types=tuple(args.features) if args.features else None,
        model_types=(MODEL_TYPE,),
        random_state=args.random_state,
        overwrite=args.overwrite,
        output_suffix=MODEL_TYPE,
    )

    summary = train_and_evaluate_models(config)
    print("Naive Bayes training complete.")
    print(f"Train/Test split: {summary['train_rows']} / {summary['test_rows']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print(f"Feature sets used: {summary['features_used']}")
    print(f"Best run: {summary['best_run']}")


if __name__ == "__main__":
    main()
