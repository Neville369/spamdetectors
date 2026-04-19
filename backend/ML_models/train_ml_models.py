from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


FEATURE_CHOICES = ("count", "tfidf")
MODEL_CHOICES = ("naive_bayes", "random_forest", "svm")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainingConfig:
    artifacts_dir: Path
    feature_types: tuple[str, ...] | None = None
    model_types: tuple[str, ...] = MODEL_CHOICES
    rf_estimators: int = 150
    random_state: int = 42
    overwrite: bool = False
    output_suffix: str | None = None


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_model(model_type: str, config: TrainingConfig):
    if model_type == "naive_bayes":
        return MultinomialNB()
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.rf_estimators,
            random_state=config.random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    if model_type == "svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=config.random_state,
            max_iter=10000,
            tol=1e-3,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def ensure_training_artifact_dirs(config: TrainingConfig) -> dict[str, Path]:
    directories = {
        "root": config.artifacts_dir,
        "features": config.artifacts_dir / "features",
        "labels": config.artifacts_dir / "labels",
        "models": config.artifacts_dir / "models",
        "metrics": config.artifacts_dir / "metrics",
    }

    required_paths = [
        directories["features"],
        directories["labels"],
        directories["labels"] / "y_train.csv",
        directories["labels"] / "y_test.csv",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing feature artifacts: {missing_text}. Run build_features.py first."
        )

    directories["models"].mkdir(parents=True, exist_ok=True)
    directories["metrics"].mkdir(parents=True, exist_ok=True)
    return directories


def load_label_data(labels_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = pd.read_csv(labels_dir / "y_train.csv")
    y_test = pd.read_csv(labels_dir / "y_test.csv")

    for name, frame in (("y_train.csv", y_train), ("y_test.csv", y_test)):
        required_columns = {"label", "label_name"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")

    y_train["label"] = y_train["label"].astype(int)
    y_test["label"] = y_test["label"].astype(int)
    return y_train, y_test


def discover_feature_types(features_dir: Path) -> tuple[str, ...]:
    available = [
        feature_type
        for feature_type in FEATURE_CHOICES
        if (features_dir / feature_type / "X_train.npz").exists()
        and (features_dir / feature_type / "X_test.npz").exists()
    ]
    return tuple(available)


def resolve_feature_types(
    requested_feature_types: tuple[str, ...] | None,
    features_dir: Path,
) -> tuple[str, ...]:
    available_feature_types = discover_feature_types(features_dir)

    if requested_feature_types is None:
        if not available_feature_types:
            raise FileNotFoundError(
                f"No saved feature matrices were found in {features_dir}. Run build_features.py first."
            )
        return available_feature_types

    missing_feature_types = [
        feature_type
        for feature_type in requested_feature_types
        if feature_type not in available_feature_types
    ]
    if missing_feature_types:
        raise FileNotFoundError(
            "Missing saved feature matrices for: "
            f"{missing_feature_types}. Re-run build_features.py with those feature types."
        )

    return requested_feature_types


def load_feature_matrices(feature_dir: Path):
    x_train_path = feature_dir / "X_train.npz"
    x_test_path = feature_dir / "X_test.npz"

    missing_paths = [path for path in (x_train_path, x_test_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing saved matrices: {missing_text}")

    return load_npz(x_train_path), load_npz(x_test_path)


def ensure_can_write(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Re-run with --overwrite to replace it."
        )


def build_output_path(metrics_dir: Path, filename: str, output_suffix: str | None) -> Path:
    if not output_suffix:
        return metrics_dir / filename

    stem, extension = filename.rsplit(".", maxsplit=1)
    return metrics_dir / f"{stem}_{output_suffix}.{extension}"


def train_and_evaluate_models(config: TrainingConfig) -> dict[str, object]:
    config.artifacts_dir = resolve_project_path(config.artifacts_dir)
    directories = ensure_training_artifact_dirs(config)
    feature_types = resolve_feature_types(config.feature_types, directories["features"])
    y_train, y_test = load_label_data(directories["labels"])

    label_lookup = (
        pd.concat(
            [
                y_train[["label", "label_name"]],
                y_test[["label", "label_name"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["label_name"]
        .to_dict()
    )
    label_order = sorted(label_lookup)
    if len(label_order) < 2:
        raise ValueError("At least two label classes are required for training.")

    target_names = [label_lookup[label] for label in label_order]

    results: list[dict[str, object]] = []
    detailed_reports: dict[str, dict[str, object]] = {}

    for feature_type in feature_types:
        feature_dir = directories["features"] / feature_type
        x_train, x_test = load_feature_matrices(feature_dir)

        if x_train.shape[0] != len(y_train) or x_test.shape[0] != len(y_test):
            raise ValueError(
                f"Row mismatch for feature type '{feature_type}': "
                f"X_train has {x_train.shape[0]} rows, y_train has {len(y_train)} rows; "
                f"X_test has {x_test.shape[0]} rows, y_test has {len(y_test)} rows."
            )

        model_dir = directories["models"] / feature_type
        model_dir.mkdir(parents=True, exist_ok=True)

        for model_type in config.model_types:
            model_path = model_dir / f"{model_type}.joblib"
            ensure_can_write(model_path, config.overwrite)

            model = build_model(model_type, config)
            started_at = time.perf_counter()
            model.fit(x_train, y_train["label"])
            y_pred = model.predict(x_test)
            fit_seconds = round(time.perf_counter() - started_at, 3)

            report = classification_report(
                y_test["label"],
                y_pred,
                labels=label_order,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(y_test["label"], y_pred, labels=label_order).tolist()
            accuracy = accuracy_score(y_test["label"], y_pred)

            joblib.dump(model, model_path)

            result_row = {
                "feature_type": feature_type,
                "model_type": model_type,
                "accuracy": round(float(accuracy), 4),
                "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
                "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
                "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
                "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
                "fit_seconds": fit_seconds,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "feature_count": int(x_train.shape[1]),
            }
            results.append(result_row)
            detailed_reports[f"{feature_type}__{model_type}"] = {
                "metrics": result_row,
                "classification_report": report,
                "confusion_matrix": matrix,
            }

    results_path = build_output_path(
        directories["metrics"],
        "model_results.csv",
        config.output_suffix,
    )
    detailed_reports_path = build_output_path(
        directories["metrics"],
        "detailed_reports.json",
        config.output_suffix,
    )
    run_summary_path = build_output_path(
        directories["metrics"],
        "run_summary.json",
        config.output_suffix,
    )

    for output_path in (results_path, detailed_reports_path, run_summary_path):
        ensure_can_write(output_path, config.overwrite)

    results_df = pd.DataFrame(results).sort_values(
        by=["weighted_f1", "accuracy"],
        ascending=False,
    )
    results_df.to_csv(results_path, index=False)

    with detailed_reports_path.open("w", encoding="utf-8") as handle:
        json.dump(detailed_reports, handle, indent=2)

    combined_labels = pd.concat(
        [y_train["label_name"], y_test["label_name"]],
        ignore_index=True,
    )
    summary = {
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "label_counts": combined_labels.value_counts().sort_index().to_dict(),
        "artifacts_dir": str(config.artifacts_dir),
        "features_used": list(feature_types),
        "models_trained": list(config.model_types),
        "best_run": results_df.iloc[0].to_dict() if not results_df.empty else None,
    }

    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ML models from saved feature artifacts. Run build_features.py first."
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
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
        help="ML models to train.",
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
        help="Random seed used for model reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting model and metric artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = TrainingConfig(
        artifacts_dir=Path(args.artifacts_dir),
        feature_types=tuple(args.features) if args.features else None,
        model_types=tuple(args.models),
        rf_estimators=args.rf_estimators,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )

    summary = train_and_evaluate_models(config)
    print("Model training complete.")
    print(f"Train/Test split: {summary['train_rows']} / {summary['test_rows']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print(f"Feature sets used: {summary['features_used']}")
    print(f"Models trained: {summary['models_trained']}")
    print(f"Best run: {summary['best_run']}")


if __name__ == "__main__":
    main()
