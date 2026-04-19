from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split

try:
    from backend.preprocess import (
        DEFAULT_MULTICLASS_LABEL_NAMES,
        PorterStemmer,
        preprocess_text,
    )
except ModuleNotFoundError:
    from preprocess import (
        DEFAULT_MULTICLASS_LABEL_NAMES,
        PorterStemmer,
        preprocess_text,
    )


FEATURE_CHOICES = ("count", "tfidf")
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class FeatureBuildConfig:
    input_path: Path
    artifacts_dir: Path
    text_column: str = "text"
    label_column: str = "label"
    label_mode: str = "multiclass"
    test_size: float = 0.10
    random_state: int = 42
    chunksize: int = 50000
    sample_size: int | None = None
    max_features: int = 20000
    min_df: int = 2
    feature_types: tuple[str, ...] = FEATURE_CHOICES
    overwrite: bool = False


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sample_dataset(
    input_path: Path,
    usecols: list[str],
    sample_size: int,
    chunksize: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sampled_chunk: pd.DataFrame | None = None

    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=chunksize):
        working = chunk.copy()
        working["_sample_key"] = rng.random(len(working))

        if sampled_chunk is None:
            combined = working
        else:
            combined = pd.concat([sampled_chunk, working], ignore_index=True)

        sampled_chunk = combined.nsmallest(sample_size, "_sample_key").reset_index(drop=True)

    if sampled_chunk is None:
        raise ValueError(f"No rows found in {input_path}.")

    return sampled_chunk.drop(columns="_sample_key")


def load_dataset(config: FeatureBuildConfig) -> pd.DataFrame:
    usecols = [config.label_column, config.text_column]

    if config.sample_size is not None:
        return sample_dataset(
            input_path=config.input_path,
            usecols=usecols,
            sample_size=config.sample_size,
            chunksize=config.chunksize,
            random_state=config.random_state,
        )

    return pd.read_csv(config.input_path, usecols=usecols)


def resolve_labels(raw_labels: pd.Series, label_mode: str) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    raw_labels = raw_labels.astype(int)

    if label_mode == "multiclass":
        label_names = raw_labels.map(DEFAULT_MULTICLASS_LABEL_NAMES)
        if label_names.isna().any():
            missing = sorted(raw_labels[label_names.isna()].unique().tolist())
            raise ValueError(f"Unknown multiclass labels found: {missing}")
        return raw_labels, label_names, None

    if label_mode == "binary":
        binary_labels = raw_labels.map({0: 0, 1: 1, 2: 1})
        label_names = binary_labels.map({0: "ham", 1: "spam_or_phish"})
        if binary_labels.isna().any():
            missing = sorted(raw_labels[binary_labels.isna()].unique().tolist())
            raise ValueError(f"Unknown binary labels found: {missing}")
        return binary_labels.astype(int), label_names, raw_labels

    label_names = raw_labels.astype(str)
    return raw_labels, label_names, None


def preprocess_dataframe(df: pd.DataFrame, config: FeatureBuildConfig) -> pd.DataFrame:
    stemmer = PorterStemmer()
    stop_words = set(ENGLISH_STOP_WORDS)

    labels, label_names, raw_labels = resolve_labels(df[config.label_column], config.label_mode)
    processed_text = df[config.text_column].fillna("").map(
        lambda value: preprocess_text(value, stemmer=stemmer, stop_words=stop_words)
    )

    processed = pd.DataFrame(
        {
            "label": labels,
            "label_name": label_names,
            "processed_text": processed_text,
        }
    )

    if raw_labels is not None:
        processed.insert(0, "raw_label", raw_labels)

    processed["text"] = df[config.text_column].fillna("")
    return processed


def build_vectorizer(feature_type: str, max_features: int, min_df: int):
    if feature_type == "count":
        return CountVectorizer(max_features=max_features, min_df=min_df)
    if feature_type == "tfidf":
        return TfidfVectorizer(max_features=max_features, min_df=min_df, sublinear_tf=True)
    raise ValueError(f"Unsupported feature type: {feature_type}")


def ensure_feature_artifact_dirs(config: FeatureBuildConfig) -> dict[str, Path]:
    if config.artifacts_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"{config.artifacts_dir} already exists. Re-run with --overwrite to replace it."
        )

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    directories = {
        "root": config.artifacts_dir,
        "processed": config.artifacts_dir / "processed",
        "splits": config.artifacts_dir / "splits",
        "labels": config.artifacts_dir / "labels",
        "features": config.artifacts_dir / "features",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_labels(y_values: pd.Series, label_names: pd.Series, output_path: Path) -> None:
    label_df = pd.DataFrame({"label": y_values, "label_name": label_names})
    label_df.to_csv(output_path, index=False)


def build_features(config: FeatureBuildConfig) -> dict[str, object]:
    config.input_path = resolve_project_path(config.input_path)
    config.artifacts_dir = resolve_project_path(config.artifacts_dir)
    directories = ensure_feature_artifact_dirs(config)
    raw_df = load_dataset(config)
    processed_df = preprocess_dataframe(raw_df, config)

    if processed_df["label"].nunique() < 2:
        raise ValueError(
            "At least two label classes are required for training. Increase --sample-size or use the full dataset."
        )

    processed_df.to_csv(directories["processed"] / "processed_dataset.csv", index=False)

    stratify_labels = processed_df["label"] if processed_df["label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        processed_df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(directories["splits"] / "train_processed.csv", index=False)
    test_df.to_csv(directories["splits"] / "test_processed.csv", index=False)
    save_labels(train_df["label"], train_df["label_name"], directories["labels"] / "y_train.csv")
    save_labels(test_df["label"], test_df["label_name"], directories["labels"] / "y_test.csv")

    feature_summaries: list[dict[str, object]] = []

    for feature_type in config.feature_types:
        feature_dir = directories["features"] / feature_type
        feature_dir.mkdir(parents=True, exist_ok=True)

        vectorizer = build_vectorizer(
            feature_type=feature_type,
            max_features=config.max_features,
            min_df=config.min_df,
        )
        x_train = vectorizer.fit_transform(train_df["processed_text"])
        x_test = vectorizer.transform(test_df["processed_text"])

        save_npz(feature_dir / "X_train.npz", x_train)
        save_npz(feature_dir / "X_test.npz", x_test)
        joblib.dump(vectorizer, feature_dir / "vectorizer.joblib")

        pd.DataFrame(
            {"feature_name": vectorizer.get_feature_names_out()}
        ).to_csv(feature_dir / "feature_names.csv", index=False)

        feature_summaries.append(
            {
                "feature_type": feature_type,
                "feature_count": int(x_train.shape[1]),
                "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
                "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
            }
        )

    summary = {
        "rows_loaded": int(len(raw_df)),
        "rows_processed": int(len(processed_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_counts": processed_df["label_name"].value_counts().sort_index().to_dict(),
        "artifacts_dir": str(config.artifacts_dir),
        "features_built": feature_summaries,
    }

    with (directories["features"] / "build_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess spam-mail data, split the dataset, and build text features."
    )
    parser.add_argument(
        "--input",
        default="processed_df.csv",
        help="Path to the source CSV dataset.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory where processed data, splits, labels, and features will be saved.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing the email text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column containing the numeric class label.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("multiclass", "binary", "raw"),
        default="multiclass",
        help="Use multiclass for ham/phish/spam, binary for ham vs phish-or-spam, or raw for untouched numeric labels.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Fraction of rows reserved for testing. Default is 0.10 for a 90/10 split.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of randomly sampled rows to build features from for faster experiments.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Rows per chunk while sampling the CSV.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Maximum vocabulary size for each vectorizer.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Ignore terms appearing in fewer than this many documents.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_CHOICES,
        default=list(FEATURE_CHOICES),
        help="Text feature extractors to build.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling, splitting, and reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing artifacts directory.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = FeatureBuildConfig(
        input_path=resolve_project_path(args.input),
        artifacts_dir=resolve_project_path(args.artifacts_dir),
        text_column=args.text_column,
        label_column=args.label_column,
        label_mode=args.label_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        max_features=args.max_features,
        min_df=args.min_df,
        feature_types=tuple(args.features),
        overwrite=args.overwrite,
    )

    summary = build_features(config)
    print("Feature pipeline complete.")
    print(f"Rows processed: {summary['rows_processed']}")
    print(f"Train/Test split: {summary['train_rows']} / {summary['test_rows']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print(f"Features built: {summary['features_built']}")


if __name__ == "__main__":
    main()
