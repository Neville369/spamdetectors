from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - import failure should be user-visible.
    raise SystemExit(
        "PyTorch is required for train_dl_models.py. Install it with `python -m pip install torch`."
    ) from exc

try:
    from backend.preprocess import DEFAULT_MULTICLASS_LABEL_NAMES, PorterStemmer, preprocess_text
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from preprocess import DEFAULT_MULTICLASS_LABEL_NAMES, PorterStemmer, preprocess_text


MODEL_CHOICES = ("cnn", "lstm")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISALLOWED_PREPROCESSED_INPUTS = {
    "processed_df.csv",
    "processed_dataset.csv",
    "train_processed.csv",
    "test_processed.csv",
}
DEFAULT_DESCRIPTION = (
    "Train CNN and LSTM text classifiers with trainable embeddings for spam-mail detection."
)


@dataclass
class DeepLearningConfig:
    input_path: Path
    artifacts_dir: Path
    text_column: str = "text"
    label_column: str = "label"
    label_mode: str = "multiclass"
    test_size: float = 0.10
    random_state: int = 42
    chunksize: int = 50000
    sample_size: int | None = None
    max_vocab_size: int = 30000
    min_frequency: int = 2
    max_sequence_length: int = 200
    embedding_dim: int = 128
    hidden_dim: int = 128
    cnn_filters: int = 128
    dropout: float = 0.30
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    model_types: tuple[str, ...] = MODEL_CHOICES
    overwrite: bool = False
    output_suffix: str | None = None


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_raw_input_dataset(input_path: Path) -> None:
    if input_path.name.lower() not in DISALLOWED_PREPROCESSED_INPUTS:
        return

    raise ValueError(
        "Deep-learning training expects the raw dataset (for example `df.csv`), "
        f"not the preprocessed file `{input_path.name}`. "
        "The DL pipeline cleans the text internally, so passing a processed CSV "
        "would preprocess the dataset twice."
    )


def build_output_path(metrics_dir: Path, filename: str, output_suffix: str | None) -> Path:
    if not output_suffix:
        return metrics_dir / filename

    stem, extension = filename.rsplit(".", maxsplit=1)
    return metrics_dir / f"{stem}_{output_suffix}.{extension}"


class TextSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int,
        dropout: float,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=size)
            for size in filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs).transpose(1, 2)
        pooled = [
            torch.max(F.relu(conv(embedded)), dim=2).values
            for conv in self.convs
        ]
        features = torch.cat(pooled, dim=1)
        return self.output(self.dropout(features))


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        _, (hidden_state, _) = self.lstm(embedded)
        features = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        return self.output(self.dropout(features))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_dataset(config: DeepLearningConfig) -> pd.DataFrame:
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


def preprocess_dataframe(df: pd.DataFrame, config: DeepLearningConfig) -> pd.DataFrame:
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

    return processed


def build_vocab(texts: pd.Series, max_vocab_size: int, min_frequency: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(str(text).split())

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, frequency in counter.most_common():
        if frequency < min_frequency:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in str(text).split()]
    token_ids = token_ids[:max_length]

    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))

    return token_ids


def vectorize_texts(texts: pd.Series, vocab: dict[str, int], max_length: int) -> np.ndarray:
    return np.asarray(
        [encode_text(text, vocab=vocab, max_length=max_length) for text in texts],
        dtype=np.int64,
    )


def create_data_loader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TextSequenceDataset(sequences, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_model(model_type: str, config: DeepLearningConfig, vocab_size: int, num_classes: int) -> nn.Module:
    if model_type == "cnn":
        return TextCNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_classes=num_classes,
            num_filters=config.cnn_filters,
            dropout=config.dropout,
        )

    if model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_classes=num_classes,
            dropout=config.dropout,
        )

    raise ValueError(f"Unsupported DL model: {model_type}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(data_loader.dataset)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_order: list[int],
    target_names: list[str],
) -> tuple[dict[str, object], list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    references: list[int] = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predicted = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(predicted)
            references.extend(labels.tolist())

    report = classification_report(
        references,
        predictions,
        labels=label_order,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report, predictions, references


def ensure_artifact_dirs(config: DeepLearningConfig) -> dict[str, Path]:
    if config.artifacts_dir.exists() and not config.overwrite:
        raise FileExistsError(
            f"{config.artifacts_dir} already exists. Re-run with --overwrite to replace it."
        )

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    directories = {
        "root": config.artifacts_dir,
        "processed": config.artifacts_dir / "processed",
        "splits": config.artifacts_dir / "splits",
        "models": config.artifacts_dir / "models",
        "metrics": config.artifacts_dir / "metrics",
        "vocab": config.artifacts_dir / "vocab",
    }

    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)

    return directories


def save_json(data: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def train_and_evaluate_models(config: DeepLearningConfig) -> dict[str, object]:
    config.input_path = resolve_project_path(config.input_path)
    config.artifacts_dir = resolve_project_path(config.artifacts_dir)
    ensure_raw_input_dataset(config.input_path)
    set_random_seed(config.random_state)
    directories = ensure_artifact_dirs(config)
    raw_df = load_dataset(config)
    processed_df = preprocess_dataframe(raw_df, config)

    if processed_df["label"].nunique() < 2:
        raise ValueError(
            "At least two label classes are required for training. Increase --sample-size or use the full dataset."
        )

    processed_df.to_csv(directories["processed"] / "processed_dataset.csv", index=False)

    label_order = sorted(processed_df["label"].unique().tolist())
    label_lookup = (
        processed_df[["label", "label_name"]]
        .drop_duplicates()
        .sort_values("label")
        .set_index("label")["label_name"]
        .to_dict()
    )
    target_names = [label_lookup[label] for label in label_order]
    label_to_index = {label: index for index, label in enumerate(label_order)}
    index_to_label_name = {index: label_lookup[label] for index, label in enumerate(label_order)}

    train_df, test_df = train_test_split(
        processed_df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=processed_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(directories["splits"] / "train_processed.csv", index=False)
    test_df.to_csv(directories["splits"] / "test_processed.csv", index=False)

    vocab = build_vocab(
        train_df["processed_text"],
        max_vocab_size=config.max_vocab_size,
        min_frequency=config.min_frequency,
    )
    save_json(vocab, directories["vocab"] / "token_to_id.json")
    save_json(index_to_label_name, directories["vocab"] / "index_to_label_name.json")

    x_train = vectorize_texts(train_df["processed_text"], vocab=vocab, max_length=config.max_sequence_length)
    x_test = vectorize_texts(test_df["processed_text"], vocab=vocab, max_length=config.max_sequence_length)
    y_train = train_df["label"].map(label_to_index).to_numpy(dtype=np.int64)
    y_test = test_df["label"].map(label_to_index).to_numpy(dtype=np.int64)

    np.save(directories["vocab"] / "X_train.npy", x_train)
    np.save(directories["vocab"] / "X_test.npy", x_test)
    np.save(directories["vocab"] / "y_train.npy", y_train)
    np.save(directories["vocab"] / "y_test.npy", y_test)

    train_loader = create_data_loader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    test_loader = create_data_loader(x_test, y_test, batch_size=config.batch_size, shuffle=False)

    class_counts = np.bincount(y_train, minlength=len(label_order))
    class_weights = len(y_train) / (len(label_order) * np.maximum(class_counts, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[dict[str, object]] = []
    histories: dict[str, dict[str, object]] = {}

    for model_type in config.model_types:
        model = build_model(
            model_type=model_type,
            config=config,
            vocab_size=len(vocab),
            num_classes=len(label_order),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
        )

        epoch_losses: list[float] = []
        started_at = time.perf_counter()

        for epoch in range(1, config.epochs + 1):
            epoch_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            epoch_losses.append(round(epoch_loss, 4))

        training_seconds = round(time.perf_counter() - started_at, 3)
        report, predictions, references = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            label_order=list(range(len(label_order))),
            target_names=target_names,
        )
        accuracy = accuracy_score(references, predictions)
        matrix = confusion_matrix(
            references,
            predictions,
            labels=list(range(len(label_order))),
        ).tolist()

        model_dir = directories["models"] / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_type": model_type,
                    "vocab_size": len(vocab),
                    "embedding_dim": config.embedding_dim,
                    "hidden_dim": config.hidden_dim,
                    "cnn_filters": config.cnn_filters,
                    "num_classes": len(label_order),
                    "max_sequence_length": config.max_sequence_length,
                },
            },
            model_dir / f"{model_type}.pt",
        )

        result_row = {
            "model_type": model_type,
            "accuracy": round(float(accuracy), 4),
            "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
            "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
            "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
            "training_seconds": training_seconds,
            "epochs": config.epochs,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "vocab_size": int(len(vocab)),
            "max_sequence_length": int(config.max_sequence_length),
        }
        results.append(result_row)
        histories[model_type] = {
            "epoch_losses": epoch_losses,
            "classification_report": report,
            "confusion_matrix": matrix,
        }

    results_path = build_output_path(
        directories["metrics"],
        "model_results.csv",
        config.output_suffix,
    )
    histories_path = build_output_path(
        directories["metrics"],
        "training_histories.json",
        config.output_suffix,
    )
    summary_path = build_output_path(
        directories["metrics"],
        "run_summary.json",
        config.output_suffix,
    )

    results_df = pd.DataFrame(results).sort_values(
        by=["weighted_f1", "accuracy"],
        ascending=False,
    )
    results_df.to_csv(results_path, index=False)
    save_json(histories, histories_path)

    summary = {
        "rows_loaded": int(len(raw_df)),
        "rows_processed": int(len(processed_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_counts": processed_df["label_name"].value_counts().sort_index().to_dict(),
        "device": str(device),
        "artifacts_dir": str(config.artifacts_dir),
        "models_trained": list(config.model_types),
        "best_run": results_df.iloc[0].to_dict() if not results_df.empty else None,
    }
    save_json(summary, summary_path)
    return summary


def add_common_arguments(
    parser: argparse.ArgumentParser,
    include_models: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input",
        default="df.csv",
        help="Path to the raw source CSV dataset. Use df.csv, not processed_df.csv.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="dl_artifacts",
        help="Directory where processed data, vocabularies, models, and metrics will be saved.",
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
        help="Optional number of randomly sampled rows to train on for faster experiments.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="Rows per chunk while sampling the CSV.",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=30000,
        help="Maximum vocabulary size for the embedding layer.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Ignore tokens seen fewer than this many times in the training data.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=200,
        help="Maximum number of tokens per email after padding and truncation.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension for trainable token vectors.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the LSTM classifier.",
    )
    parser.add_argument(
        "--cnn-filters",
        type=int,
        default=128,
        help="Number of convolution filters per kernel size for the CNN classifier.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
        help="Dropout rate applied before the output layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size used for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimization.",
    )
    if include_models:
        parser.add_argument(
            "--models",
            nargs="+",
            choices=MODEL_CHOICES,
            default=list(MODEL_CHOICES),
            help="Deep-learning models to train.",
        )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling, splitting, and initialization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing artifacts directory.",
    )
    return parser


def build_parser(
    description: str = DEFAULT_DESCRIPTION,
    include_models: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    return add_common_arguments(parser, include_models=include_models)


def config_from_args(
    args: argparse.Namespace,
    model_types: tuple[str, ...] | None = None,
    output_suffix: str | None = None,
) -> DeepLearningConfig:
    resolved_model_types = model_types or tuple(args.models)
    return DeepLearningConfig(
        input_path=resolve_project_path(args.input),
        artifacts_dir=resolve_project_path(args.artifacts_dir),
        text_column=args.text_column,
        label_column=args.label_column,
        label_mode=args.label_mode,
        test_size=args.test_size,
        random_state=args.random_state,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        max_vocab_size=args.max_vocab_size,
        min_frequency=args.min_frequency,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        cnn_filters=args.cnn_filters,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_types=resolved_model_types,
        overwrite=args.overwrite,
        output_suffix=output_suffix,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = config_from_args(args)
    summary = train_and_evaluate_models(config)
    print("Deep-learning pipeline complete.")
    print(f"Rows processed: {summary['rows_processed']}")
    print(f"Train/Test split: {summary['train_rows']} / {summary['test_rows']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Device: {summary['device']}")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print(f"Models trained: {summary['models_trained']}")
    print(f"Best run: {summary['best_run']}")


if __name__ == "__main__":
    main()
