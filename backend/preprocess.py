from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DEFAULT_MULTICLASS_LABEL_NAMES = {
    0: "ham",
    1: "phish",
    2: "spam",
}

DEFAULT_BINARY_LABEL_MAP = {
    0: 0,
    1: 1,
    2: 1,
}

BINARY_LABEL_NAMES = {
    0: "ham",
    1: "spam",
}

TOKEN_PATTERN = re.compile(r"[a-z]+")
PROJECT_ROOT = Path(__file__).resolve().parent


class PorterStemmer:
    """A lightweight Porter stemmer implementation to keep the project self-contained."""

    _vowels = set("aeiou")

    @lru_cache(maxsize=50000)
    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word

        word = word.lower()
        word = self._step_1a(word)
        word = self._step_1b(word)
        word = self._step_1c(word)
        word = self._step_2(word)
        word = self._step_3(word)
        word = self._step_4(word)
        word = self._step_5a(word)
        word = self._step_5b(word)
        return word

    def _is_consonant(self, word: str, index: int) -> bool:
        if index < 0:
            index += len(word)

        char = word[index]
        if char in self._vowels:
            return False
        if char == "y":
            if index == 0:
                return True
            return not self._is_consonant(word, index - 1)
        return True

    def _measure(self, stem: str) -> int:
        if not stem:
            return 0

        groups: list[str] = []
        for index in range(len(stem)):
            marker = "c" if self._is_consonant(stem, index) else "v"
            if not groups or marker != groups[-1]:
                groups.append(marker)

        return sum(
            1
            for index in range(len(groups) - 1)
            if groups[index] == "v" and groups[index + 1] == "c"
        )

    def _contains_vowel(self, stem: str) -> bool:
        return any(not self._is_consonant(stem, index) for index in range(len(stem)))

    def _ends_with_double_consonant(self, word: str) -> bool:
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and self._is_consonant(word, len(word) - 1)
        )

    def _ends_cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False

        if (
            not self._is_consonant(word, len(word) - 1)
            or self._is_consonant(word, len(word) - 2)
            or not self._is_consonant(word, len(word) - 3)
        ):
            return False

        return word[-1] not in {"w", "x", "y"}

    def _step_1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step_1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return stem + "ee"
            return word

        stripped = word
        changed = False

        if word.endswith("ed"):
            stem = word[:-2]
            if self._contains_vowel(stem):
                stripped = stem
                changed = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._contains_vowel(stem):
                stripped = stem
                changed = True

        if not changed:
            return word

        if stripped.endswith(("at", "bl", "iz")):
            return stripped + "e"
        if self._ends_with_double_consonant(stripped) and stripped[-1] not in {"l", "s", "z"}:
            return stripped[:-1]
        if self._measure(stripped) == 1 and self._ends_cvc(stripped):
            return stripped + "e"
        return stripped

    def _step_1c(self, word: str) -> str:
        if word.endswith("y"):
            stem = word[:-1]
            if self._contains_vowel(stem):
                return stem + "i"
        return word

    def _step_2(self, word: str) -> str:
        suffixes = [
            ("ization", "ize"),
            ("ational", "ate"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("iveness", "ive"),
            ("tional", "tion"),
            ("biliti", "ble"),
            ("lessli", "less"),
            ("entli", "ent"),
            ("ation", "ate"),
            ("alism", "al"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("ousli", "ous"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("abli", "able"),
            ("izer", "ize"),
            ("alli", "al"),
            ("ator", "ate"),
            ("eli", "e"),
            ("logi", "log"),
        ]

        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step_3(self, word: str) -> str:
        suffixes = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]

        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step_4(self, word: str) -> str:
        suffixes = [
            "ement",
            "ance",
            "ence",
            "able",
            "ible",
            "ment",
            "ant",
            "ent",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
            "al",
            "er",
            "ic",
            "ou",
        ]

        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 1:
                    return stem
                return word

        if word.endswith("ion"):
            stem = word[:-3]
            if self._measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem

        return word

    def _step_5a(self, word: str) -> str:
        if not word.endswith("e"):
            return word

        stem = word[:-1]
        measure = self._measure(stem)
        if measure > 1:
            return stem
        if measure == 1 and not self._ends_cvc(stem):
            return stem
        return word

    def _step_5b(self, word: str) -> str:
        if self._measure(word) > 1 and self._ends_with_double_consonant(word) and word.endswith("l"):
            return word[:-1]
        return word


@dataclass
class ProcessingConfig:
    input_path: Path
    output_path: Path
    text_column: str = "text"
    label_column: str = "label"
    label_mode: str = "multiclass"
    chunksize: int = 50000
    limit: int | None = None
    keep_original_text: bool = False
    overwrite: bool = False


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def preprocess_text(text: str, stemmer: PorterStemmer, stop_words: set[str]) -> str:
    tokens = TOKEN_PATTERN.findall(str(text).lower())
    cleaned_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(cleaned_tokens)


def map_labels(raw_labels: pd.Series, mode: str) -> tuple[pd.Series, pd.Series | None, pd.Series]:
    raw_labels = raw_labels.astype(int)

    if mode == "raw":
        return raw_labels, None, raw_labels.astype(str)

    if mode == "multiclass":
        label_names = raw_labels.map(DEFAULT_MULTICLASS_LABEL_NAMES)
        if label_names.isna().any():
            missing = sorted(raw_labels[label_names.isna()].dropna().unique().tolist())
            raise ValueError(f"Found unknown class labels: {missing}")
        return raw_labels, None, label_names

    mapped = raw_labels.map(DEFAULT_BINARY_LABEL_MAP)
    if mapped.isna().any():
        missing = sorted(raw_labels[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"Found unmapped raw labels for binary conversion: {missing}")

    label_names = mapped.map(BINARY_LABEL_NAMES)
    return mapped.astype(int), raw_labels, label_names


def preprocess_chunk(
    chunk: pd.DataFrame,
    stemmer: PorterStemmer,
    stop_words: set[str],
    config: ProcessingConfig,
) -> pd.DataFrame:
    processed_text = chunk[config.text_column].fillna("").map(
        lambda value: preprocess_text(value, stemmer=stemmer, stop_words=stop_words)
    )
    label_series, raw_label_series, label_name_series = map_labels(
        chunk[config.label_column],
        mode=config.label_mode,
    )

    processed = pd.DataFrame(
        {
            "label": label_series,
            "label_name": label_name_series,
            "text": processed_text,
        }
    )

    if raw_label_series is not None:
        processed.insert(0, "raw_label", raw_label_series)

    if config.keep_original_text:
        processed["text"] = chunk[config.text_column].fillna("")

    return processed


def write_chunk(chunk: pd.DataFrame, output_path: Path, first_chunk: bool) -> None:
    chunk.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)


def process_dataset(config: ProcessingConfig) -> dict[str, object]:
    config.input_path = resolve_project_path(config.input_path)
    config.output_path = resolve_project_path(config.output_path)
    if config.input_path.resolve() == config.output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    if config.output_path.exists() and not config.overwrite:
        raise FileExistsError(
            f"{config.output_path} already exists. Re-run with --overwrite to replace it."
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    stemmer = PorterStemmer()
    stop_words = set(ENGLISH_STOP_WORDS)

    raw_label_counts: Counter[int] = Counter()
    label_name_counts: Counter[str] = Counter()
    processed_rows = 0
    first_chunk = True
    preview_rows: list[dict[str, object]] = []

    reader = pd.read_csv(
        config.input_path,
        usecols=[config.label_column, config.text_column],
        chunksize=config.chunksize,
    )

    for chunk in reader:
        if config.limit is not None:
            remaining = config.limit - processed_rows
            if remaining <= 0:
                break
            if remaining < len(chunk):
                chunk = chunk.iloc[:remaining].copy()

        raw_label_counts.update(chunk[config.label_column].astype(int).tolist())

        processed_chunk = preprocess_chunk(
            chunk=chunk,
            stemmer=stemmer,
            stop_words=stop_words,
            config=config,
        )

        label_name_counts.update(processed_chunk["label_name"].tolist())
        write_chunk(processed_chunk, config.output_path, first_chunk=first_chunk)
        first_chunk = False
        processed_rows += len(processed_chunk)

        if len(preview_rows) < 3:
            for _, row in processed_chunk.head(3 - len(preview_rows)).iterrows():
                preview_rows.append(row.to_dict())

    return {
        "rows_processed": processed_rows,
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "label_name_counts": dict(sorted(label_name_counts.items())),
        "output_path": str(config.output_path),
        "preview": preview_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load df.csv and preprocess email text for spam classification."
    )
    parser.add_argument(
        "--input",
        default="df.csv",
        help="Path to the input CSV file. Defaults to df.csv in the current directory.",
    )
    parser.add_argument(
        "--output",
        default="processed_df.csv",
        help="Path for the preprocessed CSV output.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column name containing the raw email text.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column name containing labels.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("multiclass", "binary", "raw"),
        default="multiclass",
        help="Use 'multiclass' for ham/phish/spam, 'binary' for ham vs spam-or-phish, or 'raw' for untouched numeric labels.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="How many rows to process at a time.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N rows for quick experiments.",
    )
    parser.add_argument(
        "--keep-original-text",
        action="store_true",
        help="Keep the unprocessed text column in the output CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ProcessingConfig(
        input_path=resolve_project_path(args.input),
        output_path=resolve_project_path(args.output),
        text_column=args.text_column,
        label_column=args.label_column,
        label_mode=args.label_mode,
        chunksize=args.chunksize,
        limit=args.limit,
        keep_original_text=args.keep_original_text,
        overwrite=args.overwrite,
    )

    summary = process_dataset(config)

    print("Preprocessing complete.")
    print(f"Rows processed: {summary['rows_processed']}")
    print(f"Raw label counts: {summary['raw_label_counts']}")
    print(f"Class counts: {summary['label_name_counts']}")
    print(f"Output saved to: {summary['output_path']}")
    print("Preview:")
    for row in summary["preview"]:
        print(row)


if __name__ == "__main__":
    main()
