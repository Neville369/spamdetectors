from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from backend.DL_models.train_dl_models import build_model as build_dl_model
from backend.DL_models.train_dl_models import encode_text
from backend.preprocess import DEFAULT_MULTICLASS_LABEL_NAMES, PorterStemmer, preprocess_text


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
LABEL_ORDER = ["ham", "phish", "spam"]
DISPLAY_FEATURES = {
    "count": "Count Vectorizer",
    "tfidf": "TF-IDF",
    "token_sequences": "Token Sequences",
}
MODEL_PRESENTATION = {
    "naive_bayes": {
        "family": "Classical ML",
        "label": "Naive Bayes",
        "summary": "Fastest classical baseline with strong spam sensitivity on bag-of-words counts.",
    },
    "random_forest": {
        "family": "Classical ML",
        "label": "Random Forest",
        "summary": "Balanced ensemble that reacts well to mixed signals across suspicious and legitimate text.",
    },
    "svm": {
        "family": "Classical ML",
        "label": "SVM",
        "summary": "Best classical accuracy with sharp margin-based separation for phishing-heavy messages.",
    },
    "cnn": {
        "family": "Deep Learning",
        "label": "CNN",
        "summary": "Sequence model that tends to benefit from longer context windows and local token patterns.",
    },
    "lstm": {
        "family": "Deep Learning",
        "label": "LSTM",
        "summary": "Context-aware recurrent model that leans harder on phrasing like verify your or claim your.",
    },
}
ML_MODEL_ORDER = ["naive_bayes", "random_forest", "svm"]
DL_MODEL_ORDER = ["cnn", "lstm"]
STEMMER = PorterStemmer()
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def _resolve_configured_dir(value: str, default_relative: str) -> Path:
    candidate = Path(value) if value else Path(default_relative)
    if candidate.is_absolute():
        return candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    backend_candidate = BACKEND_ROOT / candidate
    return backend_candidate


def _profile_directories() -> dict[str, dict[str, Path]]:
    return {
        "main": {
            "dl": _resolve_configured_dir(
                os.getenv("BACKEND_DL_MAIN_ARTIFACT_DIR", ""),
                "backend/dl_artifacts",
            ),
            "ml": _resolve_configured_dir(
                os.getenv("BACKEND_ML_MAIN_ARTIFACT_DIR", ""),
                "backend/artifacts",
            ),
        },
        "sample": {
            "dl": _resolve_configured_dir(
                os.getenv("BACKEND_DL_SAMPLE_ARTIFACT_DIR", ""),
                "backend/dl_artifacts_small",
            ),
            "ml": _resolve_configured_dir(
                os.getenv("BACKEND_ML_SAMPLE_ARTIFACT_DIR", ""),
                "backend/artifacts_small",
            ),
        },
    }


def get_profile_directories(profile: str) -> dict[str, Path]:
    normalized = str(profile or "sample").strip().lower()
    directories = _profile_directories()
    if normalized not in directories:
        raise ValueError(f"Unsupported artifact profile: {profile}")
    return directories[normalized]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_glob(directory: Path, pattern: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def normalize_label_counts(label_counts: dict[str, Any] | None) -> dict[str, int]:
    counts = {label: 0 for label in LABEL_ORDER}
    if not label_counts:
        return counts

    for label, value in label_counts.items():
        label_key = str(label)
        if label_key in counts:
            counts[label_key] = int(value)

    return counts


def grouped_class_counts(class_counts: dict[str, int]) -> dict[str, int]:
    return {
        "ham": int(class_counts.get("ham", 0)),
        "spam": int(class_counts.get("phish", 0)) + int(class_counts.get("spam", 0)),
    }


def load_ml_build_summary(ml_dir: Path) -> dict[str, Any] | None:
    summary_path = ml_dir / "features" / "build_summary.json"
    if not summary_path.exists():
        return None
    return read_json(summary_path)


def load_dl_run_summaries(dl_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = dl_dir / "metrics"
    summaries: list[dict[str, Any]] = []

    for path in safe_glob(metrics_dir, "run_summary*.json"):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        summaries.append(payload)

    return summaries


def resolve_label_counts(ml_dir: Path, dl_dir: Path) -> dict[str, int]:
    ml_summary = load_ml_build_summary(ml_dir)
    if ml_summary and "label_counts" in ml_summary:
        return normalize_label_counts(ml_summary.get("label_counts"))

    for summary in load_dl_run_summaries(dl_dir):
        if "label_counts" in summary:
            return normalize_label_counts(summary.get("label_counts"))

    train_path = dl_dir / "splits" / "train_processed.csv"
    test_path = dl_dir / "splits" / "test_processed.csv"
    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path, usecols=["label_name"])
        test_df = pd.read_csv(test_path, usecols=["label_name"])
        label_counts = (
            pd.concat([train_df, test_df], ignore_index=True)["label_name"]
            .value_counts()
            .to_dict()
        )
        return normalize_label_counts(label_counts)

    return normalize_label_counts(None)


def load_ml_detailed_reports(ml_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_dir = ml_dir / "metrics"
    merged: dict[str, dict[str, Any]] = {}

    for path in safe_glob(metrics_dir, "detailed_reports*.json"):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            merged.update(payload)

    return merged


def load_dl_training_histories(dl_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_dir = dl_dir / "metrics"
    merged: dict[str, dict[str, Any]] = {}

    for path in safe_glob(metrics_dir, "training_histories*.json"):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            merged.update(payload)

    return merged


def load_dl_model_results(dl_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = dl_dir / "metrics"
    rows: list[dict[str, Any]] = []

    for path in safe_glob(metrics_dir, "model_results*.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        rows.extend(frame.to_dict(orient="records"))

    return rows


def select_best_run(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    return max(
        entries,
        key=lambda entry: (
            float(entry["metrics"].get("weighted_f1", 0)),
            float(entry["metrics"].get("accuracy", 0)),
        ),
    )


def build_classical_models(ml_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    detailed_reports = load_ml_detailed_reports(ml_dir)
    warnings: list[str] = []
    grouped_runs: dict[str, list[dict[str, Any]]] = {model_type: [] for model_type in ML_MODEL_ORDER}

    for run_key, payload in detailed_reports.items():
        metrics = payload.get("metrics", {})
        model_type = str(metrics.get("model_type", ""))
        if model_type not in grouped_runs:
            continue
        grouped_runs[model_type].append(
            {
                "classification_report": payload.get("classification_report", {}),
                "confusion_matrix": payload.get("confusion_matrix", []),
                "metrics": metrics,
                "run_key": run_key,
            }
        )

    models: list[dict[str, Any]] = []
    for model_type in ML_MODEL_ORDER:
        selected = select_best_run(grouped_runs[model_type])
        if not selected:
            warnings.append(
                f"No trained classical metrics were found for {model_type} in {ml_dir / 'metrics'}."
            )
            continue

        metrics = selected["metrics"]
        feature_type = str(metrics.get("feature_type", "count"))
        presentation = MODEL_PRESENTATION[model_type]
        models.append(
            {
                "accuracy": float(metrics.get("accuracy", 0)),
                "artifactFeatureType": feature_type,
                "artifactModelType": model_type,
                "classificationReport": selected.get("classification_report", {}),
                "confusionMatrix": selected.get("confusion_matrix", []),
                "family": presentation["family"],
                "featureCount": int(metrics.get("feature_count", 0)),
                "featureDescriptor": DISPLAY_FEATURES.get(feature_type, feature_type),
                "featureType": DISPLAY_FEATURES.get(feature_type, feature_type),
                "f1": float(metrics.get("weighted_f1", 0)),
                "id": model_type,
                "label": presentation["label"],
                "macroF1": float(metrics.get("macro_f1", 0)),
                "modelType": presentation["label"],
                "pipeline": DISPLAY_FEATURES.get(feature_type, feature_type),
                "precision": float(metrics.get("weighted_precision", 0)),
                "recall": float(metrics.get("weighted_recall", 0)),
                "summary": presentation["summary"],
                "testRows": int(metrics.get("test_rows", 0)),
                "trainRows": int(metrics.get("train_rows", 0)),
                "trainingSeconds": float(metrics.get("fit_seconds", 0)),
            }
        )

    return models, warnings


def build_deep_learning_models(dl_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    histories = load_dl_training_histories(dl_dir)
    result_rows = load_dl_model_results(dl_dir)
    warnings: list[str] = []

    best_results: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        model_type = str(row.get("model_type", ""))
        if model_type not in DL_MODEL_ORDER:
            continue
        current = best_results.get(model_type)
        if current is None or (
            float(row.get("weighted_f1", 0)),
            float(row.get("accuracy", 0)),
        ) > (
            float(current.get("weighted_f1", 0)),
            float(current.get("accuracy", 0)),
        ):
            best_results[model_type] = row

    models: list[dict[str, Any]] = []
    for model_type in DL_MODEL_ORDER:
        result = best_results.get(model_type)
        history = histories.get(model_type)
        if not result or not history:
            warnings.append(
                f"No trained deep-learning metrics were found for {model_type} in {dl_dir / 'metrics'}."
            )
            continue

        presentation = MODEL_PRESENTATION[model_type]
        models.append(
            {
                "accuracy": float(result.get("accuracy", 0)),
                "artifactFeatureType": "token_sequences",
                "artifactModelType": model_type,
                "classificationReport": history.get("classification_report", {}),
                "confusionMatrix": history.get("confusion_matrix", []),
                "epochs": int(result.get("epochs", 0)),
                "family": presentation["family"],
                "featureDescriptor": DISPLAY_FEATURES["token_sequences"],
                "featureType": DISPLAY_FEATURES["token_sequences"],
                "f1": float(result.get("weighted_f1", 0)),
                "id": model_type,
                "label": presentation["label"],
                "macroF1": float(result.get("macro_f1", 0)),
                "maxSequenceLength": int(result.get("max_sequence_length", 0)),
                "modelType": presentation["label"],
                "pipeline": DISPLAY_FEATURES["token_sequences"],
                "precision": float(result.get("weighted_precision", 0)),
                "recall": float(result.get("weighted_recall", 0)),
                "summary": presentation["summary"],
                "testRows": int(result.get("test_rows", 0)),
                "trainRows": int(result.get("train_rows", 0)),
                "trainingSeconds": float(result.get("training_seconds", 0)),
                "trainingHistory": history.get("epoch_losses", []),
                "vocabSize": int(result.get("vocab_size", 0)),
            }
        )

    return models, warnings


@lru_cache(maxsize=8)
def get_dashboard_payload(profile: str) -> dict[str, Any]:
    directories = get_profile_directories(profile)
    ml_dir = directories["ml"]
    dl_dir = directories["dl"]

    class_counts = resolve_label_counts(ml_dir=ml_dir, dl_dir=dl_dir)
    classical_models, classical_warnings = build_classical_models(ml_dir)
    dl_models, dl_warnings = build_deep_learning_models(dl_dir)
    ordered_models = classical_models + dl_models

    return {
        "analytics": {
            "classCounts": class_counts,
            "groupedClassCounts": grouped_class_counts(class_counts),
        },
        "artifacts": {
            "dl": str(dl_dir),
            "ml": str(ml_dir),
        },
        "metrics": {
            "labels": LABEL_ORDER,
            "models": ordered_models,
        },
        "profile": str(profile).strip().lower(),
        "warnings": classical_warnings + dl_warnings,
    }


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated)


def preprocess_message(text: str) -> str:
    return preprocess_text(str(text or ""), stemmer=STEMMER, stop_words=STOP_WORDS)


def get_model_metadata(profile: str, model_id: str) -> dict[str, Any]:
    dashboard = get_dashboard_payload(profile)
    for model in dashboard["metrics"]["models"]:
        if model["id"] == model_id:
            return model
    raise ValueError(f"Unknown model id '{model_id}' for profile '{profile}'.")


@lru_cache(maxsize=32)
def load_ml_runtime(profile: str, model_id: str, feature_type: str) -> dict[str, Any]:
    directories = get_profile_directories(profile)
    ml_dir = directories["ml"]
    vectorizer_path = ml_dir / "features" / feature_type / "vectorizer.joblib"
    model_path = ml_dir / "models" / feature_type / f"{model_id}.joblib"

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Missing vectorizer artifact: {vectorizer_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    return {
        "feature_names": feature_names,
        "model": model,
        "vectorizer": vectorizer,
    }


def top_signals_for_ml(
    processed_text: str,
    transformed_row,
    runtime: dict[str, Any],
    artifact_model_type: str,
    predicted_class: int,
) -> list[dict[str, str]]:
    indices = transformed_row.indices.tolist()
    values = transformed_row.data.tolist()
    if not indices:
        return []

    model = runtime["model"]
    feature_names = runtime["feature_names"]
    class_position = list(model.classes_).index(predicted_class)
    scored_terms: list[tuple[float, str]] = []

    for index, value in zip(indices, values, strict=False):
        term = str(feature_names[index])
        score = 0.0

        if artifact_model_type == "svm" and hasattr(model, "coef_"):
            score = float(max(0.0, model.coef_[class_position, index] * value))
        elif artifact_model_type == "random_forest" and hasattr(model, "feature_importances_"):
            score = float(model.feature_importances_[index] * 100)
        elif artifact_model_type == "naive_bayes" and hasattr(model, "feature_log_prob_"):
            score = float(math.exp(model.feature_log_prob_[class_position, index]) * 100)

        if score > 0:
            scored_terms.append((score, term))

    deduped: list[dict[str, str]] = []
    seen_terms: set[str] = set()
    for score, term in sorted(scored_terms, reverse=True):
        if term in seen_terms:
            continue
        deduped.append({"contribution": f"+{score:.2f}", "term": term})
        seen_terms.add(term)
        if len(deduped) == 6:
            break

    return deduped


def predict_with_ml(profile: str, model_meta: dict[str, Any], text: str) -> dict[str, Any]:
    artifact_model_type = model_meta["artifactModelType"]
    feature_type = model_meta["artifactFeatureType"]
    runtime = load_ml_runtime(profile, artifact_model_type, feature_type)
    processed_text = preprocess_message(text)
    transformed = runtime["vectorizer"].transform([processed_text])
    model = runtime["model"]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(transformed))
        if scores.ndim == 1:
            scores = np.expand_dims(scores, axis=0)
        probabilities = softmax(scores[0])
    else:
        predicted_class = int(model.predict(transformed)[0])
        class_names = [DEFAULT_MULTICLASS_LABEL_NAMES[int(label)] for label in model.classes_]
        probabilities = np.asarray([1.0 if class_name == DEFAULT_MULTICLASS_LABEL_NAMES[predicted_class] else 0.0 for class_name in class_names])

    class_probabilities = [
        {
            "confidence": float(probability),
            "label": DEFAULT_MULTICLASS_LABEL_NAMES[int(class_id)],
        }
        for class_id, probability in zip(model.classes_, probabilities, strict=False)
    ]
    class_probabilities.sort(key=lambda entry: entry["confidence"], reverse=True)
    top_prediction = class_probabilities[0]
    predicted_class = next(
        int(class_id)
        for class_id, probability in zip(model.classes_, probabilities, strict=False)
        if DEFAULT_MULTICLASS_LABEL_NAMES[int(class_id)] == top_prediction["label"]
    )

    return {
        "classProbabilities": class_probabilities,
        "confidence": float(top_prediction["confidence"]),
        "groupConfidence": float(
            top_prediction["confidence"]
            if top_prediction["label"] == "ham"
            else sum(
                entry["confidence"]
                for entry in class_probabilities
                if entry["label"] != "ham"
            )
        ),
        "groupLabel": "ham" if top_prediction["label"] == "ham" else "spam",
        "label": top_prediction["label"],
        "processedText": processed_text,
        "topSignals": top_signals_for_ml(
            processed_text=processed_text,
            transformed_row=transformed[0],
            runtime=runtime,
            artifact_model_type=artifact_model_type,
            predicted_class=predicted_class,
        ),
    }


@lru_cache(maxsize=16)
def load_dl_runtime(profile: str, model_id: str) -> dict[str, Any]:
    directories = get_profile_directories(profile)
    dl_dir = directories["dl"]
    checkpoint_path = dl_dir / "models" / model_id / f"{model_id}.pt"
    vocab_path = dl_dir / "vocab" / "token_to_id.json"
    label_path = dl_dir / "vocab" / "index_to_label_name.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing DL vocabulary artifact: {vocab_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing DL label mapping artifact: {label_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    runtime_config = SimpleNamespace(
        cnn_filters=int(config.get("cnn_filters", 128)),
        dropout=0.30,
        embedding_dim=int(config.get("embedding_dim", 128)),
        hidden_dim=int(config.get("hidden_dim", 128)),
    )
    model = build_dl_model(
        model_type=str(config.get("model_type", model_id)),
        config=runtime_config,
        vocab_size=int(config.get("vocab_size", 30000)),
        num_classes=int(config.get("num_classes", len(LABEL_ORDER))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "index_to_label": {
            int(index): label_name for index, label_name in read_json(label_path).items()
        },
        "max_sequence_length": int(config.get("max_sequence_length", 200)),
        "model": model,
        "token_to_id": {token: int(token_id) for token, token_id in read_json(vocab_path).items()},
    }


def predict_with_dl(profile: str, model_meta: dict[str, Any], text: str) -> dict[str, Any]:
    runtime = load_dl_runtime(profile, model_meta["artifactModelType"])
    processed_text = preprocess_message(text)
    encoded = encode_text(
        processed_text,
        vocab=runtime["token_to_id"],
        max_length=runtime["max_sequence_length"],
    )
    inputs = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        logits = runtime["model"](inputs)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_probabilities = [
        {
            "confidence": float(probability),
            "label": runtime["index_to_label"].get(index, str(index)),
        }
        for index, probability in enumerate(probabilities)
    ]
    class_probabilities.sort(key=lambda entry: entry["confidence"], reverse=True)
    top_prediction = class_probabilities[0]

    return {
        "classProbabilities": class_probabilities,
        "confidence": float(top_prediction["confidence"]),
        "groupConfidence": float(
            top_prediction["confidence"]
            if top_prediction["label"] == "ham"
            else sum(
                entry["confidence"]
                for entry in class_probabilities
                if entry["label"] != "ham"
            )
        ),
        "groupLabel": "ham" if top_prediction["label"] == "ham" else "spam",
        "label": top_prediction["label"],
        "processedText": processed_text,
        "topSignals": [],
    }


def predict_text(profile: str, model_id: str, text: str) -> dict[str, Any]:
    model_meta = get_model_metadata(profile, model_id)
    if model_meta["family"] == "Classical ML":
        prediction = predict_with_ml(profile, model_meta, text)
    else:
        prediction = predict_with_dl(profile, model_meta, text)

    return {
        "model": {
            "family": model_meta["family"],
            "id": model_meta["id"],
            "label": model_meta["label"],
            "pipeline": model_meta["pipeline"],
        },
        "prediction": prediction,
        "profile": str(profile).strip().lower(),
    }
