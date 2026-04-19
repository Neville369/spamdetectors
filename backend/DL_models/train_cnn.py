from __future__ import annotations

try:
    from backend.DL_models.train_dl_models import build_parser, config_from_args, train_and_evaluate_models
except ModuleNotFoundError:
    from train_dl_models import build_parser, config_from_args, train_and_evaluate_models


MODEL_TYPE = "cnn"


def main() -> None:
    parser = build_parser(
        description="Train the CNN text classifier with trainable embeddings for spam-mail detection.",
        include_models=False,
    )
    args = parser.parse_args()

    config = config_from_args(
        args,
        model_types=(MODEL_TYPE,),
        output_suffix=MODEL_TYPE,
    )
    summary = train_and_evaluate_models(config)
    print("CNN training complete.")
    print(f"Rows processed: {summary['rows_processed']}")
    print(f"Train/Test split: {summary['train_rows']} / {summary['test_rows']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Device: {summary['device']}")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print(f"Best run: {summary['best_run']}")


if __name__ == "__main__":
    main()
