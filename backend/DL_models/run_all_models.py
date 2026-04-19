from __future__ import annotations

try:
    from backend.DL_models.train_dl_models import build_parser, config_from_args, train_and_evaluate_models
except ModuleNotFoundError:
    from train_dl_models import build_parser, config_from_args, train_and_evaluate_models


MODEL_RUNS = (
    ("cnn", "CNN"),
    ("lstm", "LSTM"),
)


def main() -> None:
    parser = build_parser(
        description="Train the CNN and LSTM text classifiers one after another.",
        include_models=False,
    )
    args = parser.parse_args()

    for index, (model_type, label) in enumerate(MODEL_RUNS):
        config = config_from_args(
            args,
            model_types=(model_type,),
            output_suffix=model_type,
        )
        if index > 0:
            config.overwrite = True

        summary = train_and_evaluate_models(config)
        print(f"{label} training complete.")
        print(f"Best run: {summary['best_run']}")

    print("All deep-learning model runs complete.")
    print(f"Artifacts saved to: {summary['artifacts_dir']}")
    print("Models trained: CNN, LSTM")


if __name__ == "__main__":
    main()
