# Spam Model Training

## Run The Preprocessor

Generate the cleaned dataset first:

```bash
python backend/preprocess.py --input df.csv --output processed_df.csv --overwrite
```

This reads `backend/df.csv` and writes the preprocessed output to `backend/processed_df.csv`.

## Run The Pipeline

After preprocessing, build the features and then run each model script separately:

```bash
python backend/build_features.py --input processed_df.csv --artifacts-dir artifacts --overwrite
python backend/ML_models/train_naive_bayes.py --artifacts-dir artifacts --overwrite
python backend/ML_models/train_random_forest.py --artifacts-dir artifacts --overwrite
python backend/ML_models/train_svm.py --artifacts-dir artifacts --overwrite
```

If you want a faster experiment, you can reduce the dataset size during feature building with `--sample-size`. The training scripts will then train on that sampled dataset because they read the saved artifacts from the same directory:

```bash
python backend/build_features.py --input processed_df.csv --artifacts-dir artifacts_small --sample-size 10000 --overwrite
python backend/ML_models/run_all_models.py --artifacts-dir artifacts_small --overwrite
```

You can also replace `backend/ML_models/run_all_models.py` with any individual training script as long as it uses the same `--artifacts-dir`.

If you want one command for all three model runs:

```bash
python backend/ML_models/run_all_models.py --artifacts-dir artifacts --overwrite
```

## Run The Deep-Learning Models

You can now run the deep-learning models individually as well:

Use the base dataset `df.csv` here, not `processed_df.csv`. The deep-learning pipeline preprocesses the raw text internally before training.

```bash
python backend/DL_models/train_cnn.py --input df.csv --artifacts-dir dl_artifacts --overwrite
python backend/DL_models/train_lstm.py --input df.csv --artifacts-dir dl_artifacts --overwrite
```

If you want one command for both deep-learning runs:

```bash
python backend/DL_models/run_all_models.py --input df.csv --artifacts-dir dl_artifacts --overwrite
```

If you want a faster deep-learning experiment on a sample of the raw dataset, use `--sample-size` and a separate artifacts directory:

```bash
python backend/DL_models/run_all_models.py --input df.csv --artifacts-dir dl_artifacts_small --sample-size 10000 --overwrite
```

You can do the same with any individual deep-learning training script, as long as you keep the same `--artifacts-dir` for that sampled run:

```bash
python backend/DL_models/train_cnn.py --input df.csv --artifacts-dir dl_artifacts_small --sample-size 10000 --overwrite
python backend/DL_models/train_lstm.py --input df.csv --artifacts-dir dl_artifacts_small --sample-size 10000 --overwrite
```

The original combined script is still available if you want to train both models in a single run and keep the shared combined metric files:

```bash
python backend/DL_models/train_dl_models.py --input df.csv --artifacts-dir dl_artifacts --overwrite
```

## Run The API

After your artifact directories are ready, start the backend API that serves the frontend:

```bash
python -m uvicorn backend.api.app:app --reload
```

This starts the backend on `http://127.0.0.1:8000` by default.

Quick checks:

- Health: `http://127.0.0.1:8000/api/health`
- Dashboard sample: `http://127.0.0.1:8000/api/dashboard?profile=sample`

The frontend calls:

- `GET /api/dashboard?profile=sample|main`
- `POST /api/predict`

The API maps profiles to artifact folders like this:

- `main`: `backend/artifacts` and `backend/dl_artifacts`
- `sample`: `backend/artifacts_small` and `backend/dl_artifacts_small`

## Scripts

- `backend/preprocess.py`: cleans the raw email text and writes the preprocessed dataset to `backend/processed_df.csv`.
- `backend/build_features.py`: loads the processed dataset, creates the train/test split, and saves feature matrices.
- `backend/ML_models/train_naive_bayes.py`: trains only the Naive Bayes model.
- `backend/ML_models/train_random_forest.py`: trains only the Random Forest model.
- `backend/ML_models/train_svm.py`: trains only the SVM model.
- `backend/ML_models/run_all_models.py`: runs Naive Bayes, Random Forest, and SVM one after another.
- `backend/DL_models/train_cnn.py`: trains only the CNN deep-learning model.
- `backend/DL_models/train_lstm.py`: trains only the LSTM deep-learning model.
- `backend/DL_models/run_all_models.py`: runs the CNN and LSTM models one after another.
- `backend/DL_models/train_dl_models.py`: trains one or both deep-learning models from a single entrypoint.
- `backend/api/app.py`: FastAPI server that exposes dashboard data and prediction endpoints to the frontend.
