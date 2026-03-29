from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from preprocessing import load_dataset, prepare_features, split_and_scale
from train_autoencoder import Autoencoder


ARTIFACTS_DIR = Path("artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sklearn_model(filename: str):
    model_path = ARTIFACTS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def load_autoencoder_model(input_dim: int) -> Autoencoder:
    model_path = ARTIFACTS_DIR / "autoencoder.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            "Autoencoder model not found. Run train_autoencoder.py first."
        )

    model = Autoencoder(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def predict_logistic_regression(sample: pd.DataFrame, threshold: float = 0.5) -> dict:
    model = load_sklearn_model("logistic_regression.joblib")

    start = time.perf_counter()
    proba = float(model.predict_proba(sample)[0, 1])
    pred = int(proba >= threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "model": "Logistic Regression",
        "predicted_class": pred,
        "fraud_score": proba,
        "threshold": threshold,
        "latency_ms": elapsed_ms,
    }


def predict_random_forest(sample: pd.DataFrame, threshold: float = 0.5) -> dict:
    model = load_sklearn_model("random_forest.joblib")

    start = time.perf_counter()
    proba = float(model.predict_proba(sample)[0, 1])
    pred = int(proba >= threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "model": "Random Forest",
        "predicted_class": pred,
        "fraud_score": proba,
        "threshold": threshold,
        "latency_ms": elapsed_ms,
    }


def predict_isolation_forest(sample: pd.DataFrame, threshold: float = 0.0) -> dict:
    model = load_sklearn_model("isolation_forest.joblib")

    start = time.perf_counter()
    raw_score = float(-model.decision_function(sample)[0])
    pred = int(raw_score > threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "model": "Isolation Forest",
        "predicted_class": pred,
        "fraud_score": raw_score,
        "threshold": threshold,
        "latency_ms": elapsed_ms,
    }


def predict_autoencoder(sample: pd.DataFrame, threshold: float = 0.974909) -> dict:
    model = load_autoencoder_model(input_dim=sample.shape[1])

    x = torch.tensor(sample.values, dtype=torch.float32, device=DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        recon = model(x).cpu().numpy()
    error = float(((sample.values - recon) ** 2).mean(axis=1)[0])
    pred = int(error > threshold)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "model": "Autoencoder",
        "predicted_class": pred,
        "fraud_score": error,
        "threshold": threshold,
        "latency_ms": elapsed_ms,
    }


def print_prediction(result: dict, true_label: int | None = None) -> None:
    print("\n" + "=" * 60)
    print(f"MODEL: {result['model']}")
    print("=" * 60)
    if true_label is not None:
        print(f"True label        : {true_label}")
    print(f"Predicted class   : {result['predicted_class']}")
    print(f"Fraud score       : {result['fraud_score']:.6f}")
    print(f"Threshold used    : {result['threshold']:.6f}")
    print(f"Inference latency : {result['latency_ms']:.3f} ms")


def simulate_stream(
    X_stream: pd.DataFrame,
    y_stream: pd.Series,
    model_name: str = "random_forest",
    n_samples: int = 10,
) -> None:
    print("\n" + "#" * 60)
    print(f"STREAMING SIMULATION - {model_name.upper()}")
    print("#" * 60)

    predictors = {
        "logistic_regression": predict_logistic_regression,
        "random_forest": predict_random_forest,
        "isolation_forest": predict_isolation_forest,
        "autoencoder": predict_autoencoder,
    }

    if model_name not in predictors:
        raise ValueError(f"Unknown model_name: {model_name}")

    predictor = predictors[model_name]

    latencies = []
    predictions = []
    truths = []

    n = min(n_samples, len(X_stream))

    for i in range(n):
        sample = X_stream.iloc[[i]]
        true_label = int(y_stream.iloc[i])

        result = predictor(sample)
        latencies.append(result["latency_ms"])
        predictions.append(result["predicted_class"])
        truths.append(true_label)

        print(
            f"[{i:02d}] true={true_label} "
            f"pred={result['predicted_class']} "
            f"score={result['fraud_score']:.6f} "
            f"latency={result['latency_ms']:.3f} ms"
        )

    predictions = np.array(predictions)
    truths = np.array(truths)

    tp = int(((predictions == 1) & (truths == 1)).sum())
    tn = int(((predictions == 0) & (truths == 0)).sum())
    fp = int(((predictions == 1) & (truths == 0)).sum())
    fn = int(((predictions == 0) & (truths == 1)).sum())

    print("\nSTREAM SUMMARY")
    print(f"Samples processed : {n}")
    print(f"Average latency   : {np.mean(latencies):.3f} ms")
    print(f"TP                : {tp}")
    print(f"TN                : {tn}")
    print(f"FP                : {fp}")
    print(f"FN                : {fn}")


def main() -> None:
    data_path = Path("data/creditcard.csv")
    df = load_dataset(data_path)
    X, y = prepare_features(df)
    splits = split_and_scale(X, y)

    X_test = splits.X_test_scaled
    y_test = splits.y_test

    # Exemple 1 : une transaction unique
    sample = X_test.iloc[[0]]
    true_label = int(y_test.iloc[0])

    print("SINGLE TRANSACTION INFERENCE")
    print_prediction(predict_logistic_regression(sample), true_label=true_label)
    print_prediction(predict_random_forest(sample), true_label=true_label)
    print_prediction(predict_isolation_forest(sample), true_label=true_label)
    print_prediction(predict_autoencoder(sample), true_label=true_label)

    # Exemple 2 : petite simulation de flux temps réel
    simulate_stream(X_test, y_test, model_name="random_forest", n_samples=10)


if __name__ == "__main__":
    main()