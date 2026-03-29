from pathlib import Path
import argparse
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def load_data(dataset):
    dataset_dir = PROCESSED_DIR / dataset

    X_train = pd.read_csv(dataset_dir / "X_train.csv")
    X_test = pd.read_csv(dataset_dir / "X_test.csv")
    y_train = pd.read_csv(dataset_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(dataset_dir / "y_test.csv").iloc[:, 0]

    return X_train, X_test, y_train, y_test


def get_models():
    # On commence toujours avec la régression logistique comme base de comparaison.
    # class_weight="balanced" compense automatiquement si les défauts sont rares dans les données.
    models = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced")
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
    except:
        print("XGBoost non installé")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
    except:
        print(" LightGBM non installé")

    return models


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # On calcule l'AUC seulement si le modèle peut donner des probabilités (pas juste 0 ou 1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc
    }


def main(dataset):
    print(f"\n=== Modélisation : {dataset} ===")

    X_train, X_test, y_train, y_test = load_data(dataset)

    models = get_models()

    dataset_models_dir = MODELS_DIR / dataset
    dataset_results_dir = RESULTS_DIR / dataset

    dataset_models_dir.mkdir(parents=True, exist_ok=True)
    dataset_results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    # On garde une trace du meilleur modèle pour le sauvegarder séparément à la fin
    best_score = -1
    best_model = None
    best_name = None

    for name, model in models.items():
        print(f"\n--- {name} ---")

        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")
        # On sauvegarde chaque modèle individuellement pour pouvoir les comparer dans le dashboard
        joblib.dump(model, dataset_models_dir / f"{name}.pkl")

        results.append({"model": name, **metrics})
        # On préfère l'AUC pour comparer les modèles, et on prend le F1 si l'AUC n'est pas disponible
        score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1"]

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    # sauvegarde résultats
    results_df = pd.DataFrame(results)
    results_df.to_csv(dataset_results_dir / "results.csv", index=False)

    print("\n🏆 Meilleur modèle :", best_name)
    # On sauvegarde aussi le meilleur modèle sous un nom générique pour que le dashboard puisse le charger facilement
    joblib.dump(best_model, dataset_models_dir / "best_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["german", "lending_club"])
    args = parser.parse_args()

    main(args.dataset)