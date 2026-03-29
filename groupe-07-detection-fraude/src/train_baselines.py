from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import load_dataset, prepare_features, split_and_scale


def compute_cost(y_true, y_pred, cost_fp=1, cost_fn=200):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = fp * cost_fp + fn * cost_fn

    return {
        "FP": fp,
        "FN": fn,
        "Cost_FP": fp * cost_fp,
        "Cost_FN": fn * cost_fn,
        "Total_Cost": total_cost,
    }


RANDOM_STATE = 42
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name: str, y_true, y_pred, y_score) -> None:
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, digits=4))

    roc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    costs = compute_cost(y_true, y_pred, cost_fp=1, cost_fn=200)
    print("\nCOST ANALYSIS")
    for k, v in costs.items():
        print(f"{k}: {v}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_cm.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_pr_curve.png")
    plt.close()


def main() -> None:
    data_path = Path("data/creditcard.csv")
    df = load_dataset(data_path)
    X, y = prepare_features(df)
    splits = split_and_scale(X, y)

    X_train = splits.X_train_scaled
    X_test = splits.X_test_scaled
    y_train = splits.y_train
    y_test = splits.y_test

    # 1) Logistic Regression
    logreg = Pipeline(
        steps=[
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_score = logreg.predict_proba(X_test)[:, 1]
    evaluate_model("Logistic Regression", y_test, y_pred, y_score)
    joblib.dump(logreg, OUTPUT_DIR / "logistic_regression.joblib")

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]
    evaluate_model("Random Forest", y_test, y_pred, y_score)
    joblib.dump(rf, OUTPUT_DIR / "random_forest.joblib")

    # 3) Isolation Forest
    # On entraîne uniquement sur les transactions normales
    X_train_normal = X_train[y_train == 0]

    iso = IsolationForest(
        n_estimators=200,
        contamination=float(y_train.mean()),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_normal)

    # predict: 1 = normal, -1 = anomaly
    iso_pred_raw = iso.predict(X_test)
    y_pred = np.where(iso_pred_raw == -1, 1, 0)

    # decision_function: plus élevé = plus normal
    # On inverse le score pour obtenir plus élevé = plus frauduleux
    y_score = -iso.decision_function(X_test)

    evaluate_model("Isolation Forest", y_test, y_pred, y_score)
    joblib.dump(iso, OUTPUT_DIR / "isolation_forest.joblib")


if __name__ == "__main__":
    main()