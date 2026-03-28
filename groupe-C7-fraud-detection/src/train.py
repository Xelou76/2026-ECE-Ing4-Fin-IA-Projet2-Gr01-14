import numpy as np
import os
import sys
import torch
import joblib
from sklearn.metrics import average_precision_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_and_preprocess, apply_smote, print_metrics
from models import (train_isolation_forest, predict_isolation_forest,
                    train_autoencoder, predict_autoencoder,
                    train_gnn, predict_gnn,
                    train_pyod_models, predict_pyod_models)

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/creditcard.csv")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "../data/")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "../models/")

def run_pipeline():
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("🚀 DÉMARRAGE DU PIPELINE — DÉTECTION DE FRAUDE")
    print("=" * 60)

    # 1. Chargement & preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)

    # 2. SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    np.save(SAVE_PATH + "X_train_res.npy", X_train_res)
    np.save(SAVE_PATH + "y_train_res.npy", y_train_res)
    np.save(SAVE_PATH + "X_test.npy", X_test)
    np.save(SAVE_PATH + "y_test.npy", y_test)
    print("✅ Données sauvegardées !")

    resultats = {}

    # 3. Isolation Forest — données ORIGINALES
    iso_model = train_isolation_forest(X_train, y_train)
    joblib.dump(iso_model, MODELS_PATH + "isolation_forest.pkl")
    print("💾 Isolation Forest sauvegardé !")
    y_pred_iso, scores_iso = predict_isolation_forest(iso_model, X_test)
    auprc_iso = average_precision_score(y_test, scores_iso)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_iso).ravel()
    print_metrics("Isolation Forest", TP, FP, FN, auprc_iso)
    resultats['Isolation Forest'] = {'AUPRC': auprc_iso, 'TP': TP, 'FP': FP, 'FN': FN}

    # 4. Autoencoder — données SMOTE
    ae_model = train_autoencoder(X_train_res, y_train_res)
    torch.save(ae_model.state_dict(), MODELS_PATH + "autoencoder.pth")
    print("💾 Autoencoder sauvegardé !")
    y_pred_ae, errors_ae, threshold = predict_autoencoder(ae_model, X_test)
    np.save(MODELS_PATH + "ae_threshold.npy", np.array([threshold]))
    auprc_ae = average_precision_score(y_test, errors_ae)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred_ae).ravel()
    print_metrics("Autoencoder", TP, FP, FN, auprc_ae)
    resultats['Autoencoder'] = {'AUPRC': auprc_ae, 'TP': TP, 'FP': FP, 'FN': FN}

    # 5. GNN — données SMOTE
    gnn_model = train_gnn(X_train_res, y_train_res)
    torch.save(gnn_model.state_dict(), MODELS_PATH + "gnn.pth")
    print("💾 GNN sauvegardé !")
    y_pred_gnn, probs_gnn, y_true_gnn = predict_gnn(gnn_model, X_test, y_test)
    auprc_gnn = average_precision_score(y_true_gnn, probs_gnn)
    TN, FP, FN, TP = confusion_matrix(y_true_gnn, y_pred_gnn).ravel()
    print_metrics("GNN", TP, FP, FN, auprc_gnn)
    resultats['GNN'] = {'AUPRC': auprc_gnn, 'TP': TP, 'FP': FP, 'FN': FN}

    # 6. PyOD — LOF + OCSVM sur données ORIGINALES
    pyod_models = train_pyod_models(X_train, y_train)
    joblib.dump(pyod_models, MODELS_PATH + "pyod_models.pkl")
    print("💾 Modèles PyOD sauvegardés !")
    pyod_results = predict_pyod_models(pyod_models, X_test)
    for name, res in pyod_results.items():
        auprc = average_precision_score(y_test, res['scores'])
        TN, FP, FN, TP = confusion_matrix(y_test, res['y_pred']).ravel()
        print_metrics(name, TP, FP, FN, auprc)
        resultats[name] = {'AUPRC': auprc, 'TP': TP, 'FP': FP, 'FN': FN}

    # 7. Résumé final
    print("\n" + "=" * 60)
    print("   RÉSUMÉ FINAL")
    print("=" * 60)
    for model_name, metrics in resultats.items():
        cout = (metrics['FP'] * 10) + (metrics['FN'] * 500)
        print(f"  {model_name:20s} | AUPRC : {metrics['AUPRC']:.4f} | Coût : {cout}€")

    meilleur = max(resultats, key=lambda x: resultats[x]['AUPRC'])
    print(f"\n🥇 MEILLEUR MODÈLE : {meilleur} (AUPRC : {resultats[meilleur]['AUPRC']:.4f})")
    print(f"\n✅ Tous les modèles sauvegardés dans models/")

if __name__ == "__main__":
    run_pipeline()