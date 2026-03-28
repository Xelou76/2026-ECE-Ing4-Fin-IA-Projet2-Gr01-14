import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import joblib
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import FraudAutoencoder, FraudGNN, build_graph, predict_autoencoder

MODELS_PATH = os.path.join(os.path.dirname(__file__), "../models/")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/")

class AdaptiveThresholdPipeline:
    def __init__(self, model, initial_threshold, window_size=100):
        self.model = model
        self.threshold = initial_threshold
        self.window_size = window_size
        self.scores_window = deque(maxlen=window_size)
        self.decisions = []
        self.threshold_history = [initial_threshold]

    def compute_score(self, transaction):
        with torch.no_grad():
            t = torch.FloatTensor(transaction).unsqueeze(0)
            recon = self.model(t)
            return torch.mean((t - recon)**2).item()

    def adapt_threshold(self):
        if len(self.scores_window) < 10:
            return
        recent_scores = np.array(self.scores_window)
        fraud_rate = np.mean(recent_scores > self.threshold)
        if fraud_rate > 0.05:
            self.threshold *= 1.05
        elif fraud_rate < 0.01:
            self.threshold *= 0.98
        self.threshold_history.append(self.threshold)

    def process_transaction(self, transaction):
        score = self.compute_score(transaction)
        self.scores_window.append(score)
        decision = "FRAUDE" if score > self.threshold else "NORMAL"
        self.decisions.append(decision)
        if len(self.decisions) % 50 == 0:
            self.adapt_threshold()
        return decision, score


def load_models():
    print("📂 Chargement des modèles sauvegardés...")
    iso_model = joblib.load(MODELS_PATH + "isolation_forest.pkl")
    print("  ✅ Isolation Forest chargé !")

    ae_model = FraudAutoencoder(input_dim=29)
    ae_model.load_state_dict(torch.load(MODELS_PATH + "autoencoder.pth"))
    ae_model.eval()
    threshold = float(np.load(MODELS_PATH + "ae_threshold.npy")[0])
    print(f"  ✅ Autoencoder chargé ! (seuil : {threshold:.6f})")

    gnn_model = FraudGNN(input_dim=29)
    gnn_model.load_state_dict(torch.load(MODELS_PATH + "gnn.pth"))
    gnn_model.eval()
    print("  ✅ GNN chargé !")

    return iso_model, ae_model, threshold, gnn_model


def run_streaming(n_normal=480, n_fraud=20):
    print("🚀 PIPELINE DE PRÉDICTION EN TEMPS RÉEL")
    print("=" * 60)

    X_test = np.load(DATA_PATH + "X_test.npy")
    y_test = np.load(DATA_PATH + "y_test.npy")

    # Mix fraudes + normales pour démo réaliste
    fraud_idx = np.where(y_test == 1)[0][:n_fraud]
    normal_idx = np.where(y_test == 0)[0][:n_normal]
    mixed_idx = np.concatenate([fraud_idx, normal_idx])
    np.random.shuffle(mixed_idx)

    print(f"📊 Stream : {n_normal} normales + {n_fraud} fraudes = {len(mixed_idx)} transactions")

    iso_model, ae_model, threshold, gnn_model = load_models()
    pipeline = AdaptiveThresholdPipeline(ae_model, threshold)

    print(f"\n📡 Streaming de {len(mixed_idx)} transactions...\n")
    vrais_positifs = 0
    faux_positifs = 0
    faux_negatifs = 0

    for i in mixed_idx:
        decision, score = pipeline.process_transaction(X_test[i])
        true_label = y_test[i]

        if decision == "FRAUDE":
            if true_label == 1:
                vrais_positifs += 1
                print(f"  ✅ Transaction #{i:4d} | Score: {score:.4f} | FRAUDE DÉTECTÉE !")
            else:
                faux_positifs += 1
                print(f"  ❌ Transaction #{i:4d} | Score: {score:.4f} | FAUX POSITIF")
        else:
            if true_label == 1:
                faux_negatifs += 1
                print(f"  ⚠️  Transaction #{i:4d} | Score: {score:.4f} | FRAUDE MANQUÉE")

    cout_total = (faux_positifs * 10) + (faux_negatifs * 500)

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS STREAMING ({len(mixed_idx)} transactions)")
    print(f"{'='*60}")
    print(f"  Fraudes détectées : {vrais_positifs}/{n_fraud}")
    print(f"  Faux positifs     : {faux_positifs}")
    print(f"  Fraudes manquées  : {faux_negatifs}")
    print(f"  Seuil final       : {pipeline.threshold:.6f}")
    print(f"  Coût financier    : {cout_total}€")


if __name__ == "__main__":
    if not os.path.exists(MODELS_PATH + "autoencoder.pth"):
        print("⚠️  Modèles non trouvés ! Lance d'abord : py src/train.py")
    else:
        run_streaming(n_normal=480, n_fraud=20)