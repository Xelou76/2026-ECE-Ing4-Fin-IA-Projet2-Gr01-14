import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import FraudAutoencoder, predict_autoencoder

# ============================================================
# PIPELINE TEMPS RÉEL AVEC SEUILS ADAPTATIFS
# ============================================================
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


def run_streaming(n_transactions=500):
    print("🚀 PIPELINE DE PRÉDICTION EN TEMPS RÉEL")
    print("=" * 60)

    # Charger les données
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/")
    X_test = np.load(DATA_PATH + "X_test.npy")
    y_test = np.load(DATA_PATH + "y_test.npy")
    X_train_res = np.load(DATA_PATH + "X_train_res.npy")
    y_train_res = np.load(DATA_PATH + "y_train_res.npy")

    # Charger et entraîner l'autoencoder
    print("\n🤖 Chargement de l'Autoencoder...")
    from models import train_autoencoder
    model = train_autoencoder(X_train_res, y_train_res, epochs=30)
    model.eval()

    # Calculer le seuil initial
    _, _, threshold = predict_autoencoder(model, X_train_res[y_train_res == 0][:5000])
    print(f"   Seuil initial : {threshold:.6f}")

    # Lancer le streaming
    pipeline = AdaptiveThresholdPipeline(model, threshold)

    print(f"\n📡 Streaming de {n_transactions} transactions...\n")
    vrais_positifs = 0
    faux_positifs = 0
    faux_negatifs = 0

    for i in range(n_transactions):
        decision, score = pipeline.process_transaction(X_test[i])
        true_label = y_test[i]

        if decision == "FRAUDE":
            if true_label == 1:
                vrais_positifs += 1
                print(f"  ✅ Transaction #{i:4d} | Score: {score:.4f} | FRAUDE DÉTECTÉE")
            else:
                faux_positifs += 1
                print(f"  ❌ Transaction #{i:4d} | Score: {score:.4f} | FAUX POSITIF")
        else:
            if true_label == 1:
                faux_negatifs += 1

    cout_total = (faux_positifs * 10) + (faux_negatifs * 500)

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS STREAMING ({n_transactions} transactions)")
    print(f"{'='*60}")
    print(f"  Fraudes détectées : {vrais_positifs}")
    print(f"  Faux positifs     : {faux_positifs}")
    print(f"  Fraudes manquées  : {faux_negatifs}")
    print(f"  Seuil final       : {pipeline.threshold:.6f}")
    print(f"  Coût financier    : {cout_total}€")


if __name__ == "__main__":
    run_streaming(n_transactions=500)