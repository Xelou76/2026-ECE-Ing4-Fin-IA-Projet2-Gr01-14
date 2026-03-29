from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import load_dataset, prepare_features, split_and_scale


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def evaluate_autoencoder(y_true, reconstruction_error, threshold: float) -> None:
    y_pred = (reconstruction_error > threshold).astype(int)

    print("\n" + "=" * 60)
    print("MODEL: Autoencoder")
    print("=" * 60)
    print(classification_report(y_true, y_pred, digits=4))

    roc = roc_auc_score(y_true, reconstruction_error)
    pr_auc = average_precision_score(y_true, reconstruction_error)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"Threshold used: {threshold:.6f}")

    costs = compute_cost(y_true, y_pred, cost_fp=1, cost_fn=200)
    print("\nCOST ANALYSIS")
    for k, v in costs.items():
        print(f"{k}: {v}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Autoencoder")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autoencoder_cm.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, reconstruction_error)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Autoencoder")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autoencoder_pr_curve.png")
    plt.close()


def main() -> None:
    data_path = Path("data/creditcard.csv")
    df = load_dataset(data_path)
    X, y = prepare_features(df)
    splits = split_and_scale(X, y)

    X_train = splits.X_train_scaled.values.astype(np.float32)
    X_test = splits.X_test_scaled.values.astype(np.float32)
    y_train = splits.y_train.values
    y_test = splits.y_test.values

    # On entraîne l'autoencoder uniquement sur les transactions normales
    X_train_normal = X_train[y_train == 0]

    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=256,
        shuffle=True,
    )

    model = Autoencoder(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {epoch_loss:.6f}")

    # Courbe de loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("Training Loss - Autoencoder")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "autoencoder_loss.png")
    plt.close()

    # Reconstruction error sur train normal pour fixer un seuil
    model.eval()
    with torch.no_grad():
        train_recon = model(train_tensor.to(DEVICE)).cpu().numpy()
        train_error = np.mean((X_train_normal - train_recon) ** 2, axis=1)

        test_recon = model(test_tensor.to(DEVICE)).cpu().numpy()
        test_error = np.mean((X_test - test_recon) ** 2, axis=1)

    # Seuil simple : percentile 95 des normales train
    threshold = np.percentile(train_error, 95)

    evaluate_autoencoder(y_test, test_error, threshold)

    torch.save(model.state_dict(), OUTPUT_DIR / "autoencoder.pt")


if __name__ == "__main__":
    main()