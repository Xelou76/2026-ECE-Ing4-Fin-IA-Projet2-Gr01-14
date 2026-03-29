import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

# ============================================================
# ISOLATION FOREST
# ============================================================
def train_isolation_forest(X_train, y_train):
    """
    Entraîne un Isolation Forest sur données ORIGINALES (sans SMOTE)
    car c'est un algo non-supervisé qui n'a pas besoin de rééquilibrage
    """
    print("\n🌲 Entraînement Isolation Forest...")
    X_normal = X_train[y_train == 0]
    print(f"   Entraînement sur {len(X_normal)} transactions normales")

    contamination = y_train.sum() / len(y_train)
    print(f"   Contamination : {contamination:.4f}")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_normal)
    print("✅ Isolation Forest entraîné !")
    return model

def predict_isolation_forest(model, X_test):
    """Prédit avec Isolation Forest"""
    y_pred_raw = model.predict(X_test)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    scores = -model.score_samples(X_test)
    return y_pred, scores

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss : réduit le poids des exemples faciles
    et se concentre sur les exemples difficiles (fraudes)
    alpha : poids de la classe positive
    gamma : facteur de focalisation
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets.float())
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# ============================================================
# AUTOENCODER
# ============================================================
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(FraudAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(X_train, y_train, epochs=50, batch_size=256, lr=0.001):
    """Entraîne l'Autoencoder sur les transactions normales"""
    print("\n🤖 Entraînement Autoencoder avec Focal Loss...")
    X_normal = X_train[y_train == 0]

    model = FraudAutoencoder(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_normal)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_tensor, X_tensor),
                       batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, _ in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Loss : {epoch_loss/len(loader):.6f}")

    print("✅ Autoencoder entraîné !")
    return model

def predict_autoencoder(model, X_test, threshold=None):
    """Prédit avec l'Autoencoder"""
    model.eval()
    X_tensor = torch.FloatTensor(X_test)
    with torch.no_grad():
        recon = model(X_tensor)
    errors = torch.mean((X_tensor - recon)**2, dim=1).numpy()

    if threshold is None:
        threshold = np.percentile(errors, 95)

    y_pred = (errors > threshold).astype(int)
    return y_pred, errors, threshold

# ============================================================
# GNN
# ============================================================
class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(FraudGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)

def build_graph(X, y, n_neighbors=5):
    """Construit un graphe de transactions"""
    print(f"   Construction graphe sur {len(X)} transactions...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(
        x=torch.FloatTensor(X),
        edge_index=edge_index,
        y=torch.LongTensor(y.astype(int))
    )
    print(f"   ✅ Graphe : {data.num_nodes} noeuds, {data.num_edges} arêtes")
    return data

def train_gnn(X_train, y_train, epochs=100, n_sample=5000):
    """Entraîne le GNN"""
    print("\n🕸️  Entraînement GNN...")

    idx = np.random.choice(len(X_train), n_sample, replace=False)
    data_train = build_graph(X_train[idx], y_train[idx])

    model = FraudGNN(input_dim=X_train.shape[1])
    class_counts = np.bincount(y_train[idx].astype(int))
    class_weights = torch.FloatTensor([1.0, class_counts[0]/class_counts[1]])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_train)
        loss = criterion(out, data_train.y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            acc = (out.argmax(dim=1) == data_train.y).float().mean()
            print(f"   Epoch {epoch+1}/{epochs} | Loss : {loss.item():.4f} | Acc : {acc:.4f}")

    print("✅ GNN entraîné !")
    return model

def predict_gnn(model, X_test, y_test):
    """Prédit avec le GNN"""
    fraud_idx = np.where(y_test == 1)[0]
    normal_idx = np.where(y_test == 0)[0][:2000]
    test_idx = np.concatenate([fraud_idx, normal_idx])

    data_test = build_graph(X_test[test_idx], y_test[test_idx])

    model.eval()
    with torch.no_grad():
        out = model(data_test)
        probs = F.softmax(out, dim=1)[:, 1].numpy()
        y_pred = out.argmax(dim=1).numpy()

    return y_pred, probs, data_test.y.numpy()

# ============================================================
# PYOD - LOF + OCSVM avec sous-échantillonnage
# ============================================================
def train_pyod_models(X_train, y_train):
    """Entraîne LOF et OCSVM depuis PyOD sur 5000 samples max"""
    contamination = float(y_train.sum() / len(y_train))

    # Sous-échantillonnage pour OCSVM qui est très lent
    N_SAMPLE = 5000
    X_normal = X_train[y_train == 0]
    idx = np.random.choice(len(X_normal), min(N_SAMPLE, len(X_normal)), replace=False)
    X_sample = X_normal[idx]
    print(f"   PyOD entraîné sur {len(X_sample)} samples")

    models = {
        'LOF': LOF(contamination=contamination, n_neighbors=20),
        'OCSVM': OCSVM(contamination=contamination, kernel='rbf')
    }

    trained = {}
    for name, model in models.items():
        print(f"\n🔍 Entraînement {name}...")
        model.fit(X_sample)
        trained[name] = model
        print(f"✅ {name} entraîné !")

    return trained

def predict_pyod_models(models, X_test):
    """Prédit avec les modèles PyOD"""
    results = {}
    for name, model in models.items():
        scores = model.decision_function(X_test)
        y_pred = (scores > 0).astype(int)
        results[name] = {'scores': scores, 'y_pred': y_pred}
    return results