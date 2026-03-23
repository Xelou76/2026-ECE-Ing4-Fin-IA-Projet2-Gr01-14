import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# ============================================================
# ISOLATION FOREST
# ============================================================
def train_isolation_forest(X_train, y_train):
    """Entraîne un Isolation Forest"""
    print("\n🌲 Entraînement Isolation Forest...")
    contamination = y_train.sum() / len(y_train)
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    print("✅ Isolation Forest entraîné !")
    return model

def predict_isolation_forest(model, X_test):
    """Prédit avec Isolation Forest"""
    y_pred_raw = model.predict(X_test)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    scores = -model.score_samples(X_test)
    return y_pred, scores

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
    print("\n🤖 Entraînement Autoencoder...")
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