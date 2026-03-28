import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import FraudAutoencoder, FraudGNN
from predict import AdaptiveThresholdPipeline

MODELS_PATH = os.path.join(os.path.dirname(__file__), "../models/")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/")

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Detection — Groupe C7",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .fraud-alert {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 24px;
        animation: blink 0.5s infinite;
        border: 3px solid #ff6666;
        box-shadow: 0 0 20px rgba(255,0,0,0.5);
    }
    .normal-tx {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #444;
    }
    .header-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Configuration")
n_normal = st.sidebar.slider("Transactions normales", 100, 1000, 480)
n_fraud = st.sidebar.slider("Fraudes injectées", 5, 50, 20)
speed = st.sidebar.slider("Vitesse (ms)", 50, 500, 100)

st.sidebar.divider()
st.sidebar.markdown("## 🏆 Résultats réels")
st.sidebar.dataframe(pd.DataFrame({
    'Modèle': ['Isolation Forest', 'OCSVM', 'Autoencoder', 'LOF', 'GNN'],
    'AUPRC': [0.137, 0.278, 0.506, 0.604, 0.891],
    'Coût (€)': [35520, 289890, 33120, 568640, 7840]
}), hide_index=True)

st.sidebar.divider()
st.sidebar.markdown("## 👥 Groupe C7")
st.sidebar.markdown("ECE Paris — Ing4 Finance")
st.sidebar.markdown("IA Probabiliste, Théorie des Jeux et ML")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="header-card">
    <h1>🚨 Détection de Fraude Bancaire en Temps Réel</h1>
    <p style="font-size:18px">ECE Paris — Ing4 Finance — Groupe C7</p>
    <p>Dataset : Kaggle Credit Card Fraud | 284 807 transactions | 0.17% de fraudes</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📡 Streaming Live",
    "🔬 Comparaison Modèles",
    "📖 Pipeline & Méthodes"
])

# ============================================================
# ONGLET 1 — OVERVIEW
# ============================================================
with tab1:
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🏆 Meilleur AUPRC", "0.891", "GNN")
    with col2:
        st.metric("💰 Coût GNN", "7 840€", "-27 680€ vs IF")
    with col3:
        st.metric("🎯 Recall GNN", "88%", "+55% vs IF")
    with col4:
        st.metric("📊 Dataset", "284 807 tx", "Kaggle")
    with col5:
        st.metric("🚨 Taux fraude", "0.17%", "577:1")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📈 AUPRC par modèle")
        df_auprc = pd.DataFrame({
            'Modèle': ['Isolation Forest', 'OCSVM', 'Autoencoder', 'LOF', 'GNN'],
            'AUPRC': [0.137, 0.278, 0.506, 0.604, 0.891],
            'Type': ['Non-supervisé', 'PyOD', 'Deep Learning', 'PyOD', 'Graph Neural Net']
        })
        fig_auprc = px.bar(df_auprc, x='Modèle', y='AUPRC',
                          color='Type', title="Comparaison AUPRC",
                          color_discrete_map={
                              'Non-supervisé': '#FF6B6B',
                              'PyOD': '#FFA500',
                              'Deep Learning': '#4ECDC4',
                              'Graph Neural Net': '#6C5CE7'
                          })
        fig_auprc.add_hline(y=0.5, line_dash="dash", line_color="yellow",
                           annotation_text="Seuil acceptable (0.5)")
        fig_auprc.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_auprc, use_container_width=True)

    with col_b:
        st.subheader("💰 Coût financier par modèle")
        df_cout = pd.DataFrame({
            'Modèle': ['GNN', 'Autoencoder', 'Isolation Forest', 'OCSVM', 'LOF'],
            'Coût (€)': [7840, 33120, 35520, 289890, 568640]
        })
        fig_cout = px.bar(df_cout, x='Modèle', y='Coût (€)',
                         color='Coût (€)', color_continuous_scale='reds',
                         title="Coût financier (FP x10€ + FN x500€)")
        fig_cout.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_cout, use_container_width=True)

    st.divider()

    # Radar Chart
    st.subheader("🕸️ Radar Chart — Comparaison globale")
    categories = ['AUPRC', 'Recall', 'Precision', 'F1-Score', 'Coût (inv.)']
    
    max_cout = 568640
    fig_radar = go.Figure()

    models_radar = {
        'Isolation Forest': [0.137, 0.33, 0.32, 0.32, 1-35520/max_cout],
        'Autoencoder':      [0.506, 0.89, 0.03, 0.06, 1-33120/max_cout],
        'GNN':              [0.891, 0.88, 0.66, 0.75, 1-7840/max_cout],
        'LOF':              [0.604, 1.00, 0.00, 0.00, 1-568640/max_cout],
        'OCSVM':            [0.278, 0.97, 0.00, 0.01, 1-289890/max_cout],
    }
    colors_radar = ['#FF6B6B', '#4ECDC4', '#6C5CE7', '#FFA500', '#00CEC9']

    for (name, vals), color in zip(models_radar.items(), colors_radar):
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name=name,
            line_color=color, opacity=0.7
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=500,
        template="plotly_dark",
        title="Comparaison globale des 5 modèles"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ============================================================
# ONGLET 2 — STREAMING LIVE
# ============================================================
with tab2:
    st.subheader("📡 Simulation Streaming Temps Réel")
    st.markdown("Simulation d'un flux bancaire réel avec détection de fraude en temps réel")

    @st.cache_resource
    def load_models():
        ae_model = FraudAutoencoder(input_dim=29)
        ae_model.load_state_dict(torch.load(
            MODELS_PATH + "autoencoder.pth",
            map_location=torch.device('cpu')
        ))
        ae_model.eval()
        threshold = float(np.load(MODELS_PATH + "ae_threshold.npy")[0])
        return ae_model, threshold

    @st.cache_data
    def load_data():
        X_test = np.load(DATA_PATH + "X_test.npy")
        y_test = np.load(DATA_PATH + "y_test.npy")
        return X_test, y_test

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        start = st.button("🚀 Lancer", type="primary", use_container_width=True)
    with col_btn2:
        reset = st.button("🔄 Reset", use_container_width=False)

    if start:
        ae_model, threshold = load_models()
        X_test, y_test = load_data()

        fraud_idx = np.where(y_test == 1)[0][:n_fraud]
        normal_idx = np.where(y_test == 0)[0][:n_normal]
        mixed_idx = np.concatenate([fraud_idx, normal_idx])
        np.random.shuffle(mixed_idx)

        pipeline = AdaptiveThresholdPipeline(ae_model, threshold)

        # Layout
        alert_placeholder = st.empty()
        col_graph, col_stats = st.columns([3, 1])

        with col_stats:
            st.markdown("### 📊 Stats Live")
            ph_total = st.empty()
            ph_fraudes = st.empty()
            ph_fp = st.empty()
            ph_fn = st.empty()
            ph_cout = st.empty()
            ph_seuil = st.empty()

        with col_graph:
            ph_chart = st.empty()
            ph_cout_chart = st.empty()

        ph_log = st.empty()

        # Tracking
        scores_list = []
        labels_list = []
        couts_list = []
        logs = []
        vrais_positifs = 0
        faux_positifs = 0
        faux_negatifs = 0
        cout = 0

        for idx, i in enumerate(mixed_idx):
            transaction = X_test[i]
            true_label = y_test[i]
            decision, score = pipeline.process_transaction(transaction)

            scores_list.append(min(score, 50))
            labels_list.append(true_label)

            if decision == "FRAUDE":
                if true_label == 1:
                    vrais_positifs += 1
                    logs.insert(0, f"✅ #{i} | Score: {score:.2f} | FRAUDE DÉTECTÉE !")
                    alert_placeholder.markdown(
                        f'<div class="fraud-alert">🚨 FRAUDE DÉTECTÉE ! Transaction #{i} | Score Anomalie: {score:.2f}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    faux_positifs += 1
                    cout += 10
                    logs.insert(0, f"❌ #{i} | Score: {score:.2f} | Faux positif (-10€)")
                    alert_placeholder.empty()
            else:
                if true_label == 1:
                    faux_negatifs += 1
                    cout += 500
                    logs.insert(0, f"⚠️ #{i} | Score: {score:.2f} | FRAUDE MANQUÉE (-500€)")
                else:
                    alert_placeholder.empty()

            couts_list.append(cout)

            # Stats
            with ph_total:
                st.metric("Total traité", f"{idx+1}/{len(mixed_idx)}")
            with ph_fraudes:
                st.metric("✅ Fraudes détectées", f"{vrais_positifs}/{n_fraud}")
            with ph_fp:
                st.metric("❌ Faux positifs", str(faux_positifs))
            with ph_fn:
                st.metric("⚠️ Fraudes manquées", str(faux_negatifs))
            with ph_cout:
                st.metric("💰 Coût cumulé", f"{cout}€")
            with ph_seuil:
                st.metric("🎯 Seuil adaptatif", f"{pipeline.threshold:.4f}")

            # Graphique scores
            if len(scores_list) > 1:
                fig = go.Figure()
                x_range = list(range(len(scores_list)))

                fig.add_trace(go.Scatter(
                    x=x_range, y=scores_list,
                    mode='lines', name='Score anomalie',
                    line=dict(color='#4ECDC4', width=1.5),
                    fill='tozeroy', fillcolor='rgba(78,205,196,0.1)'
                ))
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=[pipeline.threshold] * len(scores_list),
                    mode='lines', name='Seuil adaptatif',
                    line=dict(color='orange', width=2, dash='dash')
                ))

                fraud_x = [j for j, l in enumerate(labels_list) if l == 1]
                fraud_y = [scores_list[j] for j in fraud_x]
                if fraud_x:
                    fig.add_trace(go.Scatter(
                        x=fraud_x, y=fraud_y,
                        mode='markers', name='Vraie fraude',
                        marker=dict(color='red', size=15,
                                   symbol='x', line=dict(width=3))
                    ))

                fig.update_layout(
                    height=300,
                    title="Scores d'anomalie en temps réel",
                    template="plotly_dark",
                    showlegend=True,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                ph_chart.plotly_chart(fig, use_container_width=True)

                # Graphique coût cumulé
                fig_cout = go.Figure()
                fig_cout.add_trace(go.Scatter(
                    x=x_range, y=couts_list,
                    mode='lines', name='Coût cumulé',
                    line=dict(color='red', width=2),
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'
                ))
                fig_cout.update_layout(
                    height=200,
                    title=f"Coût financier cumulé : {cout}€",
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )
                ph_cout_chart.plotly_chart(fig_cout, use_container_width=True)

            ph_log.markdown("**📋 Journal :**\n" + "\n".join(logs[:8]))
            time.sleep(speed / 1000)

        # Résultat final
        alert_placeholder.empty()
        if vrais_positifs == n_fraud:
            st.balloons()
            st.success(f"🎉 PARFAIT ! {vrais_positifs}/{n_fraud} fraudes détectées | Coût total : {cout}€")
        else:
            st.warning(f"⚠️ {vrais_positifs}/{n_fraud} fraudes détectées | Coût total : {cout}€")

# ============================================================
# ONGLET 3 — COMPARAISON MODELES
# ============================================================
with tab3:
    st.subheader("🔬 Comparaison détaillée des 5 modèles")

    df_full = pd.DataFrame({
        'Modèle': ['Isolation Forest', 'Autoencoder', 'GNN', 'LOF', 'OCSVM'],
        'Type': ['Non-supervisé', 'Deep Learning', 'Graph Neural Net', 'PyOD', 'PyOD'],
        'AUPRC': [0.137, 0.506, 0.891, 0.604, 0.278],
        'Recall': [0.33, 0.89, 0.88, 1.00, 0.97],
        'Precision': [0.32, 0.03, 0.66, 0.00, 0.00],
        'F1': [0.32, 0.06, 0.75, 0.00, 0.01],
        'Fraudes détectées': [32, 87, 86, 98, 95],
        'Faux Positifs': [68, 2762, 37, 56864, 28839],
        'Faux Négatifs': [66, 11, 12, 0, 3],
        'Coût (€)': [35520, 33120, 7840, 568640, 289890]
    })

    st.dataframe(df_full, hide_index=True, use_container_width=True)
    st.divider()

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_recall = px.scatter(df_full, x='Recall', y='Precision',
                               size='AUPRC', color='Modèle',
                               title="Precision vs Recall (taille = AUPRC)",
                               template="plotly_dark", height=400)
        st.plotly_chart(fig_recall, use_container_width=True)

    with col_c2:
        fig_fp = px.bar(df_full, x='Modèle', y='Faux Positifs',
                       color='Modèle', title="Faux Positifs par modèle",
                       template="plotly_dark", height=400)
        st.plotly_chart(fig_fp, use_container_width=True)

    # Gauge AUPRC GNN
    st.subheader("🎯 Performance GNN")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=0.891,
        delta={'reference': 0.5, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#6C5CE7"},
            'steps': [
                {'range': [0, 0.3], 'color': '#FF6B6B'},
                {'range': [0.3, 0.6], 'color': '#FFA500'},
                {'range': [0.6, 1], 'color': '#00C851'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 0.891
            }
        },
        title={'text': "AUPRC GNN — Niveau Production"}
    ))
    fig_gauge.update_layout(height=350, template="plotly_dark")
    st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# ONGLET 4 — PIPELINE & METHODES
# ============================================================
with tab4:
    st.subheader("📖 Pipeline & Méthodes")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("""
### 🔄 Pipeline complet
```
creditcard.csv (284 807 tx)
        ↓
   Preprocessing
   StandardScaler
        ↓
   Train/Test Split
   80% / 20%
        ↓
      SMOTE
   577:1 → 2:1
        ↓
┌─────────────────┐
│ Isolation Forest│ → AUPRC 0.137
│   Autoencoder   │ → AUPRC 0.506
│      GNN        │ → AUPRC 0.891 🏆
│      LOF        │ → AUPRC 0.604
│     OCSVM       │ → AUPRC 0.278
└─────────────────┘
        ↓
  Pipeline Streaming
  Seuils Adaptatifs
        ↓
   Décision Temps Réel
```
""")

    with col_p2:
        st.markdown("""
### 🧠 Modèles utilisés

**1. Isolation Forest**
- Algo non-supervisé baseline
- Isole les anomalies par partitionnement aléatoire
- Rapide mais limité sur données complexes

**2. Autoencoder PyTorch**
- Encoder-Decoder (29→16→8→4→8→16→29)
- Entraîné sur transactions normales uniquement
- Détection par erreur de reconstruction

**3. GNN — Graph Attention Network**
- Graphe KNN (5 voisins, 40 000 arêtes)
- 2 couches GAT (4 têtes d'attention)
- Exploite les relations entre transactions
- **Meilleur modèle : AUPRC 0.891**

**4. LOF & OCSVM (PyOD)**
- Local Outlier Factor
- One-Class SVM
- Bon rappel mais trop de faux positifs

### ⚖️ Gestion du déséquilibre
- **SMOTE** : génère des fraudes synthétiques
- **Class weights** : pénalise les fraudes manquées
- **Focal Loss** : focus sur exemples difficiles
""")

    st.divider()
    st.markdown("""
### 💡 Métriques clés

| Métrique | Pourquoi ? |
|----------|------------|
| **AUPRC** | Adaptée aux classes déséquilibrées (mieux que ROC-AUC) |
| **Recall** | Maximiser les fraudes détectées |
| **Precision** | Minimiser les faux positifs |
| **Coût financier** | FP = 10€ (blocage injustifié), FN = 500€ (fraude manquée) |

### 🔄 Streaming temps réel
- **Fenêtre glissante** de 100 transactions
- **Seuil adaptatif** : monte si trop de fraudes détectées, descend sinon
- **Résultat** : 20/20 fraudes détectées, coût minimal
""")