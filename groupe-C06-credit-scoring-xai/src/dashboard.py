# Lancer : streamlit run src/dashboard.py

import os
import json
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# CONFIG PAGE
# ============================================================

st.set_page_config(
    page_title="Credit Scoring XAI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Header */
.xai-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.8rem;
    border: 1px solid #0f3460;
}
.xai-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.xai-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.9rem;
}

/* Metric card */
.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #38bdf8;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.15rem;
}

/* Score badges */
.badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 30px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
}
.badge-ok   { background: rgba(34,197,94,0.12);  color: #22c55e; border: 1.5px solid #22c55e; }
.badge-ko   { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1.5px solid #ef4444; }
.badge-warn { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1.5px solid #fbbf24; }

/* Section title */
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE, ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")
MDL_ROOT = os.path.join(PROJECT_ROOT, "models")
RES_ROOT = os.path.join(PROJECT_ROOT, "results")

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono", color="#cbd5e1", size=11),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    margin=dict(l=10, r=10, t=35, b=10),
)

# ============================================================
# CHARGEMENT
# ============================================================

@st.cache_resource
def load_all(dataset):
    data_dir = os.path.join(DATA_ROOT, dataset)
    mdl_dir = os.path.join(MDL_ROOT, dataset)
    res_dir = os.path.join(RES_ROOT, dataset)

    try:
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        X_test_raw = pd.read_csv(os.path.join(data_dir, "X_test_raw.csv"))
        X_train_raw = pd.read_csv(os.path.join(data_dir, "X_train_raw.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
        model = joblib.load(os.path.join(mdl_dir, "best_model.pkl"))

        with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)

        extra = {}
        for name in ["xgboost", "lightgbm", "logistic"]:
            p = os.path.join(mdl_dir, f"{name}.pkl")
            if os.path.exists(p):
                extra[name] = joblib.load(p)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "X_test_raw": X_test_raw,
            "X_train_raw": X_train_raw,
            "y_train": y_train,
            "y_test": y_test,
            "model": model,
            "metadata": metadata,
            "extra": extra,
            "data_dir": data_dir,
            "mdl_dir": mdl_dir,
            "res_dir": res_dir,
        }
    except FileNotFoundError:
        return None

@st.cache_resource
def compute_shap(_model, X_test_arr):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test_arr)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    return shap_values, explainer, float(expected_value)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono;font-size:1rem;font-weight:700;
    color:#38bdf8;margin-bottom:1.2rem;'>
    🏦 Credit XAI<br>
    <span style='font-size:0.65rem;color:#64748b;'>Dashboard d'Explicabilité</span>
    </div>
    """, unsafe_allow_html=True)

    dataset = st.selectbox("Dataset", ["german", "lending_club"])

    page = st.radio("Navigation", [
        "📊  Vue d'ensemble",
        "🔍  Explicabilité SHAP",
        "🟡  LIME & Comparaison",
        "🔄  Contrefactuels",
        "⚖️  Audit Fairness",
        "👤  Scoring Individuel",
    ])

    st.markdown("---")
    data = load_all(dataset)
    if data:
        m = data["metadata"]
        st.markdown(f"""
        <div style='font-size:0.72rem;color:#64748b;font-family:IBM Plex Mono;'>
        <b style='color:#38bdf8;'>Données</b><br>
        Train : {m['n_train']:,} obs<br>
        Test  : {m['n_test']:,} obs<br>
        Features : {len(m['feature_names'])}
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(f"""
<div class='xai-header'>
    <h1>🏦 Credit Scoring-IA Explicable</h1>
    <p>{dataset} · Modèle optimal · SHAP · LIME · Contrefactuels · Fairness · RGPD Art. 22</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# GARDE-FOU
# ============================================================

if data is None:
    st.error(
        f"**Fichiers manquants pour `{dataset}`.** Lancez les scripts dans l'ordre :\n\n"
        "```bash\n"
        f"python src/exploration.py --dataset {dataset}\n"
        f"python src/modelisation.py --dataset {dataset}\n"
        f"python src/explicabilite.py --dataset {dataset}   # optionnel pour certaines pages\n"
        "```"
    )
    st.stop()

# Raccourcis
X_train = data["X_train"]
X_test = data["X_test"]
X_test_raw = data["X_test_raw"]
X_train_raw = data["X_train_raw"]
y_train = data["y_train"]
y_test = data["y_test"]
model = data["model"]
metadata = data["metadata"]

feature_names = metadata["feature_names"]
num_features = metadata["num_features"]
cat_features = metadata["cat_features"]

data_dir = data["data_dir"]
mdl_dir = data["mdl_dir"]
res_dir = data["res_dir"]

X_test_arr = X_test.values
X_train_arr = X_train.values

y_prob = model.predict_proba(X_test_arr)[:, 1]
y_pred = model.predict(X_test_arr)

shap_values, shap_explainer, expected_value = compute_shap(model, X_test_arr)

# ============================================================
# PAGE : VUE D'ENSEMBLE
# ============================================================

if "Vue d'ensemble" in page:
    st.markdown("<div class='sec-title'>Métriques du modèle</div>", unsafe_allow_html=True)

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks   = float(np.max(tpr - fpr))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{auc:.3f}")
    c2.metric("F1-Score", f"{f1:.3f}")
    c3.metric("Accuracy", f"{acc:.3f}")
    c4.metric("KS Statistic", f"{ks:.3f}")

    st.markdown("<div class='sec-title'>Courbe ROC</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}", line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aléatoire", line=dict(color="#64748b", dash="dash")))
    fig.update_layout(
        xaxis_title="Taux faux positifs",
        yaxis_title="Taux vrais positifs",
        **THEME
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("<div class='sec-title'>Distribution des scores</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=y_prob[y_test==0], name="Bon payeur", opacity=0.7,
                                marker_color="#5DCAA5", nbinsx=30))
    fig2.add_trace(go.Histogram(x=y_prob[y_test==1], name="Défaut", opacity=0.7,
                                marker_color="#E24B4A", nbinsx=30))
    fig2.update_layout(barmode="overlay", xaxis_title="Score de défaut", **THEME)
    st.plotly_chart(fig2, width='stretch')

# ============================================================
# PAGE : EXPLICABILITÉ SHAP
# ============================================================

elif "SHAP" in page:
    st.markdown("<div class='sec-title'>Importance globale des features</div>", unsafe_allow_html=True)

    mean_shap = np.abs(shap_values).mean(axis=0)
    df_shap = pd.DataFrame({"feature": feature_names, "importance": mean_shap})
    df_shap = df_shap.sort_values("importance", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=df_shap["importance"], y=df_shap["feature"],
        orientation="h", marker_color="#38bdf8"
    ))
    fig.update_layout(xaxis_title="Mean |SHAP|", **THEME)
    st.plotly_chart(fig, width='stretch')

    st.markdown("<div class='sec-title'>Explication individuelle (waterfall)</div>", unsafe_allow_html=True)
    idx = st.slider("Index du profil", 0, len(X_test)-1, 0)
    st.write(f"Score de défaut : **{y_prob[idx]:.3f}** | Vraie étiquette : **{int(y_test.iloc[idx])}**")

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_values[idx]
    }).sort_values("shap")

    colors = ["#E24B4A" if v > 0 else "#5DCAA5" for v in shap_df["shap"]]
    fig2 = go.Figure(go.Bar(
        x=shap_df["shap"], y=shap_df["feature"],
        orientation="h", marker_color=colors
    ))
    fig2.update_layout(xaxis_title="Valeur SHAP", **THEME)
    st.plotly_chart(fig2, width='stretch')

# ============================================================
# PAGE : LIME & COMPARAISON
# ============================================================

elif "LIME" in page:
    lime_path = os.path.join(res_dir, "shap_vs_lime.csv") if res_dir else None
    if lime_path and os.path.exists(lime_path):
        df_comp = pd.read_csv(lime_path)
        st.markdown("<div class='sec-title'>Concordance SHAP vs LIME</div>", unsafe_allow_html=True)
        st.dataframe(df_comp, width='stretch')

        fig = go.Figure(go.Bar(
            x=df_comp["profil"], y=df_comp["concordance"],
            marker_color=["#5DCAA5" if c >= 0.67 else "#E24B4A" for c in df_comp["concordance"]]
        ))
        fig.add_hline(y=df_comp["concordance"].mean(), line_dash="dash", line_color="#38bdf8")
        fig.update_layout(xaxis_title="Profil", yaxis_title="Concordance Top-3", **THEME)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Lance d'abord `python src/explicabilite.py` pour générer les données LIME.")

# ============================================================
# PAGE : CONTREFACTUELS
# ============================================================

elif "Contrefactuels" in page:
    cf_path = os.path.join(res_dir, "contrefactuels.csv") if res_dir else None
    if cf_path and os.path.exists(cf_path):
        st.markdown("<div class='sec-title'>Explications contrefactuelles — RGPD Art. 22</div>", unsafe_allow_html=True)
        st.markdown("*\"Que faudrait-il modifier pour ne plus être en défaut ?\"*")
        st.markdown("Tout client refusé a droit à une explication actionnable.")
        df_cf = pd.read_csv(cf_path)
        st.dataframe(df_cf, width='stretch')
    else:
        st.info("Lance d'abord `python src/explicabilite.py` pour générer les contrefactuels.")

# ============================================================
# PAGE : AUDIT FAIRNESS
# ============================================================

elif "Fairness" in page:
    st.markdown("<div class='sec-title'>Analyse de biais — âge</div>", unsafe_allow_html=True)

    if "age" in X_test_raw.columns:
        df_fair = X_test_raw.copy()
        df_fair["score"] = y_prob
        df_fair["defaut_reel"] = y_test.values
        df_fair["tranche_age"] = pd.cut(df_fair["age"], bins=[18,30,45,60,100],
                                         labels=["18-30","30-45","45-60","60+"])

        taux = df_fair.groupby("tranche_age")["score"].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=taux["tranche_age"].astype(str),
            y=taux["score"],
            marker_color="#38bdf8"
        ))
        fig.add_hline(y=y_prob.mean(), line_dash="dash", line_color="#E24B4A",
                      annotation_text=f"Moyenne globale ({y_prob.mean():.2f})")
        fig.update_layout(xaxis_title="Tranche d'âge", yaxis_title="Score moyen", **THEME)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("La feature 'age' n'est pas disponible dans ce dataset.")

# ============================================================
# PAGE : SCORING INDIVIDUEL
# ============================================================

elif "Scoring" in page:
    st.markdown("<div class='sec-title'>Saisie d'un profil client</div>", unsafe_allow_html=True)

    idx = st.slider("Choisir un profil du jeu de test", 0, len(X_test)-1, 0)

    profile = X_test.iloc[[idx]]
    score = model.predict_proba(profile.values)[0, 1]

    if score >= 0.5:
        badge = "<span class='badge badge-ko'>Refusé</span>"
    elif score >= 0.3:
        badge = "<span class='badge badge-warn'>À étudier</span>"
    else:
        badge = "<span class='badge badge-ok'>Accepté</span>"

    st.markdown(f"### Score de défaut : `{score:.3f}` {badge}", unsafe_allow_html=True)

    shap_profile = pd.DataFrame({
        "feature": feature_names,
        "shap": shap_values[idx]
    }).sort_values("shap")

    colors = ["#E24B4A" if v > 0 else "#5DCAA5" for v in shap_profile["shap"]]
    fig = go.Figure(go.Bar(
        x=shap_profile["shap"], y=shap_profile["feature"],
        orientation="h", marker_color=colors
    ))
    fig.update_layout(xaxis_title="Contribution SHAP", **THEME)
    st.plotly_chart(fig, width='stretch')

    st.markdown("<div class='sec-title'>Données du profil</div>", unsafe_allow_html=True)
    if not X_test_raw.empty:
        st.dataframe(X_test_raw.iloc[[idx]], width='stretch')
