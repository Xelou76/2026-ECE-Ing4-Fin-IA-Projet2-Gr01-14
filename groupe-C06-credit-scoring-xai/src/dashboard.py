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

# Traduction des codes du dataset German en labels lisibles pour l'affichage dans le dashboard
GERMAN_MAPPINGS = {
    "statut_compte": {
        "A11": "Compte < 0€",
        "A12": "Compte 0–200€",
        "A13": "Compte > 200€",
        "A14": "Aucun compte"
    },
    "historique_credit": {
        "A30": "Aucun crédit",
        "A31": "Crédit remboursé",
        "A32": "Retards passés",
        "A33": "Retards sévères",
        "A34": "Crédit en défaut"
    },
    "objet_credit": {
        "A40": "Voiture neuve",
        "A41": "Voiture occasion",
        "A42": "Mobilier",
        "A43": "Équipement",
        "A44": "Réparations",
        "A45": "Formation",
        "A46": "Business",
        "A47": "Autre"
    },
    "epargne": {
        "A61": "< 100€",
        "A62": "100–500€",
        "A63": "500–1000€",
        "A64": "> 1000€",
        "A65": "Aucune épargne"
    },
    "anciennete_emploi": {
        "A71": "< 1 an",
        "A72": "1–4 ans",
        "A73": "4–7 ans",
        "A74": "≥ 7 ans",
        "A75": "Sans emploi"
    },
    "statut_civil_sexe": {
        "A91": "Homme divorcé/séparé",
        "A92": "Femme divorcée/séparée/mariée",
        "A93": "Homme célibataire",
        "A94": "Homme marié/veuf",
        "A95": "Femme célibataire"
    },
    "autres_debiteurs": {
        "A101": "Aucun",
        "A102": "Co-emprunteur",
        "A103": "Garant"
    },
    "propriete": {
        "A121": "Immobilier",
        "A122": "Épargne / assurance vie",
        "A123": "Voiture / autre bien",
        "A124": "Aucun bien majeur"
    },
    "autres_credits": {
        "A141": "Banque",
        "A142": "Magasins",
        "A143": "Aucun"
    },
    "logement": {
        "A151": "Loyer",
        "A152": "Propriétaire",
        "A153": "Logé gratuitement"
    },
    "emploi": {
        "A171": "Chômeur / non qualifié",
        "A172": "Non qualifié",
        "A173": "Employé qualifié",
        "A174": "Cadre / hautement qualifié"
    },
    "telephone": {
        "A191": "Pas de téléphone",
        "A192": "Téléphone disponible"
    },
    "travailleur_etranger": {
        "A201": "Oui",
        "A202": "Non"
    }
}

def map_value(feature, value):
    if dataset == "german" and feature in GERMAN_MAPPINGS:
        return GERMAN_MAPPINGS[feature].get(str(value), value)
    return value

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
# @st.cache_resource évite de recharger les fichiers à chaque interaction de l'utilisateur.
# Tout est chargé une seule fois et gardé en mémoire tant que la session est active.
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
        # On charge aussi les modèles individuels s'ils existent, pour les comparer dans le dashboard
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
    # Calcul des valeurs SHAP mis en cache : ce calcul peut être long,donc on ne le refait pas à chaque changement de page
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
# Si les fichiers n'ont pas encore été générés par les autres scripts, on arrête là et on affiche un message d'erreur clair avec les commandes à lancer
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
        st.dataframe(df_cf, width="stretch")
    else:
        st.info("Lance d'abord `python src/explicabilite.py` pour générer les contrefactuels.")

    st.markdown("<div class='sec-title'>Simulateur interactif de profil</div>", unsafe_allow_html=True)
    st.markdown("Ajustez certaines variables numériques pour observer l’impact sur le score de défaut.")

    preprocessor = joblib.load(os.path.join(mdl_dir, "preprocessor.pkl"))

    display_feats = [f for f in num_features if f in X_test_raw.columns][:6]
    profile_vals = {}
    cols_sim = st.columns(3)

    for i, feat in enumerate(display_feats):
        col = cols_sim[i % 3]
        vmin = float(X_test_raw[feat].quantile(0.02))
        vmax = float(X_test_raw[feat].quantile(0.98))
        vmed = float(X_test_raw[feat].median())

        profile_vals[feat] = col.slider(
            feat,
            min_value=round(vmin, 1),
            max_value=round(vmax, 1),
            value=round(vmed, 1),
            key=f"sim_{feat}"
        )

    base_raw = X_test_raw.iloc[0].copy()
    for feat, val in profile_vals.items():
        base_raw[feat] = val

    profile_df = pd.DataFrame([base_raw])
    profile_proc = preprocessor.transform(profile_df)
    sim_score = float(model.predict_proba(profile_proc)[0, 1])

    col_s, col_g = st.columns([1, 2])

    with col_s:
        st.markdown("<br>", unsafe_allow_html=True)

        if sim_score >= 0.5:
            st.markdown(
                f"<div class='badge badge-ko'>❌ Risque élevé — {sim_score:.1%}</div>",
                unsafe_allow_html=True
            )
        elif sim_score >= 0.3:
            st.markdown(
                f"<div class='badge badge-warn'>⚠️ Profil intermédiaire — {sim_score:.1%}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='badge badge-ok'>✅ Faible risque — {sim_score:.1%}</div>",
                unsafe_allow_html=True
            )

    with col_g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sim_score * 100,
            number={
                "suffix": "%",
                "font": {"family": "IBM Plex Mono", "color": "#cbd5e1"}
            },
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#334155"),
                bar=dict(color="#38bdf8"),
                bgcolor="#1e293b",
                steps=[
                    {"range": [0, 30], "color": "rgba(34,197,94,0.10)"},
                    {"range": [30, 50], "color": "rgba(251,191,36,0.10)"},
                    {"range": [50, 100], "color": "rgba(239,68,68,0.10)"},
                ],
                threshold=dict(line=dict(color="#fbbf24", width=2), value=50)
            )
        ))
        fig.update_layout(**THEME, height=220)
        st.plotly_chart(fig, width="stretch")
# ============================================================
# PAGE : AUDIT FAIRNESS
# ============================================================

elif "Fairness" in page:
    st.markdown("""
    <div class='sec-title'>Audit de Fairness -biais potentiels</div>
    <p style='color:#64748b;font-size:0.88rem;'>
    Cette section analyse d’éventuels écarts de traitement entre groupes sensibles.
    Le seuil de référence pour le <i>Disparate Impact Ratio</i> est fixé à <b style='color:#fbbf24;'>0.80</b>
    (règle des 4/5).
    </p>
    """, unsafe_allow_html=True)

    from sklearn.metrics import confusion_matrix as cm_fn


    # Calcule les métriques de performance séparément pour chaque groupe
    def grp_metrics(groups):
        rows = []
        for g in np.unique(groups):
            mask = groups == g
            yt = y_test[mask]
            yp = y_pred[mask]
            ypr = y_prob[mask]

            if len(yt) < 5 or yt.nunique() < 2:
                continue

            tn, fp, fn, tp = cm_fn(yt, yp, labels=[0, 1]).ravel()

            rows.append({
                "Groupe": g,
                "N": int(mask.sum()),
                "Taux défaut réel": round(float(yt.mean()), 3),
                "Taux défaut prédit": round(float(yp.mean()), 3),
                "TPR": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 3),
                "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 3),
                "AUC": round(float(roc_auc_score(yt, ypr)), 3),
            })
        return pd.DataFrame(rows)

    # ============================================================
    # Définition des groupes sensibles selon le dataset
    # ============================================================

    groups_dict = {}

    if dataset == "german":
        if "age" in X_test_raw.columns:
            age_group = np.where(
                X_test_raw["age"].values < 30,
                "Jeune (<30)",
                "Senior (≥30)"
            )
            groups_dict["Âge"] = age_group

        if "statut_civil_sexe" in X_test_raw.columns:
            gender_group = np.where(
                X_test_raw["statut_civil_sexe"].astype(str).isin(["A92", "A95"]),
                "Femme",
                "Homme"
            )
            groups_dict["Genre"] = gender_group

    elif dataset == "lending_club":
        if "annual_inc" in X_test_raw.columns:
            income_group = np.where(
                X_test_raw["annual_inc"] < X_test_raw["annual_inc"].median(),
                "Bas revenu",
                "Haut revenu"
            )
            groups_dict["Revenu"] = income_group

        if "term" in X_test_raw.columns:
            term_group = np.where(
                X_test_raw["term"].astype(str).str.contains("60"),
                "Prêt long",
                "Prêt court"
            )
            groups_dict["Durée du prêt"] = term_group

    if not groups_dict:
        st.info("Aucun groupe sensible exploitable n'a été trouvé pour ce dataset.")
    else:
        # ============================================================
        # KPI globaux par variable sensible
        # ============================================================

        kpi_cols = st.columns(len(groups_dict))

        for i, (var, grps) in enumerate(groups_dict.items()):
            df_g = grp_metrics(grps)

            if df_g.empty or len(df_g) < 2:
                kpi_cols[i].warning(f"Données insuffisantes pour {var}")
                continue
            # En dessous de 0.80 (règle des 4/5), le modèle est considéré discriminant.
            accept_rate = 1 - df_g["Taux défaut prédit"]
            di_ratio = accept_rate.min() / accept_rate.max() if accept_rate.max() > 0 else 0

            eo_diff = max(
                abs(df_g["TPR"].max() - df_g["TPR"].min()),
                abs(df_g["FPR"].max() - df_g["FPR"].min())
            )

            compliant = di_ratio >= 0.80
            color = "#22c55e" if compliant else "#ef4444"
            status = "✅ Conforme" if compliant else "❌ À surveiller"

            kpi_cols[i].markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>DI Ratio — {var}</div>
                <div class='kpi-value' style='color:{color}'>{di_ratio:.3f}</div>
                <div class='kpi-sub'>{status} | EO diff = {eo_diff:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # Graphiques détaillés par variable
        # ============================================================

        for var, grps in groups_dict.items():
            df_g = grp_metrics(grps)

            if df_g.empty:
                continue

            st.markdown(
                f"<div class='sec-title'>Analyse détaillée -{var}</div>",
                unsafe_allow_html=True
            )

            col_a, col_b = st.columns(2)

            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Taux réel",
                    x=df_g["Groupe"],
                    y=df_g["Taux défaut réel"],
                    marker_color="#38bdf8",
                    opacity=0.85
                ))
                fig.add_trace(go.Bar(
                    name="Taux prédit",
                    x=df_g["Groupe"],
                    y=df_g["Taux défaut prédit"],
                    marker_color="#ef4444",
                    opacity=0.85
                ))
                fig.add_hline(
                    y=float(y_test.mean()),
                    line_dash="dash",
                    line_color="#94a3b8",
                    annotation_text=f"Moyenne globale ({y_test.mean():.2f})"
                )
                fig.update_layout(
                    **THEME,
                    barmode="group",
                    height=280,
                    title="Parité démographique",
                    yaxis_title="Taux de défaut"
                )
                st.plotly_chart(fig, width="stretch")

            with col_b:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="TPR",
                    x=df_g["Groupe"],
                    y=df_g["TPR"],
                    marker_color="#22c55e",
                    opacity=0.85
                ))
                fig.add_trace(go.Bar(
                    name="FPR",
                    x=df_g["Groupe"],
                    y=df_g["FPR"],
                    marker_color="#f59e0b",
                    opacity=0.85
                ))
                fig.update_layout(
                    **THEME,
                    barmode="group",
                    height=280,
                    title="Equalized Odds",
                    yaxis_title="Taux",
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig, width="stretch")

            st.dataframe(df_g, width="stretch")

        # ============================================================
        # Fichiers exportés si disponibles
        # ============================================================

        fl_path = os.path.join(res_dir, "fairness_metrics.csv")
        if os.path.exists(fl_path):
            st.markdown(
                "<div class='sec-title'>Métriques fairness exportées</div>",
                unsafe_allow_html=True
            )
            df_fl = pd.read_csv(fl_path)
            st.dataframe(df_fl, width="stretch")

        to_path = os.path.join(res_dir, "threshold_optimization.csv")
        if os.path.exists(to_path):
            st.markdown(
                "<div class='sec-title'>Optimisation de seuil - avant /après</div>",
                unsafe_allow_html=True
            )
            df_to = pd.read_csv(to_path)
            st.dataframe(df_to, width="stretch")

            required_cols = {"variable", "EO_diff_avant", "EO_diff_apres"}
            if required_cols.issubset(df_to.columns):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="EO diff avant",
                    x=df_to["variable"],
                    y=df_to["EO_diff_avant"],
                    marker_color="#ef4444",
                    opacity=0.85
                ))
                fig.add_trace(go.Bar(
                    name="EO diff après",
                    x=df_to["variable"],
                    y=df_to["EO_diff_apres"],
                    marker_color="#22c55e",
                    opacity=0.85
                ))
                fig.update_layout(
                    **THEME,
                    barmode="group",
                    height=280,
                    yaxis_title="Equalized Odds Diff (↓ meilleur)"
                )
                st.plotly_chart(fig, width="stretch")
# ============================================================
# PAGE : SCORING INDIVIDUEL
# ============================================================
# Permet de saisir les informations d'un client manuellement et d'obtenir son score en temps réel, avec une explication des variables qui ont le plus pesé dans la décision.
elif "Scoring" in page:
    st.markdown("""
    <div class='sec-title'>Scoring d'un nouveau client</div>
    <p style='color:#64748b;font-size:0.88rem;'>
    Saisissez les informations d’un client pour obtenir un score de défaut et une explication locale de la décision.
    </p>
    """, unsafe_allow_html=True)

    preprocessor = joblib.load(os.path.join(mdl_dir, "preprocessor.pkl"))

    st.markdown("**Variables numériques**")
    num_inputs = {}
    cols_n = st.columns(4)

    display_num_features = [f for f in num_features if f in X_test_raw.columns][:8]

    for i, feat in enumerate(display_num_features):
        col = cols_n[i % 4]
        vmin = float(X_test_raw[feat].quantile(0.01))
        vmax = float(X_test_raw[feat].quantile(0.99))
        vmed = float(X_test_raw[feat].median())

        num_inputs[feat] = col.number_input(
            feat,
            min_value=round(vmin, 1),
            max_value=round(vmax, 1),
            value=round(vmed, 1),
            key=f"num_{feat}"
        )

    cat_inputs = {}
    if cat_features:
        st.markdown("**Variables catégorielles**")
        cols_c = st.columns(4)

        display_cat_features = [f for f in cat_features if f in X_test_raw.columns][:8]

        for i, feat in enumerate(display_cat_features):
            col = cols_c[i % 4]
            raw_options = sorted(X_test_raw[feat].dropna().astype(str).unique().tolist())

            if raw_options:
                default_value = str(X_test_raw[feat].dropna().mode().iloc[0])
                default_index = raw_options.index(default_value) if default_value in raw_options else 0

                selected_raw = col.selectbox(
                    feat,
                    raw_options,
                    index=default_index,
                    key=f"cat_{feat}",
                    format_func=lambda x, f=feat: str(map_value(f, x))
                )

                cat_inputs[feat] = selected_raw
    if st.button("🔍 Calculer le score", type="primary"):
        base = X_test_raw.iloc[0].copy()

        for feat, val in num_inputs.items():
            base[feat] = val

        for feat, val in cat_inputs.items():
            base[feat] = val

        prof_df = pd.DataFrame([base])
        prof_proc = preprocessor.transform(prof_df)
        score = float(model.predict_proba(prof_proc)[0, 1])

        shap_explainer_local = shap.TreeExplainer(model)
        shap_ind = shap_explainer_local.shap_values(prof_proc)

        if isinstance(shap_ind, list):
            shap_ind = shap_ind[1]

        shap_arr = shap_ind[0]
        contribs = sorted(
            zip(feature_names, shap_arr),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        st.markdown("---")
        col_res, col_exp = st.columns(2)

        with col_res:
            if score >= 0.5:
                st.markdown(
                    f"<div class='badge badge-ko' style='font-size:1.1rem;padding:0.8rem 1.6rem;'>"
                    f"❌ Risque de défaut élevé<br>{score:.1%}</div>",
                    unsafe_allow_html=True
                )
            elif score >= 0.3:
                st.markdown(
                    f"<div class='badge badge-warn' style='font-size:1.1rem;padding:0.8rem 1.6rem;'>"
                    f"⚠️ Profil intermédiaire<br>{score:.1%}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='badge badge-ok' style='font-size:1.1rem;padding:0.8rem 1.6rem;'>"
                    f"✅ Faible risque de défaut<br>{score:.1%}</div>",
                    unsafe_allow_html=True
                )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number={
                    "suffix": "%",
                    "font": {"family": "IBM Plex Mono", "color": "#cbd5e1"}
                },
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#334155"),
                    bar=dict(color="#38bdf8"),
                    bgcolor="#1e293b",
                    steps=[
                        {"range": [0, 30], "color": "rgba(34,197,94,0.10)"},
                        {"range": [30, 50], "color": "rgba(251,191,36,0.10)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.10)"},
                    ],
                    threshold=dict(
                        line=dict(color="#fbbf24", width=2),
                        value=50
                    )
                )
            ))
            fig.update_layout(**THEME, height=240)
            st.plotly_chart(fig, width="stretch")

        with col_exp:
            st.markdown("**Explication locale de la décision :**")
            for feat, val in contribs[:6]:
                icon = "🟢" if val < 0 else "🔴"
                impact = "réduit le risque" if val < 0 else "augmente le risque"
                raw_val = base.get(feat, "—")
                feat_v = map_value(feat, raw_val)
                st.markdown(
                    f"- {icon} `{feat}` = **{feat_v}** → {impact} ({val:+.4f})"
                )
            # Si le client est refusé, on propose des pistes concrètes d'amélioration en comparant ses valeurs à celles des bons payeurs
            if score >= 0.5:
                st.markdown("<br>**Pistes d'amélioration possibles :**", unsafe_allow_html=True)
                bad = [(f, v) for f, v in contribs if v > 0][:3]

                for feat, _ in bad:
                    if feat in num_features and feat in X_test_raw.columns:
                        try:
                            target = float(X_test_raw[y_test == 0][feat].median())
                            st.markdown(
                                f"→ Améliorer **{feat}** vers une valeur proche de `{target:.1f}` "
                                f"(médiane observée chez les bons payeurs)."
                            )
                        except Exception:
                            pass

        st.markdown("<div class='sec-title'>Récapitulatif du profil saisi</div>", unsafe_allow_html=True)
        display_base = base.copy()
        for col in display_base.index:
            display_base[col] = map_value(col, display_base[col])

        st.dataframe(pd.DataFrame([display_base]), width="stretch")