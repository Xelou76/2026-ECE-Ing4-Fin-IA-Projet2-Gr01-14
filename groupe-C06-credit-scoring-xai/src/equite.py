# ============================================================
# dashboard.py-Streamlit interactif
# Credit Scoring XAI — Niveau Excellence
# Lancer : streamlit run src/dashboard.py.py
# ============================================================

import os
import json
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ============================================================
# CONFIG PAGE
# ============================================================

st.set_page_config(
    page_title='Credit Scoring XAI',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded',
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

/* Fairness table */
.fair-ok  { color: #22c55e; font-weight: 600; }
.fair-ko  { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

dataset = st.selectbox("Dataset", ["german", "lending_club"])

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data', 'processed', dataset)
RES_DIR  = os.path.join(BASE, '..', 'results', dataset)
MDL_DIR  = os.path.join(BASE, '..', 'models')

THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='IBM Plex Mono', color='#cbd5e1', size=11),
    xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
    yaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155'),
    margin=dict(l=10, r=10, t=35, b=10),
)

# ============================================================
# CHARGEMENT (mis en cache)
# ============================================================

@st.cache_resource
def load_all(dataset):
    try:
        X_train     = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
        X_test      = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
        X_test_raw  = pd.read_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'))
        X_train_raw = pd.read_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'))
        y_train     = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
        y_test      = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()
        model       = joblib.load(os.path.join(MDL_DIR, 'xgb_model.pkl'))

        with open(os.path.join(DATA_DIR, 'metadata.json')) as f:
            metadata = json.load(f)

        # Modèles optionnels
        extra = {}
        for name in ['lightgbm_model', 'logisticregression_model']:
            p = os.path.join(MDL_DIR, f'{name}.pkl')
            if os.path.exists(p):
                extra[name] = joblib.load(p)

        return dict(
            X_train=X_train, X_test=X_test,
            X_test_raw=X_test_raw, X_train_raw=X_train_raw,
            y_train=y_train, y_test=y_test,
            model=model, metadata=metadata, extra=extra
        )
    except FileNotFoundError:
        return None


@st.cache_resource
def compute_shap(_model, X_test_arr):
    exp  = shap.TreeExplainer(_model)
    sv   = exp.shap_values(X_test_arr)
    if isinstance(sv, list):
        sv = sv[1]
    ev = exp.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[1]
    return sv, exp, float(ev)


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

    page = st.radio('Navigation', [
        '📊  Vue d\'ensemble',
        '🔍  Explicabilité SHAP',
        '🟡  LIME & Comparaison',
        '🔄  Contrefactuels',
        '⚖️  Audit Fairness',
        '👤  Scoring Individuel',
    ])

    st.markdown('---')
    
    data = load_all(dataset)
    if data:
        m = data['metadata']
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

st.markdown("""
<div class='xai-header'>
    <h1>🏦 Credit Scoring — IA Explicable</h1>
    <p>{dataset.upper()} · XGBoost · SHAP · LIME · DiCE · Fairlearn · RGPD Art. 22</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# GARDE-FOU : fichiers manquants
# ============================================================

if data is None:
    st.error("""
**Fichiers manquants.** Lancez les scripts dans l'ordre :
```bash
python src/exploration.py
python src/modelisation.py.py
python src/explicabilite.py   # optionnel pour certaines pages
python src/equity.py         # optionnel pour la page Fairness
```
    """)
    st.stop()

# Raccourcis
X_train     = data['X_train']
X_test      = data['X_test']
X_test_raw  = data['X_test_raw']
X_train_raw = data['X_train_raw']
y_train     = data['y_train']
y_test      = data['y_test']
model       = data['model']
metadata    = data['metadata']
feature_names = metadata['feature_names']
num_features  = metadata['num_features']
cat_features  = metadata['cat_features']

X_test_arr  = X_test.values
X_train_arr = X_train.values

y_prob = model.predict_proba(X_test_arr)[:, 1]
y_pred = model.predict(X_test_arr)

shap_values, shap_explainer, expected_value = compute_shap(model, X_test_arr)

# ============================================================
# PAGE 1 — VUE D'ENSEMBLE
# ============================================================

if '📊' in page:
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix

    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    acc  = float((y_pred == y_test).mean())
    rate = float(y_pred.mean())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, fmt in [
        (c1, 'AUC-ROC',            auc,  '.4f'),
        (c2, 'F1-Score',           f1,   '.4f'),
        (c3, 'Accuracy',           acc,  '.4f'),
        (c4, 'Taux de défaut préd.', rate, '.1%'),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val:{fmt}}</div>
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Courbe ROC
    with col_l:
        st.markdown("<div class='sec-title'>Courbe ROC</div>", unsafe_allow_html=True)
        fpr_v, tpr_v, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_v, y=tpr_v, fill='tozeroy',
                                  fillcolor='rgba(56,189,248,0.1)',
                                  line=dict(color='#38bdf8', width=2),
                                  name=f'XGBoost (AUC={auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                  line=dict(color='#475569', dash='dash', width=1),
                                  name='Aléatoire', showlegend=True))
        fig.update_layout(**THEME, height=290,
                          xaxis_title='FPR', yaxis_title='TPR')
        st.plotly_chart(fig, use_container_width=True)

    # Distribution des scores
    with col_r:
        st.markdown("<div class='sec-title'>Distribution des scores</div>", unsafe_allow_html=True)
        df_dist = pd.DataFrame({
            'Score': y_prob,
            'Classe': y_test.map({0: 'Bon payeur', 1: 'Défaut'})
        })
        fig = px.histogram(df_dist, x='Score', color='Classe', nbins=40,
                           color_discrete_map={
                               'Bon payeur': '#22c55e',
                               'Défaut':     '#ef4444'
                           }, opacity=0.72, barmode='overlay')
        fig.update_layout(**THEME, height=290,
                          xaxis_title='Score de défaut (probabilité)',
                          yaxis_title='Fréquence')
        st.plotly_chart(fig, use_container_width=True)

    # Matrice de confusion + comparaison modèles
    col_cm, col_mod = st.columns([1, 2])

    with col_cm:
        st.markdown("<div class='sec-title'>Matrice de confusion</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            labels=dict(x='Prédit', y='Réel'),
            x=['Bon payeur', 'Défaut'],
            y=['Bon payeur', 'Défaut'],
            color_continuous_scale=[[0, '#0f172a'], [1, '#38bdf8']],
        )
        fig.update_layout(**THEME, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_mod:
        model_results_path = os.path.join(MDL_DIR, 'model_results.json')
        if os.path.exists(model_results_path):
            st.markdown("<div class='sec-title'>Comparaison des modèles</div>",
                        unsafe_allow_html=True)
            with open(model_results_path) as f:
                mr = json.load(f)
            df_mr = pd.DataFrame(mr['results']).T.reset_index()
            df_mr.columns = ['Modèle'] + list(df_mr.columns[1:])

            fig = go.Figure()
            colors_m = ['#38bdf8', '#22c55e', '#f59e0b']
            for metric, color in zip(['AUC-ROC', 'F1-Score', 'KS-Stat'], colors_m):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df_mr['Modèle'],
                    y=df_mr[metric].astype(float),
                    marker_color=color, opacity=0.85
                ))
            best = mr.get('best_model', '')
            fig.update_layout(**THEME, barmode='group', height=280,
                              title=f'🏆 Meilleur modèle : {best}',
                              legend=dict(orientation='h', y=1.12))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 — SHAP
# ============================================================

elif '🔍' in page:
    st.markdown("<div class='sec-title'>Importance globale — SHAP</div>",
                unsafe_allow_html=True)

    mean_shap = np.abs(shap_values).mean(axis=0)
    top_n = st.slider('Nombre de features affichées', 5,
                      min(20, len(feature_names)), 12)
    top_idx   = np.argsort(mean_shap)[::-1][:top_n]
    top_feats = [feature_names[i] for i in top_idx]
    top_vals  = mean_shap[top_idx]

    fig = go.Figure(go.Bar(
        x=top_vals[::-1], y=top_feats[::-1],
        orientation='h',
        marker=dict(
            color=top_vals[::-1],
            colorscale=[[0, '#1e293b'], [0.5, '#0284c7'], [1, '#38bdf8']],
            showscale=True,
            colorbar=dict(title='|SHAP|', tickfont=dict(size=9))
        )
    ))
    fig.update_layout(**THEME, height=420,
                      xaxis_title='Importance SHAP moyenne |Φ|')
    st.plotly_chart(fig, use_container_width=True)

    # Waterfall interactif
    st.markdown("<div class='sec-title'>Explication individuelle — Waterfall</div>",
                unsafe_allow_html=True)

    options = {
        f"Profil {i}  |  score={y_prob[i]:.3f}  |  réel={'Défaut' if y_test.iloc[i]==1 else 'Bon payeur'}": i
        for i in range(min(100, len(X_test)))
    }
    sel = st.selectbox('Choisir un profil', list(options.keys()))
    idx = options[sel]

    contribs = sorted(
        zip(feature_names, shap_values[idx]),
        key=lambda x: abs(x[1]), reverse=True
    )
    top_cf = contribs[:12]

    fig = go.Figure(go.Bar(
        x=[c[1] for c in top_cf],
        y=[c[0] for c in top_cf],
        orientation='h',
        marker_color=['#22c55e' if v > 0 else '#ef4444'
                      for _, v in top_cf],
        opacity=0.85,
    ))
    fig.add_vline(x=0, line_color='#475569', line_width=1)
    score_final = expected_value + sum(c[1] for c in contribs)
    fig.update_layout(
        **THEME, height=380,
        xaxis_title='Contribution SHAP',
        title=f'Score ≈ {score_final:.3f}  |  Base = {expected_value:.3f}'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Badge décision
    prob = y_prob[idx]
    if prob >= 0.5:
        st.markdown(f"<div class='badge badge-ko'>❌ Risque de défaut élevé — {prob:.1%}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge badge-ok'>✅ Faible risque de défaut — {prob:.1%}</div>",
                    unsafe_allow_html=True)

    st.markdown('<br>**Top 5 raisons de la décision :**', unsafe_allow_html=True)
    for feat, val in contribs[:5]:
        direction = '🟢 réduit le risque' if val < 0 else '🔴 augmente le risque'
        feat_val  = X_test_raw.iloc[idx].get(feat, X_test.iloc[idx].get(feat, 'N/A'))
        st.markdown(f'- `{feat}` = **{feat_val}** → {direction} ({val:+.4f})')

    # Scatter dépendance
    st.markdown("<div class='sec-title'>Dépendance SHAP d'une feature</div>",
                unsafe_allow_html=True)
    feat_sel = st.selectbox('Feature', feature_names, key='scatter_feat')
    fi       = feature_names.index(feat_sel)

    fig = go.Figure(go.Scatter(
        x=X_test_arr[:, fi],
        y=shap_values[:, fi],
        mode='markers',
        marker=dict(
            color=shap_values[:, fi],
            colorscale=[[0, '#ef4444'], [0.5, '#475569'], [1, '#22c55e']],
            size=5, opacity=0.55, showscale=True
        ),
    ))
    fig.add_hline(y=0, line_color='#475569', line_dash='dash')
    fig.update_layout(**THEME, height=290,
                      xaxis_title=f'{feat_sel} (normalisé)',
                      yaxis_title='Contribution SHAP')
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — LIME & COMPARAISON
# ============================================================

elif '🟡' in page:
    comp_path = os.path.join(RES_DIR, 'shap_vs_lime.csv')

    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        avg_c   = df_comp['concordance'].mean()

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Concordance moyenne</div>
        <div class='kpi-value'>{avg_c:.0%}</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Profils concordants ≥67%</div>
        <div class='kpi-value'>{(df_comp['concordance']>=0.67).sum()}/{len(df_comp)}</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='kpi-card'><div class='kpi-label'>Profils divergents</div>
        <div class='kpi-value'>{(df_comp['concordance']<0.67).sum()}/{len(df_comp)}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>Concordance par profil</div>",
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=df_comp['profil'],
            y=df_comp['concordance'],
            marker_color=['#22c55e' if c >= 0.67 else '#ef4444'
                          for c in df_comp['concordance']],
            opacity=0.85,
        ))
        fig.add_hline(y=avg_c, line_color='#fbbf24', line_dash='dash',
                      annotation_text=f'Moy. = {avg_c:.2f}',
                      annotation_font_color='#fbbf24')
        fig.add_hline(y=0.67, line_color='#64748b', line_dash='dot')
        fig.update_layout(**THEME, height=280,
                          xaxis_title='Profil', yaxis_title='Concordance Top-3')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='sec-title'>Détail par profil</div>",
                    unsafe_allow_html=True)
        st.dataframe(
            df_comp[['profil', 'pred_prob', 'shap_top3', 'lime_top3', 'concordance']],
            use_container_width=True, height=280
        )
    else:
        st.info('Lancez `explicabilite.py` pour générer la comparaison SHAP vs LIME.')

    st.markdown("<div class='sec-title'>Pourquoi comparer SHAP et LIME ?</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.markdown("""
    **SHAP** (Shapley Additive Explanations)
    - Basé sur la théorie des jeux coopératifs
    - Garantit équité, efficacité, symétrie (axiomes de Shapley)
    - Exact pour les modèles arbre (`TreeExplainer`)
    - Cohérent globalement et localement
    """)
    col_b.markdown("""
    **LIME** (Local Interpretable Model-Agnostic Explanations)
    - Approximation locale par modèle linéaire
    - Agnostique au modèle — fonctionne sur tout
    - Plus rapide mais moins précis
    - Peut diverger sur les features corrélées
    """)
    st.info("En scoring crédit, **la concordance SHAP/LIME renforce la confiance** "
            "dans l'explication donnée au client (RGPD Art. 22). "
            "Une divergence signale souvent une feature corrélée ou un profil non-linéaire.")

# ============================================================
# PAGE 4 — CONTREFACTUELS
# ============================================================

elif '🔄' in page:
    st.markdown("""
    <div class='sec-title'>Explications Contrefactuelles — RGPD Art. 22</div>
    <p style='color:#64748b;font-size:0.88rem;margin-bottom:1rem;'>
    <em>"Que faudrait-il modifier pour ne plus être en défaut ?"</em><br>
    Tout client refusé a droit à une explication actionnable.
    </p>
    """, unsafe_allow_html=True)

    cf_path = os.path.join(RES_DIR, 'contrefactuels.csv')
    if os.path.exists(cf_path):
        df_cf = pd.read_csv(cf_path)
        st.dataframe(df_cf, use_container_width=True, height=220)
    else:
        st.info('Lancez `explicabilite.py` pour générer les contrefactuels.')

    # Simulateur interactif
    st.markdown("<div class='sec-title'>Simulateur interactif de profil</div>",
                unsafe_allow_html=True)
    st.markdown('Ajustez les variables pour voir l\'impact sur le score :')

    display_feats = [f for f in num_features if f in X_test_raw.columns][:6]
    profile_vals  = {}
    cols3 = st.columns(3)

    for i, feat in enumerate(display_feats):
        col = cols3[i % 3]
        vmin = float(X_test_raw[feat].quantile(0.02))
        vmax = float(X_test_raw[feat].quantile(0.98))
        vmed = float(X_test_raw[feat].median())
        profile_vals[feat] = col.slider(
            feat,
            min_value=round(vmin, 1),
            max_value=round(vmax, 1),
            value=round(vmed, 1),
            key=f'sim_{feat}'
        )

    # Construire le profil et prédire
    base_raw = X_test_raw.iloc[0].copy()
    for feat, val in profile_vals.items():
        base_raw[feat] = val

    from sklearn.compose import ColumnTransformer
    preprocessor = joblib.load(os.path.join(MDL_DIR, 'preprocessor.pkl'))
    profile_df   = pd.DataFrame([base_raw])
    profile_proc = preprocessor.transform(profile_df)
    sim_score    = float(model.predict_proba(profile_proc)[0, 1])

    col_s, col_g = st.columns([1, 2])
    with col_s:
        st.markdown('<br>', unsafe_allow_html=True)
        if sim_score >= 0.5:
            st.markdown(f"<div class='badge badge-ko'>❌ Risque élevé — {sim_score:.1%}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge badge-ok'>✅ Faible risque — {sim_score:.1%}</div>",
                        unsafe_allow_html=True)

    with col_g:
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=sim_score * 100,
            number={'suffix': '%',
                    'font': {'family': 'IBM Plex Mono', 'color': '#cbd5e1'}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#334155'),
                bar=dict(color='#38bdf8'),
                bgcolor='#1e293b',
                steps=[
                    {'range': [0,  50], 'color': 'rgba(34,197,94,0.12)'},
                    {'range': [50, 100],'color': 'rgba(239,68,68,0.12)'},
                ],
                threshold=dict(line=dict(color='#fbbf24', width=2), value=50)
            )
        ))
        fig.update_layout(**THEME, height=200)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5 — AUDIT FAIRNESS
# ============================================================

elif '⚖️' in page:
    st.markdown("""
    <div class='sec-title'>Audit de Fairness — Biais âge & genre</div>
    <p style='color:#64748b;font-size:0.88rem;'>
    Seuil Disparate Impact Ratio : <b style='color:#fbbf24;'>≥ 0.80</b>
    (règle des 4/5 — EEOC / CJUE)
    </p>
    """, unsafe_allow_html=True)

    # ============================================================
# GROUPES SENSIBLES (ADAPTÉS AU DATASET)
# ============================================================

if dataset == "german":
    # Âge réel
    age_group = np.where(
        X_test_raw['age'].values < 30,
        'Jeune (<30)', 'Senior (≥30)'
    )

    # Genre depuis statut_civil_sexe
    gender_g = np.where(
        X_test_raw['statut_civil_sexe'].astype(str).isin(['A92', 'A95']),
        'Femme', 'Homme'
    )

    groups_dict = {
        'Âge': age_group,
        'Genre': gender_g
    }

elif dataset == "lending_club":
    # Lending Club n'a PAS de genre → on utilise revenu comme proxy
    income_group = np.where(
        X_test_raw['annual_inc'] < X_test_raw['annual_inc'].median(),
        'Low income', 'High income'
    )

    # On garde 2 variables pour garder la structure du dashboard
    groups_dict = {
        'Revenu': income_group,
        'Revenu (bis)': income_group
    }

    from sklearn.metrics import confusion_matrix as cm_fn

    def grp_metrics(groups):
        rows = []
        for g in np.unique(groups):
            mask = groups == g
            yt   = y_test[mask]
            yp   = y_pred[mask]
            ypr  = y_prob[mask]
            if len(yt) < 5 or yt.nunique() < 2:
                continue
            tn, fp, fn, tp = cm_fn(yt, yp, labels=[0, 1]).ravel()
            rows.append({
                'Groupe':             g,
                'N':                  int(mask.sum()),
                'Taux défaut réel':   round(float(yt.mean()), 3),
                'Taux défaut préd.':  round(float(yp.mean()), 3),
                'TPR':                round(tp/(tp+fn) if tp+fn>0 else 0, 3),
                'FPR':                round(fp/(fp+tn) if fp+tn>0 else 0, 3),
                'AUC':                round(float(roc_auc_score(yt, ypr)), 3),
            })
        return pd.DataFrame(rows)

    # KPI DI
    kpi_cols = st.columns(2)
    for i, (var, grps) in enumerate(groups_dict.items()):
        df_g  = grp_metrics(grps)
        acc_r = 1 - df_g['Taux défaut préd.']
        di    = acc_r.min() / acc_r.max() if acc_r.max() > 0 else 0
        conf  = di >= 0.80
        color = '#22c55e' if conf else '#ef4444'
        status = '✅ Conforme' if conf else '❌ Non-conforme'
        kpi_cols[i].markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>DI Ratio — {var}</div>
            <div class='kpi-value' style='color:{color}'>{di:.3f}</div>
            <div class='kpi-sub'>{status} | seuil = 0.80</div>
        </div>""", unsafe_allow_html=True)

    # Graphiques par variable
    for var, grps in groups_dict.items():
        df_g = grp_metrics(grps)
        if df_g.empty:
            continue

        st.markdown(f"<div class='sec-title'>Variable : {var}</div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Taux réel', x=df_g['Groupe'],
                y=df_g['Taux défaut réel'], marker_color='#38bdf8', opacity=0.85))
            fig.add_trace(go.Bar(
                name='Taux prédit', x=df_g['Groupe'],
                y=df_g['Taux défaut préd.'], marker_color='#ef4444', opacity=0.85))
            fig.add_hline(y=float(y_test.mean()), line_dash='dash',
                          line_color='#94a3b8',
                          annotation_text=f'Moy. ({y_test.mean():.2f})')
            fig.update_layout(**THEME, barmode='group', height=260,
                              title='Demographic Parity',
                              yaxis_title='Taux de défaut')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='TPR', x=df_g['Groupe'],
                y=df_g['TPR'], marker_color='#22c55e', opacity=0.85))
            fig.add_trace(go.Bar(
                name='FPR', x=df_g['Groupe'],
                y=df_g['FPR'], marker_color='#ef4444', opacity=0.85))
            fig.update_layout(**THEME, barmode='group', height=260,
                              title='Equalized Odds',
                              yaxis_title='Taux', yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_g, use_container_width=True)

    # Résultats Fairlearn si disponibles
    fl_path = os.path.join(RES_DIR, 'fairness_metrics.csv')
    if os.path.exists(fl_path):
        st.markdown("<div class='sec-title'>Métriques complètes (Fairlearn)</div>",
                    unsafe_allow_html=True)
        df_fl = pd.read_csv(fl_path)
        st.dataframe(df_fl, use_container_width=True)

    to_path = os.path.join(RES_DIR, 'threshold_optimization.csv')
    if os.path.exists(to_path):
        st.markdown("<div class='sec-title'>Threshold Optimization — Avant / Après</div>",
                    unsafe_allow_html=True)
        df_to = pd.read_csv(to_path)
        st.dataframe(df_to, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name='EO diff avant', x=df_to['variable'],
                              y=df_to['EO_diff_avant'],
                              marker_color='#ef4444', opacity=0.85))
        fig.add_trace(go.Bar(name='EO diff après', x=df_to['variable'],
                              y=df_to['EO_diff_apres'],
                              marker_color='#22c55e', opacity=0.85))
        fig.update_layout(**THEME, barmode='group', height=260,
                          yaxis_title='Equalized Odds Diff (↓ meilleur)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 6 — SCORING INDIVIDUEL
# ============================================================

elif '👤' in page:
    st.markdown("""
    <div class='sec-title'>Scoring d'un nouveau client</div>
    <p style='color:#64748b;font-size:0.88rem;'>
    Saisissez les données du client — le score et l'explication RGPD sont générés instantanément.
    </p>
    """, unsafe_allow_html=True)

    preprocessor = joblib.load(os.path.join(MDL_DIR, 'preprocessor.pkl'))

    # Formulaire numérique
    st.markdown('**Données numériques**')
    num_inputs = {}
    cols_n = st.columns(4)
    for i, feat in enumerate(num_features[:8]):
        col = cols_n[i % 4]
        if feat in X_test_raw.columns:
            vmin = float(X_test_raw[feat].quantile(0.01))
            vmax = float(X_test_raw[feat].quantile(0.99))
            vmed = float(X_test_raw[feat].median())
            num_inputs[feat] = col.number_input(
                feat, min_value=round(vmin,1),
                max_value=round(vmax,1),
                value=round(vmed,1), key=f'num_{feat}'
            )

    # Formulaire catégoriel
    if cat_features:
        st.markdown('**Données catégorielles**')
        cat_inputs = {}
        cols_c = st.columns(4)
        for i, feat in enumerate(cat_features[:8]):
            col = cols_c[i % 4]
            if feat in X_test_raw.columns:
                options = sorted(X_test_raw[feat].dropna().unique().tolist())
                cat_inputs[feat] = col.selectbox(feat, options, key=f'cat_{feat}')
    else:
        cat_inputs = {}

    if st.button('🔍 Calculer le score', type='primary'):
        base = X_test_raw.iloc[0].copy()
        for feat, val in num_inputs.items():
            base[feat] = val
        for feat, val in cat_inputs.items():
            base[feat] = val

        prof_df   = pd.DataFrame([base])
        prof_proc = preprocessor.transform(prof_df)
        score     = float(model.predict_proba(prof_proc)[0, 1])

        # SHAP local
        shap_ind = shap.TreeExplainer(model).shap_values(prof_proc)
        if isinstance(shap_ind, list):
            shap_ind = shap_ind[1]
        shap_arr = shap_ind[0]
        contribs = sorted(zip(feature_names, shap_arr),
                          key=lambda x: abs(x[1]), reverse=True)

        st.markdown('---')
        col_res, col_exp = st.columns(2)

        with col_res:
            if score >= 0.5:
                st.markdown(
                    f"<div class='badge badge-ko' style='font-size:1.1rem;padding:0.8rem 1.6rem;'>"
                    f"❌ Risque de défaut élevé<br>{score:.1%}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='badge badge-ok' style='font-size:1.1rem;padding:0.8rem 1.6rem;'>"
                    f"✅ Faible risque de défaut<br>{score:.1%}</div>",
                    unsafe_allow_html=True
                )

            # Gauge
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=score * 100,
                number={'suffix': '%',
                        'font': {'family': 'IBM Plex Mono', 'color': '#cbd5e1'}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor='#334155'),
                    bar=dict(color='#38bdf8'),
                    bgcolor='#1e293b',
                    steps=[
                        {'range': [0,  50], 'color': 'rgba(34,197,94,0.1)'},
                        {'range': [50, 100],'color': 'rgba(239,68,68,0.1)'},
                    ],
                    threshold=dict(
                        line=dict(color='#fbbf24', width=2), value=50)
                )
            ))
            fig.update_layout(**THEME, height=220)
            st.plotly_chart(fig, use_container_width=True)

        with col_exp:
            st.markdown('**Explication RGPD (Art. 22) :**')
            for feat, val in contribs[:6]:
                icon   = '🟢' if val < 0 else '🔴'
                impact = 'réduit le risque' if val < 0 else 'augmente le risque'
                feat_v = base.get(feat, '—')
                st.markdown(f'- {icon} `{feat}` = **{feat_v}** → {impact} ({val:+.4f})')

            if score >= 0.5:
                st.markdown('<br>**Leviers d\'amélioration :**',
                            unsafe_allow_html=True)
                bad = [(f, v) for f, v in contribs if v > 0][:3]
                for feat, _ in bad:
                    if feat in num_features and feat in X_test_raw.columns:
                        target = float(
                            X_test_raw[y_test == 0][feat].median()
                        )
                        st.markdown(
                            f'→ Améliorer **{feat}** vers `{target:.1f}` '
                            f'(médiane bon payeur)'
                        )