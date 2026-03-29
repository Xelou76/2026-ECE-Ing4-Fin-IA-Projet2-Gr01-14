"""
app.py
======
Application Streamlit — Visualisation des résultats du projet
Market Regime Detection (VAE-HMM) | Groupe 01

Lancement :
    streamlit run app.py

Structure des dossiers attendue :
    market_regime/
    ├── app.py                  ← ce fichier
    ├── results/
    │   ├── figures/            ← images PNG générées par main.py
    │   │   ├── 01_regimes_on_price.png
    │   │   ├── 02_equity_curves.png
    │   │   └── ...
    │   └── comparison_report.json
    └── results/run_*.log       ← fichiers de logs
"""

import json
import os
from pathlib import Path

import streamlit as st

# ===========================================================================
# CONFIGURATION GLOBALE
# ===========================================================================

# --- Chemins (à ajuster si votre structure diffère) ---
FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")

# --- Métadonnées des figures générées par main.py ---
# Format : { "nom_fichier.png" : ("Titre affiché", "Description explicative") }
FIGURE_METADATA: dict[str, tuple[str, str]] = {
    "01_regimes_on_price.png": (
        "Régimes de Marché sur le Cours SPY",
        "Visualisation des régimes de marché (Bull, Transition, Bear) détectés par le "
        "pipeline VAE-HMM sur le cours de clôture ajusté du SPY. Les fonds colorés "
        "représentent les périodes classifiées dans chaque régime : vert (Bull / "
        "basse volatilité), orange (Transition / incertitude), rouge (Bear / haute "
        "volatilité). Le sous-graphique inférieur montre les probabilités a posteriori "
        "P(régime_t | données) calculées par l'algorithme Forward-Backward du HMM.",
    ),
    "02_equity_curves.png": (
        "Comparaison des Equity Curves",
        "Évolution de la valeur de chaque stratégie (rebased à 100 au départ du "
        "test set). La stratégie VAE-HMM alloue dynamiquement entre equity, "
        "obligations et cash selon le régime détecté. Elle est comparée à la "
        "stratégie Markov-Switching (Hamilton, 1989) et au benchmark Buy-and-Hold "
        "(100% SPY). Le sous-graphique inférieur montre les drawdowns simultanés.",
    ),
    "03_transition_matrix.png": (
        "Matrice de Transition du HMM",
        "Matrice stochastique A du GaussianHMM ajusté sur l'espace latent du VAE. "
        "Chaque cellule A[i,j] représente la probabilité de transiter du régime i "
        "au régime j entre deux jours consécutifs. Les valeurs diagonales élevées "
        "(> 0.98) confirment que les régimes sont persistants — une propriété "
        "cruciale pour la validité économique du modèle.",
    ),
    "04_regime_distribution.png": (
        "Distribution des Régimes Détectés",
        "Comparaison de la distribution des régimes entre le modèle VAE-HMM (3 "
        "régimes) et la baseline Markov-Switching de Hamilton (2 régimes). Le modèle "
        "VAE-HMM capture une structure plus fine grâce à la représentation latente "
        "apprise par le LSTM-VAE, permettant d'identifier un régime intermédiaire "
        "de Transition en plus des états Bull et Bear.",
    ),
    "05_drawdowns.png": (
        "Drawdowns — Underwater Plot",
        "Représentation des pertes cumulatives de chaque stratégie relativement à "
        "leur maximum historique. Un drawdown de -20% signifie que le portefeuille "
        "vaut 20% de moins que son pic. La stratégie VAE-HMM vise à réduire ces "
        "épisodes de perte en réduisant l'exposition aux actions lors des régimes "
        "Bear détectés.",
    ),
    "06_rolling_sharpe.png": (
        "Sharpe Ratio Glissant (63 jours)",
        "Ratio de Sharpe annualisé calculé sur une fenêtre glissante de 63 jours "
        "ouvrés (≈ 1 trimestre). Un Sharpe positif indique que la stratégie génère "
        "un return supérieur au taux sans risque. Cette vue temporelle révèle les "
        "périodes où la détection de régime apporte un avantage informationnel "
        "mesurable face au benchmark passif.",
    ),
    "07_monthly_returns.png": (
        "Heatmap des Returns Mensuels",
        "Vue calendaire des returns mensuels de la stratégie VAE-HMM (en %). Les "
        "cellules vertes indiquent des mois profitables, les rouges des mois en "
        "perte. Cette visualisation permet d'identifier rapidement les années de "
        "surperformance et les périodes de stress de marché (2008, 2020) où la "
        "stratégie adaptative devrait avoir réduit les pertes.",
    ),
    "08_latent_space.png": (
        "Espace Latent VAE — Projection PCA 2D",
        "Projection en 2D (via PCA) des représentations latentes μ extraites par "
        "le LSTM-VAE sur le test set. Chaque point représente un jour de trading, "
        "coloré selon le régime assigné par le HMM. Des clusters bien séparés "
        "indiquent que le VAE a appris une représentation discriminante des états "
        "de marché dans l'espace latent de dimension 8.",
    ),
    "09_training_history.png": (
        "Courbes d'Apprentissage du VAE",
        "Évolution des métriques d'entraînement du LSTM-VAE epoch par epoch. À "
        "gauche : l'ELBO totale (train vs validation). Au centre : la reconstruction "
        "loss (MSE). À droite : la KL divergence avec le β-schedule (KL annealing "
        "de 0 → 1 sur 50 epochs). L'early stopping arrête l'entraînement lorsque "
        "la validation loss ne s'améliore plus.",
    ),
    "10_full_dashboard.png": (
        "Dashboard Complet — Vue Synthétique",
        "Vue agrégée 3×2 combinant les visualisations clés du projet : régimes sur "
        "le prix, matrice de transition, equity curves, drawdowns, Sharpe glissant "
        "et distribution des régimes. Cette figure est conçue pour être insérée "
        "directement dans un rapport ou une présentation.",
    ),
}

# ===========================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ===========================================================================

st.set_page_config(
    page_title="Market Regime Detection — VAE-HMM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CSS PERSONNALISÉ
# ===========================================================================

st.markdown("""
<style>
    /* Police principale */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Fond général */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Titres */
    h1 { color: #58a6ff !important; font-family: 'IBM Plex Mono', monospace !important; }
    h2 { color: #79c0ff !important; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
    h3 { color: #c9d1d9 !important; }

    /* Cards métriques */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #58a6ff;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-value.positive { color: #3fb950; }
    .metric-value.negative { color: #f85149; }
    .metric-value.neutral  { color: #ffa657; }

    /* Badge régime */
    .regime-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        margin: 2px;
    }
    .badge-bull  { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; }
    .badge-trans { background: #3a2a0a; color: #ffa657; border: 1px solid #ffa657; }
    .badge-bear  { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; }

    /* Caption sous les images */
    .fig-caption {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 10px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #8b949e;
        margin-top: 6px;
        line-height: 1.6;
    }

    /* Séparateur */
    hr { border-color: #30363d !important; }

    /* Selectbox */
    .stSelectbox label { color: #c9d1d9 !important; }

    /* Bouton download */
    .stDownloadButton button {
        background: #1f2937;
        border: 1px solid #30363d;
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# FONCTIONS UTILITAIRES
# ===========================================================================

def load_report() -> dict | None:
    """Charge le rapport JSON de comparaison des modèles si disponible."""
    report_path = RESULTS_DIR / "comparison_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_available_figures() -> list[Path]:
    """Retourne la liste triée des images PNG disponibles dans FIGURES_DIR."""
    if not FIGURES_DIR.exists():
        return []
    return sorted(FIGURES_DIR.glob("*.png"))


def get_log_files() -> list[Path]:
    """Retourne les fichiers de log disponibles."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.log"))


def metric_card(label: str, value: str, color_class: str = "") -> str:
    """Génère le HTML d'une carte de métrique."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
    </div>
    """


def fmt_pct(value: float | None, invert: bool = False) -> tuple[str, str]:
    """
    Formate une valeur flottante en pourcentage avec couleur.

    Parameters
    ----------
    value : float | None
        Valeur numérique (ex: 0.145 → "14.5%").
    invert : bool
        Si True, une valeur négative est "positive" (ex: drawdown).

    Returns
    -------
    tuple[str, str]
        (texte formaté, classe CSS de couleur)
    """
    if value is None:
        return "N/A", "neutral"
    pct = value * 100
    text = f"{pct:+.2f}%"
    if invert:
        cls = "positive" if pct > 0 else ("negative" if pct < 0 else "neutral")
    else:
        cls = "positive" if pct > 0 else ("negative" if pct < 0 else "neutral")
    return text, cls


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("## 📈 VAE-HMM")
    st.markdown("**Market Regime Detection**")
    st.markdown("<span class='regime-badge badge-bull'>● Bull</span>"
                "<span class='regime-badge badge-trans'>● Transition</span>"
                "<span class='regime-badge badge-bear'>● Bear</span>",
                unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Accueil & Introduction",
            "📊  Résultats Visuels",
            "📋  Métriques de Performance",
            "📁  Fichiers & Logs",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Infos du projet
    st.markdown("**Projet**")
    st.caption("Détection de Régimes de Marché")
    st.caption("Pipeline : VAE → HMM → Backtest")

    st.markdown("**Modèles**")
    st.caption("LSTM-VAE (PyTorch)")
    st.caption("GaussianHMM (hmmlearn)")
    st.caption("Baseline : Hamilton (1989)")

    st.markdown("**Données**")
    st.caption("SPY · TLT · GLD · ^VIX")
    st.caption("2005 – 2024 · Fréquence J")

    st.markdown("---")
    figures = get_available_figures()
    report = load_report()
    st.caption(f"✅ {len(figures)} figure(s) disponible(s)")
    st.caption(f"{'✅' if report else '⚠️'} Rapport JSON {'chargé' if report else 'absent'}")


# ===========================================================================
# PAGE 1 — ACCUEIL & INTRODUCTION
# ===========================================================================

if "Accueil" in page:

    st.markdown("# Market Regime Detection")
    st.markdown("### Approche Hybride LSTM-VAE + Hidden Markov Model")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 🎯 Objectif du Projet")
        st.markdown("""
        Ce projet implémente un système de **détection automatique de régimes de marché**
        en combinant deux approches complémentaires de l'apprentissage automatique :

        1. **Un Variational Autoencoder (LSTM-VAE)** apprend une représentation latente
           compacte des conditions de marché à partir de séries temporelles financières
           multivariées. L'encodeur compresse 30 jours de features (rendements,
           volatilités, indicateurs techniques) en un vecteur de dimension 8.

        2. **Un Hidden Markov Model (GaussianHMM)** modélise les transitions entre
           régimes discrets dans cet espace latent. Les émissions gaussiennes capturent
           la structure de covariance de chaque régime de marché.

        3. **Une stratégie de trading adaptative** ajuste l'allocation du portefeuille
           (equity / obligations / cash) en temps réel selon le régime détecté, avec
           des coûts de transaction réalistes (10 bps par rebalancement).
        """)

        st.markdown("## 🔬 Pipeline Technique")

        steps = [
            ("📥", "Ingestion des Données",
             "Téléchargement via `yfinance` pour SPY, TLT, GLD et ^VIX sur 2005–2024 "
             "(~5 000 jours de trading). Mise en cache locale pour éviter les appels "
             "redondants à l'API."),
            ("⚙️", "Feature Engineering",
             "Construction de 46 features : log-rendements multi-horizons (1j, 5j, 21j), "
             "volatilités réalisées annualisées (5j, 21j, 63j), RSI(14), Bandes de "
             "Bollinger, tendance EMA50/200, ratios de volatilité et corrélations "
             "glissantes inter-actifs."),
            ("🧠", "Entraînement du LSTM-VAE",
             "Architecture LSTM bidirectionnel (2 couches, 128 unités) avec KL annealing "
             "(β : 0 → 1 sur 50 epochs), early stopping (patience 20) et gradient "
             "clipping. Split chronologique strict : 70% train / 15% val / 15% test."),
            ("🎯", "Fitting du HMM",
             "GaussianHMM à 3 régimes avec matrice de covariance `full`, ajusté sur les "
             "représentations latentes μ du VAE (train set). 10 restarts EM pour la "
             "robustesse. Tri canonique des régimes par variance croissante."),
            ("📈", "Backtest & Évaluation",
             "Backtest vectorisé (NumPy) de la stratégie adaptative sur le test set. "
             "Comparaison scientifique avec la baseline Markov-Switching (Hamilton, 1989) "
             "et Buy-and-Hold. Métriques : Sharpe, Sortino, Calmar, MaxDD, IC."),
        ]

        for icon, title, desc in steps:
            with st.expander(f"{icon}  {title}", expanded=False):
                st.markdown(desc)

    with col2:
        st.markdown("## 🗂️ Architecture")
        st.code("""market_regime/
├── config/        # Pydantic settings
├── data/
│   ├── downloader.py
│   ├── features.py
│   └── processor.py
├── models/
│   ├── vae.py     # LSTM-VAE
│   ├── trainer.py
│   ├── hmm.py
│   └── markov_switching.py
├── strategy/
│   └── backtester.py
├── evaluation/
│   └── comparator.py
├── utils/
│   ├── metrics.py
│   └── plotting.py
└── main.py""", language="text")

        st.markdown("## 📦 Stack Technique")
        techs = [
            ("🔥", "PyTorch", "LSTM-VAE"),
            ("🔗", "hmmlearn", "GaussianHMM"),
            ("📊", "statsmodels", "Baseline Hamilton"),
            ("💹", "yfinance", "Données marché"),
            ("⚡", "NumPy", "Backtest vectorisé"),
            ("📐", "scikit-learn", "RobustScaler, PCA"),
            ("🎨", "Matplotlib", "Visualisations"),
            ("⚙️", "Pydantic v2", "Configuration"),
        ]
        for icon, lib, usage in techs:
            st.markdown(
                f"<div class='metric-card' style='padding:10px 14px;margin:4px 0'>"
                f"<span style='color:#58a6ff;font-family:monospace'>{icon} {lib}</span>"
                f"<br><span style='font-size:0.8rem;color:#8b949e'>{usage}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## 📚 Références")
    refs = [
        "Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. ICLR.",
        "Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary "
        "Time Series and the Business Cycle*. Econometrica, 57(2).",
        "Higgins, I. et al. (2017). *β-VAE: Learning Basic Visual Concepts with a "
        "Constrained Variational Framework*. ICLR.",
        "Jobson, J. D., & Korkie, B. (1981). *Performance Hypothesis Testing with the "
        "Sharpe and Treynor Measures*. Journal of Finance.",
    ]
    for ref in refs:
        st.markdown(f"- {ref}")


# ===========================================================================
# PAGE 2 — RÉSULTATS VISUELS
# ===========================================================================

elif "Résultats" in page:

    st.markdown("## 📊 Résultats Visuels")
    st.markdown(
        "Toutes les figures ci-dessous sont générées automatiquement par `main.py` "
        "et sauvegardées dans `results/figures/`. Elles couvrent l'ensemble du pipeline : "
        "détection des régimes, performance du backtest et analyse de l'espace latent."
    )
    st.markdown("---")

    figures = get_available_figures()

    if not figures:
        st.warning(
            "⚠️ Aucune image trouvée dans `results/figures/`. "
            "Lancez d'abord `python main.py` pour générer les figures.",
            icon="⚠️",
        )
        st.markdown("**Figures attendues après exécution :**")
        for fname, (title, _) in FIGURE_METADATA.items():
            st.markdown(f"- `{fname}` — {title}")
        st.stop()

    # --- Sélecteur de vue ---
    col_mode, col_select = st.columns([1, 3])
    with col_mode:
        view_mode = st.radio(
            "Mode d'affichage",
            ["Défilement complet", "Sélection individuelle"],
            horizontal=False,
        )

    # --- MODE DÉFILEMENT : toutes les figures en séquence ---
    if view_mode == "Défilement complet":

        with col_select:
            categories = {
                "Toutes les figures": None,
                "Régimes & Marché": ["01", "03", "04", "08"],
                "Performance & Backtest": ["02", "05", "06", "07"],
                "Apprentissage VAE": ["09"],
                "Dashboard": ["10"],
            }
            cat_choice = st.selectbox(
                "Filtrer par catégorie", list(categories.keys())
            )

        prefix_filter = categories[cat_choice]

        for fig_path in figures:
            fname = fig_path.name
            # Filtre par catégorie
            if prefix_filter is not None:
                if not any(fname.startswith(p) for p in prefix_filter):
                    continue

            title, description = FIGURE_METADATA.get(
                fname, (fname.replace(".png", "").replace("_", " ").title(), "")
            )

            st.markdown(f"### {title}")
            st.image(str(fig_path), width='stretch')

            if description:
                st.markdown(
                    f"<div class='fig-caption'>📝 {description}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("")

    # --- MODE SÉLECTION : une figure à la fois ---
    else:
        with col_select:
            fig_names = [f.name for f in figures]
            display_names = [
                FIGURE_METADATA.get(n, (n, ""))[0]
                for n in fig_names
            ]
            name_map = dict(zip(display_names, fig_names))

            selected_display = st.selectbox(
                "Choisir une figure",
                display_names,
                format_func=lambda x: f"  {x}",
            )

        selected_fname = name_map[selected_display]
        selected_path = FIGURES_DIR / selected_fname

        if selected_path.exists():
            title, description = FIGURE_METADATA.get(
                selected_fname, (selected_display, "")
            )
            st.markdown(f"### {title}")
            st.image(str(selected_path), width='stretch')

            if description:
                st.markdown(
                    f"<div class='fig-caption'>📝 {description}</div>",
                    unsafe_allow_html=True,
                )

            # Navigation précédent / suivant
            idx = fig_names.index(selected_fname)
            col_prev, col_info, col_next = st.columns([1, 2, 1])
            with col_prev:
                if idx > 0:
                    if st.button("← Précédent"):
                        st.session_state["selected"] = display_names[idx - 1]
            with col_info:
                st.caption(f"Figure {idx + 1} / {len(figures)}")
            with col_next:
                if idx < len(figures) - 1:
                    if st.button("Suivant →"):
                        st.session_state["selected"] = display_names[idx + 1]


# ===========================================================================
# PAGE 3 — MÉTRIQUES DE PERFORMANCE
# ===========================================================================

elif "Métriques" in page:

    st.markdown("## 📋 Métriques de Performance")
    st.markdown(
        "Tableau comparatif des stratégies sur le **test set** (15% des données, "
        "jamais vues pendant l'entraînement). Toutes les métriques sont annualisées "
        "sur 252 jours de trading."
    )
    st.markdown("---")

    report = load_report()

    if report is None:
        st.warning(
            "⚠️ `results/comparison_report.json` introuvable. "
            "Exécutez d'abord `python main.py`.",
            icon="⚠️",
        )
        st.stop()

    metrics_data = report.get("metrics", {})

    if not metrics_data:
        st.error("Le rapport ne contient pas de métriques.")
        st.stop()

    # --- Tableau comparatif ---
    st.markdown("### 📊 Vue Comparative")

    METRIC_LABELS = {
        "annualized_return":    ("CAGR",              "annualized_return",  True),
        "annualized_volatility":("Volatilité ann.",    "annualized_volatility", False),
        "sharpe_ratio":         ("Sharpe Ratio",       "sharpe_ratio",       True),
        "sortino_ratio":        ("Sortino Ratio",      "sortino_ratio",      True),
        "calmar_ratio":         ("Calmar Ratio",       "calmar_ratio",       True),
        "max_drawdown":         ("Max Drawdown",       "max_drawdown",       False),
        "total_return":         ("Return Total",       "total_return",       True),
        "win_rate":             ("Win Rate",           "win_rate",           True),
    }

    model_names = list(metrics_data.keys())
    n_models = len(model_names)
    cols = st.columns(n_models)

    COLOR_MAP = {
        "VAE-HMM":                 "#58a6ff",
        "Markov-Switching":        "#f78166",
        "Buy & Hold":              "#ffa657",
    }

    for col, model_name in zip(cols, model_names):
        color = next(
            (v for k, v in COLOR_MAP.items() if k in model_name),
            "#79c0ff",
        )
        m = metrics_data[model_name]

        col.markdown(
            f"<div style='text-align:center;padding:8px;background:#161b22;"
            f"border-top:3px solid {color};border-radius:4px;margin-bottom:12px'>"
            f"<span style='color:{color};font-weight:600;font-family:monospace'>"
            f"{model_name}</span></div>",
            unsafe_allow_html=True,
        )

        for key, (label, field, higher_is_better) in METRIC_LABELS.items():
            val = m.get(field)
            if val is None:
                text, cls = "N/A", "neutral"
            elif field in ("sharpe_ratio", "sortino_ratio", "calmar_ratio"):
                text = f"{val:+.3f}"
                cls = "positive" if val > 0 else "negative"
            elif field == "win_rate":
                text = f"{val:.1%}"
                cls = "positive" if val > 0.5 else "neutral"
            else:
                text = f"{val:+.2%}"
                cls = "positive" if val > 0 else "negative"
                if field == "max_drawdown":
                    cls = "negative" if val < -0.15 else "neutral"

            col.markdown(
                metric_card(label, text, cls),
                unsafe_allow_html=True,
            )

    # --- Test statistique ---
    st.markdown("---")
    st.markdown("### 🔬 Test de Significativité (Jobson-Korkie)")
    jk = report.get("sharpe_significance_test", {})
    if jk:
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(metric_card("ΔSharpe", f"{jk.get('delta_sharpe', 0):+.4f}",
                                  "positive" if jk.get('delta_sharpe', 0) > 0 else "negative"),
                      unsafe_allow_html=True)
        col2.markdown(metric_card("z-statistic", f"{jk.get('z_statistic', 0):.3f}", "neutral"),
                      unsafe_allow_html=True)
        col3.markdown(metric_card("p-value", f"{jk.get('p_value', 1):.4f}", "neutral"),
                      unsafe_allow_html=True)
        sig = jk.get("significant_5pct", False)
        col4.markdown(metric_card("Significatif (5%)", "OUI ★" if sig else "non",
                                  "positive" if sig else "neutral"),
                      unsafe_allow_html=True)

        st.markdown(
            "<div class='fig-caption'>📝 H₀ : Sharpe(VAE-HMM) = Sharpe(Buy-and-Hold). "
            "Un p-value < 0.05 indique que la surperformance en Sharpe est statistiquement "
            "significative et non due au hasard (test de Jobson-Korkie, 1981).</div>",
            unsafe_allow_html=True,
        )

    # --- IC ---
    st.markdown("---")
    st.markdown("### 📡 Information Coefficient du Signal de Régime")
    ic_data = report.get("information_coefficient", {})
    if ic_data:
        col1, col2 = st.columns(2)
        ic1 = ic_data.get("ic_1d", 0)
        ic5 = ic_data.get("ic_5d", 0)
        col1.markdown(
            metric_card("IC à 1 jour", f"{ic1:.4f}",
                        "positive" if ic1 > 0 else "negative"),
            unsafe_allow_html=True,
        )
        col2.markdown(
            metric_card("IC à 5 jours", f"{ic5:.4f}",
                        "positive" if ic5 > 0 else "negative"),
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='fig-caption'>📝 L'Information Coefficient (corrélation de Spearman) "
            "mesure la capacité prédictive du signal de régime sur les returns futurs. "
            "Un IC > 0.05 est considéré comme économiquement utile en gestion quantitative.</div>",
            unsafe_allow_html=True,
        )

    # --- Stats régimes ---
    st.markdown("---")
    st.markdown("### 🎯 Statistiques Conditionnelles par Régime (VAE-HMM)")
    regime_stats = report.get("regime_stats", {}).get("vae_hmm", {})
    if regime_stats:
        import pandas as pd
        rows = []
        for regime_name, stats in regime_stats.items():
            rows.append({
                "Régime": regime_name,
                "Fréquence (%)": f"{stats.get('frequency_pct', 0):.1f}%",
                "Return moy. / jour": f"{stats.get('mean_daily_return_pct', 0):+.4f}%",
                "Volatilité ann.": f"{stats.get('ann_volatility_pct', 0):.1f}%",
                "Durée moy. (j)": f"{stats.get('mean_duration_days', 0):.0f}",
                "Durée max (j)": f"{stats.get('max_duration_days', 0):.0f}",
                "Nb épisodes": stats.get("n_episodes", 0),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)


# ===========================================================================
# PAGE 4 — FICHIERS & LOGS
# ===========================================================================

elif "Fichiers" in page:

    st.markdown("## 📁 Fichiers Générés & Logs")
    st.markdown("Téléchargez les fichiers de résultats produits par le pipeline.")
    st.markdown("---")

    # --- Rapport JSON ---
    st.markdown("### 📄 Rapport de Comparaison (JSON)")
    report_path = RESULTS_DIR / "comparison_report.json"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button(
                label="⬇️  Télécharger comparison_report.json",
                data=f,
                file_name="comparison_report.json",
                mime="application/json",
            )
        with st.expander("👁️  Aperçu du rapport"):
            report = load_report()
            if report:
                st.json({k: v for k, v in report.items()
                         if k != "metrics_table"})
    else:
        st.info("comparison_report.json non disponible — exécutez `python main.py`.")

    # --- Logs ---
    st.markdown("---")
    st.markdown("### 📋 Fichiers de Log")
    log_files = get_log_files()

    if log_files:
        selected_log = st.selectbox(
            "Choisir un fichier de log",
            log_files,
            format_func=lambda p: p.name,
        )
        with open(selected_log, "rb") as f:
            st.download_button(
                label=f"⬇️  Télécharger {selected_log.name}",
                data=f,
                file_name=selected_log.name,
                mime="text/plain",
            )
        with st.expander("👁️  Aperçu (100 dernières lignes)"):
            log_content = selected_log.read_text(encoding="utf-8", errors="replace")
            lines = log_content.splitlines()
            st.code("\n".join(lines[-100:]), language="text")
    else:
        st.info(
            "Aucun fichier .log trouvé dans `results/`. "
            "Ils sont créés automatiquement par `main.py`."
        )

    # --- Figures individuelles ---
    st.markdown("---")
    st.markdown("### 🖼️  Téléchargement des Figures")
    figures = get_available_figures()
    if figures:
        col1, col2 = st.columns(2)
        for i, fig_path in enumerate(figures):
            col = col1 if i % 2 == 0 else col2
            title = FIGURE_METADATA.get(fig_path.name, (fig_path.name, ""))[0]
            with open(fig_path, "rb") as f:
                col.download_button(
                    label=f"⬇️  {title}",
                    data=f,
                    file_name=fig_path.name,
                    mime="image/png",
                    key=f"dl_{fig_path.name}",
                )
    else:
        st.info("Aucune figure disponible. Lancez `python main.py`.")

    # --- Guide de lancement ---
    st.markdown("---")
    st.markdown("### 🚀 Commandes utiles")
    st.code("""# Lancer le pipeline complet
python main.py

# Lancer le pipeline sans réentraîner le VAE
python main.py --skip-training

# Lancer l'application web
streamlit run app.py

# Exécuter les tests unitaires
pytest tests/ -v

# Tests avec couverture HTML
pytest tests/ --cov=. --cov-report=html
""", language="bash")
