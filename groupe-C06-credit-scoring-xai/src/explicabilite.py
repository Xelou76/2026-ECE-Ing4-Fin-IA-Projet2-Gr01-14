from pathlib import Path
import ast
import json
import joblib
import warnings
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

RANDOM_STATE = 42
N_LIME_PROFILES = 10

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_ROOT = BASE_DIR / "models"
RESULTS_ROOT = BASE_DIR / "results"

# AJOUT ARGUMENT DATASET
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["german", "lending_club"])
args = parser.parse_args()

DATASET = args.dataset

PROCESSED_DIR = DATA_DIR / DATASET
MODELS_DIR = MODELS_ROOT / DATASET
RESULTS_DIR = RESULTS_ROOT / DATASET

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Setup OK")
print(f"BASE_DIR = {BASE_DIR}")

# ============================================================
# CHARGEMENT
# ============================================================

X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
X_test_raw = pd.read_csv(PROCESSED_DIR / "X_test_raw.csv")
X_train_raw = pd.read_csv(PROCESSED_DIR / "X_train_raw.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
# On charge le meilleur modèle sélectionné lors de l'étape de modélisation
model = joblib.load(MODELS_DIR / "best_model.pkl")

with open(PROCESSED_DIR / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]
num_features = metadata["num_features"]
cat_features = metadata["cat_features"]

print(f"Dataset : {DATASET}")
print(f"X_test : {X_test.shape}")
print(f"Features : {len(feature_names)}")

# ============================================================
# PRÉDICTIONS DE BASE
# ============================================================

X_test_arr = X_test.values
X_train_arr = X_train.values
# y_prob : la probabilité estimée de défaut pour chaque client (entre 0 et 1)
y_prob = model.predict_proba(X_test_arr)[:, 1]
y_pred = model.predict(X_test_arr)

print(f"Score moyen (test) : {y_prob.mean():.3f}")

# ============================================================
# SHAP — EXPLAINER
# ============================================================
# SHAP calcule pour chaque variable sa contribution à la décision du modèle.
# C'est l'outil principal pour comprendre "pourquoi le modèle a dit oui ou non".
print("\n─── SHAP : calcul des valeurs ───")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_arr)

# Certains modèles retournent les valeurs SHAP pour chaque classe : on prend celle de la classe "défaut" (index 1)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[1]

print(f"  Valeur de base (expected value) : {expected_value:.4f}")
print(f"  Shape shap_values : {shap_values.shape}")

# ============================================================
# SHAP GLOBAL — BEESWARM
# ============================================================
# Vue d'ensemble : quelles variables ont le plus d'impact sur l'ensemble des clients ?
print("\n─── SHAP Global : Beeswarm ───")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test_arr,
    feature_names=feature_names,
    show=False,
    plot_type="dot",
    max_display=15,
)
plt.title("SHAP Beeswarm — Impact global des features", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_beeswarm.png", bbox_inches="tight")
plt.close()
print(f"  ✓ {RESULTS_DIR / 'shap_beeswarm.png'}")

# ============================================================
# SHAP GLOBAL — BAR PLOT
# ============================================================

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test_arr,
    feature_names=feature_names,
    show=False,
    plot_type="bar",
    max_display=15,
)
plt.title("SHAP Feature Importance moyenne |Φ|", fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_bar.png", bbox_inches="tight")
plt.close()
print(f"  ✓ {RESULTS_DIR / 'shap_bar.png'}")

# ============================================================
# SHAP LOCAL — WATERFALL (3 profils)
# ============================================================
# On explique 3 cas concrets : le client le plus sûr, le plus risqué, et un cas limite
print("\n─── SHAP Local : Waterfall (3 profils) ───")

idx_accepted = int(np.argmax(y_prob))
idx_refused = int(np.argmin(y_prob))
idx_borderline = int(np.argmin(np.abs(y_prob - 0.5)))

profiles = {
    "accepte": idx_accepted,
    "refuse": idx_refused,
    "borderline": idx_borderline,
}

for ptype, idx in profiles.items():
    try:
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_test_arr[idx],
            feature_names=feature_names,
        )
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(exp, show=False, max_display=12)
        plt.title(
            f"SHAP Waterfall — Profil {ptype} (score={y_prob[idx]:.3f})",
            fontweight="bold",
            fontsize=11,
        )
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_waterfall_{ptype}.png", bbox_inches="tight")
        plt.close()
        print(f"  ✓ {RESULTS_DIR / f'shap_waterfall_{ptype}.png'}  (prob={y_prob[idx]:.3f})")
    except Exception as e:
        print(f"  ⚠ Waterfall {ptype} : {e}")

# ============================================================
# SHAP SCATTER — DÉPENDANCE
# ============================================================
# On regarde comment les 2 variables les plus importantes influencent le score selon leur valeur
print("\n─── SHAP Scatter : dépendance ───")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top2_idx = np.argsort(mean_abs_shap)[::-1][:2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, feat_idx in zip(axes, top2_idx):
    feat_name = feature_names[feat_idx]
    sc = ax.scatter(
        X_test_arr[:, feat_idx],
        shap_values[:, feat_idx],
        c=shap_values[:, feat_idx],
        cmap="RdYlGn",
        alpha=0.6,
        s=20,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"Valeur de {feat_name} (normalisée)")
    ax.set_ylabel("Contribution SHAP")
    ax.set_title(f"Dépendance SHAP — {feat_name}", fontweight="bold")
    plt.colorbar(sc, ax=ax, shrink=0.8)

plt.suptitle("Relation valeur → contribution SHAP", fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_scatter.png", bbox_inches="tight")
plt.close()
print(f"  ✓ {RESULTS_DIR / 'shap_scatter.png'}")

# ============================================================
# LIME — 10 PROFILS
# ============================================================
# LIME est une alternative à SHAP
# On l'utilise ici sur 10 profils aléatoires pour comparer ses conclusions avec celles de SHAP.
print(f"\n─── LIME sur {N_LIME_PROFILES} profils ───")

try:
    from lime import lime_tabular

    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_arr,
        feature_names=feature_names,
        class_names=["Bon payeur", "Défaut"],
        mode="classification",
        random_state=RANDOM_STATE,
    )

    np.random.seed(RANDOM_STATE)
    profile_indices = np.random.choice(len(X_test), N_LIME_PROFILES, replace=False)

    lime_results = []
    for i, idx in enumerate(profile_indices):
        try:
            exp = lime_explainer.explain_instance(
                X_test_arr[idx],
                model.predict_proba,
                num_features=8,
                num_samples=500,
            )
            top_feats = exp.as_list(label=1)
            lime_results.append({
                "profile_idx": int(idx),
                "true_label": int(y_test.iloc[idx]),
                "pred_prob": round(float(y_prob[idx]), 3),
                "lime_top_features": top_feats,
            })
            # On génère un graphique seulement pour le premier profil, comme exemple visuel
            if i == 0:
                fig = exp.as_pyplot_figure(label=1)
                plt.title(f"LIME — Profil exemple (prob={y_prob[idx]:.3f})", fontweight="bold")
                plt.tight_layout()
                plt.savefig(RESULTS_DIR / "lime_exemple.png", bbox_inches="tight")
                plt.close()
                print(f"  ✓ {RESULTS_DIR / 'lime_exemple.png'}")

        except Exception as e:
            print(f"  ⚠ LIME profil {idx} : {e}")

    print(f"  LIME calculé sur {len(lime_results)}/{N_LIME_PROFILES} profils")
except ImportError:
    print("   LIME non installé. Installez avec : pip install lime")
    lime_results = []
    np.random.seed(RANDOM_STATE)
    profile_indices = np.random.choice(len(X_test), N_LIME_PROFILES, replace=False)

# ============================================================
# COMPARAISON SHAP VS LIME
# ============================================================
# On vérifie si SHAP et LIME sont d'accord sur les 3 variables les plus importantes pour chaque profil.
print("\n─── Comparaison SHAP vs LIME ───")

rows_comp = []
for i, (lime_info, idx) in enumerate(zip(lime_results, profile_indices)):
    shap_top_idx = np.argsort(np.abs(shap_values[idx]))[::-1][:3]
    shap_top_names = [feature_names[j] for j in shap_top_idx]
    lime_top_names = [f[0] for f in lime_info["lime_top_features"][:3]]
    concordance = len(set(shap_top_names) & set(lime_top_names)) / 3.0

    rows_comp.append({
        "profil": i,
        "idx": idx,
        "pred_prob": round(float(y_prob[idx]), 3),
        "shap_top3": " | ".join(shap_top_names),
        "lime_top3": " | ".join(lime_top_names),
        "concordance": round(concordance, 2),
    })

if rows_comp:
    df_comp = pd.DataFrame(rows_comp)
    df_comp.to_csv(RESULTS_DIR / "shap_vs_lime.csv", index=False)

    avg_conc = df_comp["concordance"].mean()
    print(f"  Concordance moyenne Top-3 : {avg_conc:.0%}")
    print(f"  ✓ {RESULTS_DIR / 'shap_vs_lime.csv'}")

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_colors = ["#5DCAA5" if c >= 0.67 else "#E24B4A" for c in df_comp["concordance"]]
    ax.bar(df_comp["profil"], df_comp["concordance"], color=bar_colors, alpha=0.85)
    ax.axhline(avg_conc, color="navy", linestyle="--", lw=2, label=f"Moyenne = {avg_conc:.2f}")
    ax.axhline(0.67, color="gray", linestyle=":", lw=1.5, label="Seuil 67%")
    ax.set_xlabel("Profil")
    ax.set_ylabel("Concordance Top-3 features")
    ax.set_title("Concordance SHAP vs LIME par profil", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_lime_concordance.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {RESULTS_DIR / 'shap_lime_concordance.png'}")
else:
    print("  ⚠ Pas de données LIME — comparaison ignorée")

# ============================================================
# CONTREFACTUELS DiCE
# ============================================================
# Les contrefactuels répondent à la question : "Qu'aurait-il fallu changer pour que la décision soit différente ?"
print("\n─── Contrefactuels DiCE ───")
# On cible les clients les plus à risque pour leur proposer des pistes d'amélioration
refused_indices = np.where(y_prob < 0.4)[0][:3]
if len(refused_indices) == 0:
    refused_indices = np.argsort(y_prob)[:3]

dice_ok = False
try:
    import dice_ml
    from dice_ml import Dice

    X_train_dice = X_train.copy()
    X_train_dice["defaut"] = y_train.values

    d = dice_ml.Data(
        dataframe=X_train_dice,
        continuous_features=num_features,
        outcome_name="defaut",
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp_dice = Dice(d, m, method="random")
    # On ne fait pas varier les variables qu'on ne peut pas changer (âge, genre, etc.)
    immutable = [f for f in feature_names if any(k in f.lower() for k in ["age", "statut_civil"])]

    cf_results = []
    for idx in refused_indices:
        query = X_test.iloc[[idx]].copy()
        try:
            cfs = exp_dice.generate_counterfactuals(
                query,
                total_CFs=3,
                desired_class="opposite",
                features_to_vary=[f for f in feature_names if f not in immutable],
            )
            cf_df = cfs.cf_examples_list[0].final_cfs_df.copy()
            cf_df.insert(0, "profile_idx", idx)
            cf_df.insert(1, "score_original", round(float(y_prob[idx]), 3))
            cf_results.append(cf_df)
            print(f"  ✓ Profil {idx} (score={y_prob[idx]:.3f}) : {len(cf_df)} contrefactuels générés")
        except Exception as e:
            print(f"   DiCE profil {idx} : {e}")

    if cf_results:
        all_cfs = pd.concat(cf_results, ignore_index=True)
        all_cfs.to_csv(RESULTS_DIR / "contrefactuels.csv", index=False)
        print(f"  ✓ {RESULTS_DIR / 'contrefactuels.csv'}")
        dice_ok = True

except ImportError:
    print("   DiCE non installé. Installez avec : pip install dice-ml")
except Exception as e:
    print(f"  DiCE erreur : {e}")
# Si DiCE n'est pas disponible, on génère des suggestions manuellement en testant de petites modifications
if not dice_ok:
    print("  → Contrefactuels manuels (fallback)...")

    cf_rows = []
    for idx in refused_indices:
        original = X_test_arr[idx].copy()
        score_orig = float(y_prob[idx])
        suggestions = {}
        # Pour chaque variable numérique, on cherche la plus petite modification qui améliore le score
        for j, feat in enumerate(feature_names):
            if feat not in num_features:
                continue
            std_j = X_train_arr[:, j].std()
            for delta in [0.5, 1.0, 1.5, 2.0]:
                candidate = original.copy()
                candidate[j] = original[j] + delta * std_j
                new_score = model.predict_proba(candidate.reshape(1, -1))[0, 1]
                if new_score < score_orig - 0.05:
                    suggestions[feat] = {
                        "delta_std": round(delta, 1),
                        "nouveau_score": round(new_score, 3),
                    }
                    break

        phrases = []
        for feat, info in suggestions.items():
            phrases.append(
                f"Modifier '{feat}' de {info['delta_std']} écart-type "
                f"ferait passer le score à {info['nouveau_score']:.2f}"
            )
        suggestion_texte = " | ".join(phrases) if phrases else "Aucune suggestion trouvée"

        cf_rows.append({
            "profile_idx": int(idx),
            "score_original": round(score_orig, 3),
            "suggestions": suggestion_texte,
        })

    df_cf_manual = pd.DataFrame(cf_rows)
    df_cf_manual.to_csv(RESULTS_DIR / "contrefactuels.csv", index=False)
    print(f"  ✓ {RESULTS_DIR / 'contrefactuels.csv'} (méthode manuelle)")

    fig, axes = plt.subplots(1, min(3, len(cf_rows)), figsize=(15, 5))
    if len(cf_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, cf_rows):
        try:
            sugg = ast.literal_eval(row["suggestions"])
            feats = list(sugg.keys())[:6]
            deltas = [sugg[f]["delta_std"] for f in feats]
            ax.barh(feats, deltas, color="#5DCAA5", alpha=0.85)
            ax.set_xlabel("Variation (en σ)")
            ax.set_title(
                f"Profil {row['profile_idx']}\nScore: {row['score_original']:.3f} → ?",
                fontsize=10,
                fontweight="bold",
            )
        except Exception:
            ax.set_visible(False)

    plt.suptitle("Suggestions contrefactuelles ('que changer pour éviter le défaut ?')",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "contrefactuels_viz.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {RESULTS_DIR / 'contrefactuels_viz.png'}")

# ============================================================
# RÉCAPITULATIF
# ============================================================

print("\n" + "=" * 55)
print("RÉCAPITULATIF — EXPLICABILITE")
print("=" * 55)
print("Fichiers générés dans results/ :")
for fname in [
    "shap_beeswarm.png", "shap_bar.png",
    "shap_waterfall_accepte.png", "shap_waterfall_refuse.png",
    "shap_waterfall_borderline.png", "shap_scatter.png",
    "lime_exemple.png", "shap_vs_lime.csv",
    "shap_lime_concordance.png", "contrefactuels.csv",
]:
    status = "✓" if (RESULTS_DIR / fname).exists() else "✗"
    print(f"  {status} {fname}")

print("\n Explicabilité terminée.")