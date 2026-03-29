from pathlib import Path
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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame):
    # On sépare les colonnes numériques des colonnes texte/catégorielles
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Pour les nombres : on remplace les valeurs manquantes par la médiane, puis on normalise

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # Pour les catégories : on remplace les valeurs manquantes par la plus fréquente, puis on encode en chiffres
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features)
    ])

    return preprocessor, num_features, cat_features


def prepare_german():
    german_dir = RAW_DIR / "german_credit"
    raw_path_data = german_dir / "german.data"
    raw_path_csv = german_dir / "german_credit.csv"

    column_names = [
        "statut_compte", "duree_mois", "historique_credit", "objet_credit",
        "montant_credit", "epargne", "anciennete_emploi", "taux_versement",
        "statut_civil_sexe", "autres_debiteurs", "anciennete_residence",
        "propriete", "age", "autres_credits", "logement", "nb_credits",
        "emploi", "nb_personnes_charge", "telephone", "travailleur_etranger",
        "defaut"
    ]
    # On accepte le fichier CSV ou le fichier .data brut, selon ce qui est disponible
    if raw_path_csv.exists():
        df = pd.read_csv(raw_path_csv)
    elif raw_path_data.exists():
        df = pd.read_csv(raw_path_data, sep=r"\s+", header=None, names=column_names)
    else:
        raise FileNotFoundError(
            f"Aucun fichier German Credit trouvé dans {german_dir}. "
            "Attendu : german.data ou german_credit.csv"
        )

    if "defaut" not in df.columns:
        raise ValueError("La colonne cible 'defaut' est absente du dataset German.")

    # 1 = bon payeur -> 0 | 2 = mauvais payeur -> 1
    if sorted(df["defaut"].dropna().unique().tolist()) == [1, 2]:
        df["defaut"] = (df["defaut"] == 2).astype(int)

    df_feat = df.copy()
    df_feat["charge_mensuelle"] = df_feat["montant_credit"] / df_feat["duree_mois"]
    df_feat["jeune_emprunteur"] = (df_feat["age"] < 30).astype(int)
    df_feat["credit_long"] = (df_feat["duree_mois"] > 24).astype(int)

    sensitive_features = {
        "age": "age",
        "gender": "statut_civil_sexe"
    }

    return df_feat, "defaut", sensitive_features


def prepare_lending_club():
    lending_dir = RAW_DIR / "lending_club"
    raw_path = lending_dir / "lending_club.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Fichier Lending Club introuvable : {raw_path}"
        )

    print("Chargement Lending Club...")
    df = pd.read_csv(raw_path, low_memory=False)

    print(f"Dataset brut : {df.shape}")

    # On garde uniquement les prêts avec un résultat clair : remboursé ou en défaut
    valid_status = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(valid_status)].copy()

    # 1 = défaut / prêt problématique, 0 = non défaut
    df["defaut"] = (df["loan_status"] == "Charged Off").astype(int)


    drop_cols = [
        "loan_status",
        "id",
        "member_id",
        "url",
        "desc",
        "emp_title",
        "title",
        "zip_code",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    keep_cols = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "defaut",
    ]
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    existing_keep_cols = list(dict.fromkeys(existing_keep_cols))  # supprime doublons éventuels
    df = df[existing_keep_cols].copy()
    df = df[existing_keep_cols].copy()

    # Nettoyage de quelques colonnes typiquement textuelles
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.replace(" months", "", regex=False).str.strip()

    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .astype(str)
            .str.replace(r"\+ years", "", regex=True)
            .str.replace(r" years", "", regex=True)
            .str.replace(r" year", "", regex=True)
            .str.replace(r"< 1", "0", regex=False)
            .str.replace(r"10\+", "10", regex=True)
            .str.strip()
        )
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    if "int_rate" in df.columns:
        df["int_rate"] = (
            df["int_rate"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    if "revol_util" in df.columns:
        df["revol_util"] = (
            df["revol_util"].astype(str).str.replace("%", "", regex=False).str.strip()
        )
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    if "term" in df.columns:
        df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # Si le fichier est très grand, on prend un échantillon de 50 000 lignes pour ne pas saturer la mémoire
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).copy()

    print(f"Dataset nettoyé : {df.shape}")
    print(f"Taux de défaut : {df['defaut'].mean():.2%}")

    # Lending Club n'a pas d'info directe sur l'âge ou le genre
    sensitive_features = {
        "age": None,
        "gender": None
    }

    return df, "defaut", sensitive_features

def save_outputs(
    dataset_name: str,
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    preprocessor,
    num_features,
    cat_features,
    target_name: str,
    sensitive_features: dict
):

    dataset_processed_dir = PROCESSED_DIR / dataset_name
    dataset_models_dir = MODELS_DIR / dataset_name

    dataset_processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_models_dir.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(dataset_processed_dir / "X_train.csv", index=False)
    X_test_df.to_csv(dataset_processed_dir / "X_test.csv", index=False)
    y_train.reset_index(drop=True).to_csv(dataset_processed_dir / "y_train.csv", index=False, header=[target_name])
    y_test.reset_index(drop=True).to_csv(dataset_processed_dir / "y_test.csv", index=False, header=[target_name])
    # Sauvegarde des données brutes (pour les afficher lisiblement dans le dashboard)
    X_train_raw.reset_index(drop=True).to_csv(dataset_processed_dir / "X_train_raw.csv", index=False)
    X_test_raw.reset_index(drop=True).to_csv(dataset_processed_dir / "X_test_raw.csv", index=False)
    # On sauvegarde aussi l'objet de transformation pour pouvoir l'appliquer sur de nouvelles données plus tard
    joblib.dump(preprocessor, dataset_models_dir / "preprocessor.pkl")
    # Le fichier metadata regroupe toutes les infos utiles sur le dataset (noms de colonnes, taux de défaut, etc.)
    metadata = {
        "dataset_name": dataset_name,
        "target_name": target_name,
        "feature_names": X_train_df.columns.tolist(),
        "num_features": num_features,
        "cat_features": cat_features,
        "n_train": int(X_train_df.shape[0]),
        "n_test": int(X_test_df.shape[0]),
        "target_rate_train": float(y_train.mean()),
        "target_rate_test": float(y_test.mean()),
        "sensitive_features": sensitive_features,
    }

    with open(dataset_processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Données sauvegardées pour {dataset_name}")
    print(f"📁 Processed : {dataset_processed_dir}")
    print(f"📁 Models    : {dataset_models_dir}")


def run_pipeline(dataset_name: str):
    print(f"\n=== Préparation dataset : {dataset_name} ===")

    if dataset_name == "german":
        df, target_name, sensitive_features = prepare_german()
    elif dataset_name == "lending_club":
        df, target_name, sensitive_features = prepare_lending_club()
    else:
        raise ValueError("dataset_name doit être 'german' ou 'lending_club'")

    print(f"Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    X = df.drop(columns=[target_name])
    y = df[target_name]

    preprocessor, num_features, cat_features = build_preprocessor(X)

    print(f"Numériques ({len(num_features)}) : {num_features}")
    print(f"Catégorielles ({len(cat_features)}) : {cat_features}")
    # On divise en 80% pour entraîner et 20% pour tester, en gardant le même ratio de défauts dans chaque partie
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
    # On applique la transformation sur le train, puis on la réutilise telle quelle sur le test
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    feature_names_out = num_features + cat_features

    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names_out)
    X_test_df = pd.DataFrame(X_test_arr, columns=feature_names_out)

    save_outputs(
        dataset_name=dataset_name,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        preprocessor=preprocessor,
        num_features=num_features,
        cat_features=cat_features,
        target_name=target_name,
        sensitive_features=sensitive_features
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["german", "lending_club"])
    args = parser.parse_args()

    run_pipeline(args.dataset)