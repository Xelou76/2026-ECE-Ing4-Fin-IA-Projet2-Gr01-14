import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

def load_and_preprocess(data_path):
    """Charge et préprocesse le dataset"""
    print("📂 Chargement des données...")
    df = pd.read_csv(data_path)
    
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"✅ Données chargées : {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"   Fraudes train : {y_train.sum()} | Fraudes test : {y_test.sum()}")
    return X_train.values, X_test.values, y_train.values, y_test.values

def apply_smote(X_train, y_train):
    """Applique SMOTE pour rééquilibrer les classes"""
    print("\n⚖️  Application SMOTE...")
    print(f"   Avant : {Counter(y_train)}")
    
    over = SMOTE(sampling_strategy=0.1, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    pipeline = Pipeline([('over', over), ('under', under)])
    
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print(f"   Après : {Counter(y_res)}")
    return X_res, y_res

def print_metrics(model_name, TP, FP, FN, auprc):
    """Affiche les métriques d'un modèle"""
    cout_total = (FP * 10) + (FN * 500)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  AUPRC              : {auprc:.4f}")
    print(f"  Fraudes détectées  : {TP}")
    print(f"  Faux Positifs      : {FP}")
    print(f"  Faux Négatifs      : {FN}")
    print(f"  Coût financier     : {cout_total}€")