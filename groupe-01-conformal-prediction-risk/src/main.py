import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from mapie.regression import MapieRegressor

def run_analysis():
    print("--- Analyse de Risque   ---")
    
    # 1. Données S&P 500
    df = yf.download("^GSPC", start="2023-01-01", end="2026-03-25")
    df['Target'] = df['Close'].shift(-1)
    
    #  Retours décalés
    for i in range(1, 4):
        df[f'Lag_{i}'] = df['Close'].shift(i)
    
    df = df.dropna()
    X = df[['Lag_1', 'Lag_2', 'Lag_3']]
    y = df['Target']

    # 2. Split Temporel
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 3. Modèle Random 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    mapie = MapieRegressor(estimator=model, cv="split")
    mapie.fit(X_train, y_train)

    # Prédiction avec Intervalle de Confiance 95%
    y_pred, y_pis = mapie.predict(X_test, alpha=[0.05])

    # 4. Graphique de sortie
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Prix Réel", color="black", alpha=0.7)
    plt.fill_between(
        y_test.index, 
        y_pis[:, 0, 0], 
        y_pis[:, 1, 0], 
        color="red", alpha=0.2, label="Tunnel de Risque Garanti (95%)"
    )
    plt.title("Gestion du Risque : Conformal Prediction sur S&P 500")
    plt.legend()
    # On sauvegarde l'image
    plt.savefig("groupe-01-conformal-prediction-risk/resultat_final.png")
    print("Graphique généré.")

if __name__ == "__main__":
    run_analysis()
