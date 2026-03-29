"""
data_loader.py
==============
Télécharge les données de prix des 30 actions du DJIA via yfinance,
calcule les rendements log et génère les statistiques de base.

Usage:
    from src.data_loader import DataLoader
    loader = DataLoader()
    prices, returns = loader.load_djia()
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Liste des 30 composants du DJIA (2024)
# ─────────────────────────────────────────────────────────────────────────────
DJIA_TICKERS = [
    'AAPL', 'AMGN', 'AXP',  'BA',   'CAT',
    'CRM',  'CSCO', 'CVX',  'DIS',  'DOW',
    'GS',   'HD',   'HON',  'IBM',  'INTC',
    'JNJ',  'JPM',  'KO',   'MCD',  'MMM',
    'MRK',  'MSFT', 'NKE',  'PG',   'TRV',
    'UNH',  'V',    'VZ',   'WBA',  'WMT'
]

# Secteurs pour les features du graphe
DJIA_SECTORS = {
    'AAPL': 'Technology',   'MSFT': 'Technology',  'CRM': 'Technology',
    'CSCO': 'Technology',   'IBM': 'Technology',    'INTC': 'Technology',
    'JPM':  'Financials',   'GS':  'Financials',    'AXP': 'Financials',
    'V':    'Financials',   'TRV': 'Financials',
    'JNJ':  'Healthcare',   'UNH': 'Healthcare',    'AMGN': 'Healthcare',
    'MRK':  'Healthcare',   'WBA': 'Healthcare',
    'HD':   'Consumer Disc','MCD': 'Consumer Disc',  'NKE': 'Consumer Disc',
    'KO':   'Consumer Stap','DIS': 'Consumer Disc',  'WMT': 'Consumer Stap',
    'PG':   'Consumer Stap',
    'CAT':  'Industrials',  'HON': 'Industrials',   'BA': 'Industrials',
    'MMM':  'Industrials',
    'DOW':  'Materials',
    'CVX':  'Energy',
    'VZ':   'Telecom',
}

# Mapping secteur -> indice entier (pour les features du GNN)
SECTOR_TO_IDX = {s: i for i, s in enumerate(sorted(set(DJIA_SECTORS.values())))}


class DataLoader:
    """
    Charge et prépare les données financières du DJIA.

    Paramètres
    ----------
    tickers : list, optionnel
        Liste des tickers. Par défaut : les 30 actions du DJIA.
    start : str
        Date de début au format 'YYYY-MM-DD'.
    end : str
        Date de fin au format 'YYYY-MM-DD'.
    """

    def __init__(
        self,
        tickers: list = None,
        start: str = '2018-01-01',
        end: str   = '2023-12-31',
    ):
        self.tickers = tickers if tickers else list(DJIA_TICKERS)
        self.start   = start
        self.end     = end
        self.sectors = DJIA_SECTORS

        # Données brutes (rempli après load_djia)
        self.prices  = None
        self.returns = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Téléchargement des prix
    # ─────────────────────────────────────────────────────────────────────────
    def download_prices(self) -> pd.DataFrame:
        """
        Télécharge les prix de clôture ajustés via yfinance.

        Retourne
        --------
        prices : DataFrame de forme (n_jours, n_actifs)
        """
        print(f"[DataLoader] Téléchargement de {len(self.tickers)} tickers "
              f"({self.start} → {self.end})...")

        raw = yf.download(
            tickers     = self.tickers,
            start       = self.start,
            end         = self.end,
            auto_adjust = True,   # ajuste pour dividendes et splits
            progress    = False,
        )

        # yfinance renvoie un MultiIndex si plusieurs tickers
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw['Close']
        else:
            prices = raw[['Close']]
            prices.columns = self.tickers

        # Réordonner les colonnes selon l'ordre original
        available = [t for t in self.tickers if t in prices.columns]
        prices = prices[available]

        # Supprimer les colonnes avec trop de NaN (>5%)
        threshold = 0.05
        too_many_nan = prices.columns[prices.isna().mean() > threshold].tolist()
        if too_many_nan:
            print(f"[DataLoader] Suppression de {too_many_nan} (trop de NaN)")
            prices = prices.drop(columns=too_many_nan)
            self.tickers = list(prices.columns)

        # Forward-fill puis back-fill pour les NaN restants (jours fériés)
        prices = prices.ffill().bfill()

        # Supprimer les jours où TOUS les prix sont NaN
        prices = prices.dropna(how='all')

        print(f"[DataLoader] ✓ {prices.shape[0]} jours × {prices.shape[1]} actifs chargés")
        self.prices = prices
        return prices

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Calcul des rendements
    # ─────────────────────────────────────────────────────────────────────────
    def compute_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calcule les rendements à partir des prix.

        Paramètres
        ----------
        method : 'log' (rendements log-normaux, recommandé)
                 ou 'simple' (r = P_t/P_{t-1} - 1)

        Retourne
        --------
        returns : DataFrame de même forme que prices, moins la première ligne.

        Note sur les rendements log :
            r_t = ln(P_t / P_{t-1})
            Avantage : additivité dans le temps, distribution plus proche de la normale.
        """
        if self.prices is None:
            raise RuntimeError("Appelez d'abord download_prices().")

        if method == 'log':
            returns = np.log(self.prices / self.prices.shift(1)).dropna()
        elif method == 'simple':
            returns = self.prices.pct_change().dropna()
        else:
            raise ValueError("method doit être 'log' ou 'simple'")

        print(f"[DataLoader] ✓ Rendements ({method}) calculés : {returns.shape}")
        self.returns = returns
        return returns

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Statistiques descriptives
    # ─────────────────────────────────────────────────────────────────────────
    def descriptive_stats(self) -> pd.DataFrame:
        """
        Retourne un DataFrame avec les statistiques annualisées de chaque actif.

        Métriques calculées :
            - Rendement annualisé   : μ × 252
            - Volatilité annualisée : σ × √252
            - Sharpe (rf=0)         : (μ / σ) × √252
            - Skewness / Kurtosis   : mesures de forme de la distribution
            - Rendement cumulé      : valeur finale du portefeuille 1€ investi
        """
        if self.returns is None:
            raise RuntimeError("Appelez d'abord compute_returns().")

        ann = 252  # jours de trading par an

        stats = pd.DataFrame({
            'Rendement annualisé (%)'  : (self.returns.mean() * ann * 100).round(2),
            'Volatilité annualisée (%)': (self.returns.std() * np.sqrt(ann) * 100).round(2),
            'Sharpe (rf=0)'            : ((self.returns.mean() / self.returns.std())
                                           * np.sqrt(ann)).round(3),
            'Skewness'                 : self.returns.skew().round(3),
            'Kurtosis excès'           : self.returns.kurt().round(3),
            'Rendement cumulé (×)'     : ((1 + self.returns).cumprod().iloc[-1]).round(3),
            'Secteur'                  : [
                self.sectors.get(t, 'Unknown') for t in self.returns.columns
            ],
        })
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Pipeline principal
    # ─────────────────────────────────────────────────────────────────────────
    def load_djia(self) -> tuple:
        """
        Pipeline complet : téléchargement + calcul des rendements.

        Retourne
        --------
        (prices, returns) : tuple de DataFrames
        """
        prices  = self.download_prices()
        returns = self.compute_returns(method='log')
        return prices, returns

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Préparation des features pour le GNN
    # ─────────────────────────────────────────────────────────────────────────
    def build_node_features(self, window: int = 20) -> np.ndarray:
        """
        Construit la matrice de features initiales des nœuds pour le GNN.

        Pour chaque actif, on calcule sur une fenêtre glissante de `window` jours :
            - Rendement moyen
            - Volatilité
            - Momentum (rendement cumulé sur la fenêtre)
            - Z-score du volume (si disponible)

        Retourne
        --------
        X : ndarray de forme (n_actifs, n_features)
        """
        if self.returns is None:
            raise RuntimeError("Appelez d'abord compute_returns().")

        r = self.returns.iloc[-window:]  # dernière fenêtre

        features = []
        for ticker in r.columns:
            ret_series = r[ticker]
            feat = [
                ret_series.mean(),           # rendement moyen
                ret_series.std(),            # volatilité
                ret_series.sum(),            # momentum (rendement cumulé log)
                ret_series.skew(),           # asymétrie
                # One-hot simplifié : indice de secteur normalisé
                SECTOR_TO_IDX.get(
                    self.sectors.get(ticker, 'Unknown'), 0
                ) / max(len(SECTOR_TO_IDX), 1),
            ]
            features.append(feat)

        X = np.array(features, dtype=np.float32)
        print(f"[DataLoader] ✓ Features nœuds : {X.shape} "
              f"({X.shape[1]} features par actif)")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Visualisations
    # ─────────────────────────────────────────────────────────────────────────
    def plot_prices(self, normalize: bool = True, figsize=(14, 6)):
        """
        Trace les prix normalisés (base 100) ou bruts, colorés par secteur.
        """
        if self.prices is None:
            raise RuntimeError("Appelez d'abord download_prices().")

        fig, ax = plt.subplots(figsize=figsize)
        data = self.prices / self.prices.iloc[0] * 100 if normalize else self.prices

        sector_colors = {
            'Technology':    '#3266ad',
            'Financials':    '#E85D24',
            'Healthcare':    '#1D9E75',
            'Consumer Disc': '#EF9F27',
            'Consumer Stap': '#97C459',
            'Industrials':   '#7F77DD',
            'Energy':        '#D4537E',
            'Materials':     '#888780',
            'Telecom':       '#B0B0A0',
        }

        for ticker in data.columns:
            sector = self.sectors.get(ticker, 'Unknown')
            color  = sector_colors.get(sector, '#cccccc')
            ax.plot(data.index, data[ticker],
                    linewidth=0.9, alpha=0.65, color=color)

        # Légende secteurs
        from matplotlib.lines import Line2D
        present_sectors = {self.sectors.get(t, 'Unknown') for t in data.columns}
        legend_elements = [
            Line2D([0], [0], color=c, linewidth=2, label=s)
            for s, c in sector_colors.items()
            if s in present_sectors
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  fontsize=9, ncol=2, framealpha=0.9)

        ylabel = 'Prix normalisé (base 100)' if normalize else 'Prix de clôture ($)'
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title('Évolution des 30 actions du DJIA — coloré par secteur',
                     fontsize=13, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        plt.tight_layout()
        return fig

    def plot_returns_distribution(self, top_n: int = 9, figsize=(14, 10)):
        """
        Histogrammes des rendements journaliers pour les top_n actifs (par Sharpe).
        """
        if self.returns is None:
            raise RuntimeError("Appelez d'abord compute_returns().")

        stats       = self.descriptive_stats()
        top_tickers = stats['Sharpe (rf=0)'].nlargest(top_n).index.tolist()

        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, ticker in enumerate(top_tickers):
            r = self.returns[ticker]
            axes[i].hist(r, bins=60, color='#3266ad', alpha=0.75,
                         edgecolor='white', linewidth=0.3)
            axes[i].axvline(r.mean(), color='#E85D24', lw=1.5,
                            linestyle='--', label=f'μ={r.mean():.4f}')
            axes[i].set_title(
                f'{ticker} — σ={r.std()*100:.2f}%/j', fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].set_xlabel('Rendement log', fontsize=9)
            axes[i].grid(True, alpha=0.25, linewidth=0.5)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Distribution des rendements journaliers (top actifs par Sharpe)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, figsize=(12, 10)):
        """
        Heatmap de corrélation, triée par secteur pour faire apparaître les clusters.

        Interprétation :
            - Blocs diagonaux rouges = corrélations fortes intra-secteur
            - Hors-diagonale verte = actifs peu corrélés (diversification)
        """
        if self.returns is None:
            raise RuntimeError("Appelez d'abord compute_returns().")

        corr = self.returns.corr()

        # Tri par secteur
        sector_order = sorted(
            self.returns.columns,
            key=lambda t: self.sectors.get(t, 'Unknown')
        )
        corr_sorted = corr.loc[sector_order, sector_order]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_sorted.values, cmap='RdYlGn',
                       vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(len(sector_order)))
        ax.set_yticks(range(len(sector_order)))
        ax.set_xticklabels(sector_order, rotation=90, fontsize=9)
        ax.set_yticklabels(sector_order, fontsize=9)

        plt.colorbar(im, ax=ax, label='Corrélation de Pearson', shrink=0.8)
        ax.set_title(
            'Matrice de corrélation des rendements (triée par secteur)\n'
            'Les blocs rouges révèlent les clusters à modéliser par le graphe',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide si exécuté directement
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    loader = DataLoader(start='2020-01-01', end='2023-12-31')
    prices, returns = loader.load_djia()

    print("\n=== Statistiques descriptives (top 5 par Sharpe) ===")
    stats = loader.descriptive_stats()
    print(stats.nlargest(5, 'Sharpe (rf=0)').to_string())

    print(f"\nFormes : prix={prices.shape}, rendements={returns.shape}")

    X = loader.build_node_features(window=20)
    print(f"\nFeatures GNN : {X.shape}")

    print("\n✓ DataLoader fonctionne correctement")
