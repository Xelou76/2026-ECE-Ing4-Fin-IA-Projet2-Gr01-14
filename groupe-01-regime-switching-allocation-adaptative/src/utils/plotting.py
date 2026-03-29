"""
utils/plotting.py
=================
Visualisations publication-ready pour le projet VAE-HMM.

Interface principale : ``RegimePlotter.plot_all(...)`` — appelé par ``main.py``
pour générer toutes les figures en une seule instruction.

Figures générées
----------------
1. Régimes de marché sur le prix (cours SPY + fonds colorés par régime)
2. Equity curves comparées (VAE-HMM vs Hamilton vs Buy-and-Hold)
3. Matrice de transition HMM (heatmap annotée)
4. Distribution des régimes (barres + fréquences)
5. Drawdowns — Underwater plot
6. Sharpe ratio glissant multi-stratégies
7. Returns mensuels — heatmap (années × mois)
8. Espace latent VAE — PCA 2D coloré par régime
9. Dashboard agrégé 3×2 pour le rapport final

Classes
-------
RegimePlotter
    Classe principale de visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from config.constants import (
    BASELINE_REGIME_COLORS,
    BASELINE_REGIME_LABELS,
    MODEL_BUY_HOLD,
    MODEL_MARKOV_SWITCHING,
    MODEL_VAE_HMM,
    REGIME_COLORS,
    REGIME_LABELS,
)
from strategy.backtester import BacktestResult

# Backend non-interactif (compatible serveurs CI/CD)
matplotlib.use("Agg")

# Palette de stratégies
_STRATEGY_PALETTE: Dict[str, str] = {
    MODEL_VAE_HMM: "#58a6ff",
    MODEL_MARKOV_SWITCHING: "#f78166",
    MODEL_BUY_HOLD: "#ffa657",
}
_DEFAULT_COLOR = "#79c0ff"

# Style global sombre
_RC = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#e6edf3",
    "axes.grid": True,
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "legend.facecolor": "#21262d",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "figure.dpi": 150,
}


class RegimePlotter:
    """
    Générateur complet de visualisations VAE-HMM.

    Parameters
    ----------
    output_dir : Path
        Dossier de sauvegarde des figures.
    dpi : int, optional
        Résolution des figures sauvegardées. Par défaut 150.
    fmt : str, optional
        Format de fichier : 'png', 'pdf', 'svg'. Par défaut 'png'.

    Examples
    --------
    >>> plotter = RegimePlotter(Path("results/figures"))
    >>> plotter.plot_all(
    ...     prices=prices_test,
    ...     regimes_vae_hmm=regimes_test,
    ...     regimes_baseline=regimes_baseline,
    ...     backtest_results=results,
    ...     train_history=history,
    ... )
    """

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 150,
        fmt: str = "png",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.fmt = fmt

    # ------------------------------------------------------------------
    # Public — API principale (appelée par main.py)
    # ------------------------------------------------------------------

    def plot_all(
        self,
        prices: pd.Series,
        regimes_vae_hmm: np.ndarray,
        regimes_baseline: np.ndarray,
        backtest_results: Dict[str, BacktestResult],
        train_history: Optional[Dict] = None,
        regime_proba: Optional[np.ndarray] = None,
        latent_vectors: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Génère et sauvegarde toutes les figures du projet.

        Point d'entrée unique appelé par ``main.py``. Génère en séquence
        toutes les visualisations définies dans cette classe.

        Parameters
        ----------
        prices : pd.Series
            Prix du benchmark (ex: SPY) sur la période test.
        regimes_vae_hmm : np.ndarray
            Régimes prédits par VAE-HMM — shape (N,).
        regimes_baseline : np.ndarray
            Régimes prédits par Markov-Switching — shape (N,).
        backtest_results : Dict[str, BacktestResult]
            Résultats des backtests (VAE-HMM + Buy-and-Hold + optionnellement Hamilton).
        train_history : Dict, optional
            Historique d'entraînement du VAE (courbes de loss).
        regime_proba : np.ndarray, optional
            Probabilités a posteriori des régimes — shape (N, K).
        latent_vectors : np.ndarray, optional
            Représentations latentes du VAE — shape (N, latent_dim).
        transition_matrix : np.ndarray, optional
            Matrice de transition du HMM — shape (K, K).
        """
        logger.info("RegimePlotter — génération de toutes les figures...")

        # 1. Régimes sur le prix
        self.plot_regimes_on_price(
            prices=prices,
            regimes=regimes_vae_hmm,
            regime_proba=regime_proba,
            title="Régimes VAE-HMM sur le Cours SPY",
            filename="01_regimes_on_price",
        )

        # 2. Equity curves
        self.plot_equity_curves(
            backtest_results=backtest_results,
            filename="02_equity_curves",
        )

        # 3. Matrice de transition
        if transition_matrix is not None:
            self.plot_transition_matrix(
                matrix=transition_matrix,
                filename="03_transition_matrix",
            )

        # 4. Distribution des régimes
        self.plot_regime_distribution(
            regimes_vae=regimes_vae_hmm,
            regimes_baseline=regimes_baseline,
            filename="04_regime_distribution",
        )

        # 5. Drawdowns
        self.plot_drawdowns(
            backtest_results=backtest_results,
            filename="05_drawdowns",
        )

        # 6. Rolling Sharpe
        self.plot_rolling_sharpe(
            backtest_results=backtest_results,
            filename="06_rolling_sharpe",
        )

        # 7. Monthly returns heatmap
        if MODEL_VAE_HMM in backtest_results:
            self.plot_monthly_returns(
                result=backtest_results[MODEL_VAE_HMM],
                filename="07_monthly_returns",
            )

        # 8. Espace latent (si disponible)
        if latent_vectors is not None:
            self.plot_latent_space(
                latent=latent_vectors,
                regimes=regimes_vae_hmm,
                filename="08_latent_space",
            )

        # 9. Training history (si disponible)
        if train_history is not None:
            self.plot_training_history(
                history=train_history,
                filename="09_training_history",
            )

        # 10. Dashboard complet
        self.plot_dashboard(
            prices=prices,
            regimes=regimes_vae_hmm,
            backtest_results=backtest_results,
            transition_matrix=transition_matrix,
            filename="10_full_dashboard",
        )

        logger.success(f"  {10 if train_history else 9} figures sauvegardées → {self.output_dir}")

    # ------------------------------------------------------------------
    # Figures individuelles
    # ------------------------------------------------------------------

    def plot_regimes_on_price(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        regime_proba: Optional[np.ndarray] = None,
        title: str = "Régimes de Marché (VAE-HMM)",
        filename: str = "regimes_on_price",
    ) -> None:
        """
        Visualise les régimes en surimpression du cours du benchmark.

        Chaque période est colorée selon le régime actif (fond semi-transparent).
        Si les probabilités sont fournies, un subplot affiche leur évolution.

        Parameters
        ----------
        prices : pd.Series
            Cours de clôture du benchmark.
        regimes : np.ndarray
            Séquence de régimes — shape (N,).
        regime_proba : np.ndarray, optional
            Probabilités a posteriori — shape (N, K).
        title : str
            Titre principal.
        filename : str
            Nom du fichier (sans extension).
        """
        with plt.rc_context(_RC):
            nrows = 2 if regime_proba is not None else 1
            ratios = [3, 1] if regime_proba is not None else [1]
            fig, axes = plt.subplots(
                nrows, 1, figsize=(16, 9 if nrows == 2 else 6),
                gridspec_kw={"height_ratios": ratios},
            )
            if nrows == 1:
                axes = [axes]

            ax1 = axes[0]
            ax1.plot(
                prices.index, prices.values,
                color="#e6edf3", linewidth=1.2, zorder=5,
                label=prices.name or "Benchmark",
            )
            self._shade_regimes(ax1, prices.index, regimes)

            patches = [
                mpatches.Patch(
                    facecolor=REGIME_COLORS.get(k, "#888"),
                    alpha=0.45,
                    label=REGIME_LABELS.get(k, f"R{k}"),
                )
                for k in sorted(set(int(r) for r in regimes))
            ]
            ax1.legend(handles=patches, fontsize=8, loc="upper left")
            ax1.set_title(title, fontsize=14, fontweight="bold", pad=10)
            ax1.set_ylabel("Prix ($)", fontsize=10)
            ax1.tick_params(labelbottom=(nrows == 1))

            if regime_proba is not None and nrows == 2:
                ax2 = axes[1]
                n_r = regime_proba.shape[1]
                bottom = np.zeros(len(prices))
                for k in range(n_r):
                    c = REGIME_COLORS.get(k, "#888")
                    ax2.fill_between(
                        prices.index, bottom, bottom + regime_proba[:, k],
                        color=c, alpha=0.75,
                        label=REGIME_LABELS.get(k, f"R{k}"),
                    )
                    bottom += regime_proba[:, k]
                ax2.set_ylabel("P(Régime)", fontsize=9)
                ax2.set_ylim(0, 1)

            plt.tight_layout()
            self._save(fig, filename)

    def plot_equity_curves(
        self,
        backtest_results: Dict[str, BacktestResult],
        filename: str = "equity_curves",
    ) -> None:
        """
        Trace les equity curves de toutes les stratégies.

        Parameters
        ----------
        backtest_results : Dict[str, BacktestResult]
            Résultats des backtests.
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(16, 10),
                gridspec_kw={"height_ratios": [3, 1]},
            )

            for name, res in backtest_results.items():
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                lw = 2.2 if MODEL_VAE_HMM in name else 1.3
                ls = "-" if MODEL_VAE_HMM in name else ("--" if MODEL_BUY_HOLD in name else "-.")
                sr = res.metrics.get("sharpe_ratio", 0)
                ax1.plot(
                    res.equity_curve.index, res.equity_curve.values,
                    color=color, linewidth=lw, linestyle=ls,
                    label=f"{name}  (Sharpe: {sr:.2f})",
                )

            ax1.axhline(100, color="#555", linewidth=0.8, linestyle=":", alpha=0.6)
            ax1.set_title("Comparaison des Stratégies — Equity Curves", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Valeur (base 100)", fontsize=10)
            ax1.legend(fontsize=9, loc="upper left")

            # Drawdown subplot
            for name, res in backtest_results.items():
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                dd = (res.equity_curve / res.equity_curve.cummax() - 1) * 100
                ax2.fill_between(dd.index, dd.values, 0, color=color, alpha=0.2)
                ax2.plot(dd.index, dd.values, color=color, linewidth=0.8)

            ax2.axhline(0, color="#555", linewidth=0.6)
            ax2.set_ylabel("Drawdown (%)", fontsize=9)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

            plt.tight_layout()
            self._save(fig, filename)

    def plot_transition_matrix(
        self,
        matrix: np.ndarray,
        filename: str = "transition_matrix",
    ) -> None:
        """
        Heatmap annotée de la matrice de transition du HMM.

        Parameters
        ----------
        matrix : np.ndarray
            Matrice de transition — shape (K, K), stochastique par ligne.
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            n = matrix.shape[0]
            labels_short = [f"R{k}" for k in range(n)]
            labels_long = [REGIME_LABELS.get(k, f"R{k}") for k in range(n)]

            fig, ax = plt.subplots(figsize=(8, 6))
            cmap = LinearSegmentedColormap.from_list("hmm", ["#0d1117", "#388bfd"])

            sns.heatmap(
                matrix, annot=True, fmt=".3f", cmap=cmap,
                vmin=0, vmax=1, ax=ax,
                xticklabels=labels_short, yticklabels=labels_short,
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Matrice de Transition HMM (VAE-HMM)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Régime à t+1", fontsize=10)
            ax.set_ylabel("Régime à t", fontsize=10)

            legend = "\n".join(f"R{k} = {l}" for k, l in enumerate(labels_long))
            ax.text(
                1.25, 0.5, legend, transform=ax.transAxes,
                fontsize=8, va="center",
                bbox=dict(boxstyle="round", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
                color="#c9d1d9",
            )

            plt.tight_layout()
            self._save(fig, filename)

    def plot_regime_distribution(
        self,
        regimes_vae: np.ndarray,
        regimes_baseline: np.ndarray,
        filename: str = "regime_distribution",
    ) -> None:
        """
        Distribution comparée des régimes : VAE-HMM vs Markov-Switching.

        Parameters
        ----------
        regimes_vae : np.ndarray
            Régimes VAE-HMM.
        regimes_baseline : np.ndarray
            Régimes Markov-Switching (Hamilton).
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))

            for ax, regimes, labels, colors, model_name in [
                (axes[0], regimes_vae, REGIME_LABELS, REGIME_COLORS, "VAE-HMM"),
                (axes[1], regimes_baseline, BASELINE_REGIME_LABELS, BASELINE_REGIME_COLORS, "Hamilton"),
            ]:
                n_r = int(max(regimes)) + 1
                counts = [(regimes == k).sum() for k in range(n_r)]
                freqs = [c / sum(counts) * 100 for c in counts]
                names = [labels.get(k, f"R{k}") for k in range(n_r)]
                bar_colors = [colors.get(k, "#888") for k in range(n_r)]

                bars = ax.bar(range(n_r), freqs, color=bar_colors, alpha=0.85,
                              edgecolor="#30363d", linewidth=0.8)
                ax.set_xticks(range(n_r))
                ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
                ax.set_ylabel("Fréquence (%)", fontsize=9)
                ax.set_title(f"Distribution des Régimes\n({model_name})", fontsize=11, fontweight="bold")

                for bar, freq in zip(bars, freqs):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{freq:.1f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="#e6edf3",
                    )

            plt.tight_layout()
            self._save(fig, filename)

    def plot_drawdowns(
        self,
        backtest_results: Dict[str, BacktestResult],
        filename: str = "drawdowns",
    ) -> None:
        """
        Underwater plot des drawdowns de toutes les stratégies.

        Parameters
        ----------
        backtest_results : Dict[str, BacktestResult]
        filename : str
        """
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(16, 6))
            for name, res in backtest_results.items():
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                dd = (res.equity_curve / res.equity_curve.cummax() - 1) * 100
                mdd = res.metrics.get("max_drawdown", 0) * 100
                ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.2)
                ax.plot(dd.index, dd.values, color=color, linewidth=1.2,
                        label=f"{name}  (MaxDD: {mdd:.1f}%)")

            ax.axhline(0, color="#555", linewidth=0.6)
            ax.set_title("Drawdowns — Underwater Plot", fontsize=13, fontweight="bold")
            ax.set_ylabel("Drawdown (%)", fontsize=10)
            ax.legend(fontsize=9, loc="lower left")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            plt.tight_layout()
            self._save(fig, filename)

    def plot_rolling_sharpe(
        self,
        backtest_results: Dict[str, BacktestResult],
        filename: str = "rolling_sharpe",
    ) -> None:
        """
        Sharpe ratio glissant (63 jours) de toutes les stratégies.

        Parameters
        ----------
        backtest_results : Dict[str, BacktestResult]
        filename : str
        """
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(16, 5))
            for name, res in backtest_results.items():
                rs = res.rolling_sharpe_63d.dropna()
                if rs.empty:
                    continue
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                ax.plot(rs.index, rs.values, color=color, linewidth=1.4, label=name)
                ax.fill_between(rs.index, 0, rs.values, color=color, alpha=0.07)

            ax.axhline(0, color="#555", linewidth=0.6)
            ax.axhline(1, color="#555", linewidth=0.5, linestyle=":", alpha=0.4)
            ax.set_title("Sharpe Ratio Glissant (63 jours)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Sharpe (ann.)", fontsize=10)
            ax.legend(fontsize=9)
            plt.tight_layout()
            self._save(fig, filename)

    def plot_monthly_returns(
        self,
        result: BacktestResult,
        filename: str = "monthly_returns",
    ) -> None:
        """
        Heatmap des returns mensuels (années × mois).

        Parameters
        ----------
        result : BacktestResult
            Résultat d'une stratégie.
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            returns = result.returns
            if not isinstance(returns.index, pd.DatetimeIndex):
                logger.warning("Index non DatetimeIndex — heatmap mensuelle ignorée.")
                return

            monthly = (1 + returns).resample("ME").prod() - 1
            pivot = monthly.groupby(
                [monthly.index.year, monthly.index.month]
            ).first().unstack(level=1) * 100

            months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                      "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
            pivot.columns = [months[m - 1] for m in pivot.columns]

            fig, ax = plt.subplots(figsize=(15, max(4, len(pivot) * 0.5)))
            finite_vals = pivot.values[np.isfinite(pivot.values)]
            max_val = max(abs(finite_vals).max(), 1) if len(finite_vals) > 0 else 5

            sns.heatmap(
                pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, vmin=-max_val, vmax=max_val,
                linewidths=0.4, linecolor="#30363d",
                ax=ax, annot_kws={"size": 8},
                cbar_kws={"label": "Return (%)"},
            )
            ax.set_title(
                f"Returns Mensuels — {result.strategy_name}",
                fontsize=13, fontweight="bold",
            )
            ax.set_ylabel("Année", fontsize=9)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            self._save(fig, filename)

    def plot_latent_space(
        self,
        latent: np.ndarray,
        regimes: np.ndarray,
        filename: str = "latent_space",
    ) -> None:
        """
        Projection 2D de l'espace latent VAE via PCA, coloré par régime.

        Parameters
        ----------
        latent : np.ndarray
            Représentations latentes — shape (N, latent_dim).
        regimes : np.ndarray
            Régimes associés — shape (N,).
        filename : str
            Nom du fichier de sortie.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        with plt.rc_context(_RC):
            X = StandardScaler().fit_transform(latent)
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_ * 100

            fig, ax = plt.subplots(figsize=(9, 7))
            for k in sorted(set(int(r) for r in regimes)):
                mask = regimes == k
                color = REGIME_COLORS.get(k, "#888")
                label = REGIME_LABELS.get(k, f"R{k}")
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=color, alpha=0.45, s=12,
                    label=f"{label} (n={mask.sum()})",
                    rasterized=True,
                )

            ax.set_title(
                f"Espace Latent VAE — PCA 2D\n"
                f"(PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            ax.legend(fontsize=9, markerscale=2)
            plt.tight_layout()
            self._save(fig, filename)

    def plot_training_history(
        self,
        history: Dict,
        filename: str = "training_history",
    ) -> None:
        """
        Courbes d'apprentissage du VAE : total loss, reconstruction, KL.

        Parameters
        ----------
        history : Dict
            Historique d'entraînement (train_loss, val_loss, beta_values, etc.).
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            for ax, train_key, val_key, title in [
                (axes[0], "train_loss", "val_loss", "ELBO (Loss totale)"),
                (axes[1], "train_recon", "val_recon", "Reconstruction Loss (MSE)"),
                (axes[2], "train_kl", "val_kl", "KL Divergence"),
            ]:
                if train_key in history:
                    ax.plot(history[train_key], color="#58a6ff", linewidth=1.5, label="Train")
                if val_key in history:
                    ax.plot(history[val_key], color="#ffa657", linewidth=1.5, label="Val")
                if "best_epoch" in history:
                    ax.axvline(history["best_epoch"], color="#f78166",
                               linewidth=1, linestyle="--", label=f"Best epoch ({history['best_epoch']})")
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel("Epoch", fontsize=9)
                ax.legend(fontsize=8)

            if "beta_values" in history and history["beta_values"]:
                ax_beta = axes[2].twinx()
                ax_beta.plot(
                    history["beta_values"], color="#79c0ff",
                    linewidth=1, linestyle=":", alpha=0.6, label="β (KL weight)",
                )
                ax_beta.set_ylabel("β", color="#79c0ff", fontsize=8)
                ax_beta.tick_params(axis="y", colors="#79c0ff")

            fig.suptitle("Historique d'Entraînement du VAE", fontsize=14, fontweight="bold")
            plt.tight_layout()
            self._save(fig, filename)

    def plot_dashboard(
        self,
        prices: pd.Series,
        regimes: np.ndarray,
        backtest_results: Dict[str, BacktestResult],
        transition_matrix: Optional[np.ndarray] = None,
        filename: str = "full_dashboard",
    ) -> None:
        """
        Dashboard agrégé 3×2 pour le rapport final.

        Layout :
        ┌─────────────────────┬─────────────────────┐
        │  Régimes / Prix     │  Matrice Transition │
        ├─────────────────────┼─────────────────────┤
        │  Equity Curves      │  Drawdowns          │
        ├─────────────────────┼─────────────────────┤
        │  Rolling Sharpe     │  Distribution Régim │
        └─────────────────────┴─────────────────────┘

        Parameters
        ----------
        prices : pd.Series
            Cours du benchmark.
        regimes : np.ndarray
            Régimes VAE-HMM.
        backtest_results : Dict[str, BacktestResult]
            Résultats des stratégies.
        transition_matrix : np.ndarray, optional
            Matrice de transition HMM.
        filename : str
            Nom du fichier de sortie.
        """
        with plt.rc_context(_RC):
            fig = plt.figure(figsize=(22, 17))
            fig.suptitle(
                "VAE-HMM Market Regime Detection — Dashboard Complet",
                fontsize=15, fontweight="bold", y=0.98,
            )
            gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.22)

            # --- 1. Régimes sur prix ---
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(prices.index, prices.values, color="#e6edf3", linewidth=1.0, zorder=5)
            self._shade_regimes(ax1, prices.index, regimes)
            patches = [
                mpatches.Patch(facecolor=REGIME_COLORS.get(k, "#888"), alpha=0.4,
                               label=REGIME_LABELS.get(k, f"R{k}"))
                for k in sorted(set(int(r) for r in regimes))
            ]
            ax1.legend(handles=patches, fontsize=7, loc="upper left")
            ax1.set_title("Régimes VAE-HMM / Prix SPY", fontsize=10, fontweight="bold")
            ax1.set_ylabel("Prix ($)", fontsize=8)

            # --- 2. Matrice de transition ---
            ax2 = fig.add_subplot(gs[0, 1])
            if transition_matrix is not None:
                n = transition_matrix.shape[0]
                cmap_tm = LinearSegmentedColormap.from_list("tm", ["#0d1117", "#388bfd"])
                sns.heatmap(
                    transition_matrix, annot=True, fmt=".3f", cmap=cmap_tm,
                    vmin=0, vmax=1,
                    xticklabels=[f"R{k}" for k in range(n)],
                    yticklabels=[f"R{k}" for k in range(n)],
                    ax=ax2, linewidths=0.4, linecolor="#30363d",
                    annot_kws={"size": 9}, cbar_kws={"shrink": 0.8},
                )
                ax2.set_title("Matrice de Transition HMM", fontsize=10, fontweight="bold")
            else:
                ax2.text(0.5, 0.5, "Matrice non disponible", ha="center", va="center",
                         color="#8b949e", fontsize=10)
                ax2.set_title("Matrice de Transition HMM", fontsize=10, fontweight="bold")

            # --- 3. Equity curves ---
            ax3 = fig.add_subplot(gs[1, 0])
            for name, res in backtest_results.items():
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                lw = 2.0 if MODEL_VAE_HMM in name else 1.1
                ls = "-" if MODEL_VAE_HMM in name else ("--" if MODEL_BUY_HOLD in name else "-.")
                sr = res.metrics.get("sharpe_ratio", 0)
                ax3.plot(res.equity_curve.index, res.equity_curve.values,
                         color=color, linewidth=lw, linestyle=ls,
                         label=f"{name} (SR={sr:.2f})")
            ax3.axhline(100, color="#555", linewidth=0.6, linestyle=":")
            ax3.set_title("Equity Curves", fontsize=10, fontweight="bold")
            ax3.set_ylabel("Base 100", fontsize=8)
            ax3.legend(fontsize=7, loc="upper left")

            # --- 4. Drawdowns ---
            ax4 = fig.add_subplot(gs[1, 1])
            for name, res in backtest_results.items():
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                dd = (res.equity_curve / res.equity_curve.cummax() - 1) * 100
                ax4.fill_between(dd.index, dd.values, 0, color=color, alpha=0.2)
                ax4.plot(dd.index, dd.values, color=color, linewidth=0.9, label=name)
            ax4.axhline(0, color="#555", linewidth=0.6)
            ax4.set_title("Drawdowns", fontsize=10, fontweight="bold")
            ax4.set_ylabel("Drawdown (%)", fontsize=8)
            ax4.legend(fontsize=7, loc="lower left")
            ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

            # --- 5. Rolling Sharpe ---
            ax5 = fig.add_subplot(gs[2, 0])
            for name, res in backtest_results.items():
                rs = res.rolling_sharpe_63d.dropna()
                if rs.empty:
                    continue
                color = _STRATEGY_PALETTE.get(name, _DEFAULT_COLOR)
                ax5.plot(rs.index, rs.values, color=color, linewidth=1.2, label=name)
            ax5.axhline(0, color="#555", linewidth=0.6)
            ax5.set_title("Sharpe Glissant (63j)", fontsize=10, fontweight="bold")
            ax5.set_ylabel("Sharpe (ann.)", fontsize=8)
            ax5.legend(fontsize=7)

            # --- 6. Distribution régimes ---
            ax6 = fig.add_subplot(gs[2, 1])
            n_r = int(max(regimes)) + 1
            counts = [(regimes == k).sum() for k in range(n_r)]
            total = sum(counts)
            freqs = [c / total * 100 for c in counts]
            bar_colors = [REGIME_COLORS.get(k, "#888") for k in range(n_r)]
            bars = ax6.bar(range(n_r), freqs, color=bar_colors, alpha=0.85,
                           edgecolor="#30363d", linewidth=0.7)
            ax6.set_xticks(range(n_r))
            ax6.set_xticklabels(
                [REGIME_LABELS.get(k, f"R{k}") for k in range(n_r)],
                rotation=12, ha="right", fontsize=7,
            )
            for bar, freq in zip(bars, freqs):
                ax6.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{freq:.1f}%", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#e6edf3",
                )
            ax6.set_title("Distribution des Régimes VAE-HMM", fontsize=10, fontweight="bold")
            ax6.set_ylabel("Fréquence (%)", fontsize=8)

            self._save(fig, filename)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _shade_regimes(
        self,
        ax: plt.Axes,
        index: pd.Index,
        regimes: np.ndarray,
    ) -> None:
        """Colore le fond de l'axe selon le régime actif."""
        n = len(index)
        if n == 0:
            return

        regime_arr = np.asarray(regimes)
        changes = np.where(np.diff(regime_arr, prepend=regime_arr[0] - 1) != 0)[0]
        changes = np.append(changes, n)

        for i in range(len(changes) - 1):
            start = changes[i]
            end = changes[i + 1]
            k = int(regime_arr[start])
            color = REGIME_COLORS.get(k, "#888888")
            ax.axvspan(
                index[start], index[min(end, n - 1)],
                alpha=0.15, color=color, zorder=1, linewidth=0,
            )

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Sauvegarde la figure et ferme."""
        path = self.output_dir / f"{filename}.{self.fmt}"
        fig.savefig(
            path, dpi=self.dpi, format=self.fmt,
            bbox_inches="tight", facecolor=fig.get_facecolor(),
        )
        logger.debug(f"  → {path.name}")
        plt.close(fig)
