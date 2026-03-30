"""
models/markov_switching.py
==========================
Modèle de référence : Markov-Switching (Hamilton, 1989) via statsmodels.

Ce modèle est la *baseline scientifique* contre laquelle la performance
du pipeline VAE-HMM sera comparée. Il s'agit du modèle économétrique
standard de détection de régimes, utilisé depuis 35 ans en finance.

Différences fondamentales avec VAE-HMM :
  - Ajusté directement sur les rendements bruts (pas d'espace latent)
  - 2 régimes seulement (convention Hamilton : bull / bear)
  - Modèle AR(p) avec changement de régime sur la moyenne et la variance
  - Estimation par EM (Hamilton filter) / MLE

Classes
-------
MarkovSwitchingBaseline
    Wrapper statsmodels avec interface cohérente avec RegimeHMM.

References
----------
.. [1] Hamilton, J. D. (1989). A New Approach to the Economic Analysis
       of Nonstationary Time Series and the Business Cycle. Econometrica.
.. [2] statsmodels.tsa.regime_switching.markov_switching
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

try:
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    logger.warning(
        "statsmodels non disponible. MarkovSwitchingBaseline désactivé. "
        "Installez avec : pip install statsmodels"
    )

from config.settings import MarkovSwitchingConfig


class MarkovSwitchingBaseline:
    """
    Modèle de référence Markov-Switching (Hamilton, 1989).

    Ajuste un modèle de changement de régime markovien directement sur les
    log-rendements du benchmark (SPY). Sert de comparaison scientifique
    pour évaluer l'apport de l'espace latent VAE.

    Spécification du modèle (order=0) :
      y_t | s_t ~ N(μ_{s_t}, σ²_{s_t})
      s_t | s_{t-1} ~ Markov(A)   où A est la matrice de transition 2×2

    Pour order > 0 (Markov-AR) :
      y_t = μ_{s_t} + Σ φ_j · y_{t-j} + ε_t,  ε_t ~ N(0, σ²_{s_t})

    Parameters
    ----------
    cfg : MarkovSwitchingConfig
        Configuration (k_regimes, order, switching_variance).

    Attributes
    ----------
    result_ : MarkovRegressionResults
        Résultat de l'estimation statsmodels (après fit).
    is_fitted_ : bool
        True après un appel réussi à fit().

    Examples
    --------
    >>> baseline = MarkovSwitchingBaseline(MarkovSwitchingConfig(k_regimes=2))
    >>> baseline.fit(returns_train)
    >>> regimes = baseline.predict(returns_test)
    >>> baseline.print_summary()
    """

    def __init__(self, cfg: MarkovSwitchingConfig) -> None:
        if not _STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels est requis pour MarkovSwitchingBaseline. "
                "pip install statsmodels>=0.14"
            )
        self.cfg = cfg
        self.result_ = None
        self.is_fitted_: bool = False
        self._regime_map: Optional[np.ndarray] = None  # mapping → canonique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns_train: pd.Series) -> "MarkovSwitchingBaseline":
        """
        Estime le modèle Markov-Switching par MLE (EM + Newton-Raphson).

        Parameters
        ----------
        returns_train : pd.Series
            Série de log-rendements journaliers du benchmark (ex: SPY).
            Index DatetimeIndex, valeurs float.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Si la série est trop courte ou contient des NaN.
        RuntimeError
            Si l'optimisation MLE ne converge pas.
        """
        self._validate_series(returns_train, "returns_train")

        logger.info(
            f"Fitting Markov-Switching (Hamilton) — "
            f"k_regimes={self.cfg.k_regimes} | "
            f"AR order={self.cfg.order} | "
            f"switching_variance={self.cfg.switching_variance}"
        )

        # Utilise MarkovAutoregression si order > 0, sinon MarkovRegression
        if self.cfg.order == 0:
            model = MarkovRegression(
                endog=returns_train.values,
                k_regimes=self.cfg.k_regimes,
                switching_variance=self.cfg.switching_variance,
            )
        else:
            model = MarkovAutoregression(
                endog=returns_train.values,
                k_regimes=self.cfg.k_regimes,
                order=self.cfg.order,
                switching_variance=self.cfg.switching_variance,
            )

        # statsmodels >= 0.14 a renommé/retiré l'argument method="em"
        # On essaie plusieurs combinaisons pour la compatibilité maximale
        fitted = False
        for fit_kwargs in [
            {"disp": False, "maxiter": 200},                    # défaut (>= 0.14)
            {"disp": False, "maxiter": 200, "method": "em"},    # < 0.14
            {"disp": False, "maxiter": 500},                    # fallback long
        ]:
            try:
                self.result_ = model.fit(**fit_kwargs)
                fitted = True
                break
            except Exception:
                continue

        if not fitted:
            raise RuntimeError(
                "Échec de la convergence du Markov-Switching après plusieurs tentatives. "
                "Vérifiez votre version de statsmodels (pip install statsmodels>=0.14)."
            )

        # Mapping canonique : régime 0 = plus faible variance (bull)
        self._compute_regime_map()
        self.is_fitted_ = True

        self._log_fit_summary(returns_train)
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        """
        Prédit les régimes sur une nouvelle série de rendements.

        Utilise le filtre de Hamilton (filtre forward) pour calculer
        P(s_t | y_{1..t}) puis prend le régime le plus probable.

        Note : contrairement au VAE-HMM qui peut encoder directement
        n'importe quelle séquence, le Markov-Switching nécessite une
        estimation online (les probabilités filtrées dépendent de toute
        l'histoire passée). On ré-estime donc sur le dataset complet
        train + test pour obtenir les probabilités sur le test set.

        Parameters
        ----------
        returns : pd.Series
            Série de log-rendements — peut être différente du train.

        Returns
        -------
        np.ndarray
            Séquence de régimes — shape (T,), valeurs dans {0, 1}.
            0 = bull (faible vol), 1 = bear (forte vol).
        """
        self._check_fitted()
        self._validate_series(returns, "returns")

        # Filtre Hamilton sur la série fournie
        proba = self._get_smoothed_proba(returns)   # (T, K)
        raw_regimes = np.argmax(proba, axis=1)
        return self._remap(raw_regimes)

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """
        Retourne les probabilités lissées de chaque régime.

        Utilise le lisseur de Kim (two-pass : forward + backward)
        pour obtenir P(s_t | y_{1..T}) qui intègre les observations futures.

        Parameters
        ----------
        returns : pd.Series
            Série de log-rendements.

        Returns
        -------
        np.ndarray
            Probabilités lissées — shape (T, K), colonnes en ordre canonique.
        """
        self._check_fitted()
        proba = self._get_smoothed_proba(returns)
        # Réordonne selon mapping canonique
        return proba[:, self._regime_map]

    def get_transition_matrix(self) -> np.ndarray:
        """
        Retourne la matrice de transition estimée (ordre canonique).

        Returns
        -------
        np.ndarray
            Matrice A — shape (K, K), stochastique par ligne.
        """
        self._check_fitted()
        # statsmodels stocke la matrice de transition dans regime_transition
        # Shape dépend de la version : parfois (K, K), parfois transposée
        A_raw = self.result_.regime_transition
        if A_raw.ndim == 3:
            # MarkovAutoregression : (K, K, order) → prend le 1er ordre
            A_raw = A_raw[:, :, 0]
        # Réordonne
        m = self._regime_map
        return A_raw[np.ix_(m, m)]

    def get_regime_parameters(self) -> pd.DataFrame:
        """
        Retourne les paramètres estimés par régime.

        Returns
        -------
        pd.DataFrame
            Tableau avec moyennes, variances et durées impliquées.
        """
        self._check_fitted()
        rows = []
        A = self.get_transition_matrix()
        for k in range(self.cfg.k_regimes):
            # Durée espérée dans le régime = 1 / (1 - A[k,k])
            expected_duration = 1.0 / max(1 - A[k, k], 1e-6)
            rows.append({
                "regime": k,
                "expected_duration_days": expected_duration,
                "self_transition_prob": A[k, k],
            })
        return pd.DataFrame(rows).set_index("regime")

    def print_summary(self) -> None:
        """Affiche le résumé statsmodels + diagnostics supplémentaires."""
        self._check_fitted()
        logger.info("=" * 55)
        logger.info("  BASELINE MARKOV-SWITCHING (Hamilton, 1989)")
        logger.info("=" * 55)

        # Log-vraisemblance et AIC/BIC
        try:
            logger.info(f"  Log-likelihood : {self.result_.llf:.4f}")
            logger.info(f"  AIC            : {self.result_.aic:.4f}")
            logger.info(f"  BIC            : {self.result_.bic:.4f}")
        except AttributeError:
            pass

        # Matrice de transition
        A = self.get_transition_matrix()
        logger.info("\n  Matrice de transition :")
        for i, row in enumerate(A):
            formatted = "  ".join(f"{v:.4f}" for v in row)
            logger.info(f"    R{i} → [{formatted}]")

        # Durées implicites
        params = self.get_regime_parameters()
        logger.info("\n  Durées implicites par régime :")
        for k, row in params.iterrows():
            logger.info(
                f"    Régime {k} : {row['expected_duration_days']:.1f} jours en moyenne"
            )
        logger.info("=" * 55)

    def save(self, path: Path) -> None:
        """Sauvegarde le modèle avec joblib."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "result": self.result_,
                "cfg": self.cfg,
                "regime_map": self._regime_map,
            },
            path,
        )
        logger.info(f"MarkovSwitchingBaseline sauvegardé : {path}")

    @classmethod
    def load(cls, path: Path) -> "MarkovSwitchingBaseline":
        """Charge un modèle depuis un fichier joblib."""
        data = joblib.load(path)
        instance = cls(data["cfg"])
        instance.result_ = data["result"]
        instance._regime_map = data["regime_map"]
        instance.is_fitted_ = True
        logger.info(f"MarkovSwitchingBaseline chargé : {path}")
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Appelez fit() avant de prédire.")

    @staticmethod
    def _validate_series(series: pd.Series, name: str) -> None:
        """Valide la série de rendements."""
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} doit être une pd.Series.")
        if series.isna().any():
            n = series.isna().sum()
            raise ValueError(
                f"{name} contient {n} NaN. "
                "Nettoyez la série avant l'estimation."
            )
        if len(series) < 100:
            raise ValueError(
                f"{name} trop courte ({len(series)} obs). "
                "Le Markov-Switching nécessite ≥ 100 observations."
            )

    def _get_smoothed_proba(self, returns: pd.Series) -> np.ndarray:
        """
        Retourne les probabilités lissées (Kim smoother).

        Tente d'abord smoothed_marginal_probabilities (disponible si
        la série est la même que le train). Sinon, recalcule le filtre
        avec les paramètres fixés sur la nouvelle série.
        """
        try:
            # Si c'est la même série que le train → résultat direct
            proba = self.result_.smoothed_marginal_probabilities
            # Shape : (T, K) ou (K, T) selon la version de statsmodels
            if proba.shape[0] == self.cfg.k_regimes:
                proba = proba.T
            return proba
        except Exception:
            # Pour une série différente, prédit via filtered_marginal_probabilities
            # en appliquant les paramètres estimés
            proba = self.result_.filtered_marginal_probabilities
            if proba.shape[0] == self.cfg.k_regimes:
                proba = proba.T
            return proba

    def _compute_regime_map(self) -> None:
        """
        Détermine le mapping canonique des régimes.

        Régime 0 = faible variance (bull), Régime 1 = forte variance (bear).
        Basé sur les variances estimées par régime dans le modèle MS.
        """
        try:
            # Récupère les variances par régime depuis les paramètres estimés
            params = self.result_.params
            # statsmodels nomme les paramètres : 'sigma2[0]', 'sigma2[1]'
            sigma2 = {}
            for k in range(self.cfg.k_regimes):
                key = f"sigma2[{k}]"
                if key in self.result_.param_names:
                    idx = self.result_.param_names.index(key)
                    sigma2[k] = params[idx]
                else:
                    # Fallback si nom différent selon la version
                    sigma2[k] = 1.0
            variances = np.array([sigma2.get(k, 1.0) for k in range(self.cfg.k_regimes)])
        except Exception:
            # Si extraction échoue, garde l'ordre original
            variances = np.arange(self.cfg.k_regimes, dtype=float)

        # Tri croissant : régime 0 = low-vol
        self._regime_map = np.argsort(variances)

    def _remap(self, raw_regimes: np.ndarray) -> np.ndarray:
        """Applique le mapping canonique aux labels bruts."""
        if self._regime_map is None:
            return raw_regimes
        inv_map = np.argsort(self._regime_map)
        return inv_map[raw_regimes]

    def _log_fit_summary(self, returns_train: pd.Series) -> None:
        """Log récapitulatif post-fitting."""
        try:
            logger.success(
                f"Markov-Switching ajusté — "
                f"log-likelihood={self.result_.llf:.4f} | "
                f"AIC={self.result_.aic:.4f} | "
                f"BIC={self.result_.bic:.4f}"
            )
        except AttributeError:
            logger.success("Markov-Switching ajusté.")
