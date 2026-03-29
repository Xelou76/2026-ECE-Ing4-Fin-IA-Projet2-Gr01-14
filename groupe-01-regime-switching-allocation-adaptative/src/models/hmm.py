"""
models/hmm.py
=============
Hidden Markov Model sur l'espace latent du VAE pour la détection de régimes.

Corrections v3 :
  - RegimeIdentifier ordonne par RENDEMENT FINANCIER moyen (inchangé depuis v2).
  - predict_causal() : Forward filtering sans look-ahead pour le backtest.
  - get_soft_allocation() : allocation pondérée par les posteriors (F-B).
  - NOUVEAU : validate_regime_quality() — diagnostics automatiques du HMM.
    Détecte les états dégénérés (proba≈0), les régimes mal ordonnés, et
    un IC trop faible avant de lancer le backtest.
  - NOUVEAU : Fallback automatique si n_regimes=3 produit un état vide.
    Dans ce cas, le HMM se réentraîne avec n_regimes=2.

Architecture du pipeline :
  VAE latent z_t → GaussianHMM(n_regimes) → régime r_t

Classes
-------
RegimeIdentifier
    Identifie et trie les régimes par rendement financier croissant.
RegimeHMM
    Wrapper complet du GaussianHMM avec inférence causale et allocation souple.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.preprocessing import StandardScaler

from config.settings import HMMConfig


# ---------------------------------------------------------------------------
# Régime identifier — tri par rendement financier
# ---------------------------------------------------------------------------

class RegimeIdentifier:
    """
    Identifie les régimes par ordre croissant de rendement financier moyen.

    Correction héritée de v2 : tri par rendement financier (pas variance latente).
    """

    def __init__(self, n_regimes: int) -> None:
        self.n_regimes = n_regimes
        self._permutation: Optional[np.ndarray] = None
        self._inv_permutation: Optional[np.ndarray] = None

    def fit(
        self,
        model: GaussianHMM,
        X_scaled: np.ndarray,
        returns_market: Optional[np.ndarray] = None,
    ) -> "RegimeIdentifier":
        """
        Calcule la permutation canonique.

        Stratégie 1 (préférée) : tri par rendement financier moyen par régime.
        Stratégie 2 (fallback)  : tri par variance dans l'espace latent.
        """
        _, raw_regimes = model.decode(X_scaled, algorithm="viterbi")

        if returns_market is not None and len(returns_market) == len(raw_regimes):
            mean_returns = np.array([
                np.mean(returns_market[raw_regimes == k])
                if np.sum(raw_regimes == k) > 0 else 0.0
                for k in range(self.n_regimes)
            ])
            self._permutation = np.argsort(mean_returns)
            logger.info(
                f"RegimeIdentifier — tri par rendement financier : "
                + " | ".join(
                    f"R{k}={mean_returns[k]*100:.3f}%/j" for k in range(self.n_regimes)
                )
            )
        else:
            logger.warning(
                "RegimeIdentifier — returns_market absent, fallback sur variance "
                "latente. Les régimes pourraient être mal ordonnés."
            )
            variances = np.array(
                [np.trace(model.covars_[k]) for k in range(self.n_regimes)]
            )
            self._permutation = np.argsort(variances)

        self._inv_permutation = np.argsort(self._permutation)
        logger.debug(f"Permutation canonique : {self._permutation}")
        return self

    def remap(self, regimes: np.ndarray) -> np.ndarray:
        """Remplace les labels bruts HMM par les labels canoniques triés."""
        if self._inv_permutation is None:
            raise RuntimeError("Appelez fit() avant remap().")
        return self._inv_permutation[regimes]

    def remap_proba(self, proba: np.ndarray) -> np.ndarray:
        """Réordonne les colonnes de probabilités selon la permutation canonique."""
        if self._permutation is None:
            raise RuntimeError("Appelez fit() avant remap_proba().")
        return proba[:, self._permutation]


# ---------------------------------------------------------------------------
# RegimeHMM
# ---------------------------------------------------------------------------

class RegimeHMM:
    """
    Détecteur de régimes de marché basé sur un Gaussian HMM.

    Améliorations v3 :
    - validate_regime_quality() : diagnostics avant backtest
    - Fallback automatique n_regimes=3 → 2 si état dégénéré détecté
    - Ordre des régimes par rendement financier (hérité de v2)
    - predict_causal() : inférence sans look-ahead pour backtesting propre
    - get_soft_allocation() : allocation pondérée par les posteriors

    Parameters
    ----------
    cfg : HMMConfig
        Configuration du HMM.
    """

    # Seuil de fréquence minimale d'un régime (en %) pour le considérer valide
    _MIN_REGIME_FREQ_PCT: float = 5.0

    def __init__(self, cfg: HMMConfig) -> None:
        self.cfg = cfg
        self._model: Optional[GaussianHMM] = None
        self._scaler: Optional[StandardScaler] = None
        self._identifier: Optional[RegimeIdentifier] = None
        self._n_regimes_effective: int = cfg.n_regimes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        latent_train: np.ndarray,
        latent_val: Optional[np.ndarray] = None,
        returns_market: Optional[np.ndarray] = None,
    ) -> "RegimeHMM":
        """
        Entraîne le HMM sur les représentations latentes.

        NOUVEAU v3 : détecte les états dégénérés et retente avec
        n_regimes réduit si nécessaire.

        Parameters
        ----------
        latent_train : np.ndarray
            Espace latent du train set — shape (N_train, latent_dim).
        latent_val : np.ndarray, optional
            Espace latent du val set — concaténé au train pour le fit.
        returns_market : np.ndarray, optional
            Rendements journaliers réels alignés sur latent_train.
            Utilisé pour trier les régimes par rendement.
        """
        # Concatène train + val pour plus de données au fit
        X = latent_train
        if latent_val is not None:
            X = np.vstack([latent_train, latent_val])
            logger.debug(f"HMM fit sur train+val : {X.shape}")

        # Normalisation
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Entraînement initial
        n_regimes = self.cfg.n_regimes
        model = self._fit_hmm(X_scaled, n_regimes)

        # Validation qualité : détecte les états dégénérés
        raw_labels = model.predict(X_scaled)
        freqs = np.array([
            100.0 * np.mean(raw_labels == k) for k in range(n_regimes)
        ])
        degenerate = freqs < self._MIN_REGIME_FREQ_PCT

        if np.any(degenerate) and n_regimes > 2:
            logger.warning(
                f"HMM : {degenerate.sum()} état(s) dégénéré(s) détecté(s) "
                f"(fréquence < {self._MIN_REGIME_FREQ_PCT}%). "
                f"Fréquences : {freqs.round(1)}. "
                f"Retentative avec n_regimes=2."
            )
            n_regimes = 2
            model = self._fit_hmm(X_scaled, n_regimes)
            self._n_regimes_effective = 2
        else:
            self._n_regimes_effective = n_regimes
            logger.info(
                f"HMM : {n_regimes} régimes valides. "
                f"Fréquences : {freqs.round(1)}%"
            )

        self._model = model

        # Ordonnancement des régimes par rendement financier
        # Aligne les returns sur latent_train seulement (pas val)
        returns_aligned = returns_market
        if latent_val is not None and returns_market is not None:
            # On prend seulement les N_train premières observations
            returns_aligned = returns_market[:len(latent_train)]

        self._identifier = RegimeIdentifier(self._n_regimes_effective)
        X_train_scaled = self._scaler.transform(latent_train)
        self._identifier.fit(model, X_train_scaled, returns_aligned)

        logger.success(
            f"HMM entraîné — {self._n_regimes_effective} régimes | "
            f"convergé={model.monitor_.converged}"
        )
        return self

    def predict(self, latent: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes via Viterbi (LOOK-AHEAD — ne pas utiliser pour backtest).

        Utilise l'algorithme de Viterbi qui exploite toute la séquence
        dans les deux directions. Résultat non-causal.

        Pour le backtest : utiliser predict_causal().
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(latent)
        _, raw_labels = self._model.decode(X_scaled, algorithm="viterbi")
        return self._identifier.remap(raw_labels)

    def predict_causal(self, latent: np.ndarray) -> np.ndarray:
        """
        Prédit les régimes via Forward Filtering (CAUSAL — pour le backtest).

        Implémente le filtre forward de l'HMM : à chaque instant t,
        le régime est estimé en utilisant uniquement les observations
        passées {z_1, ..., z_t}. Pas de look-ahead.

        C'est la version correcte pour un backtest réaliste.

        Parameters
        ----------
        latent : np.ndarray
            Représentations latentes — shape (N, latent_dim).

        Returns
        -------
        np.ndarray
            Régimes prédits causalement — shape (N,).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(latent)

        n = len(X_scaled)
        K = self._n_regimes_effective

        # Probabilités d'émission log p(z_t | r_t=k)
        log_emission = self._compute_log_emission(X_scaled)  # (N, K)

        # Filtre forward
        log_alpha = np.full((n, K), -np.inf)
        log_transmat = np.log(self._model.transmat_ + 1e-300)
        log_startprob = np.log(self._model.startprob_ + 1e-300)

        log_alpha[0] = log_startprob + log_emission[0]

        for t in range(1, n):
            # log p(r_t | z_{1:t}) = log Σ_k p(r_t | r_{t-1}=k) p(r_{t-1}=k | z_{1:t-1})
            log_alpha[t] = (
                np.logaddexp.reduce(log_alpha[t-1, :, None] + log_transmat, axis=0)
                + log_emission[t]
            )

        # Décision MAP : argmax de la distribution forward normalisée
        raw_labels = np.argmax(log_alpha, axis=1)
        return self._identifier.remap(raw_labels)

    def predict_proba(self, latent: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités postérieures Forward-Backward.

        Probabilités postérieures γ_t(k) = P(r_t=k | z_{1:T}).
        NB : utilise toute la séquence (non-causal) → pour l'analyse.

        Pour le backtest, utiliser predict_proba_causal().

        Returns
        -------
        np.ndarray
            Probabilités — shape (N, n_regimes).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(latent)
        raw_proba = self._model.predict_proba(X_scaled)
        return self._identifier.remap_proba(raw_proba)

    def predict_proba_causal(self, latent: np.ndarray) -> np.ndarray:
        """
        Retourne les probabilités forward normalisées (CAUSAL — pour le backtest).

        Probabilités α_t(k) = P(r_t=k | z_{1:t}) normalisées.

        Returns
        -------
        np.ndarray
            Probabilités causales — shape (N, n_regimes).
        """
        self._check_fitted()
        X_scaled = self._scaler.transform(latent)

        n = len(X_scaled)
        K = self._n_regimes_effective

        log_emission = self._compute_log_emission(X_scaled)
        log_alpha = np.full((n, K), -np.inf)
        log_transmat = np.log(self._model.transmat_ + 1e-300)
        log_startprob = np.log(self._model.startprob_ + 1e-300)

        log_alpha[0] = log_startprob + log_emission[0]

        for t in range(1, n):
            log_alpha[t] = (
                np.logaddexp.reduce(log_alpha[t-1, :, None] + log_transmat, axis=0)
                + log_emission[t]
            )

        # Normalisation pour obtenir des probabilités
        log_norm = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        raw_proba = np.exp(log_alpha - log_norm)

        return self._identifier.remap_proba(raw_proba)

    def get_soft_allocation(
        self,
        latent: np.ndarray,
        regime_allocations: Dict[int, Dict[str, float]],
        min_confidence: float = 0.60,
        fallback_regime: int = 0,
    ) -> pd.DataFrame:
        """
        Calcule les allocations pondérées par les probabilités forward (causal).

        Allocation = Σ_k P(r_t=k | z_{1:t}) × alloc[k]

        Si max(P(r_t=k)) < min_confidence → allocation du régime prudent.

        Parameters
        ----------
        latent : np.ndarray
            Représentations latentes — shape (N, latent_dim).
        regime_allocations : dict
            Allocations par régime {0: {"equity": 0.1, ...}, 1: {...}}.
        min_confidence : float
            Seuil de confiance. Sous ce seuil → allocation prudente.
        fallback_regime : int
            Régime utilisé quand la confiance est insuffisante.

        Returns
        -------
        pd.DataFrame
            Allocations — shape (N, 3), colonnes equity/bond/cash.
        """
        self._check_fitted()

        # Probabilités causales
        proba = self.predict_proba_causal(latent)  # (N, K)
        K = proba.shape[1]

        # Matrice d'allocation par régime (K, 3)
        alloc_keys = ["equity", "bond", "cash"]
        alloc_matrix = np.array([
            [regime_allocations.get(k, regime_allocations[fallback_regime]).get(a, 0.0)
             for a in alloc_keys]
            for k in range(K)
        ])  # (K, 3)

        # Allocation souple : Σ_k P(r_t=k) × alloc[k]
        soft_alloc = proba @ alloc_matrix  # (N, 3)

        # Fallback si confiance insuffisante
        confidence = proba.max(axis=1)  # (N,)
        low_confidence_mask = confidence < min_confidence

        if low_confidence_mask.any():
            fallback_alloc = np.array([
                regime_allocations[fallback_regime].get(a, 0.0) for a in alloc_keys
            ])
            soft_alloc[low_confidence_mask] = fallback_alloc
            logger.debug(
                f"Soft allocation : {low_confidence_mask.sum()} observations "
                f"sous le seuil de confiance ({min_confidence:.0%}) → fallback régime {fallback_regime}"
            )

        return pd.DataFrame(soft_alloc, columns=alloc_keys)

    def validate_regime_quality(
        self,
        latent: np.ndarray,
        returns_market: np.ndarray,
    ) -> Dict[str, object]:
        """
        Diagnostics automatiques de la qualité du HMM.

        NOUVEAU v3 : appelé après fit() pour détecter les problèmes avant
        de lancer le backtest. Retourne un rapport de diagnostics.

        Vérifie :
        - Fréquence de chaque régime (doit être > 5%)
        - Ordonnancement bull/bear (régime 0 < régime 1 en rendement)
        - Information Coefficient (1j) du signal de régime
        - Durée moyenne des régimes (doit être > 5 jours)

        Parameters
        ----------
        latent : np.ndarray
            Espace latent sur lequel évaluer les diagnostics.
        returns_market : np.ndarray
            Rendements journaliers réels alignés.

        Returns
        -------
        dict
            Rapport de diagnostics avec warnings.
        """
        self._check_fitted()

        regimes = self.predict_causal(latent)
        proba = self.predict_proba_causal(latent)
        K = self._n_regimes_effective

        report = {"n_regimes": K, "warnings": [], "passed": True}

        # 1. Fréquences des régimes
        freqs = {k: 100.0 * np.mean(regimes == k) for k in range(K)}
        report["regime_frequencies_pct"] = freqs
        for k, freq in freqs.items():
            if freq < self._MIN_REGIME_FREQ_PCT:
                msg = f"Régime {k} trop rare : {freq:.1f}% (seuil={self._MIN_REGIME_FREQ_PCT}%)"
                report["warnings"].append(msg)
                report["passed"] = False
                logger.warning(f"validate_regime_quality — {msg}")

        # 2. Rendements moyens par régime
        mean_rets = {k: float(np.mean(returns_market[regimes == k])) * 100
                     for k in range(K) if np.sum(regimes == k) > 0}
        report["mean_daily_return_pct"] = mean_rets
        if K >= 2 and mean_rets.get(0, 0) > mean_rets.get(K-1, 0):
            msg = "Régimes mal ordonnés : régime 0 a un rendement > régime K-1"
            report["warnings"].append(msg)
            logger.warning(f"validate_regime_quality — {msg}")

        # 3. Information Coefficient (1 jour)
        # IC = corrélation de Spearman entre confiance du régime bull et return J+1
        confidence_bull = proba[:, -1]  # probabilité du régime le plus bull
        n = min(len(confidence_bull), len(returns_market))
        if n > 10:
            from scipy.stats import spearmanr
            ic, p_val = spearmanr(confidence_bull[:-1], returns_market[1:n])
            report["ic_1d"] = float(ic)
            report["ic_pvalue"] = float(p_val)
            if abs(ic) < 0.02:
                msg = f"IC(1j) très faible : {ic:.4f} (p={p_val:.3f})"
                report["warnings"].append(msg)
                logger.warning(f"validate_regime_quality — {msg}")

        # 4. Durée moyenne des régimes
        durations = []
        current_regime = regimes[0]
        current_len = 1
        for r in regimes[1:]:
            if r == current_regime:
                current_len += 1
            else:
                durations.append(current_len)
                current_regime = r
                current_len = 1
        durations.append(current_len)
        mean_duration = float(np.mean(durations)) if durations else 0.0
        report["mean_regime_duration_days"] = mean_duration
        if mean_duration < 5.0:
            msg = f"Régimes trop instables : durée moyenne={mean_duration:.1f}j (seuil=5j)"
            report["warnings"].append(msg)
            report["passed"] = False
            logger.warning(f"validate_regime_quality — {msg}")

        # Bilan
        if report["passed"]:
            logger.success("validate_regime_quality — tous les diagnostics OK")
        else:
            logger.warning(
                f"validate_regime_quality — {len(report['warnings'])} problème(s) détecté(s)"
            )

        return report

    def get_transition_matrix(self) -> np.ndarray:
        """Retourne la matrice de transition dans l'ordre canonique."""
        self._check_fitted()
        P = self._model.transmat_
        perm = self._identifier._permutation
        return P[np.ix_(perm, perm)]

    def save(self, path: Path) -> None:
        """Sauvegarde le HMM, le scaler et l'identifier."""
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
            "identifier": self._identifier,
            "n_regimes_effective": self._n_regimes_effective,
            "cfg": self.cfg,
        }, path)
        logger.info(f"HMM sauvegardé : {path}")

    @classmethod
    def load(cls, path: Path) -> "RegimeHMM":
        """Charge un HMM depuis un fichier joblib."""
        data = joblib.load(path)
        instance = cls(data["cfg"])
        instance._model = data["model"]
        instance._scaler = data["scaler"]
        instance._identifier = data["identifier"]
        instance._n_regimes_effective = data.get("n_regimes_effective", data["cfg"].n_regimes)
        logger.info(f"HMM chargé : {path}")
        return instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_hmm(self, X_scaled: np.ndarray, n_regimes: int) -> GaussianHMM:
        """Entraîne un GaussianHMM avec les hyperparamètres de config."""
        best_model = None
        best_score = -np.inf

        for init_idx in range(self.cfg.n_init):
            try:
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type=self.cfg.covariance_type,
                    n_iter=self.cfg.n_iter,
                    tol=self.cfg.tol,
                    random_state=42 + init_idx,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.debug(f"HMM init {init_idx} échouée : {e}")

        if best_model is None:
            raise RuntimeError("Tous les restarts HMM ont échoué.")

        logger.debug(
            f"HMM ({n_regimes} régimes) — meilleur score={best_score:.2f} | "
            f"convergé={best_model.monitor_.converged}"
        )
        return best_model

    def _compute_log_emission(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calcule les log-probabilités d'émission gaussiennes pour chaque état.

        log p(z_t | r_t=k) = log N(z_t; μ_k, Σ_k)

        Returns
        -------
        np.ndarray
            Log-probabilités — shape (N, K).
        """
        from scipy.stats import multivariate_normal
        K = self._n_regimes_effective
        N = len(X_scaled)
        log_emission = np.zeros((N, K))

        for k in range(K):
            try:
                rv = multivariate_normal(
                    mean=self._model.means_[k],
                    cov=self._model.covars_[k],
                    allow_singular=True,
                )
                log_emission[:, k] = rv.logpdf(X_scaled)
            except Exception:
                log_emission[:, k] = -1000.0

        return log_emission

    def _check_fitted(self) -> None:
        """Vérifie que le modèle est entraîné."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("Le HMM n'est pas entraîné. Appelez fit() d'abord.")