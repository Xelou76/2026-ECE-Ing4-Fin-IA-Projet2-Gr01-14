"""
data/downloader.py
==================
Téléchargement et mise en cache des données de marché via yfinance.

Classes
-------
MarketDataDownloader
    Télécharge, valide, nettoie et met en cache les séries OHLCV.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class MarketDataDownloader:
    """
    Télécharge et met en cache les données OHLCV via yfinance.

    Le cache est un fichier pickle indexé par un hash des paramètres
    (tickers + dates) : deux appels identiques ne téléchargent jamais
    deux fois.

    Parameters
    ----------
    cache_dir : Path
        Répertoire de stockage du cache disque.

    Examples
    --------
    >>> dl = MarketDataDownloader(Path("data/cache"))
    >>> prices = dl.download(["SPY", "TLT"], "2010-01-01", "2023-12-31")
    >>> prices.shape
    (3521, 2)
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        column: str = "Close",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Télécharge les prix de clôture ajustés pour une liste de tickers.

        Tente d'abord le cache disque. En cas d'échec ou si
        ``force_refresh=True``, appelle l'API yfinance puis sauvegarde.

        Parameters
        ----------
        tickers : list[str]
            Liste des symboles boursiers (ex: ["SPY", "TLT", "GLD"]).
        start_date : str
            Date de début au format ISO 8601 ("YYYY-MM-DD").
        end_date : str
            Date de fin au format ISO 8601 ("YYYY-MM-DD").
        column : str, optional
            Colonne OHLCV à extraire. Par défaut "Close".
        force_refresh : bool, optional
            Si True, ignore le cache et re-télécharge. Par défaut False.

        Returns
        -------
        pd.DataFrame
            DataFrame indexé par date, une colonne par ticker,
            sans NaN (forward-fill puis drop).

        Raises
        ------
        ValueError
            Si aucune donnée n'est disponible pour les paramètres donnés.
        RuntimeError
            Si le téléchargement yfinance échoue.
        """
        cache_key = self._build_cache_key(tickers, start_date, end_date, column)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if cache_path.exists() and not force_refresh:
            logger.debug(f"Cache hit : {cache_path.name}")
            return self._load_cache(cache_path)

        logger.info(
            f"Téléchargement yfinance : {tickers} | {start_date} → {end_date}"
        )
        raw = self._fetch_from_yfinance(tickers, start_date, end_date)
        prices = self._extract_column(raw, tickers, column)
        prices = self._clean(prices)
        self._validate(prices, tickers)
        self._save_cache(prices, cache_path)
        logger.success(
            f"Téléchargement réussi : {prices.shape[0]} jours × {prices.shape[1]} tickers"
        )
        return prices

    def download_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Télécharge les données OHLCV complètes pour un seul ticker.

        Utile pour les calculs de volatilité réalisée intraday.

        Parameters
        ----------
        ticker : str
            Symbole boursier.
        start_date : str
            Date de début.
        end_date : str
            Date de fin.

        Returns
        -------
        pd.DataFrame
            DataFrame OHLCV nettoyé et validé.
        """
        cache_key = self._build_cache_key([ticker], start_date, end_date, "ohlcv")
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        if cache_path.exists():
            logger.debug(f"Cache OHLCV hit : {cache_path.name}")
            return self._load_cache(cache_path)

        logger.info(f"Téléchargement OHLCV : {ticker}")
        raw = self._fetch_from_yfinance([ticker], start_date, end_date)

        # Aplatir le MultiIndex si plusieurs tickers
        if isinstance(raw.columns, pd.MultiIndex):
            ohlcv = raw.xs(ticker, axis=1, level=1)
        else:
            ohlcv = raw.copy()

        ohlcv = ohlcv.dropna()
        self._save_cache(ohlcv, cache_path)
        return ohlcv

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_from_yfinance(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Appel brut à yfinance avec gestion des erreurs."""
        try:
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,   # Prix ajustés des dividendes et splits
                progress=False,
                threads=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Échec du téléchargement yfinance pour {tickers}: {exc}"
            ) from exc

        if data.empty:
            raise ValueError(
                f"Aucune donnée retournée par yfinance pour {tickers} "
                f"({start_date} → {end_date})."
            )
        return data

    def _extract_column(
        self,
        raw: pd.DataFrame,
        tickers: list[str],
        column: str,
    ) -> pd.DataFrame:
        """Extrait une colonne OHLCV du DataFrame MultiIndex yfinance."""
        if len(tickers) == 1:
            # Un seul ticker : pas de MultiIndex
            if column in raw.columns:
                df = raw[[column]].rename(columns={column: tickers[0]})
            else:
                raise ValueError(
                    f"Colonne '{column}' absente pour {tickers[0]}. "
                    f"Colonnes disponibles : {list(raw.columns)}"
                )
        else:
            # MultiIndex (colonne, ticker)
            if column in raw.columns.get_level_values(0):
                df = raw[column].copy()
                df.columns = [str(c) for c in df.columns]
            else:
                raise ValueError(
                    f"Colonne '{column}' absente. "
                    f"Colonnes disponibles : {list(raw.columns.get_level_values(0).unique())}"
                )
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données : forward-fill, suppression des NaN résiduels.

        Stratégie : on complète les trous courts par forward-fill (jusqu'à
        5 jours : week-end, jours fériés décalés) puis on supprime les lignes
        avec NaN sur tous les tickers.
        """
        n_before = len(df)
        df = df.ffill(limit=5)
        df = df.dropna(how="all")
        n_after = len(df)
        if n_before > n_after:
            logger.warning(
                f"  Nettoyage : {n_before - n_after} lignes supprimées (NaN persistants)"
            )
        return df

    def _validate(self, df: pd.DataFrame, tickers: list[str]) -> None:
        """Vérifie l'intégrité minimale des données téléchargées."""
        if df.empty:
            raise ValueError("DataFrame vide après nettoyage.")

        missing_tickers = [t for t in tickers if t not in df.columns]
        if missing_tickers:
            raise ValueError(
                f"Tickers manquants dans les données : {missing_tickers}"
            )

        # Vérifie qu'on a au moins 252 jours de trading
        if len(df) < 252:
            raise ValueError(
                f"Seulement {len(df)} jours disponibles. "
                "Le modèle nécessite au minimum 252 jours (1 an)."
            )

        # Vérifie la présence de valeurs négatives (anomalie de données)
        if (df < 0).any().any():
            logger.warning(
                "Des prix négatifs ont été détectés — vérifiez les données source."
            )

        nan_pct = df.isna().mean() * 100
        for ticker, pct in nan_pct.items():
            if pct > 5:
                logger.warning(
                    f"  {ticker} : {pct:.1f}% de valeurs manquantes après cleaning."
                )

    @staticmethod
    def _build_cache_key(
        tickers: list[str],
        start_date: str,
        end_date: str,
        column: str,
    ) -> str:
        """Génère un hash SHA-256 court pour identifier le cache."""
        payload = f"{'_'.join(sorted(tickers))}_{start_date}_{end_date}_{column}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _save_cache(df: pd.DataFrame, path: Path) -> None:
        """Sérialise le DataFrame en pickle."""
        with open(path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Cache sauvegardé : {path.name}")

    @staticmethod
    def _load_cache(path: Path) -> pd.DataFrame:
        """Désérialise le DataFrame depuis le cache pickle."""
        with open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301
