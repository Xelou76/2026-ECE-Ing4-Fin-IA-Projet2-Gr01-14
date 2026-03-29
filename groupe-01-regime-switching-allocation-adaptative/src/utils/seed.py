"""
utils/seed.py
=============
Gestion centralisée de la reproductibilité : fixation des seeds pour
Python, NumPy, PyTorch (CPU et CUDA) et hmmlearn.

Appeler `set_all_seeds(seed)` en début de `main.py` garantit la
reproductibilité totale des expériences.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
from loguru import logger


def set_all_seeds(seed: int = 42) -> None:
    """
    Fixe tous les seeds aléatoires pour garantir la reproductibilité.

    Couvre : Python `random`, NumPy, PyTorch (CPU + tous GPUs disponibles),
    et les variables d'environnement PYTHONHASHSEED.

    Parameters
    ----------
    seed : int, optional
        Valeur du seed global. Par défaut 42.

    Notes
    -----
    Pour une reproductibilité maximale sur GPU, des opérations non-déterministes
    peuvent être désactivées via `torch.use_deterministic_algorithms(True)`, au
    prix de performances légèrement réduites.

    Examples
    --------
    >>> set_all_seeds(42)
    >>> import numpy as np
    >>> np.random.rand(3)  # séquence identique à chaque exécution
    array([...])
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (import conditionnel pour ne pas le rendre obligatoire au setup)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
        # Déterminisme au coût de la performance (activable pour debug)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug(f"PyTorch seeds fixés (seed={seed})")
    except ImportError:
        warnings.warn("PyTorch non disponible. Seeds PyTorch non fixés.", stacklevel=2)

    logger.info(f"Tous les seeds aléatoires fixés à {seed}")


def get_numpy_rng(seed: int = 42) -> np.random.Generator:
    """
    Retourne un générateur NumPy isolé (pour les tests unitaires).

    Parameters
    ----------
    seed : int
        Seed du générateur.

    Returns
    -------
    np.random.Generator
        Générateur NumPy déterministe.
    """
    return np.random.default_rng(seed)
