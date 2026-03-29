"""config package - Centralisation des paramètres."""

from config.constants import REGIME_COLORS, REGIME_LABELS, TRADING_DAYS_PER_YEAR
from config.settings import ProjectConfig, get_settings

__all__ = ["get_settings", "ProjectConfig", "REGIME_LABELS", "REGIME_COLORS", "TRADING_DAYS_PER_YEAR"]
