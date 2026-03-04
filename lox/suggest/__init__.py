"""Cross-asset suggest: ticker + direction recommendations from macro thesis."""

from lox.suggest.cross_asset import suggest_cross_asset
from lox.suggest.scoring import SuggestResult

__all__ = ["suggest_cross_asset", "SuggestResult"]
