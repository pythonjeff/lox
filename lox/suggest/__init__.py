"""Cross-asset suggest: ticker + direction recommendations from macro thesis."""

from lox.suggest.cross_asset import suggest_cross_asset
from lox.suggest.reversion import run_reversion_screen, ReversionScreenResult
from lox.suggest.scanner import run_opportunity_scan, ScannerResult
from lox.suggest.scoring import SuggestResult

__all__ = [
    "suggest_cross_asset",
    "SuggestResult",
    "run_reversion_screen",
    "ReversionScreenResult",
    "run_opportunity_scan",
    "ScannerResult",
]
