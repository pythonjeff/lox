import sys
from pathlib import Path


def pytest_configure():
    """
    Ensure `src/` is on sys.path for the src-layout package import (`ai_options_trader`).
    This keeps tests runnable without requiring an editable install.
    """
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


