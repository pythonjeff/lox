"""Command registrations for the Typer CLI.

We keep `ai_options_trader/cli.py` as the entrypoint module (because
`pyproject.toml` points scripts at `ai_options_trader.cli:app`).

To keep that file small, commands are defined in this package and registered
from the entrypoint.
"""


