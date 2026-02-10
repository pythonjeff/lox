"""Command registrations for the Typer CLI.

We keep `lox/cli.py` as the entrypoint module (because
`pyproject.toml` points scripts at `lox.cli:app`).

To keep that file small, commands are defined in this package and registered
from the entrypoint.
"""


