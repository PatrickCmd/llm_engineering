"""
Package entry point — enables ``python -m brochure_cli``.

Invoking the package as a module delegates immediately to the Typer
application defined in :mod:`brochure_cli.main`.
"""

from brochure_cli.main import app

if __name__ == "__main__":
    app()
