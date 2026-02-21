"""
CLI entry point for the brochure generator.

Defines the Typer application and the ``generate`` command, which runs the
full pipeline: scrape → filter links → generate brochure → export to file.

Usage
-----
Run as a module from the ``week1/`` directory::

    python -m brochure_cli generate "HuggingFace" https://huggingface.co
    python -m brochure_cli generate "OpenAI" https://openai.com --format txt
    python -m brochure_cli generate "Sunbird AI" https://sunbird.ai --stream --output reports/sunbird

Run ``python -m brochure_cli --help`` for the full option list.
"""

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .exporter import ExportFormat, export_brochure
from .generator import generate_brochure, stream_brochure

# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="brochure",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

console = Console()


@app.callback()
def _root() -> None:
    """Generate AI-powered company brochures from any website."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_client() -> OpenAI:
    """Load environment variables and return an initialised OpenAI client.

    Reads ``OPENAI_API_KEY`` from the environment (or a ``.env`` file in the
    working directory). Exits with a clear error message if the key is missing.

    Returns:
        An initialised ``openai.OpenAI`` client ready for API calls.

    Raises:
        SystemExit: If ``OPENAI_API_KEY`` is not set.
    """
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY not found. "
            "Add it to your [bold].env[/bold] file and try again."
        )
        raise typer.Exit(code=1)
    return OpenAI(api_key=api_key)


def _default_output_path(company_name: str, fmt: ExportFormat) -> Path:
    """Derive a default output file path from the company name and format.

    Converts the company name to a filesystem-safe slug (lowercase, spaces
    replaced with underscores) and appends the appropriate extension.

    Args:
        company_name: Human-readable company name (e.g. ``"HuggingFace"``).
        fmt:          Export format (``"md"`` or ``"txt"``).

    Returns:
        A ``Path`` object such as ``huggingface.md``.
    """
    slug = company_name.lower().replace(" ", "_")
    return Path(f"{slug}.{fmt}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def generate(
    company_name: str = typer.Argument(
        ...,
        help="Name of the company (e.g. 'HuggingFace').",
    ),
    url: str = typer.Argument(
        ...,
        help="Full URL of the company website (e.g. https://huggingface.co).",
    ),
    fmt: ExportFormat = typer.Option(
        "md",
        "--format",
        "-f",
        help="Output file format: 'md' (Markdown) or 'txt' (plain text).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Destination file path. Extension is added/corrected automatically. "
            "Defaults to '<company-slug>.<format>' in the current directory."
        ),
    ),
    stream: bool = typer.Option(
        False,
        "--stream/--no-stream",
        help="Stream tokens to the terminal as the brochure is generated.",
    ),
) -> None:
    """Generate an AI-powered brochure for a company from its website.

    The command runs a three-stage pipeline:

    \b
    1. Scrape  — fetch all links from the homepage with a headless browser.
    2. Filter  — ask GPT to select only brochure-relevant links (About,
                 Careers, Products, etc.) using Structured Outputs.
    3. Generate — scrape each selected page, assemble the content, and have
                  GPT write a polished Markdown brochure.

    The result is saved to disk in the chosen format.

    \b
    Examples:
      python -m brochure_cli generate "HuggingFace" https://huggingface.co
      python -m brochure_cli generate "OpenAI" https://openai.com -f txt -o reports/openai
      python -m brochure_cli generate "Sunbird AI" https://sunbird.ai --stream
    """
    client = _load_client()
    output_path = output if output is not None else _default_output_path(company_name, fmt)

    console.print(
        Panel(
            f"[bold]Company:[/bold]  {company_name}\n"
            f"[bold]Website:[/bold]  {url}\n"
            f"[bold]Format:[/bold]   {fmt}\n"
            f"[bold]Output:[/bold]   {output_path}\n"
            f"[bold]Streaming:[/bold] {'yes' if stream else 'no'}",
            title="[cyan]Brochure Generator[/cyan]",
            expand=False,
        )
    )

    if stream:
        console.print("\n[bold]Generating brochure (streaming)...[/bold]\n")
        brochure = stream_brochure(company_name, url, client)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(
                "Scraping website and generating brochure — this may take a minute...",
                total=None,
            )
            brochure = generate_brochure(company_name, url, client)

    export_brochure(brochure, output_path, fmt)
    console.print(f"\n[bold green]Done![/bold green] Brochure saved to [bold]{output_path.resolve()}[/bold]")
