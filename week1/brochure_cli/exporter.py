"""
Brochure export module.

Handles saving the generated Markdown brochure to disk in the user's chosen
format. Two formats are supported:

- ``md``  — raw Markdown, written as-is.
- ``txt`` — plain text with Markdown symbols stripped via regex.
"""

import re
from pathlib import Path
from typing import Literal

ExportFormat = Literal["md", "txt"]
"""Supported export formats: ``"md"`` for Markdown, ``"txt"`` for plain text."""

# Regex substitutions applied when exporting to plain text, in order.
_MD_STRIP_RULES: list[tuple[str, str]] = [
    (r"#{1,6}\s+", ""),            # ATX headings  → bare text
    (r"\*\*(.+?)\*\*", r"\1"),     # **bold**       → text
    (r"\*(.+?)\*", r"\1"),         # *italic*       → text
    (r"__(.+?)__", r"\1"),         # __bold__       → text
    (r"_(.+?)_", r"\1"),           # _italic_       → text
    (r"\[(.+?)\]\(.+?\)", r"\1"),  # [text](url)    → text
    (r"`(.+?)`", r"\1"),           # `code`         → text
    (r"^\s*[-*+]\s+", "- ", re.MULTILINE),  # normalise list bullets
]


def _strip_markdown(text: str) -> str:
    """Remove common Markdown formatting symbols from *text*.

    Applies a sequence of regex substitutions to produce clean plain text
    suitable for a ``.txt`` file. Does not handle all possible Markdown
    constructs — complex tables or fenced code blocks may retain some symbols.

    Args:
        text: Markdown-formatted string to clean.

    Returns:
        Plain text string with common Markdown syntax removed.
    """
    for rule in _MD_STRIP_RULES:
        pattern, replacement, *flags = rule
        flag = flags[0] if flags else 0
        text = re.sub(pattern, replacement, text, flags=flag)
    return text


def export_brochure(content: str, path: Path, fmt: ExportFormat) -> None:
    """Save a brochure to disk in the specified format.

    Ensures the output file has the correct extension (adding or replacing it
    if necessary) and creates any missing parent directories automatically.

    Args:
        content: The brochure text in Markdown format.
        path:    Destination file path. If the path has no extension, or an
                 extension that does not match *fmt*, the correct extension is
                 appended automatically.
        fmt:     Output format — ``"md"`` writes raw Markdown; ``"txt"`` strips
                 Markdown symbols first.

    Raises:
        OSError: If the file cannot be written (e.g. permissions error).

    Example:
        >>> from pathlib import Path
        >>> export_brochure("# My Company\\n...", Path("output/brochure"), "md")
        # Writes to output/brochure.md
    """
    expected_suffix = f".{fmt}"
    if path.suffix != expected_suffix:
        path = path.with_suffix(expected_suffix)

    path.parent.mkdir(parents=True, exist_ok=True)

    text_to_write = _strip_markdown(content) if fmt == "txt" else content
    path.write_text(text_to_write, encoding="utf-8")

    print(f"[exporter] Brochure saved → {path.resolve()}")
