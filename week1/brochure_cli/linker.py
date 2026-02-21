"""
Link-filtering module using GPT Structured Outputs.

Sends the full list of links scraped from a company's homepage to a fast
GPT model, which decides which links are relevant for a business brochure
(About, Careers, Products, Blog, etc.). The model responds with structured
JSON validated against a Pydantic schema, eliminating the need for manual
JSON parsing or error handling.
"""

from openai import OpenAI
from pydantic import BaseModel

from .config import LINK_SYSTEM_PROMPT, MODEL_FAST
from .scraper import fetch_page_links


# ---------------------------------------------------------------------------
# Pydantic schemas for Structured Outputs
# ---------------------------------------------------------------------------

class RelevantLink(BaseModel):
    """A single brochure-relevant hyperlink returned by the model.

    Attributes:
        type: Human-readable label for the page type (e.g. ``"about page"``).
        url:  Absolute URL of the link.
    """

    type: str
    url: str


class RelevantLinks(BaseModel):
    """Top-level schema for the model's structured response.

    Attributes:
        links: List of relevant links identified by the model.
    """

    links: list[RelevantLink]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(url: str) -> str:
    """Build the user message containing all page links for the model to classify.

    Fetches raw links from *url* and appends them to the classification
    instruction so the static system prompt never needs to change.

    Args:
        url: The company homepage URL whose links should be evaluated.

    Returns:
        A formatted string listing all scraped links, ready to send as the
        ``user`` role message.
    """
    links = fetch_page_links(url, include_external=True)
    body = "\n".join(links)
    return (
        f"Here is the list of links on the website {url} —\n"
        "Please decide which of these are relevant web links for a brochure "
        "about the company, respond with the full https URL in JSON format.\n"
        "Do not include Terms of Service, Privacy, or email links.\n\n"
        f"Links:\n\n{body}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_relevant_links(url: str, client: OpenAI) -> list[dict]:
    """Filter a page's links to only those relevant for a company brochure.

    Scrapes all hyperlinks from *url*, sends them to ``MODEL_FAST`` using
    OpenAI Structured Outputs (``responses.parse``), and returns the subset
    the model deems relevant (About, Careers, Products, Blog, etc.).

    Args:
        url:    The company homepage URL to analyse.
        client: An initialised ``openai.OpenAI`` client.

    Returns:
        A list of dicts with keys ``"type"`` (str) and ``"url"`` (str),
        representing the brochure-relevant links identified by the model.

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> links = select_relevant_links("https://huggingface.co", client)
        >>> for link in links:
        ...     print(link["type"], "→", link["url"])
    """
    response = client.responses.parse(
        model=MODEL_FAST,
        input=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(url)},
        ],
        text_format=RelevantLinks,
    )
    links = response.output_parsed.model_dump()["links"]
    print(f"[linker] {len(links)} relevant link(s) identified.")
    return links
