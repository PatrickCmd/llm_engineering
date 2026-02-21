"""
Brochure generation module.

Orchestrates the full pipeline:
1. Scrape the company landing page.
2. Filter its links to brochure-relevant pages (via GPT Structured Outputs).
3. Scrape each relevant sub-page.
4. Assemble all content into a single prompt.
5. Call GPT to generate a polished Markdown brochure.

Two generation modes are supported:
- ``generate_brochure`` — waits for the full response, then returns it.
- ``stream_brochure``   — streams tokens to stdout as they arrive and returns
  the complete text when done.
"""

from openai import OpenAI

from .config import BROCHURE_SYSTEM_PROMPT, MODEL_BROCHURE
from .linker import select_relevant_links
from .scraper import fetch_page_content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assemble_content(company_name: str, url: str, client: OpenAI) -> str:
    """Scrape the landing page and all relevant sub-pages into a single string.

    Fetches the main page content, identifies brochure-relevant links via GPT,
    then fetches each of those sub-pages and concatenates everything with
    descriptive section headers.

    Args:
        company_name: Human-readable name of the company (used for context).
        url:          Company homepage URL.
        client:       Initialised ``openai.OpenAI`` client.

    Returns:
        A multi-section string ready to be inserted into the brochure prompt.
    """
    print(f"[generator] Fetching landing page: {url}")
    landing = fetch_page_content(url)

    relevant_links = select_relevant_links(url, client)

    sections = [f"## Landing Page:\n\n{landing}\n\n## Relevant Pages:"]
    for link in relevant_links:
        print(f"[generator] Fetching {link['type']}: {link['url']}")
        page_text = fetch_page_content(link["url"])
        sections.append(f"\n### {link['type'].title()}\n\n{page_text}")

    return "\n".join(sections)


def _build_user_prompt(company_name: str, url: str, client: OpenAI) -> str:
    """Build the full user message for the brochure generation call.

    Args:
        company_name: Human-readable name of the company.
        url:          Company homepage URL.
        client:       Initialised ``openai.OpenAI`` client.

    Returns:
        The complete user prompt string containing company context and all
        scraped page content.
    """
    content = _assemble_content(company_name, url, client)
    return (
        f"You are looking at a company called: {company_name}\n\n"
        "Here are the contents of its landing page and other relevant pages. "
        "Use this information to build a short brochure of the company in "
        "markdown without code blocks.\n\n"
        + content
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_brochure(company_name: str, url: str, client: OpenAI) -> str:
    """Generate a company brochure and return the full Markdown string.

    Runs the complete scrape → filter → generate pipeline and blocks until
    the model returns the full response.

    Args:
        company_name: Human-readable name of the company (e.g. ``"HuggingFace"``).
        url:          Full URL of the company website (e.g. ``"https://huggingface.co"``).
        client:       Initialised ``openai.OpenAI`` client.

    Returns:
        The generated brochure as a Markdown-formatted string.

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> md = generate_brochure("HuggingFace", "https://huggingface.co", client)
        >>> print(md[:500])
    """
    user_prompt = _build_user_prompt(company_name, url, client)
    response = client.responses.create(
        model=MODEL_BROCHURE,
        input=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text


def stream_brochure(company_name: str, url: str, client: OpenAI) -> str:
    """Stream-generate a company brochure, printing tokens to stdout as they arrive.

    Runs the same pipeline as :func:`generate_brochure` but uses the streaming
    Responses API so output appears progressively in the terminal rather than
    all at once after a long wait.

    Args:
        company_name: Human-readable name of the company.
        url:          Full URL of the company website.
        client:       Initialised ``openai.OpenAI`` client.

    Returns:
        The complete generated brochure as a Markdown-formatted string
        (identical content to :func:`generate_brochure`, just delivered
        incrementally to stdout during generation).

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> md = stream_brochure("OpenAI", "https://openai.com", client)
    """
    user_prompt = _build_user_prompt(company_name, url, client)
    stream = client.responses.create(
        model=MODEL_BROCHURE,
        input=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    result = ""
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
            result += event.delta

    print()  # trailing newline after stream completes
    return result
