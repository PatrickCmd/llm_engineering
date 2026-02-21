"""
Configuration constants for the brochure CLI application.

Centralises model identifiers and system prompts so they can be updated
in one place without touching any generation or filtering logic.
"""

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

MODEL_FAST = "gpt-5-nano"
"""Model used for the link-filtering step.

Chosen for speed and low cost — classifying links is a simple task that
does not require a high-capability model.
"""

MODEL_BROCHURE = "gpt-4.1-mini"
"""Model used for brochure generation.

A stronger model is used here because the output is customer/investor-facing
prose that benefits from higher quality writing.
"""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

LINK_SYSTEM_PROMPT = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include
in a brochure about the company, such as links to an About page, a Company
page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

BROCHURE_SYSTEM_PROMPT = """
You are an assistant that analyses the contents of several relevant pages
from a company website and creates a short brochure about the company for
prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have
the information.
"""
