"""
Web scraping module using Playwright.

Launches a headless Chromium browser to fully execute JavaScript before
extracting page content or links. This makes it suitable for modern
single-page applications (React, Vue, Angular) that plain HTTP clients
such as ``requests`` cannot render correctly.

All public functions are synchronous wrappers around private async
implementations, so callers do not need to manage an event loop.
"""

import asyncio
from urllib.parse import urlparse

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_WAIT_FOR = "domcontentloaded"
_TIMEOUT = 120_000          # milliseconds
_CONTENT_LIMIT = 10_000     # characters


# ---------------------------------------------------------------------------
# Private async implementations
# ---------------------------------------------------------------------------

async def _async_fetch_content(url: str) -> str:
    """Async implementation — fetch rendered page text from *url*."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(user_agent=_USER_AGENT)
            page = await context.new_page()
            await page.goto(url, wait_until=_WAIT_FOR, timeout=_TIMEOUT)

            result = await page.evaluate("""
                () => {
                    const title = document.title ?? "";

                    // Strip non-content elements before reading innerText
                    ["script", "style", "noscript", "svg", "img"].forEach(tag =>
                        document.querySelectorAll(tag).forEach(el => el.remove())
                    );

                    const content = document.body ? document.body.innerText : "";
                    return { title, content };
                }
            """)

            lines = [line.strip() for line in result["content"].splitlines()]
            text = "\n".join(line for line in lines if line)
            return (result["title"].strip() + "\n\n" + text)[:_CONTENT_LIMIT]

        except PlaywrightTimeoutError:
            print(f"[scraper] Timeout fetching {url} — returning empty string.")
            return ""
        except Exception as exc:
            print(f"[scraper] Error fetching {url}: {exc}")
            return ""
        finally:
            await browser.close()


async def _async_fetch_links(url: str, include_external: bool) -> list[str]:
    """Async implementation — extract all unique hyperlinks from *url*."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(user_agent=_USER_AGENT)
            page = await context.new_page()
            await page.goto(url, wait_until=_WAIT_FOR, timeout=_TIMEOUT)

            raw: list[str] = await page.evaluate("""
                () => {
                    const seen = new Set();
                    const results = [];

                    document.querySelectorAll("a[href]").forEach(el => {
                        const raw = el.getAttribute("href").trim();

                        // Skip blank, fragment-only, and javascript: hrefs
                        if (!raw || raw === "#" || raw.startsWith("javascript:")) return;

                        let href;
                        try {
                            href = new URL(raw, window.location.href).href;
                        } catch {
                            return;  // malformed href — skip
                        }

                        if (seen.has(href)) return;
                        seen.add(href);
                        results.push(href);
                    });

                    return results;
                }
            """)

            if not include_external:
                base_netloc = urlparse(url).netloc
                raw = [h for h in raw if urlparse(h).netloc == base_netloc]

            return raw

        except PlaywrightTimeoutError:
            print(f"[scraper] Timeout fetching links from {url} — returning empty list.")
            return []
        except Exception as exc:
            print(f"[scraper] Error fetching links from {url}: {exc}")
            return []
        finally:
            await browser.close()


# ---------------------------------------------------------------------------
# Public synchronous API
# ---------------------------------------------------------------------------

def fetch_page_content(url: str) -> str:
    """Fetch and return the rendered text content of a webpage.

    Launches a headless Chromium browser, waits for the DOM to load, strips
    non-content elements (scripts, styles, images), and returns the visible
    text truncated to ``_CONTENT_LIMIT`` characters.

    Args:
        url: The full URL of the page to fetch (e.g. ``"https://example.com"``).

    Returns:
        A string containing the page title followed by its visible body text,
        truncated to ``_CONTENT_LIMIT`` characters. Returns an empty string
        if the page cannot be fetched.

    Example:
        >>> text = fetch_page_content("https://huggingface.co")
        >>> print(text[:200])
    """
    return asyncio.run(_async_fetch_content(url))


def fetch_page_links(url: str, include_external: bool = False) -> list[str]:
    """Fetch and return all unique, resolved hyperlinks from a webpage.

    Relative URLs are resolved to absolute form against the page's own origin.
    Blank, fragment-only (``#``), and ``javascript:`` hrefs are filtered out,
    and results are deduplicated.

    Args:
        url: The full URL of the page to scrape.
        include_external: When ``True``, links to other domains are included.
            Defaults to ``False`` (internal links only).

    Returns:
        A deduplicated list of absolute URL strings found on the page.

    Example:
        >>> links = fetch_page_links("https://huggingface.co")
        >>> print(links[:5])
    """
    return asyncio.run(_async_fetch_links(url, include_external))
