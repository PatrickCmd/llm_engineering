import asyncio
from typing import TypedDict
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError


class PageData(TypedDict):
    """Structured result returned by the scraper."""
    url: str
    title: str
    content: str


async def scrape_webpage(url: str, wait_for: str = "networkidle", timeout: int = 30000) -> PageData:
    """
    Scrape the title and text content of a webpage using Playwright.

    Playwright launches a real browser (Chromium by default), which fully executes
    JavaScript — making it ideal for React, Vue, Angular, and other JS-heavy sites
    that plain HTTP clients like `requests` cannot render.

    Args:
        url (str): The full URL of the webpage to scrape
                   (e.g. "https://example.com").
        wait_for (str): The condition to wait for before extracting content.
                        Options:
                          - "networkidle"      → waits until network has been idle
                                                 for 500ms (best for SPAs). [default]
                          - "domcontentloaded" → waits for the HTML to be parsed.
                          - "load"             → waits for the load event.
                          - "commit"           → waits until the response starts arriving.
        timeout (int): Maximum time in milliseconds to wait for the page to load.
                       Defaults to 30000 (30 seconds).

    Returns:
        PageData: A TypedDict containing:
            - ``url``     (str): The URL that was scraped.
            - ``title``   (str): The page's <title> tag value, or an empty string
                                 if no title was found.
            - ``content`` (str): The visible text content of the fully rendered
                                 page, with whitespace normalized.

    Raises:
        PlaywrightTimeoutError: If the page does not load within the timeout period.
        Exception: For any other browser or network related errors.

    Example:
        >>> import asyncio
        >>> data = asyncio.run(scrape_webpage("https://news.ycombinator.com"))
        >>> print(data["title"])
        >>> print(data["content"][:500])
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

            page = await context.new_page()
            await page.goto(url, wait_until=wait_for, timeout=timeout)

            # Extract both title and body text in a single evaluate call
            # to avoid multiple round trips to the browser
            result = await page.evaluate("""
                () => {
                    const title = document.title ?? "";

                    // Remove non-visible / non-content elements
                    const tagsToRemove = ['script', 'style', 'noscript', 'svg', 'img'];
                    tagsToRemove.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });

                    const content = document.body.innerText ?? "";

                    return { title, content };
                }
            """)

            # Normalize excessive whitespace and blank lines
            lines = [line.strip() for line in result["content"].splitlines()]
            cleaned_content = "\n".join(line for line in lines if line)

            return PageData(
                url=url,
                title=result["title"].strip(),
                content=cleaned_content,
            )

        except PlaywrightTimeoutError:
            raise PlaywrightTimeoutError(
                f"Page '{url}' did not finish loading within {timeout}ms. "
                "Try increasing the timeout or using a different wait_for strategy."
            )
        finally:
            await browser.close()


def scrape(url: str, wait_for: str = "networkidle", timeout: int = 30000) -> PageData:
    """
    Synchronous wrapper around :func:`scrape_webpage`.

    Useful when you are not inside an async context and want a simple
    one-liner call.

    Args:
        url (str): The full URL of the webpage to scrape.
        wait_for (str): Load condition — see :func:`scrape_webpage` for options.
        timeout (int): Timeout in milliseconds. Defaults to 30000.

    Returns:
        PageData: A TypedDict containing ``url``, ``title``, and ``content``.
                  See :func:`scrape_webpage` for full field descriptions.

    Example:
        >>> from scraper import scrape
        >>> data = scrape("https://news.ycombinator.com")
        >>> print(data["title"])
        >>> print(data["content"][:300])
    """
    return asyncio.run(scrape_webpage(url, wait_for=wait_for, timeout=timeout))


if __name__ == "__main__":
    # url = "https://news.ycombinator.com"
    url = "https://openai.com/index/introducing-openai-frontier/"
    # url = "https://openai.com"
    print(f"Scraping: {url}\n{'=' * 50}")
    data = scrape(url, wait_for="domcontentloaded", timeout=120000)  # Increase timeout for slower pages

    print(f"Title   : {data['title']}")
    print(f"URL     : {data['url']}")
    print(f"{'=' * 50}")
    print(data["content"][:5000])
    print(f"\n{'=' * 50}")
    print(f"Total characters scraped: {len(data['content'])}")
