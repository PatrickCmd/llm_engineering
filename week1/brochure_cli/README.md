# brochure_cli

A modular CLI tool that generates AI-powered company brochures from any website.
Given a company name and URL it scrapes the site with a real browser, filters
the links to only those relevant for a brochure, and uses GPT to write a polished
Markdown document targeting prospective customers, investors, and recruits.

---

## How it works

```
Website URL
    │
    ▼
[scraper.py]  ── Playwright headless browser fetches all page links
    │
    ▼
[linker.py]   ── GPT (gpt-5-nano) classifies links via Structured Outputs
    │               keeping only About, Careers, Products, Blog, etc.
    ▼
[scraper.py]  ── Playwright fetches full text of each relevant page
    │
    ▼
[generator.py]── GPT (gpt-4.1-mini) writes the brochure from assembled content
    │
    ▼
[exporter.py] ── Saves to .md (Markdown) or .txt (plain text)
```

---

## Project structure

```
brochure_cli/
├── __init__.py       # Package marker
├── __main__.py       # Enables: python -m brochure_cli
├── config.py         # Model names and system prompts
├── scraper.py        # Playwright: fetch rendered page content and links
├── linker.py         # GPT Structured Outputs link filtering
├── generator.py      # Content assembly and brochure generation
├── exporter.py       # Save output to .md or .txt
├── main.py           # Typer CLI app and command definitions
└── README.md
```

---

## Prerequisites

- Python 3.11+
- `uv` or `pip` with the project dependencies installed
- Playwright browsers installed (`playwright install chromium`)
- An OpenAI API key

---

## Configuration

Create a `.env` file in the `week1/` directory (or any parent directory):

```env
OPENAI_API_KEY=sk-proj-...
```

The key is loaded automatically at runtime; it is never hard-coded.

---

## Usage

Run all commands from the `week1/` directory.

### Basic — save as Markdown (default)

```bash
python -m brochure_cli generate "HuggingFace" https://huggingface.co
# Output: huggingface.md
```

### Save as plain text

```bash
python -m brochure_cli generate "OpenAI" https://openai.com --format txt
# Output: openai.txt
```

### Custom output path

```bash
python -m brochure_cli generate "Sunbird AI" https://sunbird.ai \
    --format md \
    --output reports/sunbird_ai
# Output: reports/sunbird_ai.md  (directories are created automatically)
```

### Stream tokens to the terminal

```bash
python -m brochure_cli generate "HuggingFace" https://huggingface.co --stream
```

Tokens are printed to stdout as they arrive. The brochure is still saved to
the output file when generation completes.

### Help

```bash
python -m brochure_cli --help
python -m brochure_cli generate --help
```

---

## Options reference

| Option | Short | Default | Description |
|---|---|---|---|
| `--format` | `-f` | `md` | Output format: `md` or `txt` |
| `--output` | `-o` | `<slug>.<fmt>` | Destination file path |
| `--stream` | | `False` | Stream tokens to stdout during generation |
| `--no-stream` | | — | Disable streaming (default behaviour) |

---

## Models used

| Stage | Model | Reason |
|---|---|---|
| Link filtering | `gpt-5-nano` | Simple classification — fast and cheap |
| Brochure generation | `gpt-4.1-mini` | Requires high-quality prose writing |

---

## Notes

- Each run makes several Playwright browser sessions (one per page fetched) plus
  a few API calls, so expect 30–90 seconds depending on the target site.
- Set `--format txt` if you plan to paste the output into a plain-text system
  (email, Slack, etc.). Markdown symbols are stripped automatically.
- Pages that block headless browsers or require authentication will return empty
  content — the brochure will still be generated from whatever was accessible.
