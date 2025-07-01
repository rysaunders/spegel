# Spegel – Reflect the web through AI

Automatically rewrites the websites into markdown optimised for viewing in the terminal.
Read intro blog post [here](https://simedw.com/2025/06/23/introducing-spegel/)

This is a proof-of-concept, bugs are to be expected but feel free to raise an issue or pull request.

##  Screenshot
Sometimes you don't want to read through someone's life story just to get to a recipe
![Recipe Example](https://simedw.com/2025/06/23/introducing-spegel/images/recipe_example.png)


## Installation

Requires Python 3.11+

```
pip install spegel
```
or clone the repo and install it editable mode

```bash
# Clone and enter the directory
$ git clone <repo-url>
$ cd spegel

# Install dependencies and the CLI
$ pip install -e .
```

## API Keys
Spegel supports both Claude and Gemini. You need at least one API key:

**For Claude (recommended):**
```
ANTHROPIC_API_KEY=sk-ant-...
```

**For Google Gemini:**
```
GEMINI_API_KEY=...
```

By default, Spegel will use Claude if available, otherwise Gemini. You can force a specific provider in your config file.


## Usage

### Launch the browser

```bash
spegel                # Start with welcome screen
spegel bbc.com        # Open a URL immediately
```

Or, equivalently:

```bash
python -m spegel      # Start with welcome screen
python -m spegel bbc.com
```

### Basic controls
- `/`         – Open URL input
- `Tab`/`Shift+Tab` – Cycle links
- `Enter`     – Open selected link
- `e`         – Edit LLM prompt for current view
- `b`         – Go back
- `q`         – Quit

## Editing settings

Spegel loads settings from a TOML config file. You can customize views, prompts, and UI options.

**Config file search order:**
1. `./.spegel.toml` (current directory)
2. `~/.spegel.toml`
3. `~/.config/spegel/config.toml`

To edit settings:
1. Copy the example config:
   ```bash
   cp example_config.toml .spegel.toml
   # or create ~/.spegel.toml
   ```
2. Edit `.spegel.toml` in your favorite editor.

Example snippet:
```toml
[settings]
default_view = "terminal"
app_title = "Spegel"

[[views]]
id = "raw"
name = "Raw View"
prompt = ""

[[views]]
id = "terminal"
name = "Terminal"
prompt = "Transform this webpage into the perfect terminal browsing experience! ..."
```

---

For more, see the code or open an issue!
