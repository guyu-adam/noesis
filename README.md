# JARVES v6 "Secretary"

**Claude Code's local assistant.** Offloads file operations, shell execution, and routine LLM tasks to a local model — saving Claude API tokens on the tasks where a local model or zero-LLM path does the job.

---

## What it saves

The biggest Claude token costs in daily use:

| Task | Without JARVES | With JARVES | Saving |
|------|---------------|-------------|--------|
| Read a 600-line Python file | ~12,000 tokens | — | — |
| `/outline` that file | — | ~640 tokens returned | **~11,360 tokens** |
| `/grep` for one function | — | ~130 tokens returned | **~11,870 tokens** |
| `/summarize` a config file | — | ~200 tokens returned | **~11,800 tokens** |
| `/run` git log, ls, find | — | 0 tokens | **100%** |
| `/write` or `/patch` a file | Edit + Read round-trip | 0 tokens | **100%** |

Zero-LLM endpoints (/grep, /outline, /tree, /exists, /write, /patch, /run, /read) never touch any model — they're pure Python and respond in <50ms.

LLM endpoints (/summarize, /codegen, /ask) route to a local Ollama model. No Claude API call, no cost.

---

## Architecture

```
Claude Code
    │  HTTP POST (localhost:7860)
    ▼
jarves.py  (Flask)
    │
    ├── Zero-LLM path (instant, no model)
    │     /run /read /grep /outline /tree /exists /write /patch
    │
    └── Local-LLM path (Ollama, no cloud)
          /ask /summarize /codegen /batch
```

---

## Quick start

### 1. Install Ollama and pull a model

```bash
# https://ollama.com
ollama pull qwen3:4b
ollama create qwen3-4b-jarves -f Modelfile.qwen3-4b

# Semantic memory (optional but recommended)
ollama pull nomic-embed-text
```

### 2. Start the server

```bash
pip install flask requests numpy rich
python jarves.py
# Server at http://localhost:7860
```

### 3. Use the client

```python
import sys; sys.path.insert(0, '/path/to/jarves')
from j import J

# Zero-LLM — instant, no model cost
J.exists("~/project/file.py")                      # existence check
J.outline("~/project/app.py")                      # function/class map
J.grep("~/project/app.py", "def process", context=3)  # search with context
J.tree("~/project", depth=2)                       # directory tree
J.write("~/project/config.py", "KEY = 'value'")   # write file
J.patch("~/project/config.py", "old_val", "new")  # find-and-replace
J.run("git log --oneline -5")                      # shell command

# Local-LLM — no Claude API tokens
J.summarize("~/project/big_file.py", focus="error handling")
J.codegen("write a function to flatten a nested list")
J.ask("what does this regex do: r'\\d{3}-\\d{4}'")

# Batch multiple ops in one call
J.batch([
    ("outline", "~/project/app.py"),
    ("run", "pytest --tb=short"),
    ("exists", "~/project/.env"),
])
```

---

## Endpoint reference

### Zero-LLM (no model involved)

| Endpoint | Method | Key params | Returns |
|----------|--------|-----------|---------|
| `/run` | POST | `cmd`, `timeout` | `{output}` |
| `/read` | POST | `path`, `limit` | `{content}` |
| `/grep` | POST | `path`, `pattern`, `context` | `{matches}` |
| `/outline` | POST | `path` | `{outline}` — func/class map |
| `/tree` | POST | `path`, `depth` | `{tree}` |
| `/exists` | POST | `path` | `{exists, is_file, size}` |
| `/write` | POST | `path`, `content` | `{result}` |
| `/patch` | POST | `path`, `old`, `new` | `{result}` |

### Local-LLM (Ollama, no cloud)

| Endpoint | Method | Key params | Returns |
|----------|--------|-----------|---------|
| `/ask` | POST | `task`, `max_tokens` | `{result}` |
| `/summarize` | POST | `path` or `text`, `focus` | `{summary}` |
| `/codegen` | POST | `task`, `lang` | `{code}` |
| `/batch` | POST | `{tasks: [...]}` | `{results: [...]}` |
| `/note` | POST | `key`, `value` | `{saved}` |
| `/memory/clear` | POST | — | `{cleared}` |
| `/status` | GET | — | `{status, model, tokens_saved_est}` |

---

## Models

| Modelfile | Base | Size | Notes |
|-----------|------|------|-------|
| `Modelfile.qwen3-4b` | qwen3:4b | 2.5 GB | **Recommended** — good for Apple Silicon |
| `Modelfile.qwen3` | qwen3:8b | 5.2 GB | Better quality, slower |
| `Modelfile.gemma3` | gemma3:4b | 3.3 GB | Fallback |

**Tested on Apple Silicon (M-series).** Runs entirely on-device via Ollama.

---

## Benchmark results (tested on Apple M-series, qwen3:4b)

```
Zero-LLM ops:  7/7 passed   avg response: 0.02s
Local-LLM ops: 3/3 passed   avg response: 20-37s

Tokens saved estimate (one session): ~30,000+
Saving per /outline call: ~11,600 tokens
Saving per /grep call:    ~11,900 tokens
```

Zero-LLM endpoints are always <50ms. LLM endpoints (summarize, codegen) take 15-40s on qwen3:4b due to chain-of-thought — use them for background tasks, not interactive queries.

---

## Best use cases for Claude Code

1. **"Does this file have a `process_data` function?"** → `J.grep("file.py", "def process_data")` — 0 tokens, instant
2. **"What's in this project?"** → `J.tree("~/project")` — 0 tokens, compact output
3. **"I need to understand this 800-line file"** → `J.summarize("file.py", focus="main logic")` — local LLM, no API cost
4. **"Write a helper function for X"** → `J.codegen("...")` — local LLM, no API cost
5. **"Patch this config value"** → `J.patch("config.py", "old", "new")` — 0 tokens, instant

---

## Requirements

- Python 3.9+
- Ollama running locally
- `flask requests numpy rich`

```bash
pip install flask requests numpy rich
```

---

## Version history

| Version | Changes |
|---------|---------|
| v6 "Secretary" | +5 new zero-LLM endpoints: /grep, /outline, /tree, /exists, /write, /patch; qwen3:4b; token savings counter |
| v5 | Core architecture: /ask auto-routing, /run, /read, /summarize, /codegen, semantic memory |

---

## License

MIT
