# JARVES

Minimal local AI execution server. Runs a small LLM (via Ollama) behind a Flask API with semantic memory. Designed to be called by Claude Code or any orchestrator that needs a cheap local execution layer.

**Why**: Offloads file reading, shell execution, and routine LLM tasks to a local model, saving API tokens on repetitive work.

---

## Architecture

```
Claude Code / orchestrator
        │  HTTP (localhost:7860)
        ▼
  jarves.py  (Flask server)
        │
        ├── /ask    → Ollama LLM (qwen3:8b or similar)
        ├── /run    → subprocess shell execution
        ├── /read   → file reading with line limit
        ├── /summarize → LLM-powered file/text summary
        ├── /batch  → parallel multi-task execution
        └── /memory → semantic memory (nomic-embed-text embeddings)
```

---

## Quick start

### 1. Install Ollama and pull a model

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:8b

# Create the JARVES model with strict no-fence system prompt
ollama create qwen3-jarves -f Modelfile.qwen3
```

### 2. Start the server

```bash
pip install flask requests numpy
python jarves.py
# Listening on http://localhost:7860
```

### 3. Use the client

```python
import sys; sys.path.insert(0, '/path/to/jarves')
from j import J

J.ask("write a Python function to flatten a nested list")
J.run("ls ~/Desktop")
J.read("~/some/file.py")
J.summarize("~/some/big_file.py", focus="error handling")
J.batch([("run", "whoami"), ("ask", "current date")])
J.status()
J.clear()   # clear conversation history
```

---

## Modelfiles

Two included modelfiles:

| File | Base model | Size | Notes |
|------|-----------|------|-------|
| `Modelfile.qwen3` | qwen3:8b | 5.2 GB | Recommended — clean output, no fences |
| `Modelfile.gemma3` | gemma3:4b | 3.3 GB | Lighter — use if RAM is tight |

Both configure the model to output **raw results only** — no markdown fences, no explanations, no `<think>` tags.

---

## API reference

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/ask` | POST | `{"task": "...", "max_tokens": 600}` | `{"result": "..."}` |
| `/run` | POST | `{"cmd": "shell command"}` | `{"output": "..."}` |
| `/read` | POST | `{"path": "~/file", "limit": 8000}` | `{"content": "..."}` |
| `/summarize` | POST | `{"path": "~/file", "focus": "..."}` | `{"summary": "..."}` |
| `/batch` | POST | `{"tasks": [...]}` | `{"results": [...]}` |
| `/note` | POST | `{"key": "k", "value": "v"}` | `{"saved": true}` |
| `/memory/clear` | POST | — | `{"cleared": true}` |
| `/status` | GET | — | `{"model": "...", "history": N}` |

---

## Semantic memory

JARVES stores conversation history and retrieves relevant context using `nomic-embed-text` embeddings (via Ollama). Memory persists in `memory.json` and `embeddings.json`.

```bash
ollama pull nomic-embed-text
```

If `nomic-embed-text` is unavailable, the server falls back to simple recency-based context.

---

## Requirements

- Python 3.9+
- Ollama running locally
- `flask requests numpy`

---

## License

MIT
