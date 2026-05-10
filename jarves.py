"""
JARVES v5 - Claude's execution layer
Two paths only:
  1. Deterministic (no LLM): file I/O, shell, math
  2. LLM direct call: code gen, summarize, Q&A
No agent loop. Fast. Reliable.
"""

import os, json, threading, time, re, subprocess, math
from datetime import datetime
from pathlib import Path

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import requests as req
from flask import Flask, request, jsonify
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()
app = Flask(__name__)
MEMORY_FILE = Path(__file__).parent / "memory.json"
EMBED_FILE  = Path(__file__).parent / "embeddings.json"
OLLAMA      = "http://localhost:11434/api/chat"
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
MODEL       = "qwen3-jarves"

# ── memory ─────────────────────────────────────────────────────────────────────

class Memory:
    def __init__(self):
        self.notes: dict = {}
        self.history: list = []
        self.embeddings: list = []  # [{id, task, result, emb}]
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if MEMORY_FILE.exists():
            try:
                d = json.loads(MEMORY_FILE.read_text())
                self.notes   = d.get("notes", {})
                self.history = d.get("history", [])
            except Exception:
                pass
        if EMBED_FILE.exists():
            try:
                self.embeddings = json.loads(EMBED_FILE.read_text())
            except Exception:
                pass

    def _save(self):
        MEMORY_FILE.write_text(json.dumps(
            {"notes": self.notes, "history": self.history[-40:]},
            ensure_ascii=False, indent=2
        ))

    def _save_embeddings(self):
        EMBED_FILE.write_text(json.dumps(self.embeddings[-40:], ensure_ascii=False))

    def _embed(self, text: str) -> list:
        try:
            r = req.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=10)
            return r.json().get("embedding", [])
        except Exception:
            return []

    def _cosine(self, a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a))
        nb  = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def clear(self):
        with self._lock:
            self.history = []
            self.embeddings = []
            self._save()
            if EMBED_FILE.exists():
                EMBED_FILE.unlink()

    def save(self, key: str, val: str):
        with self._lock:
            self._load()
            self.notes[key] = val
            self._save()

    def record(self, tid: int, task: str, result: str):
        with self._lock:
            self._load()
            self.history.append({
                "id": tid,
                "time": datetime.now().strftime("%m-%d %H:%M"),
                "task": task[:100],
                "result": result[:200],
            })
            self._save()
        # embed async so it doesn't block the response
        def _do_embed():
            emb = self._embed(task)
            if emb:
                with self._lock:
                    self.embeddings.append({
                        "id": tid, "task": task[:100],
                        "result": result[:200], "emb": emb
                    })
                    self._save_embeddings()
        threading.Thread(target=_do_embed, daemon=True).start()

    def ctx(self, current_task: str = "") -> str:
        out = []
        if self.notes:
            out.append("Notes: " + " | ".join(f"{k}={v}" for k, v in list(self.notes.items())[-6:]))
        if not self.history:
            return "\n".join(out)

        if current_task and self.embeddings:
            q_emb = self._embed(current_task)
            if q_emb:
                scored = sorted(
                    self.embeddings, key=lambda e: self._cosine(q_emb, e["emb"]), reverse=True
                )[:3]
                out.append("Relevant: " + " | ".join(
                    f"#{e['id']} \"{e['task'][:50]}\"→{e['result'][:60]}" for e in scored
                ))
                return "\n".join(out)

        # fallback: chronological last 3
        out.append("Recent: " + " | ".join(
            f"#{h['id']} \"{h['task'][:50]}\"→{h['result'][:60]}"
            for h in self.history[-3:]
        ))
        return "\n".join(out)

mem = Memory()

# ── state ───────────────────────────────────────────────────────────────────────

class State:
    def __init__(self):
        self.status = "IDLE"
        self.task   = "—"
        self.result = ""
        self.count  = 0
        self._lock  = threading.Lock()
    def set(self, status, task=None):
        with self._lock:
            self.status = status
            if task is not None: self.task = task

st = State()

# ── deterministic tools (no LLM) ──────────────────────────────────────────────

def _shell(cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip() or "(no output)"

def _read(path: str, limit: int = 8000) -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"File not found: {path}"
    text = p.read_text(errors="replace")
    return text[:limit] + (f"\n...[truncated, total {len(text)} chars]" if len(text) > limit else "")

def _ls(path: str, pattern: str = "*") -> str:
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return f"Path not found: {path}"
    items = sorted(p.glob(pattern))
    return "\n".join(
        f"{'📁' if i.is_dir() else '📄'} {i.name}  ({i.stat().st_size//1024}KB)"
        for i in items
    ) or "(empty)"

# ── LLM direct call ────────────────────────────────────────────────────────────

def llm(task: str, system: str = "", max_tokens: int = 600) -> str:
    base_system = (
        "You are JARVES, Claude's local execution assistant.\n"
        "Output ONLY the raw result. No preamble, no explanation, no markdown headers.\n"
        "No code fences (never use ``` or ~~~). No docstrings. No comments.\n"
        "For code: bare function body only, starting with 'def'.\n"
        "For facts: one sentence.\n"
    )
    ctx = mem.ctx(task)
    if ctx:
        base_system += f"Context:\n{ctx}\n"
    if system:
        base_system += system

    # thinking eats ~150-300 tokens before content starts — budget accordingly
    total_predict = max_tokens + 300

    for attempt in range(3):
        try:
            resp = req.post(OLLAMA, json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": base_system},
                    {"role": "user",   "content": task},
                ],
                "options": {"num_predict": total_predict},
                "stream": False,
            }, timeout=180)
            msg = resp.json().get("message", {})
            # content is the real answer; thinking is internal monologue — ignore it
            content = msg.get("content", "").strip()
            if content:
                # strip markdown code fences if model ignores instructions
                content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
                return content.strip()
            console.print(f"[dim yellow]empty content, retry {attempt+1}/3[/dim yellow]")
        except Exception as e:
            if attempt == 2:
                return f"ERROR: {e}"
    return "(no response)"

# ── routing ─────────────────────────────────────────────────────────────────────

# Patterns → direct deterministic execution, zero LLM cost
DIRECT_ROUTES = [
    # ls / list files
    (re.compile(r"(ls|list|列出?|有什么|有哪些).{0,20}?(文件|folder|目录|dir|~/|/\w)", re.I),
     lambda t: _ls(_extract_path(t, "~/Desktop"))),
    # shell exec explicit
    (re.compile(r"^(run|exec|执行|运行)[：:\s]+(.+)", re.I | re.S),
     lambda t: _shell(re.search(r"^(?:run|exec|执行|运行)[：:\s]+(.+)", t, re.I | re.S).group(1))),
    # time
    (re.compile(r"(几点|current time|what time|现在时间)", re.I),
     lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    # math
    (re.compile(r"^[\d\s\+\-\*\/\.\^\(\)]+$"),
     lambda t: str(eval(t.replace("^", "**")))),
]

def _extract_path(task: str, default: str) -> str:
    m = re.search(r"(~/[^\s]+|/[^\s]+)", task)
    if m: return m.group(1)
    for kw, path in [("桌面","~/Desktop"),("desktop","~/Desktop"),("下载","~/Downloads")]:
        if kw.lower() in task.lower(): return path
    return default

def route(task: str) -> tuple[str, str]:
    """Returns (mode, result_or_sentinel)."""
    for pattern, fn in DIRECT_ROUTES:
        if pattern.search(task):
            try:
                return "direct", fn(task)
            except Exception as e:
                return "direct", f"Error: {e}"
    return "llm", ""

# ── task runner ─────────────────────────────────────────────────────────────────

def run_task(task: str, sender: str, system: str = "", max_tokens: int = 600) -> str:
    st.count += 1
    st.set("WORKING", task)
    ts = datetime.now().strftime("%H:%M:%S")

    mode, pre = route(task)

    console.print()
    console.print(Rule(f"[cyan]#{st.count}  {ts}  [{mode}]  {sender}[/cyan]"))
    console.print(f"[yellow]▶ {task[:120]}[/yellow]\n")

    try:
        result = pre if mode == "direct" else llm(task, system, max_tokens)
        st.result = result
        mem.record(st.count, task, result)
        console.print(Panel(result[:1000], title="[green]✓[/green]", border_style="green"))
    except Exception as e:
        result = f"ERROR: {e}"
        st.result = result
        console.print(Panel(result, title="[red]✗[/red]", border_style="red"))
    finally:
        st.set("IDLE", "—")

    return result

# ── endpoints ───────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify({"status": st.status, "task": st.task, "count": st.count, "last": st.result})

@app.route("/memory")
def memory():
    return jsonify({"notes": mem.notes, "history": mem.history[-10:]})

@app.route("/ask", methods=["POST"])
def ask():
    """General purpose. Auto-routes between direct and LLM."""
    d = request.json or {}
    task = d.get("task","").strip()
    if not task: return jsonify({"error":"task required"}), 400
    if st.status == "WORKING": return jsonify({"error":"busy"}), 429
    result = run_task(task, d.get("from","?"), d.get("system",""), d.get("max_tokens",600))
    ok = not result.startswith("ERROR:")
    return jsonify({"result": result} if ok else {"error": result}), (200 if ok else 500)

@app.route("/chat", methods=["POST"])
def chat():
    """Non-blocking /ask."""
    d = request.json or {}
    task = d.get("task","").strip()
    if not task: return jsonify({"error":"task required"}), 400
    if st.status == "WORKING": return jsonify({"error":"busy"}), 429
    threading.Thread(target=run_task, args=(task, d.get("from","?"),
                     d.get("system",""), d.get("max_tokens",600)), daemon=True).start()
    return jsonify({"accepted": True})

@app.route("/run", methods=["POST"])
def run_cmd():
    """Direct shell execution, no LLM. Fastest path."""
    d = request.json or {}
    cmd = d.get("cmd","").strip()
    if not cmd: return jsonify({"error":"cmd required"}), 400
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(Rule(f"[green]shell  {ts}[/green]"))
    console.print(f"[dim]$ {cmd}[/dim]")
    try:
        out = _shell(cmd, timeout=d.get("timeout", 30))
        console.print(f"[dim]{out[:300]}[/dim]")
        return jsonify({"output": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/read", methods=["POST"])
def read():
    """Read file, return content. No LLM."""
    d = request.json or {}
    path = d.get("path","").strip()
    if not path: return jsonify({"error":"path required"}), 400
    limit = d.get("limit", 8000)
    content = _read(path, limit)
    console.print(Rule(f"[green]read  {path}[/green]"))
    console.print(f"[dim]{content[:200]}...[/dim]")
    return jsonify({"content": content, "path": path})

@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Compress file or text → bullet points.
    Claude calls this instead of reading large files directly.
    {"path":"~/foo.py", "focus":"what to extract"}
    {"text":"...", "focus":"..."}
    """
    d = request.json or {}
    focus = d.get("focus", "key logic and structure")

    if "path" in d:
        content = _read(d["path"], limit=7000)
        label = d["path"]
    elif "text" in d:
        content = d["text"][:7000]
        label = "text"
    else:
        return jsonify({"error": "path or text required"}), 400

    if content.startswith("File not found"):
        return jsonify({"error": content}), 404

    ts = datetime.now().strftime("%H:%M:%S")
    console.print(Rule(f"[magenta]summarize  {ts}[/magenta]"))
    console.print(f"[yellow]▶ {label} | focus: {focus}[/yellow]\n")

    prompt = f"Focus on: {focus}\n\nContent:\n{content}"
    result = llm(prompt,
                 system="Summarize in ≤6 concise bullet points. Facts only. No preamble.\n",
                 max_tokens=400)
    console.print(Panel(result, title="[magenta]summary[/magenta]", border_style="magenta"))
    return jsonify({"summary": result, "source": label})

@app.route("/codegen", methods=["POST"])
def codegen():
    """
    Code generation optimized path.
    {"task":"write X", "lang":"python"}
    """
    d = request.json or {}
    task = d.get("task","").strip()
    lang = d.get("lang","python")
    if not task: return jsonify({"error":"task required"}), 400
    if st.status == "WORKING": return jsonify({"error":"busy"}), 429

    ts = datetime.now().strftime("%H:%M:%S")
    console.print(Rule(f"[cyan]codegen  {ts}[/cyan]"))
    console.print(f"[yellow]▶ {task}[/yellow]\n")

    result = llm(task,
                 system=f"Output {lang} code only. No explanation. No markdown fences.\n",
                 max_tokens=700)
    console.print(Panel(result, title="[cyan]code[/cyan]", border_style="cyan"))
    st.count += 1
    mem.record(st.count, task, result[:200])
    return jsonify({"code": result, "lang": lang})

@app.route("/batch", methods=["POST"])
def batch():
    """Run multiple shell cmds or ask tasks at once. Returns list of results.
    {"tasks": [{"type":"run","cmd":"ls ~"},{"type":"ask","task":"..."}]}
    """
    d = request.json or {}
    tasks = d.get("tasks", [])
    if not tasks: return jsonify({"error": "tasks required"}), 400
    results = []
    for t in tasks:
        typ = t.get("type", "ask")
        try:
            if typ == "run":
                results.append({"type": "run", "result": _shell(t.get("cmd",""))})
            elif typ == "read":
                results.append({"type": "read", "result": _read(t.get("path",""))})
            else:
                results.append({"type": "ask", "result": run_task(t.get("task",""), "batch")})
        except Exception as e:
            results.append({"type": typ, "error": str(e)})
    return jsonify({"results": results})

@app.route("/memory/clear", methods=["POST"])
def memory_clear():
    """Clear history from both memory and disk."""
    mem.clear()
    console.print("[yellow]memory history cleared[/yellow]")
    return jsonify({"cleared": True, "notes": mem.notes})

@app.route("/note", methods=["POST"])
def note():
    """Save a key-value note to persistent memory."""
    d = request.json or {}
    key = d.get("key","").strip()
    val = d.get("value","").strip()
    if not key or not val: return jsonify({"error":"key and value required"}), 400
    mem.save(key, val)
    console.print(f"[green]note saved:[/green] {key} = {val}")
    return jsonify({"saved": {key: val}})

# ── main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=7860, threaded=True),
        daemon=True
    ).start()

    console.print(Panel(
        "[bold cyan]JARVES v5[/bold cyan]  ·  Claude's execution layer\n\n"
        "[bold]Endpoints:[/bold]\n"
        "  [cyan]/ask[/cyan]       auto-route (direct or LLM)\n"
        "  [cyan]/run[/cyan]       shell command, no LLM\n"
        "  [cyan]/read[/cyan]      read file, no LLM\n"
        "  [cyan]/summarize[/cyan] compress file/text → bullets\n"
        "  [cyan]/codegen[/cyan]   code generation\n"
        "  [cyan]/note[/cyan]      save to memory\n"
        "  [cyan]/chat[/cyan]      non-blocking /ask\n\n"
        f"[bold]Memory:[/bold] {len(mem.notes)} notes · {len(mem.history)} past tasks\n"
        "[dim]http://localhost:7860[/dim]",
        border_style="cyan", title="[bold]Ready[/bold]"
    ))
    console.print("[green]✓ Waiting...[/green]\n")

    while True:
        time.sleep(1)
