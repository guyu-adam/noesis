"""
Noesis — A computational framework for GWT–IIT integration.

Multi-agent consciousness simulation where specialized agents compete
for global workspace access. Φ (integrated information) is measured
across broadcast cycles to test whether GWT mechanisms produce high-Φ states.

Two theories, one computational testbed.
"""

import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from flask import Flask, request, jsonify
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from workspace import GlobalWorkspace
from memory import SemanticMemory

console = Console()
app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
MODEL = os.environ.get("NOESIS_MODEL", "qwen3:4b")

# ── Core components ────────────────────────────────────────────────────────────
memory = SemanticMemory()
workspace = GlobalWorkspace(memory)

# ── State ──────────────────────────────────────────────────────────────────────
class ExperimentState:
    def __init__(self):
        self.status = "IDLE"
        self.cycle = 0
        self.broadcast_history: list = []
        self.phi_history: list = []
        self._lock = threading.Lock()

    def record_broadcast(self, agent_name: str, proposal: str, phi: float):
        with self._lock:
            self.cycle += 1
            entry = {
                "cycle": self.cycle,
                "time": datetime.now().isoformat(),
                "winner": agent_name,
                "content": proposal[:300],
                "phi": phi,
            }
            self.broadcast_history.append(entry)
            self.phi_history.append(phi)

state = ExperimentState()

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify({
        "status": state.status,
        "cycle": state.cycle,
        "phi_history": state.phi_history[-20:],
        "model": MODEL,
    })

@app.route("/experiment/run", methods=["POST"])
def run_experiment():
    """
    Run one cycle of the GWT+IIT experiment.

    Request:
        { "stimulus": "The input text all agents process" }

    Response:
        {
            "cycle": N,
            "proposals": { "perceptor": "...", "reasoner": "...", ... },
            "winner": "reasoner",
            "phi": 0.42,
            "broadcast": "content that won...",
            "narrative": "first-person report..."
        }
    """
    d = request.json or {}
    stimulus = d.get("stimulus", "").strip()
    if not stimulus:
        return jsonify({"error": "stimulus required"}), 400

    state.status = "RUNNING"

    try:
        from experiment import run_cycle
        result = run_cycle(stimulus, workspace, memory, MODEL)
        state.record_broadcast(
            result["winner"], result["broadcast"], result["phi"]
        )
        state.status = "IDLE"
        return jsonify(result)
    except Exception as e:
        state.status = "IDLE"
        return jsonify({"error": str(e)}), 500

@app.route("/experiment/batch", methods=["POST"])
def run_batch():
    """
    Run multiple cycles across different stimuli.

    Request:
        { "stimuli": ["stimulus 1", "stimulus 2", ...] }
    """
    d = request.json or {}
    stimuli = d.get("stimuli", [])
    if not stimuli:
        return jsonify({"error": "stimuli required"}), 400

    state.status = "RUNNING"
    results = []
    try:
        from experiment import run_cycle
        for stim in stimuli:
            result = run_cycle(stim, workspace, memory, MODEL)
            state.record_broadcast(
                result["winner"], result["broadcast"], result["phi"]
            )
            results.append(result)
        state.status = "IDLE"
        return jsonify({"results": results, "phi_trace": state.phi_history})
    except Exception as e:
        state.status = "IDLE"
        return jsonify({"error": str(e)}), 500

@app.route("/experiment/reset", methods=["POST"])
def reset_experiment():
    """Clear history and reset for fresh experiment."""
    global state
    state = ExperimentState()
    workspace.reset()
    memory.clear()
    return jsonify({"reset": True})

@app.route("/memory")
def get_memory():
    return jsonify(memory.summary())

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=7860, threaded=True),
        daemon=True,
    ).start()

    console.print(Panel(
        "[bold cyan]Noesis v0.1[/bold cyan]  ·  GWT–IIT Integration Framework\n\n"
        "[bold]Endpoints:[/bold]\n"
        "  [cyan]POST /experiment/run[/cyan]    — run one GWT+IIT cycle\n"
        "  [cyan]POST /experiment/batch[/cyan]  — run multiple stimuli\n"
        "  [cyan]POST /experiment/reset[/cyan]  — reset experiment state\n"
        "  [cyan]GET  /status[/cyan]            — view Φ trace\n"
        "  [cyan]GET  /memory[/cyan]            — semantic memory view\n\n"
        f"[bold]Model:[/bold] {MODEL}\n"
        "[dim]http://localhost:7860[/dim]",
        border_style="cyan", title="[bold]Ready[/bold]"
    ))
    console.print("[green]Waiting for experiments...[/green]\n")

    while True:
        time.sleep(1)
