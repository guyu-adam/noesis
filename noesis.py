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
from rich.table import Table

from workspace import GlobalWorkspace
from memory import SemanticMemory
from metrics import consciousness_profile, lz_complexity, semantic_entropy

console = Console()
app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"
MODEL = os.environ.get("NOESIS_MODEL", "qwen3:4b")

# ── Core components ────────────────────────────────────────────────────────
memory = SemanticMemory()
workspace = GlobalWorkspace(memory)


# ── State ──────────────────────────────────────────────────────────────────
class ExperimentState:
    def __init__(self):
        self.status = "IDLE"
        self.cycle = 0
        self.broadcast_history: list = []
        self.phi_history: list = []
        self.complexity_history: list = []
        self.profile_history: list = []
        self._lock = threading.Lock()

    def record_cycle(self, result: dict):
        with self._lock:
            self.cycle += 1
            entry = {
                "cycle": self.cycle,
                "time": datetime.now().isoformat(),
                "winner": result.get("winner", "none"),
                "content": str(result.get("broadcast", ""))[:300],
                "phi": result.get("phi_after", 0),
                "phi_delta": result.get("phi_delta", 0),
                "broadcasted": result.get("broadcasted", False),
                "complexity": result.get("complexity", 0),
                "mode": result.get("mode", "unknown"),
            }
            self.broadcast_history.append(entry)
            self.phi_history.append(result.get("phi_after", 0))
            self.complexity_history.append(result.get("complexity", 0))
            if "conscious_access_confidence" in str(result):
                self.profile_history.append(result.get("profile", {}))

    def summary(self) -> dict:
        from iit import phi_trace
        return {
            "status": self.status,
            "total_cycles": self.cycle,
            "phi_analysis": phi_trace(self.phi_history),
            "complexity_analysis": phi_trace(self.complexity_history),
            "model": MODEL,
        }


state = ExperimentState()


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify(state.summary())


@app.route("/experiment/run", methods=["POST"])
def run_experiment():
    """
    Run one cycle of the GWT+IIT experiment.

    Request:
        {
            "stimulus": "The input text all agents process",
            "mode": "competitive"  // optional: competitive|random|no_broadcast|single_agent
        }

    Response:
        {cycle data including phi, proposals, winner, broadcast, narrative, profile}
    """
    d = request.json or {}
    stimulus = d.get("stimulus", "").strip()
    mode = d.get("mode", "competitive")

    if not stimulus:
        return jsonify({"error": "stimulus required"}), 400
    if mode not in ("competitive", "random", "no_broadcast", "single_agent"):
        return jsonify({"error": f"unknown mode: {mode}"}), 400

    state.status = "RUNNING"

    try:
        from experiment import run_cycle
        result = run_cycle(stimulus, workspace, memory, MODEL, mode=mode)

        # Attach consciousness profile
        result["profile"] = consciousness_profile(
            phi=result["phi_after"],
            complexity=result["complexity"],
            fisher_trace=result["fisher_trace"],
            broadcasted=result["broadcasted"],
            attention_score=result["attention_score"],
        )

        state.record_cycle(result)
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
        {
            "stimuli": ["s1", "s2", ...],
            "mode": "competitive",
            "cycles_per_stimulus": 3
        }
    """
    d = request.json or {}
    stimuli = d.get("stimuli", [])
    mode = d.get("mode", "competitive")
    cycles_per = d.get("cycles_per_stimulus", 1)

    if not stimuli:
        return jsonify({"error": "stimuli required"}), 400

    state.status = "RUNNING"
    results = []
    try:
        from experiment import run_cycle
        for stim in stimuli:
            for _ in range(cycles_per):
                result = run_cycle(stim, workspace, memory, MODEL, mode=mode)
                result["profile"] = consciousness_profile(
                    phi=result["phi_after"],
                    complexity=result["complexity"],
                    fisher_trace=result["fisher_trace"],
                    broadcasted=result["broadcasted"],
                    attention_score=result["attention_score"],
                )
                state.record_cycle(result)
                results.append(result)
        state.status = "IDLE"
        return jsonify({
            "results": results,
            "summary": state.summary(),
        })
    except Exception as e:
        state.status = "IDLE"
        return jsonify({"error": str(e)}), 500


@app.route("/experiment/compare", methods=["POST"])
def run_comparison():
    """
    Run the FULL comparison experiment — same stimuli across all modes.

    This is the definitive experiment that tests all three hypotheses:
      1. Φ_competitive > Φ_random (competition matters)
      2. Φ_competitive > Φ_no_broadcast (broadcast creates integration)
      3. Φ_competitive > Φ_single_agent (multi-agent matters)

    Request:
        {
            "stimuli": ["s1", "s2", ...],
            "modes": ["competitive", "random", "no_broadcast"],
            "cycles_per_stimulus": 3
        }

    Response:
        {mode: [results], analysis: {hypothesis tests}}
    """
    d = request.json or {}
    stimuli = d.get("stimuli", [])
    modes = d.get("modes", ["competitive", "random", "no_broadcast"])
    cycles_per = d.get("cycles_per_stimulus", 2)

    if len(stimuli) < 2:
        return jsonify({"error": "at least 2 stimuli required"}), 400

    state.status = "RUNNING"
    try:
        from experiment import run_comparison_experiment, analyze_comparison
        results = run_comparison_experiment(stimuli, MODEL, cycles_per, modes)
        analysis = analyze_comparison(results)
        state.status = "IDLE"
        return jsonify({
            "results": results,
            "analysis": analysis,
            "n_stimuli": len(stimuli),
            "cycles_per_stimulus": cycles_per,
            "total_cycles": len(stimuli) * cycles_per * len(modes),
        })
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
    # Clear agent internal states
    for agents_dict in _agent_cache_global.values():
        for agent in agents_dict.values():
            agent.reset_state()
    return jsonify({"reset": True})


@app.route("/memory")
def get_memory():
    return jsonify(memory.summary())


@app.route("/profile")
def get_profile():
    """Get full consciousness profile trace across cycles."""
    return jsonify({
        "profiles": state.profile_history[-20:],
        "summary": state.summary(),
    })


# Global reference for agent cache (used by reset)
_agent_cache_global = {}


# ── Monkey-patch agent cache ───────────────────────────────────────────────
import experiment as _exp_mod
_exp_mod._agent_cache = _agent_cache_global


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=7860, threaded=True),
        daemon=True,
    ).start()

    console.print(Panel(
        "[bold cyan]Noesis v0.2[/bold cyan]  ·  GWT–IIT Integration Framework\n\n"
        "[bold]Endpoints:[/bold]\n"
        "  [cyan]POST /experiment/run[/cyan]       — run one GWT+IIT cycle\n"
        "  [cyan]POST /experiment/batch[/cyan]     — run multiple stimuli\n"
        "  [cyan]POST /experiment/compare[/cyan]   — full comparison experiment\n"
        "  [cyan]POST /experiment/reset[/cyan]     — reset experiment state\n"
        "  [cyan]GET  /status[/cyan]               — Φ trace + summary\n"
        "  [cyan]GET  /profile[/cyan]              — consciousness profile trace\n"
        "  [cyan]GET  /memory[/cyan]               — semantic memory view\n\n"
        f"[bold]Model:[/bold] {MODEL}\n"
        "[dim]http://localhost:7860[/dim]",
        border_style="cyan", title="[bold]Ready[/bold]"
    ))
    console.print("[green]Waiting for experiments...[/green]\n")

    while True:
        time.sleep(1)
