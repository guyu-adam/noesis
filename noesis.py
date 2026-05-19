"""
Noesis — A computational framework for GWT–IIT integration.

Computational consciousness framework: specialized processors compete for
global workspace access. Φ (integrated information) is measured across
broadcast cycles to test whether GWT mechanisms produce high-Φ states.

Two backends:
  - LLM agents (noesis-llm branch): Ollama-based, token-distribution Φ proxy
  - Neural processors (main branch): Small RNNs, causal TPM-based Φ

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

from workspace import GlobalWorkspace, CollaborativeWorkspace
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


# ── LLM Endpoints ──────────────────────────────────────────────────────────

@app.route("/status")
def status():
    return jsonify(state.summary())


@app.route("/experiment/run", methods=["POST"])
def run_experiment():
    """
    Run one cycle of the GWT+IIT experiment (LLM agents).

    Request:
        {
            "stimulus": "The input text all agents process",
            "mode": "competitive"
        }
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
    global state
    state = ExperimentState()
    workspace.reset()
    memory.clear()
    for agents_dict in _agent_cache_global.values():
        for agent in agents_dict.values():
            agent.reset_state()
    return jsonify({"reset": True})


@app.route("/memory")
def get_memory():
    return jsonify(memory.summary())


@app.route("/profile")
def get_profile():
    return jsonify({
        "profiles": state.profile_history[-20:],
        "summary": state.summary(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Neural experiment endpoints — main branch (RNN processors, causal TPM-based Φ)
# ══════════════════════════════════════════════════════════════════════════════

_neural_state = None


def _get_neural_state():
    global _neural_state
    if _neural_state is None:
        _neural_state = ExperimentState()
    return _neural_state


@app.route("/neural/run", methods=["POST"])
def neural_run():
    """
    Run one neural GWT+IIT cycle using small RNN processors.

    Request:
        {
            "stimulus": "text to encode into stimulus vector",
            "mode": "competitive",
            "n_neurons": 32,
            "n_input": 16
        }
    """
    d = request.json or {}
    stimulus_text = d.get("stimulus", "").strip()
    mode = d.get("mode", "competitive")
    n_neurons = d.get("n_neurons") or int(os.environ.get("NOESIS_N_NEURONS", "256"))
    n_input = d.get("n_input") or int(os.environ.get("NOESIS_N_INPUT", "32"))

    if not stimulus_text:
        return jsonify({"error": "stimulus required"}), 400
    if mode not in ("competitive", "random", "no_broadcast", "single_agent",
                    "collaborative", "hybrid"):
        return jsonify({"error": f"unknown mode: {mode}"}), 400

    from agents.neural_base import encode_stimulus
    import numpy as np

    nstate = _get_neural_state()
    nstate.status = "RUNNING"

    try:
        from experiment import run_neural_cycle
        stimulus_vec = encode_stimulus(stimulus_text, dim=n_input)

        if mode in ("collaborative", "hybrid"):
            if not isinstance(workspace, CollaborativeWorkspace):
                _neural_ws = CollaborativeWorkspace(memory)
            else:
                _neural_ws = workspace
        else:
            _neural_ws = workspace

        result = run_neural_cycle(
            stimulus_vec, _neural_ws, memory, mode=mode,
            n_neurons=n_neurons, n_input=n_input,
        )
        nstate.record_cycle(result)
        nstate.status = "IDLE"
        return jsonify(result)
    except Exception as e:
        nstate.status = "IDLE"
        return jsonify({"error": str(e)}), 500


@app.route("/neural/compare", methods=["POST"])
def neural_compare():
    d = request.json or {}
    stimuli = d.get("stimuli", [])
    modes = d.get("modes", ["competitive", "random", "no_broadcast", "collaborative"])
    cycles_per = d.get("cycles_per", 2)

    if len(stimuli) < 2:
        return jsonify({"error": "at least 2 stimuli required"}), 400

    from agents.neural_base import encode_stimulus
    import numpy as np

    nstate = _get_neural_state()
    nstate.status = "RUNNING"

    try:
        from experiment import run_neural_cycle

        all_results = {}
        for mode in modes:
            mem = SemanticMemory()
            ws = CollaborativeWorkspace(mem) if mode in ("collaborative", "hybrid") else GlobalWorkspace(mem)
            mode_results = []
            n_input = 16

            for stim_text in stimuli:
                stim_vec = encode_stimulus(stim_text, dim=n_input)
                for _ in range(cycles_per):
                    result = run_neural_cycle(stim_vec, ws, mem, mode=mode)
                    mode_results.append(result)

            all_results[mode] = mode_results
            mem.clear()

        analysis = {}
        for mode, cycles in all_results.items():
            deltas = [c["phi_delta"] for c in cycles]
            analysis[mode] = {
                "mean_phi_delta": round(float(np.mean(deltas)), 6),
                "n_cycles": len(cycles),
            }

        nstate.status = "IDLE"
        return jsonify({"results": all_results, "analysis": analysis})
    except Exception as e:
        nstate.status = "IDLE"
        return jsonify({"error": str(e)}), 500


@app.route("/neural/reset", methods=["POST"])
def neural_reset():
    global _neural_state
    _neural_state = ExperimentState()
    from agents.neural_base import NeuralProcessor
    from experiment import _neural_processor_cache
    for proc in _neural_processor_cache.values():
        if isinstance(proc, NeuralProcessor):
            proc.reset_state()
    workspace.reset()
    return jsonify({"reset": True, "processor_type": "neural_rnn"})


# Global reference for agent cache (used by reset)
_agent_cache_global = {}

# Monkey-patch agent cache
import experiment as _exp_mod
_exp_mod._agent_cache = _agent_cache_global


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    # Hardware-aware config
    from config import get_config, _GPU as _HAS_GPU
    cfg = get_config()
    if _HAS_GPU:
        gpu_info = cfg.summary()["gpu"]
        gpu_line = f"[bold green]GPU:[/bold green] {gpu_info['name']} ({gpu_info['vram_mb']}MB VRAM)"
    else:
        gpu_line = "[bold yellow]GPU:[/bold yellow] none (CPU-only)"

    n_default = int(os.environ.get("NOESIS_N_NEURONS", "256"))
    n_proc_default = int(os.environ.get("NOESIS_N_PROCESSORS",
                          os.environ.get("NOESIS_N_AGENTS", "5")))

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=7860, threaded=True),
        daemon=True,
    ).start()

    console.print(Panel(
        "[bold cyan]Noesis v0.3[/bold cyan]  ·  GWT–IIT Integration Framework\n\n"
        f"  {gpu_line}\n"
        f"  [bold]Neural:[/bold] {n_default} neurons/processor × {n_proc_default} processors "
        f"= {n_default * n_proc_default} total\n"
        f"  [bold]Backend:[/bold] {'CuPy GPU' if _HAS_GPU else 'NumPy CPU'}\n\n"
        "[bold]Neural Endpoints (main branch):[/bold]\n"
        "  [cyan]POST /neural/run[/cyan]          — run one cycle (RNN processors, causal Phi)\n"
        "  [cyan]POST /neural/compare[/cyan]      — comparison (all modes)\n"
        "  [cyan]POST /neural/reset[/cyan]        — reset neural experiment\n\n"
        "[bold]LLM Endpoints (legacy):[/bold]\n"
        "  [cyan]POST /experiment/run[/cyan]       — run one cycle (LLM agents)\n"
        "  [cyan]POST /experiment/compare[/cyan]   — full comparison experiment\n\n"
        "[bold]Shared:[/bold]\n"
        "  [cyan]GET  /status[/cyan]               — Phi trace + summary\n"
        "  [cyan]GET  /profile[/cyan]              — consciousness profile trace\n\n"
        f"[bold]LLM Model:[/bold] {MODEL}\n"
        "[dim]http://localhost:7860[/dim]\n\n"
        "[dim]Set NOESIS_N_NEURONS, NOESIS_N_AGENTS, NOESIS_N_UNROLL env vars to scale[/dim]",
        border_style="cyan", title="[bold]Ready[/bold]"
    ))
    console.print("[green]Waiting for experiments...[/green]\n")

    while True:
        time.sleep(1)
