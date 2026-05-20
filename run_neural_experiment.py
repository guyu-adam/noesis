"""
Neural experiment runner — batch comparison across 6 modes using RNN processors.

This is the MAIN BRANCH experiment script. It uses:
  - 5 specialized RNN processors (Perceptor, Reasoner, Evaluator, Integrator, Predictor)
  - Real causal TPM-based Φ computation (neural_iit.py)
  - CuPy GPU acceleration for RTX 5060 Ti 16GB

Usage:
    python run_neural_experiment.py

Environment variables:
    NOESIS_N_NEURONS   — override neuron count per processor (default: 512 on 16GB)
    NOESIS_N_INPUT     — stimulus vector dimension (default: 32)
    NOESIS_N_CYCLES    — cycles per stimulus per mode (default: 5)
    NOESIS_N_UNROLL    — RNN unroll steps (default: 25)
    NOESIS_BATCH_SIZE  — parallel cycles (default: 8 on 16GB)
"""

import sys
import json
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import run_neural_comparison, analyze_neural_comparison
from agents.neural_base import encode_stimulus
from config import get_config


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# Stimuli — same 20 questions as LLM experiments for comparability
# ══════════════════════════════════════════════════════════════════════════════

STIMULI = [
    # Consciousness & cognition (8)
    "What is the relationship between perception and consciousness?",
    "How does attention shape what we become aware of?",
    "Can a system be conscious without self-awareness?",
    "What distinguishes integrated information from mere computation?",
    "How does memory influence the quality of conscious experience?",
    "Is the global workspace a necessary condition for consciousness?",
    "What role does prediction error play in conscious perception?",
    "Can multiple processors jointly produce a unified conscious state?",
    # Multi-agent coordination (12 — generalizability)
    "How do agents reach consensus when their observations conflict?",
    "What mechanisms allow distributed agents to coordinate without a central authority?",
    "When does coalition formation improve collective decision quality?",
    "How should information be shared to maximise collective intelligence?",
    "What is the binding problem and how might it be solved computationally?",
    "How does information integration differ from information broadcasting?",
    "Under what conditions does multi-agent communication increase system coherence?",
    "What role does redundancy play in robust collective representations?",
    "How should an agent weight novel information versus prior consensus?",
    "What distinguishes useful surprise from disruptive noise in a multi-agent system?",
    "Can competition and collaboration coexist in a broadcast architecture?",
    "How does the structure of a communication channel shape the information that flows through it?",
]

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

CYCLES = int(os.environ.get("NOESIS_N_CYCLES", "5"))
MODES = ["competitive", "random", "no_broadcast", "collaborative", "hybrid", "single_processor"]

cfg = get_config()
cfg.print()

# ══════════════════════════════════════════════════════════════════════════════
# Encode stimuli as neural vectors
# ══════════════════════════════════════════════════════════════════════════════

stimuli_vecs = [encode_stimulus(s, dim=cfg.n_input) for s in STIMULI]
print(f"[ENCODE] {len(stimuli_vecs)} stimuli → {cfg.n_input}-dim vectors")

# ══════════════════════════════════════════════════════════════════════════════
# Run experiment
# ══════════════════════════════════════════════════════════════════════════════

total = len(STIMULI) * CYCLES * len(MODES)
print(f"[START] {time.strftime('%H:%M:%S')}  {len(STIMULI)} stimuli × {CYCLES} cycles × {len(MODES)} modes = {total} total cycles")
print(f"        {cfg.n_neurons} neurons/processor × {cfg.n_processors} processors = {cfg.total_neurons} total")
print(f"        batch_size={cfg.batch_size}, n_state_clusters={cfg.n_state_clusters}")
print(f"        GPU: {'yes' if cfg.gpu else 'no'}", flush=True)

results = run_neural_comparison(
    stimuli_vecs,
    modes=MODES,
    cycles_per_stimulus=CYCLES,
    n_neurons=cfg.n_neurons,
    n_input=cfg.n_input,
)

print(f"[EXPERIMENTS DONE] {time.strftime('%H:%M:%S')}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

analysis = analyze_neural_comparison(results)

# ══════════════════════════════════════════════════════════════════════════════
# Build output
# ══════════════════════════════════════════════════════════════════════════════

output = {
    "results_summary": {},
    "analysis": analysis,
    "metadata": {
        "branch": "main",
        "processor_type": "neural_rnn",
        "n_neurons": cfg.n_neurons,
        "n_input": cfg.n_input,
        "n_unroll": cfg.n_unroll,
        "n_processors": cfg.n_processors,
        "total_neurons": cfg.total_neurons,
        "n_state_clusters": cfg.n_state_clusters,
        "gpu": cfg.gpu,
        "gpu_info": cfg.summary().get("gpu", {}),
        "n_stimuli": len(STIMULI),
        "cycles_per_stimulus": CYCLES,
        "modes": MODES,
        "total_cycles": total,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
    },
}

for mode, cycles in results.items():
    phi_deltas = [c["phi_delta"] for c in cycles]
    phi_afters = [c["phi_after"] for c in cycles]
    phi_befores = [c["phi_before"] for c in cycles]
    complexities = [c.get("complexity", 0) for c in cycles]
    output["results_summary"][mode] = {
        "mean_phi_after": round(float(np.mean(phi_afters)), 6),
        "std_phi_after": round(float(np.std(phi_afters)), 6),
        "mean_phi_before": round(float(np.mean(phi_befores)), 6),
        "mean_phi_delta": round(float(np.mean(phi_deltas)), 6),
        "std_phi_delta": round(float(np.std(phi_deltas)), 6),
        "max_phi_delta": round(float(np.max(phi_deltas)), 6),
        "mean_complexity": round(float(np.mean(complexities)), 6),
        "n_cycles": len(cycles),
        "broadcast_rate": round(
            sum(1 for c in cycles if c.get("broadcasted")) / max(len(cycles), 1), 4
        ),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
os.makedirs(out_dir, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')

# Summary JSON
out_path = os.path.join(out_dir, f"neural_results_{ts}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, cls=NpEncoder, ensure_ascii=False)

# Raw cycles JSONL (for ablation analysis)
raw_path = os.path.join(out_dir, f"neural_raw_{ts}.jsonl")
with open(raw_path, "w", encoding="utf-8") as f:
    for mode, cycles in results.items():
        for c in cycles:
            f.write(json.dumps(c, cls=NpEncoder, ensure_ascii=False) + "\n")

print(f"[SAVED] {out_path}", flush=True)
print(f"[SAVED] {raw_path}", flush=True)
print(f"[DONE] {time.strftime('%H:%M:%S')}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Print summary
# ══════════════════════════════════════════════════════════════════════════════

print("\n=== Results Summary ===")
print(json.dumps(output["results_summary"], indent=2))
print("\n=== Hypothesis Tests ===")
for k, v in analysis.get("hypothesis_tests", {}).items():
    if isinstance(v, dict):
        print(f"  {k}: p={v.get('p_value', '?')}, supported={v.get('supported', '?')}")
