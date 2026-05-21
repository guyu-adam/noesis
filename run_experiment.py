"""
Neural experiment runner — batch comparison across 6 modes using RNN processors.

This is the MAIN BRANCH experiment script. It uses:
  - 5 specialized RNN processors (Perceptor, Reasoner, Evaluator, Integrator, Predictor)
  - Real causal TPM-based Φ computation (neural_iit.py)
  - CuPy GPU acceleration for RTX 5060 Ti 16GB

Usage:
    python run_experiment.py

Environment variables:
    NOESIS_N_NEURONS   — override neuron count per processor (default: 512 on 16GB)
    NOESIS_N_INPUT     — stimulus vector dimension (default: 32)
    NOESIS_N_CYCLES    — cycles per stimulus per mode (default: 5)
    NOESIS_N_UNROLL    — RNN unroll steps (default: 25)
    NOESIS_BATCH_SIZE  — parallel cycles (default: 8 on 16GB)
"""

import sys, json, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import run_neural_comparison, analyze_neural_comparison
from agents.neural_base import encode_stimulus
from config import get_config


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# Stimuli
# ══════════════════════════════════════════════════════════════════════════════

STIMULI = [
    # Consciousness & cognition (8 — replicates prior work)
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
MODES = ["competitive", "random", "no_broadcast", "collaborative", "hybrid",
         "adaptive", "single_processor", "single_self_broadcast", "homogeneous_competitive"]

cfg = get_config()

# ══════════════════════════════════════════════════════════════════════════════
# Encode stimuli → neural vectors
# ══════════════════════════════════════════════════════════════════════════════

stimuli_vecs = [encode_stimulus(s, dim=cfg.n_input) for s in STIMULI]

total = len(STIMULI) * CYCLES * len(MODES)
print(f"[START] {time.strftime('%H:%M:%S')}  {len(STIMULI)} stimuli x {CYCLES} cycles x {len(MODES)} modes", flush=True)
print(f"Total cycles: {total}", flush=True)
print(f"Neurons: {cfg.n_neurons}/processor x {cfg.n_processors} processors = {cfg.total_neurons} total", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

results = run_neural_comparison(
    stimuli_vecs,
    modes=MODES,
    cycles_per_stimulus=CYCLES,
    n_neurons=cfg.n_neurons,
    n_input=cfg.n_input,
)
print(f"[EXPERIMENTS DONE] {time.strftime('%H:%M:%S')}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Analysis & output
# ══════════════════════════════════════════════════════════════════════════════

analysis = analyze_neural_comparison(results)

# Processor differentiation validation
_processor_similarity = {}
try:
    from experiment import _get_neural_processors as _gnp
    _procs = _gnp(cfg.n_neurons, cfg.n_input)
    _names = ["perceptor", "reasoner", "evaluator", "integrator", "predictor"]
    for i, ni in enumerate(_names):
        for j, nj in enumerate(_names):
            if i < j and hasattr(_procs[ni], 'read_proposal') and hasattr(_procs[nj], 'read_proposal'):
                vi = _procs[ni].read_proposal()
                vj = _procs[nj].read_proposal()
                sim = float(np.dot(vi.flatten(), vj.flatten()) /
                           max(np.linalg.norm(vi) * np.linalg.norm(vj), 1e-10))
                _processor_similarity[f"{ni}_vs_{nj}"] = round(sim, 4)
except Exception as e:
    _processor_similarity = {"error": f"similarity_computation_failed: {e}"}

# Experiment config lock
import subprocess
_pip_freeze = ""
try:
    _pip_freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=10).stdout.strip()[:5000]
except Exception:
    _pip_freeze = "pip_freeze_unavailable"

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
        "merge_strategy": "attention_weighted",
        "n_stimuli": len(STIMULI),
        "cycles_per_stimulus": CYCLES,
        "modes": MODES,
        "total_cycles": total,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "processor_pairwise_similarity": _processor_similarity,
        "dependencies": _pip_freeze[:3000],
    },
}

for mode, cycles in results.items():
    phi_deltas = [c["phi_delta"] for c in cycles]
    phi_afters = [c["phi_after"] for c in cycles]
    phi_befores = [c["phi_before"] for c in cycles]
    complexities = [c.get("complexity", 0) for c in cycles]
    phi_withins = [c.get("phi_decomposed", {}).get("phi_within", 0) for c in cycles]
    phi_betweens = [c.get("phi_decomposed", {}).get("phi_between", 0) for c in cycles]
    merge_rets = [c.get("merge_metadata", {}).get("differentiation_retention", 0) for c in cycles]
    output["results_summary"][mode] = {
        "mean_phi_after":  round(float(np.mean(phi_afters)), 6),
        "std_phi_after":   round(float(np.std(phi_afters)), 6),
        "mean_phi_before": round(float(np.mean(phi_befores)), 6),
        "mean_phi_delta":  round(float(np.mean(phi_deltas)), 6),
        "std_phi_delta":   round(float(np.std(phi_deltas)), 6),
        "max_phi_delta":   round(float(np.max(phi_deltas)), 6),
        "mean_complexity": round(float(np.mean(complexities)), 6),
        "mean_phi_within": round(float(np.mean(phi_withins)), 6),
        "mean_phi_between": round(float(np.mean(phi_betweens)), 6),
        "mean_merge_diff_retention": round(float(np.mean(merge_rets)), 6),
        "attention_weight_mean": (round(float(np.mean(
            [c.get("merge_metadata", {}).get("attention_weight_mean", 0)
             for c in cycles if c.get("merge_metadata", {}).get("strategy") == "attention"]
        )), 4) if any(c.get("merge_metadata", {}).get("strategy") == "attention" for c in cycles) else None),
        "attention_weight_std": (round(float(np.mean(
            [c.get("merge_metadata", {}).get("attention_weight_std", 0)
             for c in cycles if c.get("merge_metadata", {}).get("strategy") == "attention"]
        )), 4) if any(c.get("merge_metadata", {}).get("strategy") == "attention" for c in cycles) else None),
        "n_cycles":        len(cycles),
        "broadcast_rate":  round(sum(1 for c in cycles if c.get("broadcasted")) / max(len(cycles), 1), 4),
    }

# Save to timestamped file
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
os.makedirs(out_dir, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = os.path.join(out_dir, f"neural_results_{ts}.json")

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, cls=NpEncoder, ensure_ascii=False)

# Save raw cycles for ablation analysis
raw_path = os.path.join(out_dir, f"neural_raw_{ts}.jsonl")
with open(raw_path, "w", encoding="utf-8") as f:
    for mode, cycles in results.items():
        for c in cycles:
            f.write(json.dumps(c, cls=NpEncoder, ensure_ascii=False) + "\n")

# Save experiment config lock (full reproducibility)
config_path = os.path.join(out_dir, f"experiment_config_{ts}.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(output["metadata"], f, indent=2, cls=NpEncoder, ensure_ascii=False)

print(f"[SAVED] {out_path}", flush=True)
print(f"[SAVED] {raw_path}", flush=True)
print(f"[DONE] {time.strftime('%H:%M:%S')}", flush=True)
print(json.dumps(output["results_summary"], indent=2))
print("Hypothesis tests (Mann-Whitney U on phi_delta):")
for k, v in analysis.get("hypothesis_tests", {}).items():
    if isinstance(v, dict):
        print(f"  {k}: p={v.get('p_value','?')}, d={v.get('cohens_d','?')}, "
              f"supported={v.get('supported','?')}, n_for_80pct={v.get('n_for_80pct_power','?')}")
ancova = analysis.get("ancova", {})
if isinstance(ancova, dict) and "F_statistic" in ancova:
    print(f"\nANCOVA: F({ancova.get('df_between',0)},{ancova.get('df_within',0)})={ancova.get('F_statistic','?')}, "
          f"p={ancova.get('p_value','?')}, significant={ancova.get('significant','?')}")
    print("Adjusted means:", ancova.get("adjusted_means", {}))
cal = analysis.get("phi_calibration", {})
if isinstance(cal, dict) and "random_system" in cal:
    print(f"\nΦ calibration: random={cal.get('random_system',0)}, deterministic={cal.get('deterministic_cycle',0)}, noise={cal.get('noise_driven',0)}")
sim = output["metadata"].get("processor_pairwise_similarity", {})
if sim and "error" not in sim:
    mean_sim = float(np.mean(list(sim.values()))) if sim else 0
    print(f"Processor pairwise similarity: mean={mean_sim:.4f}, all={sim}")
# Attention weight distribution (supplementary per reviewer request)
for mode, summary in output["results_summary"].items():
    awm = summary.get("attention_weight_mean")
    aws = summary.get("attention_weight_std")
    if awm is not None:
        print(f"  {mode}: attention_weights mean={awm}, std={aws} (effective_n≈{1.0/max(aws**2*5, 1e-4):.1f} of 5)")
print("\nClustered analysis (20 stimulus-level means per mode):")
clustered = analysis.get("clustered_analysis", {})
for k, v in clustered.get("hypothesis_tests", {}).items():
    if isinstance(v, dict) and "p_value" in v:
        print(f"  {k}: p={v['p_value']}, d={v.get('cohens_d','?')}")
if "clustered_summary" in clustered:
    for mode, s in clustered["clustered_summary"].items():
        print(f"  {mode}: {s['mean_phi_after']} +- {s['se_phi_after']} (SE)")
