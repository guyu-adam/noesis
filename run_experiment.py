import sys, json, time, os
import numpy as np
sys.path.insert(0, '/home/zhang/noesis')

from experiment import run_comparison_experiment, analyze_comparison

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

STIMULI = [
    # Perception & awareness
    "What is the relationship between perception and consciousness?",
    "How does attention shape what we become aware of?",
    "Can a system be conscious without self-awareness?",
    "What distinguishes integrated information from mere computation?",
    # Memory & time
    "How does memory influence the quality of conscious experience?",
    "Is the global workspace a necessary condition for consciousness?",
    "What role does prediction error play in conscious perception?",
    "Can multiple agents jointly produce a unified conscious state?",
    # Coordination & agency
    "How do agents reach consensus when their observations conflict?",
    "What mechanisms allow distributed agents to coordinate without a central authority?",
    "When does coalition formation improve collective decision quality?",
    "How should information be shared to maximise collective intelligence?",
    # Integration & binding
    "What is the binding problem and how might it be solved computationally?",
    "How does information integration differ from information broadcasting?",
    "Under what conditions does multi-agent communication increase system coherence?",
    "What role does redundancy play in robust collective representations?",
    # Novelty & surprise
    "How should an agent weight novel information versus prior consensus?",
    "What distinguishes useful surprise from disruptive noise in a multi-agent system?",
    "Can competition and collaboration coexist in a broadcast architecture?",
    "How does the structure of a communication channel shape the information that flows through it?",
]

MODEL = "qwen3.5:4b"
CYCLES = 5
MODES = ["competitive", "random", "no_broadcast", "collaborative", "hybrid"]

total = len(STIMULI) * CYCLES * len(MODES)
print(f"[START] {time.strftime('%H:%M:%S')} Running {len(STIMULI)} stimuli x {CYCLES} cycles x {len(MODES)} modes", flush=True)
print(f"Total cycles: {total}", flush=True)

results = run_comparison_experiment(STIMULI, MODEL, cycles_per_stimulus=CYCLES, modes=MODES)
print(f"[EXPERIMENTS DONE] {time.strftime('%H:%M:%S')}", flush=True)

analysis = analyze_comparison(results)

output = {"results_summary": {}, "analysis": analysis, "metadata": {
    "model": MODEL,
    "n_stimuli": len(STIMULI),
    "cycles_per_stimulus": CYCLES,
    "modes": MODES,
    "total_cycles": total,
    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
}}
for mode, cycles in results.items():
    phi_deltas = [c["phi_delta"] for c in cycles]
    phi_afters = [c["phi_after"] for c in cycles]
    phi_befores = [c["phi_before"] for c in cycles]
    output["results_summary"][mode] = {
        "mean_phi_after":  round(float(np.mean(phi_afters)), 6),
        "std_phi_after":   round(float(np.std(phi_afters)), 6),
        "mean_phi_before": round(float(np.mean(phi_befores)), 6),
        "mean_phi_delta":  round(float(np.mean(phi_deltas)), 6),
        "std_phi_delta":   round(float(np.std(phi_deltas)), 6),
        "max_phi_delta":   round(float(np.max(phi_deltas)), 6),
        "n_cycles":        len(cycles),
        "broadcast_rate":  round(sum(1 for c in cycles if c.get("broadcasted")) / len(cycles), 4),
    }

# Save to timestamped file
out_dir = "/home/zhang/noesis/experiments"
os.makedirs(out_dir, exist_ok=True)
ts = time.strftime('%Y%m%d_%H%M%S')
out_path = f"{out_dir}/results_{ts}.json"

with open(out_path, "w") as f:
    json.dump(output, f, indent=2, cls=NpEncoder)

# Also save raw cycles for ablation analysis
raw_path = f"{out_dir}/raw_{ts}.jsonl"
with open(raw_path, "w") as f:
    for mode, cycles in results.items():
        for c in cycles:
            f.write(json.dumps(c, cls=NpEncoder) + "\n")

print(f"[SAVED] {out_path}", flush=True)
print("[DONE]", time.strftime('%H:%M:%S'), flush=True)
print(json.dumps(output["results_summary"], indent=2))
print("Hypothesis tests:")
for k, v in analysis.get("hypothesis_tests", {}).items():
    if isinstance(v, dict):
        print(f"  {k}: p={v.get('p_value','?')}, supported={v.get('supported','?')}")
