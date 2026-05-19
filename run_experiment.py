import sys, json, time
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
    "What is the relationship between perception and consciousness?",
    "How does attention shape what we become aware of?",
    "Can a system be conscious without self-awareness?",
    "What distinguishes integrated information from mere computation?",
    "How does memory influence the quality of conscious experience?",
    "Is the global workspace a necessary condition for consciousness?",
    "What role does prediction error play in conscious perception?",
    "Can multiple processors jointly produce a unified conscious state?",
]

MODEL = "qwen3.5:4b"
CYCLES = 3
MODES = ["competitive", "random", "no_broadcast", "collaborative", "hybrid"]

print(f"[START] {time.strftime('%H:%M:%S')} Running {len(STIMULI)} stimuli x {CYCLES} cycles x {len(MODES)} modes", flush=True)
print(f"Total cycles: {len(STIMULI)*CYCLES*len(MODES)}", flush=True)

results = run_comparison_experiment(STIMULI, MODEL, cycles_per_stimulus=CYCLES, modes=MODES)
print(f"[EXPERIMENTS DONE] {time.strftime('%H:%M:%S')}", flush=True)

analysis = analyze_comparison(results)

output = {"results_summary": {}, "analysis": analysis}
for mode, cycles in results.items():
    phi_deltas = [c["phi_delta"] for c in cycles]
    phi_afters = [c["phi_after"] for c in cycles]
    output["results_summary"][mode] = {
        "mean_phi_after": round(float(np.mean(phi_afters)), 6),
        "mean_phi_delta": round(float(np.mean(phi_deltas)), 6),
        "max_phi_delta":  round(float(np.max(phi_deltas)), 6),
        "n_cycles": len(cycles),
    }

with open("/home/zhang/noesis/experiment_results.json", "w") as f:
    json.dump(output, f, indent=2, cls=NpEncoder)

print("[DONE]", time.strftime('%H:%M:%S'), flush=True)
print(json.dumps(output["results_summary"], indent=2))
print("Hypothesis tests:")
for k, v in analysis.get("hypothesis_tests", {}).items():
    if isinstance(v, dict):
        print(f"  {k}: p={v.get('p_value','?')}, supported={v.get('supported','?')}")
