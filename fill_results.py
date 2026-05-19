"""
fill_results.py — parse experiment JSON and generate filled LaTeX table snippets.

Usage (after experiments complete):
    python3 fill_results.py /path/to/results_YYYYMMDD_HHMMSS.json
"""
import json, sys, re

def fill(results_path: str, paper_path: str = None):
    with open(results_path) as f:
        data = json.load(f)

    summary = data.get("results_summary", {})
    tests = data.get("analysis", {}).get("hypothesis_tests", {})

    mode_order = ["collaborative", "hybrid", "competitive", "random", "no_broadcast"]
    mode_labels = {
        "collaborative": "Collaborative",
        "hybrid": "Hybrid",
        "competitive": "Competitive",
        "random": "Random",
        "no_broadcast": "No-broadcast",
    }

    print("=" * 60)
    print("TABLE 1: Phi comparison")
    print("=" * 60)
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Mode} & \textbf{Mean $\hat{\Phi}_{\text{after}}$} & \textbf{SD} &"
          r" \textbf{Mean $\Delta\hat{\Phi}$} & \textbf{SD} & \textbf{$n$} \\")
    print(r"\midrule")
    best_phi = max(summary[m]["mean_phi_after"] for m in mode_order if m in summary)
    for mode in mode_order:
        if mode not in summary:
            continue
        s = summary[mode]
        phi_str = f"{s['mean_phi_after']:.4f}"
        if s["mean_phi_after"] == best_phi:
            phi_str = r"\textbf{" + phi_str + "}"
        print(f"{mode_labels[mode]} & {phi_str} & {s.get('std_phi_after', 0):.4f} & "
              f"{s['mean_phi_delta']:.4f} & {s.get('std_phi_delta', 0):.4f} & "
              f"{s['n_cycles']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print()
    print("=" * 60)
    print("TABLE 2: Hypothesis tests")
    print("=" * 60)
    pair_labels = {
        "collaborative_vs_competitive": ("Collaborative vs.\\ Competitive", "H1"),
        "collaborative_vs_random":      ("Collaborative vs.\\ Random", "H2"),
        "hybrid_vs_competitive":        ("Hybrid vs.\\ Competitive", "H3"),
        "competitive_vs_random":        ("Competitive vs.\\ Random", "H4"),
        "competitive_vs_no_broadcast":  ("Competitive vs.\\ No-broadcast", "H5"),
    }
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"\textbf{Comparison} & \textbf{Hyp.} & \textbf{$U$} & \textbf{$p$} & \textbf{Supported} \\")
    print(r"\midrule")
    for key, (label, hyp) in pair_labels.items():
        if key not in tests:
            continue
        t = tests[key]
        p = t["p_value"]
        U = t.get("U_statistic", "?")
        sup = r"\textbf{Yes}" if t.get("supported") else "No"
        p_str = f"$<$0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{label} & {hyp} & {U:.0f} & {p_str} & {sup} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print()
    print("=" * 60)
    print("ABSTRACT snippet (update numbers):")
    print("=" * 60)
    best_mode = max(mode_order, key=lambda m: summary.get(m, {}).get("mean_phi_after", 0))
    competitive = summary.get("competitive", {})
    random_ = summary.get("random", {})
    H4 = tests.get("competitive_vs_random", {})
    H1 = tests.get("collaborative_vs_competitive", {})
    print(f"  collaborative mean Phi_after = {summary.get('collaborative', {}).get('mean_phi_after', 'TBD')}")
    print(f"  competitive mean Phi_after   = {competitive.get('mean_phi_after', 'TBD')}")
    print(f"  random mean Phi_after        = {random_.get('mean_phi_after', 'TBD')}")
    print(f"  H1 (collab > comp): p={H1.get('p_value', 'TBD')}, supported={H1.get('supported', 'TBD')}")
    print(f"  H4 (comp > rand):   p={H4.get('p_value', 'TBD')}, supported={H4.get('supported', 'TBD')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fill_results.py results.json")
        sys.exit(1)
    fill(sys.argv[1])
