"""
Experiment runner — orchestrates one GWT+IIT cycle.

Each cycle:
  1. Φ measured before broadcast (baseline)
  2. All specialized agents process the same stimulus → produce proposals
  3. Attention controller selects a winner (competitive or random baseline)
  4. Winner broadcast to global workspace if salience exceeds ignition threshold
  5. Φ measured after broadcast
  6. Information geometry metrics computed
  7. Narrator generates a first-person phenomenological report

Baseline modes (for hypothesis testing):
  - "competitive": standard GWT competition (the experimental condition)
  - "random": winner selected randomly (control — tests if competition matters)
  - "no_broadcast": no broadcast at all (control — tests if broadcast creates Φ)
  - "single_agent": only one agent (control — tests if multi-agent matters)
"""

import random
import time
from workspace import GlobalWorkspace, AttentionController
from iit import phi_proxy, information_geometry_metric

_agent_cache: dict = {}


def _get_agents(model: str):
    """Initialize or retrieve agent instances for a given model."""
    if model not in _agent_cache:
        from agents.perceptor import Perceptor
        from agents.reasoner import Reasoner
        from agents.evaluator import Evaluator
        from agents.narrator import Narrator

        _agent_cache[model] = {
            "perceptor": Perceptor(model),
            "reasoner": Reasoner(model),
            "evaluator": Evaluator(model),
            "narrator": Narrator(model),
        }
    return _agent_cache[model]


def run_cycle(
    stimulus: str,
    workspace: GlobalWorkspace,
    memory,  # SemanticMemory
    model: str,
    mode: str = "competitive",
) -> dict:
    """
    Run one complete GWT+IIT experimental cycle.

    Args:
        stimulus: The input text that all agents process.
        workspace: The global workspace instance.
        memory: Semantic memory instance.
        model: Ollama model name.
        mode: "competitive" | "random" | "no_broadcast" | "single_agent"

    Returns:
        Dict with full cycle data.
    """
    agents = _get_agents(model)
    controller = AttentionController(memory)
    workspace_context = workspace.get_context()

    # ── Phase 1: Pre-broadcast Φ measurement ───────────────────────────
    phi_before = phi_proxy(workspace.read(), workspace.history)

    # ── Phase 2: Agents process stimulus in parallel ───────────────────
    proposals = {}

    if mode == "single_agent":
        # Only use one agent — tests if multi-agent competition is needed
        try:
            proposals["reasoner"] = agents["reasoner"].process(stimulus, workspace_context)
        except Exception as e:
            proposals["reasoner"] = f"[reasoner error: {e}]"
    else:
        for name, agent in agents.items():
            if name == "narrator":
                continue
            try:
                proposals[name] = agent.process(stimulus, workspace_context)
            except Exception as e:
                proposals[name] = f"[{name} error: {e}]"

    # ── Phase 3: Attention competition (or baseline selection) ──────────
    if mode == "random":
        # Random selection baseline — tests if competition matters
        agent_names = list(proposals.keys())
        winner_name = random.choice(agent_names) if agent_names else "none"
        winner_content = proposals.get(winner_name, "")
        attention_score = 0.0
    elif mode == "no_broadcast":
        # No broadcast at all — tests if broadcast creates Φ
        winner_name = "none"
        winner_content = ""
        attention_score = 0.0
    else:
        # Competitive selection (the experimental condition)
        winner_name, winner_content, attention_score = controller.select(
            proposals, workspace_context, workspace_for_suppression=workspace
        )

    # ── Phase 4: Global broadcast ──────────────────────────────────────
    broadcasted = False
    if mode != "no_broadcast" and winner_name != "none" and winner_content:
        workspace.broadcast(winner_name, winner_content, attention_score)
        broadcasted = True

    # ── Phase 5: Post-broadcast Φ measurement ──────────────────────────
    phi_after = phi_proxy(
        workspace.read(), workspace.history, proposals=proposals
    )

    # ── Phase 6: Information geometry metrics ─────────────────────────
    geo_metrics = information_geometry_metric(workspace.history)

    # ── Phase 7: Narrative generation ──────────────────────────────────
    try:
        narrative = agents["narrator"].generate(
            stimulus=stimulus,
            broadcast_history=workspace.history,
            phi_before=phi_before,
            phi_after=phi_after,
            winner=winner_name,
        )
    except Exception:
        narrative = "[narrator unavailable]"

    return {
        "stimulus": stimulus[:200],
        "proposals": {k: v[:200] for k, v in proposals.items()},
        "winner": winner_name,
        "attention_score": attention_score,
        "broadcast": winner_content[:400] if broadcasted else "(no broadcast)",
        "broadcasted": broadcasted,
        "phi_before": phi_before,
        "phi_after": phi_after,
        "phi_delta": round(phi_after - phi_before, 6),
        "fisher_trace": geo_metrics["fisher_trace"],
        "complexity": geo_metrics["complexity"],
        "narrative": narrative,
        "mode": mode,
        "ignition_threshold": controller.ignition_threshold,
    }


def run_comparison_experiment(
    stimuli: list[str],
    model: str,
    cycles_per_stimulus: int = 3,
    modes: list[str] = None,
) -> dict[str, list[dict]]:
    """
    Run the same stimuli across all experimental modes for comparison.

    This is the definitive experiment — it tests whether competitive GWT
    mechanisms produce higher Φ than baselines (random, no_broadcast, single_agent).

    Returns:
        {mode_name: [results_per_cycle], ...}
    """
    if modes is None:
        modes = ["competitive", "random", "no_broadcast"]

    from memory import SemanticMemory
    from workspace import GlobalWorkspace

    all_results = {}

    for mode in modes:
        mem = SemanticMemory()
        ws = GlobalWorkspace(mem)
        mode_results = []

        for stim in stimuli:
            for _ in range(cycles_per_stimulus):
                result = run_cycle(stim, ws, mem, model, mode=mode)
                mode_results.append(result)

        all_results[mode] = mode_results
        mem.clear()

    return all_results


def analyze_comparison(results: dict[str, list[dict]]) -> dict:
    """
    Statistical comparison of Φ across experimental modes.

    Tests the three central hypotheses:
      1. Φ_competitive > Φ_random  (competition matters)
      2. Φ_competitive > Φ_no_broadcast  (broadcast creates integration)
      3. Φ_competitive > Φ_single_agent  (multi-agent matters)
    """
    import numpy as np

    analysis = {}
    for mode, cycles in results.items():
        phi_deltas = [c["phi_delta"] for c in cycles]
        phi_afters = [c["phi_after"] for c in cycles]
        analysis[mode] = {
            "mean_phi_delta": round(float(np.mean(phi_deltas)), 6),
            "max_phi_delta": round(float(np.max(phi_deltas)), 6),
            "mean_phi_after": round(float(np.mean(phi_afters)), 6),
            "n_cycles": len(cycles),
            "broadcast_rate": sum(1 for c in cycles if c.get("broadcasted")) / len(cycles),
        }

    # Hypothesis tests
    if "competitive" in analysis and "random" in analysis:
        comp_deltas = [c["phi_delta"] for c in results["competitive"]]
        rand_deltas = [c["phi_delta"] for c in results["random"]]
        # Mann-Whitney U test (non-parametric)
        try:
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(comp_deltas, rand_deltas, alternative="greater")
            analysis["hypothesis_tests"] = {
                "competitive_vs_random": {
                    "U_statistic": float(stat),
                    "p_value": round(float(pval), 6),
                    "significant": pval < 0.05,
                    "interpretation": (
                        "Competition significantly increases Φ" if pval < 0.05
                        else "No significant Φ advantage from competition"
                    ),
                }
            }
        except Exception:
            analysis["hypothesis_tests"] = {"error": "scipy not available"}

    return analysis
