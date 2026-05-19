"""
Experiment runner — orchestrates GWT+IIT and CGWT cycles.

Each cycle:
  1. Φ measured before broadcast (baseline)
  2. All specialized agents process the same stimulus → produce proposals
  3. Selection: competitive (GWT), random, collaborative (CGWT), or hybrid
  4. Broadcast to global workspace
  5. Φ measured after broadcast (phi_proxy for competitive; phi_collaborative for CGWT)
  6. Information geometry metrics computed
  7. Narrator generates a first-person phenomenological report

Modes:
  - "competitive": standard GWT winner-take-all (control)
  - "random": random winner selection (control)
  - "no_broadcast": no broadcast (control)
  - "single_agent": one agent only (control)
  - "collaborative": CGWT coalition broadcast using ConsensusController
  - "hybrid": competitive narrows to top-2, then collaborative merge
"""

import random
import time
from workspace import GlobalWorkspace, AttentionController, CollaborativeWorkspace, ConsensusController
from iit import phi_proxy, phi_collaborative, information_geometry_metric

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
    workspace,  # GlobalWorkspace or CollaborativeWorkspace
    memory,
    model: str,
    mode: str = "competitive",
) -> dict:
    """
    Run one complete experimental cycle (GWT or CGWT depending on mode).

    Args:
        stimulus: The input text that all agents process.
        workspace: GlobalWorkspace (competitive/random/no_broadcast/single_agent)
                   or CollaborativeWorkspace (collaborative/hybrid).
        memory: SemanticMemory instance.
        model: Ollama model name.
        mode: "competitive" | "random" | "no_broadcast" | "single_agent"
              | "collaborative" | "hybrid"

    Returns:
        Dict with full cycle data including mode-appropriate Φ.
    """
    agents = _get_agents(model)
    workspace_context = workspace.get_context()

    # ── Phase 1: Pre-broadcast Φ ───────────────────────────────────────
    phi_before = phi_proxy(workspace.read(), workspace.history)

    # ── Phase 2: All agents process stimulus ──────────────────────────
    proposals = {}
    if mode == "single_agent":
        try:
            proposals["reasoner"] = agents["reasoner"].process(stimulus, workspace_context)
        except Exception as e:
            proposals["reasoner"] = f"[error: {e}]"
    else:
        for name, agent in agents.items():
            if name == "narrator":
                continue
            try:
                proposals[name] = agent.process(stimulus, workspace_context)
            except Exception as e:
                proposals[name] = f"[{name} error: {e}]"

    # ── Phase 3: Selection ─────────────────────────────────────────────
    broadcasted = False
    coalition_names = []
    winner_name = "none"
    winner_content = ""
    attention_score = 0.0

    if mode == "random":
        agent_names = list(proposals.keys())
        winner_name = random.choice(agent_names) if agent_names else "none"
        winner_content = proposals.get(winner_name, "")

    elif mode == "no_broadcast":
        pass  # nothing broadcast

    elif mode in ("collaborative", "hybrid"):
        # CGWT: ConsensusController selects coalition
        ctrl = ConsensusController(memory, coalition_size=2, agreement_threshold=0.25)
        cws = workspace if isinstance(workspace, CollaborativeWorkspace) else workspace

        if mode == "hybrid":
            # First do competitive pass to narrow to top-2 candidates
            comp_ctrl = AttentionController(memory)
            top_name, top_content, top_score = comp_ctrl.select(
                proposals, workspace_context, workspace_for_suppression=workspace
            )
            # Then collaborative among top-2 only
            top2_proposals = {top_name: top_content}
            scored_all = [(comp_ctrl._score(v, workspace_context), k, v)
                          for k, v in proposals.items() if k != top_name]
            if scored_all:
                scored_all.sort(reverse=True)
                runner_name, runner_content = scored_all[0][1], scored_all[0][2]
                top2_proposals[runner_name] = runner_content

            world_model = getattr(cws, 'world_model', None)
            if world_model is None:
                from world_model import WorldModel
                world_model = WorldModel()
            coalition_names, winner_content, attention_score = ctrl.select_coalition(
                top2_proposals, world_model, workspace_context, workspace
            )
        else:
            # Full collaborative: ConsensusController across all proposals
            world_model = getattr(cws, 'world_model', None)
            if world_model is None:
                from world_model import WorldModel
                world_model = WorldModel()
            coalition_names, winner_content, attention_score = ctrl.select_coalition(
                proposals, world_model, workspace_context, workspace
            )

        winner_name = f"coalition:{'+'.join(coalition_names)}" if coalition_names else "none"

        # Broadcast coalition
        if coalition_names and winner_content and isinstance(workspace, CollaborativeWorkspace):
            workspace.collaborative_broadcast(
                coalition_names, winner_content, attention_score, proposals
            )
            broadcasted = True
        elif coalition_names and winner_content:
            workspace.broadcast(winner_name, winner_content, attention_score)
            broadcasted = True

    else:
        # Standard competitive (GWT)
        ctrl = AttentionController(memory)
        winner_name, winner_content, attention_score = ctrl.select(
            proposals, workspace_context, workspace_for_suppression=workspace
        )
        if winner_name != "none" and winner_content:
            workspace.broadcast(winner_name, winner_content, attention_score)
            broadcasted = True

    # ── Phase 4: Standard broadcast for non-collaborative modes ───────
    if mode in ("random", "single_agent") and winner_content:
        workspace.broadcast(winner_name, winner_content, attention_score)
        broadcasted = True

    # ── Phase 5: Post-broadcast Φ (mode-appropriate) ──────────────────
    if mode in ("collaborative", "hybrid") and coalition_names:
        phi_after = phi_collaborative(
            workspace.read(), workspace.history, proposals, coalition_names
        )
    else:
        phi_after = phi_proxy(workspace.read(), workspace.history, proposals=proposals)

    # Update world model for collaborative workspace
    if isinstance(workspace, CollaborativeWorkspace) and mode not in ("collaborative", "hybrid"):
        workspace.world_model.update(stimulus, proposals)

    # ── Phase 6: Information geometry ─────────────────────────────────
    geo_metrics = information_geometry_metric(workspace.history)

    # ── Phase 7: Narrative ─────────────────────────────────────────────
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

    result = {
        "stimulus": stimulus[:200],
        "proposals": {k: v[:200] for k, v in proposals.items()},
        "winner": winner_name,
        "coalition": coalition_names,
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
    }
    if mode == "competitive":
        result["ignition_threshold"] = AttentionController(memory).ignition_threshold
    return result


def run_comparison_experiment(
    stimuli: list[str],
    model: str,
    cycles_per_stimulus: int = 3,
    modes: list[str] = None,
) -> dict[str, list[dict]]:
    """
    Run the same stimuli across all experimental modes for comparison.

    Tests the central CGWT hypothesis:
        Φ_collaborative > Φ_competitive > Φ_random > Φ_no_broadcast

    Returns:
        {mode_name: [results_per_cycle], ...}
    """
    if modes is None:
        modes = ["competitive", "random", "no_broadcast", "collaborative", "hybrid"]

    from memory import SemanticMemory
    from workspace import GlobalWorkspace, CollaborativeWorkspace

    # Modes that need CollaborativeWorkspace
    collaborative_modes = {"collaborative", "hybrid"}
    all_results = {}

    for mode in modes:
        mem = SemanticMemory()
        ws = CollaborativeWorkspace(mem) if mode in collaborative_modes else GlobalWorkspace(mem)
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

    # Hypothesis tests (Mann-Whitney U, non-parametric)
    try:
        from scipy.stats import mannwhitneyu
        tests = {}
        pairs = [
            ("collaborative", "competitive", "H1: Collaborative Φ > Competitive Φ"),
            ("collaborative", "random",      "H2: Collaborative Φ > Random Φ"),
            ("hybrid",        "competitive", "H3: Hybrid Φ > Competitive Φ"),
            ("competitive",   "random",      "H4: Competitive Φ > Random Φ"),
            ("competitive",   "no_broadcast","H5: Broadcast Φ > No-broadcast Φ"),
        ]
        for mode_a, mode_b, label in pairs:
            if mode_a in results and mode_b in results:
                a = [c["phi_delta"] for c in results[mode_a]]
                b = [c["phi_delta"] for c in results[mode_b]]
                stat, pval = mannwhitneyu(a, b, alternative="greater")
                tests[f"{mode_a}_vs_{mode_b}"] = {
                    "label": label,
                    "U_statistic": float(stat),
                    "p_value": round(float(pval), 6),
                    "significant": pval < 0.05,
                    "supported": pval < 0.05,
                }
        analysis["hypothesis_tests"] = tests
    except Exception as e:
        analysis["hypothesis_tests"] = {"error": str(e)}

    return analysis
