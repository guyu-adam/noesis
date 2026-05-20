"""
Experiment runner — orchestrates GWT+IIT and CGWT cycles.

Each cycle:
  1. Φ measured before broadcast (baseline)
  2. All specialized processors process the same stimulus → produce proposals
  3. Selection: competitive (GWT), random, collaborative (CGWT), or hybrid
  4. Broadcast to global workspace
  5. Φ measured after broadcast (phi_proxy for competitive; phi_collaborative for CGWT)
  6. Information geometry metrics computed
  7. Narrator generates a first-person phenomenological report

Modes:
  - "competitive": standard GWT winner-take-all (control)
  - "random": random winner selection (control)
  - "no_broadcast": no broadcast (control)
  - "single_agent": one processor only (control)
  - "collaborative": CGWT coalition broadcast using ConsensusController
  - "hybrid": competitive narrows to top-2, then collaborative merge
"""

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from workspace import GlobalWorkspace, AttentionController, CollaborativeWorkspace, ConsensusController
from iit import phi_proxy, phi_collaborative, information_geometry_metric

_agent_cache: dict = {}
EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def _save_cycle_result(result: dict, mode: str, cycle_id: int):
    """Persist a single cycle result to experiments/<date>/<mode>.jsonl."""
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        dir_path = EXPERIMENTS_DIR / date_str
        dir_path.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            **result,
        }
        # Convert numpy values to native types for JSON
        for key, val in record.items():
            if hasattr(val, 'tolist'):
                record[key] = val.tolist() if hasattr(val, 'tolist') else str(val)

        filepath = dir_path / f"{mode}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass  # never let persistence failure crash the experiment


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

        world_model = workspace.world_model

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

            coalition_names, winner_content, attention_score = ctrl.select_coalition(
                top2_proposals, world_model, workspace_context, workspace
            )
        else:
            # Full collaborative: ConsensusController across all proposals
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

    _save_cycle_result(result, mode, workspace._cycle_count)
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


# ══════════════════════════════════════════════════════════════════════════════
# Neural experiment runner — for main branch (RNN processors, real causal Φ)
# ══════════════════════════════════════════════════════════════════════════════

_neural_processor_cache: dict = {}


def _get_neural_processors(n_neurons: int = None, n_input: int = None):
    """Initialize or retrieve neural processor instances (5 processor types)."""
    if "perceptor" not in _neural_processor_cache:
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))

        from agents.neural_agents import (
            NeuralPerceptor, NeuralReasoner, NeuralEvaluator,
            NeuralIntegrator, NeuralPredictor, NeuralNarrator,
        )

        _neural_processor_cache["perceptor"] = NeuralPerceptor(n_neurons, n_input, seed=100)
        _neural_processor_cache["reasoner"] = NeuralReasoner(n_neurons, n_input, seed=200)
        _neural_processor_cache["evaluator"] = NeuralEvaluator(n_neurons, n_input, seed=300)
        _neural_processor_cache["integrator"] = NeuralIntegrator(n_neurons, n_input, seed=400)
        _neural_processor_cache["predictor"] = NeuralPredictor(n_neurons, n_input, seed=500)
        _neural_processor_cache["narrator"] = NeuralNarrator()
        _neural_processor_cache["n_neurons"] = n_neurons
        _neural_processor_cache["n_input"] = n_input
    return _neural_processor_cache


def run_neural_cycle(
    stimulus_vec: "np.ndarray",
    workspace,     # GlobalWorkspace or CollaborativeWorkspace
    memory,
    mode: str = "competitive",
    n_neurons: int = 32,
    n_input: int = 16,
) -> dict:
    """
    Run one experimental cycle using neural processors (small RNNs), not LLMs.

    This is the MAIN BRANCH experiment runner. Key differences from run_cycle():

      - Processors are small RNNs with real recurrent causal structure
      - Proposals are activation vectors (numpy arrays), not text strings
      - Φ is computed from neural activation TPMs (neural_iit.py)
      - The "broadcast" is an activation pattern injected into all processors

    The experimental protocol (GWT/IIT cycle structure) is identical to the
    LLM version — only the processor implementation differs.

    Args:
        stimulus_vec: Shape (n_input,). Encoded stimulus vector.
        workspace: GlobalWorkspace or CollaborativeWorkspace.
        memory: SemanticMemory instance.
        mode: "competitive" | "random" | "no_broadcast" | "single_agent" | "single_processor"
              | "collaborative" | "hybrid"
        n_neurons: Neurons per processor (default 32).
        n_input: Stimulus vector dimension (default 16).

    Returns:
        Dict with full cycle data including neural Φ.
    """
    import numpy as np
    from neural_iit import (
        neural_phi, neural_phi_approx, phi_sensitivity,
        neural_information_geometry, cluster_activation_states,
    )

    processors = _get_neural_processors(n_neurons, n_input)
    workspace_context_vec = workspace.read_vec() if hasattr(workspace, 'read_vec') else np.zeros(n_neurons)

    # ── Phase 1: Pre-broadcast Φ ───────────────────────────────────────
    if hasattr(workspace, 'read_vec'):
        phi_before = neural_phi(
            workspace.read_vec(),
            {k: v.read_proposal() for k, v in processors.items()
             if hasattr(v, 'read_proposal')},
            workspace.history if hasattr(workspace, 'history') else [],
        )
    else:
        phi_before = 0.0

    # ── Phase 2: Neural processors process stimulus ────────────────────
    proposals = {}
    processor_names = ["perceptor", "reasoner", "evaluator", "integrator", "predictor"]

    if mode in ("single_agent", "single_processor"):
        try:
            proposals["reasoner"] = processors["reasoner"].process(
                stimulus_vec, workspace_context_vec
            )
        except Exception:
            proposals["reasoner"] = np.zeros(n_neurons)
    else:
        for name in processor_names:
            try:
                proposals[name] = processors[name].process(
                    stimulus_vec, workspace_context_vec
                )
            except Exception:
                proposals[name] = np.zeros(n_neurons)

    # ── Phase 3: Selection (reuses same AttentionController/ConsensusController) ──
    broadcasted = False
    coalition_names = []
    winner_name = "none"
    winner_vec = np.zeros(n_neurons)
    attention_score = 0.0

    # Convert neural proposals to text summaries for the text-based controllers
    # (AttentionController and ConsensusController operate on text)
    text_proposals = {}
    for name, vec in proposals.items():
        text_proposals[name] = _neural_vec_to_text(vec, name)

    if mode == "random":
        processor_names_shuffled = list(text_proposals.keys())
        winner_name = random.choice(processor_names_shuffled) if processor_names_shuffled else "none"
        winner_vec = proposals.get(winner_name, np.zeros(n_neurons))

    elif mode == "no_broadcast":
        pass

    elif mode in ("collaborative", "hybrid"):
        ctrl = ConsensusController(memory, coalition_size=2, agreement_threshold=0.25)
        cws = workspace if isinstance(workspace, CollaborativeWorkspace) else workspace

        world_model = workspace.world_model

        if mode == "hybrid":
            comp_ctrl = AttentionController(memory)
            top_name, top_content, top_score = comp_ctrl.select(
                text_proposals, "", workspace_for_suppression=workspace
            )
            top2 = {top_name: top_content}
            scored_all = [(comp_ctrl._score(v, ""), k, v)
                          for k, v in text_proposals.items() if k != top_name]
            if scored_all:
                scored_all.sort(reverse=True)
                top2[scored_all[0][1]] = scored_all[0][2]
            coalition_names, merged_text, attention_score = ctrl.select_coalition(
                top2, world_model, "", workspace
            )
        else:
            coalition_names, merged_text, attention_score = ctrl.select_coalition(
                text_proposals, world_model, "", workspace
            )

        winner_name = f"coalition:{'+'.join(coalition_names)}" if coalition_names else "none"
        # Merge coalition activation vectors
        if coalition_names:
            coalition_vecs = [proposals[n] for n in coalition_names if n in proposals]
            winner_vec = np.mean(coalition_vecs, axis=0) if coalition_vecs else np.zeros(n_neurons)

        if coalition_names and isinstance(workspace, CollaborativeWorkspace):
            workspace.collaborative_broadcast(
                coalition_names,
                _neural_vec_to_text(winner_vec, "coalition"),
                attention_score,
                text_proposals,
            )
            broadcasted = True
        elif coalition_names:
            workspace.broadcast(winner_name, _neural_vec_to_text(winner_vec, winner_name), attention_score)
            broadcasted = True

    else:
        # Competitive (GWT)
        ctrl = AttentionController(memory)
        winner_name, winner_content, attention_score = ctrl.select(
            text_proposals, "", workspace_for_suppression=workspace
        )
        winner_vec = proposals.get(winner_name, np.zeros(n_neurons))
        if winner_name != "none":
            workspace.broadcast(winner_name, winner_content, attention_score)
            broadcasted = True

    # Standard broadcast for non-collaborative modes
    if mode in ("random", "single_agent", "single_processor") and winner_name != "none":
        workspace.broadcast(winner_name, _neural_vec_to_text(winner_vec, winner_name), attention_score)
        broadcasted = True

    # Store activation vector in workspace history (for neural Φ)
    if hasattr(workspace, 'history') and workspace.history:
        workspace.history[-1]["content_vec"] = winner_vec

    # ── Phase 4: Post-broadcast neural Φ ───────────────────────────────
    phi_after = neural_phi(
        winner_vec,
        {k: v for k, v in proposals.items()},
        workspace.history if hasattr(workspace, 'history') else [],
    )

    # ── Phase 5: Information geometry ─────────────────────────────────
    if hasattr(workspace, 'history'):
        hist_activations = [
            h.get("content_vec", np.zeros(n_neurons))
            for h in workspace.history[-10:]
        ]
    else:
        hist_activations = [winner_vec]
    geo_metrics = neural_information_geometry(hist_activations)

    # ── Phase 6: Neural narrative ─────────────────────────────────────
    try:
        narrative = processors["narrator"].generate(
            winner_vec, phi_before, phi_after, winner_name, broadcasted
        )
        processors["narrator"].cycle_count += 1
    except Exception:
        narrative = "[narrator unavailable]"

    # ── Phase 7: Complexity (differentiation) ─────────────────────────
    processor_stds = [float(np.std(p)) for p in proposals.values() if len(p) > 0]
    complexity = float(np.mean(processor_stds)) if processor_stds else 0.0

    # ── Phase 8: Φ weight sensitivity ─────────────────────────────────
    try:
        sensitivity = phi_sensitivity(
            winner_vec, {k: v for k, v in proposals.items()},
            workspace.history if hasattr(workspace, 'history') else [],
        )
    except Exception:
        sensitivity = {"error": "sensitivity_analysis_failed"}

    result = {
        "stimulus_vec_shape": list(stimulus_vec.shape),
        "proposals": {k: _neural_vec_summary(v) for k, v in proposals.items()},
        "winner": winner_name,
        "coalition": coalition_names,
        "attention_score": attention_score,
        "broadcast": _neural_vec_to_text(winner_vec, winner_name) if broadcasted else "(no broadcast)",
        "broadcasted": broadcasted,
        "phi_before": phi_before,
        "phi_after": phi_after,
        "phi_delta": round(phi_after - phi_before, 6),
        "phi_sensitivity": sensitivity,
        "fisher_trace": geo_metrics["fisher_trace"],
        "complexity": round(complexity, 6),
        "narrative": narrative,
        "mode": mode,
        "processor_type": "neural_rnn",
    }

    _save_cycle_result(result, mode, workspace._cycle_count)
    return result


def run_neural_comparison(
    stimuli_vecs: list["np.ndarray"],
    modes: list[str] = None,
    cycles_per_stimulus: int = 5,
    n_neurons: int = None,
    n_input: int = None,
) -> dict[str, list[dict]]:
    """
    Batch neural experiment: run same stimuli across all modes for comparison.

    This is the neural counterpart to run_comparison_experiment() — instead of
    LLM agents, it uses small RNN processors with real causal structure and
    computes Φ from neural activation TPMs via neural_iit.py.

    Tests the central CGWT hypothesis:
        Φ_collaborative > Φ_competitive > Φ_random > Φ_no_broadcast

    Args:
        stimuli_vecs: List of encoded stimulus vectors, each shape (n_input,).
        modes: Modes to compare. Default: all 6.
        cycles_per_stimulus: Repetitions per stimulus per mode.
        n_neurons: Neurons per processor (reads NOESIS_N_NEURONS or config default).
        n_input: Stimulus dimension (reads NOESIS_N_INPUT or config default).

    Returns:
        {mode_name: [result_dicts], ...}
    """
    import numpy as np
    from config import get_config

    if modes is None:
        modes = ["competitive", "random", "no_broadcast",
                 "collaborative", "hybrid", "single_processor"]

    cfg = get_config()
    n_neurons = n_neurons or cfg.n_neurons
    n_input = n_input or cfg.n_input

    from memory import SemanticMemory
    from workspace import GlobalWorkspace, CollaborativeWorkspace

    collaborative_modes = {"collaborative", "hybrid"}
    all_results = {}

    total_cycles = len(stimuli_vecs) * cycles_per_stimulus * len(modes)
    print(f"[NEURAL] {len(stimuli_vecs)} stimuli × {cycles_per_stimulus} cycles × {len(modes)} modes = {total_cycles} total cycles")
    print(f"[NEURAL] {n_neurons} neurons/processor, {n_input}-dim input")
    if cfg.gpu:
        info = cfg.summary()
        gpu_info = info["gpu"]
        print(f"[NEURAL] GPU: {gpu_info.get('name', '?')} ({gpu_info.get('vram_mb', 0)}MB VRAM)")
    else:
        print("[NEURAL] CPU-only mode")

    cycle_idx = 0
    for mode in modes:
        mem = SemanticMemory()
        ws = CollaborativeWorkspace(mem) if mode in collaborative_modes else GlobalWorkspace(mem)
        mode_results = []

        for stim_idx, stim_vec in enumerate(stimuli_vecs):
            for _ in range(cycles_per_stimulus):
                result = run_neural_cycle(
                    stim_vec, ws, mem, mode=mode,
                    n_neurons=n_neurons, n_input=n_input,
                )
                mode_results.append(result)
                cycle_idx += 1
                if cycle_idx % 10 == 0:
                    print(f"[NEURAL] {cycle_idx}/{total_cycles} cycles complete", flush=True)

        all_results[mode] = mode_results
        mem.clear()

    return all_results


def analyze_neural_comparison(results: dict[str, list[dict]]) -> dict:
    """
    Statistical comparison of neural Φ across experimental modes.

    Uses Mann-Whitney U (non-parametric, one-tailed) on phi_delta values.
    Also reports phi_sensitivity CV distributions and complexity metrics.
    """
    import numpy as np

    analysis = {}
    for mode, cycles in results.items():
        phi_deltas = [c["phi_delta"] for c in cycles]
        phi_afters = [c["phi_after"] for c in cycles]
        phi_befores = [c["phi_before"] for c in cycles]
        phi_sensitivities = [
            c.get("phi_sensitivity", {}).get("cv", 0)
            for c in cycles
            if isinstance(c.get("phi_sensitivity"), dict)
        ]
        complexities = [c.get("complexity", 0) for c in cycles]

        analysis[mode] = {
            "mean_phi_delta": round(float(np.mean(phi_deltas)), 6),
            "std_phi_delta": round(float(np.std(phi_deltas)), 6),
            "mean_phi_after": round(float(np.mean(phi_afters)), 6),
            "std_phi_after": round(float(np.std(phi_afters)), 6),
            "mean_phi_before": round(float(np.mean(phi_befores)), 6),
            "max_phi_delta": round(float(np.max(phi_deltas)), 6),
            "mean_complexity": round(float(np.mean(complexities)), 6),
            "mean_sensitivity_cv": round(float(np.mean(phi_sensitivities)), 6) if phi_sensitivities else 0,
            "n_cycles": len(cycles),
            "broadcast_rate": round(
                sum(1 for c in cycles if c.get("broadcasted")) / max(len(cycles), 1), 4
            ),
        }

    # Hypothesis tests
    try:
        from scipy.stats import mannwhitneyu
        tests = {}
        pairs = [
            ("collaborative", "competitive", "H1: Collab Φ > Competitive Φ"),
            ("collaborative", "random", "H2: Collab Φ > Random Φ"),
            ("hybrid", "competitive", "H3: Hybrid Φ > Competitive Φ"),
            ("competitive", "random", "H4: Competitive Φ > Random Φ"),
            ("competitive", "no_broadcast", "H5: Broadcast Φ > No-broadcast Φ"),
        ]
        for mode_a, mode_b, label in pairs:
            if mode_a in results and mode_b in results:
                a = [c["phi_delta"] for c in results[mode_a]]
                b = [c["phi_delta"] for c in results[mode_b]]
                if len(a) >= 4 and len(b) >= 4:
                    stat, pval = mannwhitneyu(a, b, alternative="greater")
                    tests[f"{mode_a}_vs_{mode_b}"] = {
                        "label": label,
                        "U_statistic": float(stat),
                        "p_value": round(float(pval), 6),
                        "significant": pval < 0.05,
                        "supported": pval < 0.05,
                        "n_a": len(a),
                        "n_b": len(b),
                    }
        analysis["hypothesis_tests"] = tests
    except Exception as e:
        analysis["hypothesis_tests"] = {"error": str(e)}

    return analysis


def _neural_vec_to_text(vec: "np.ndarray", label: str = "") -> str:
    """Convert neural activation vector to human-readable summary."""
    import numpy as np
    v = np.asarray(vec).flatten()
    active = int(np.sum(np.abs(v) > 0.5))
    total = len(v)
    sparsity = 1.0 - active / total
    mean_abs = float(np.mean(np.abs(v)))
    top_dims = np.argsort(-np.abs(v))[:5]
    top_str = ",".join(f"d{i}({v[i]:+.2f})" for i in top_dims)
    return f"[{label}] {active}/{total} active (sp={sparsity:.2f}, μ|a|={mean_abs:.3f}) top: {top_str}"


def _neural_vec_summary(vec: "np.ndarray") -> str:
    """Short summary of a neural activation vector (for JSON serialization)."""
    import numpy as np
    v = np.asarray(vec).flatten()
    return f"{int(np.sum(np.abs(v) > 0.5))}/{len(v)} active, μ|a|={float(np.mean(np.abs(v))):.3f}"
