"""
Experiment runner — orchestrates one GWT+IIT cycle.

Each cycle:
  1. All specialized agents process the same stimulus → produce proposals
  2. Attention controller selects a winner
  3. Winner is broadcast to the global workspace
  4. Φ is measured before and after broadcast
  5. Narrator generates a first-person phenomenological report

This is the core experimental loop that tests:
    Does GWT-style competition + broadcast produce measurable Φ increases?
"""

import time
from workspace import GlobalWorkspace, AttentionController
from memory import SemanticMemory
from iit import phi_proxy

# Lazy-loaded agent instances keyed by model name.
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
    memory: SemanticMemory,
    model: str,
) -> dict:
    """
    Run one complete GWT+IIT experimental cycle.

    Args:
        stimulus: The input text that all agents process.
        workspace: The global workspace instance.
        memory: Semantic memory instance.
        model: Ollama model name.

    Returns:
        Dict with full cycle data — proposals, winner, Φ, broadcast, narrative.
    """
    agents = _get_agents(model)
    controller = AttentionController(memory)
    workspace_context = workspace.get_context()

    # ── Phase 1: Pre-broadcast Φ measurement ───────────────────────────────
    phi_before = phi_proxy(workspace.read(), workspace.history)

    # ── Phase 2: All agents process stimulus in parallel ────────────────────
    proposals = {}
    for name, agent in agents.items():
        if name == "narrator":
            continue  # Narrator runs after broadcast
        try:
            proposals[name] = agent.process(stimulus, workspace_context)
        except Exception as e:
            proposals[name] = f"[{name} error: {e}]"

    # ── Phase 3: Attention competition ─────────────────────────────────────
    winner_name, winner_content, attention_score = controller.select(
        proposals, workspace_context
    )

    # ── Phase 4: Global broadcast ──────────────────────────────────────────
    workspace.broadcast(winner_name, winner_content)

    # ── Phase 5: Post-broadcast Φ measurement ──────────────────────────────
    phi_after = phi_proxy(workspace.read(), workspace.history)

    # ── Phase 6: Narrative generation ──────────────────────────────────────
    narrative = agents["narrator"].generate(
        stimulus=stimulus,
        broadcast_history=workspace.history,
        phi_before=phi_before,
        phi_after=phi_after,
        winner=winner_name,
    )

    return {
        "stimulus": stimulus[:200],
        "proposals": {k: v[:200] for k, v in proposals.items()},
        "winner": winner_name,
        "attention_score": attention_score,
        "broadcast": winner_content[:400],
        "phi_before": phi_before,
        "phi_after": phi_after,
        "phi_delta": phi_after - phi_before,
        "narrative": narrative,
    }


def run_multi_cycle(
    stimuli: list[str],
    model: str,
    cycles_per_stimulus: int = 3,
) -> list[dict]:
    """
    Run multiple cycles per stimulus — the same stimulus goes through
    multiple rounds of competition and broadcast, simulating sustained
    conscious processing of a single input.
    """
    from memory import SemanticMemory
    from workspace import GlobalWorkspace

    mem = SemanticMemory()
    ws = GlobalWorkspace(mem)
    results = []

    for stim in stimuli:
        for _ in range(cycles_per_stimulus):
            result = run_cycle(stim, ws, mem, model)
            results.append(result)

    return results
