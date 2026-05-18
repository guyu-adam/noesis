"""
Global Workspace + Attention Controller.

Core GWT mechanism: multiple specialized agents produce proposals;
the attention controller picks a winner; the winner is "broadcast"
so all agents can access it in subsequent cycles.

Φ is measured on the workspace state before and after broadcast.
"""

import threading
from typing import Optional


class GlobalWorkspace:
    """
    The shared space where broadcast content lives.

    In GWT terms, this is the "stage" — information that enters here
    is globally available (conscious). In IIT terms, this is where
    we measure the causal integration (Φ) of the system.

    Attributes:
        current_content: The currently broadcast content (or None if idle).
        history: All broadcast entries from previous cycles.
    """

    def __init__(self, memory):
        self.memory = memory
        self.current_content: Optional[str] = None
        self.history: list[dict] = []
        self._lock = threading.Lock()

    def broadcast(self, agent_name: str, content: str) -> dict:
        """
        Place content into the global workspace. This is the moment
        of "conscious access" in GWT.

        Returns the broadcast entry (also appended to history).
        """
        with self._lock:
            entry = {
                "agent": agent_name,
                "content": content,
            }
            self.current_content = content
            self.history.append(entry)
        self.memory.store(f"broadcast:{agent_name}", content)
        return entry

    def read(self) -> Optional[str]:
        """Return the currently broadcast content (for agents to read)."""
        return self.current_content

    def get_context(self, n: int = 3) -> str:
        """Build context string of recent broadcasts for agent prompts."""
        if not self.history:
            return ""
        recent = self.history[-n:]
        return "\n".join(
            f"[{h['agent']}] {h['content'][:200]}" for h in recent
        )

    def reset(self):
        with self._lock:
            self.current_content = None
            self.history = []


class AttentionController:
    """
    Decides which agent's proposal wins conscious access.

    Combines:
      - Bottom-up salience (novelty, surprise — how different from memory?)
      - Top-down relevance (goal alignment — does this help the system's current task?)
      - Emotional weight (affective charge from evaluator agent)

    This is the "spotlight" of GWT. The hypothesis is that this
    competitive selection mechanism is what creates the conditions
    for high Φ — parallel processing becomes serially integrated.
    """

    def __init__(self, memory):
        self.memory = memory

    def select(self, proposals: dict[str, str], context: str = "") -> tuple[str, str, float]:
        """
        Select a winner from agent proposals.

        Args:
            proposals: {agent_name: proposal_text} from all agents.
            context: Current workspace context (for relevance scoring).

        Returns:
            (winner_name, winner_content, attention_score)
        """
        if not proposals:
            return ("none", "", 0.0)

        scored = []
        for name, content in proposals.items():
            if not content:
                continue
            score = self._score(content, context)
            scored.append((score, name, content))

        if not scored:
            return ("none", "", 0.0)

        scored.sort(reverse=True)
        winner = scored[0]
        return (winner[1], winner[2], winner[0])

    def _score(self, content: str, context: str = "") -> float:
        """
        Compute attention salience for a proposal.

        Future implementation will combine:
          - Novelty: cosine distance from nearest memory embedding
          - Relevance: similarity to current task/goal
          - Intensity: raw length / information density
        """
        return len(content) / 100.0  # placeholder: raw magnitude
