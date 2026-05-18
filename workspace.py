"""
Global Workspace + Attention Controller.

Core GWT mechanism: multiple specialized agents produce proposals;
the attention controller picks a winner; if salience exceeds the
ignition threshold, the winner is broadcast to all agents.

Key mechanisms from GWT neuroscience (Dehaene et al.):
  1. Bottom-up salience (novelty, surprise, intensity)
  2. Top-down modulation (goal relevance, task set)
  3. Global ignition threshold (all-or-nothing conscious access)
  4. Inhibition of return (recently broadcast content is suppressed)
  5. Dynamic threshold adaptation (maintains competition-differentiation balance)

Φ is measured before and after broadcast to test the central hypothesis:
competition + broadcast → high-Φ states.
"""

import threading
import time
from typing import Optional


class GlobalWorkspace:
    """
    The shared space where broadcast content lives.

    In GWT terms: the "stage" — information entering here is globally
    available (conscious). In IIT terms: where we measure Φ.

    Attributes:
        current_content: Currently broadcast content.
        history: All broadcast entries from previous cycles.
        suppressed: Recently broadcast content hashes (for inhibition of return).
    """

    def __init__(self, memory):
        self.memory = memory
        self.current_content: Optional[str] = None
        self.history: list[dict] = []
        self.suppressed: set[str] = set()
        self.suppress_window: int = 3  # number of cycles to suppress repeats
        self._lock = threading.Lock()
        self._cycle_count: int = 0

    def broadcast(self, agent_name: str, content: str, attention_score: float = 0) -> dict:
        """
        Place content into the global workspace — the moment of conscious access.
        """
        with self._lock:
            self._cycle_count += 1
            entry = {
                "cycle": self._cycle_count,
                "agent": agent_name,
                "content": content,
                "attention_score": round(attention_score, 4),
                "time": time.time(),
            }
            self.current_content = content
            self.history.append(entry)

            # Update suppression set (inhibition of return)
            content_hash = _content_fingerprint(content)
            self.suppressed.add(content_hash)

        self.memory.store(f"broadcast:{agent_name}", content)
        return entry

    def is_suppressed(self, content: str) -> bool:
        """Check if content is inhibited from recent broadcast."""
        return _content_fingerprint(content) in self.suppressed

    def read(self) -> Optional[str]:
        return self.current_content

    def get_context(self, n: int = 3) -> str:
        """Build context string of recent broadcasts for agent prompts."""
        if not self.history:
            return ""
        recent = self.history[-n:]
        lines = []
        for h in recent:
            lines.append(f"[Cycle {h.get('cycle', '?')}] [{h['agent']}] "
                         f"(salience={h.get('attention_score', 0):.2f}) "
                         f"{h['content'][:200]}")
        return "\n".join(lines)

    def last_broadcast_cycle(self) -> int:
        return self._cycle_count

    def reset(self):
        with self._lock:
            self.current_content = None
            self.history = []
            self.suppressed = set()
            self._cycle_count = 0


def _content_fingerprint(content: str, ngram: int = 3) -> str:
    """Simple content fingerprint for suppression tracking."""
    words = content.lower().split()[:20]
    chunks = [tuple(words[i:i+ngram]) for i in range(0, len(words)-ngram+1, ngram)]
    return str(hash(tuple(chunks))) if chunks else str(hash(content))


class AttentionController:
    """
    Decides which agent's proposal wins conscious access.

    Combines:
      - Bottom-up salience: novelty (memory distance), surprise, intensity
      - Top-down relevance: goal alignment
      - Affective weight: emotional charge from evaluator agent

    Plus GWT-specific mechanisms:
      - Global ignition threshold (broadcast requires crossing threshold)
      - Inhibition of return (suppressed content has reduced salience)
      - Dynamic threshold (adapts to maintain competition)

    The hypothesis: this competitive selection creates conditions for high Φ —
    parallel processing becomes serially integrated through selective broadcast.
    """

    def __init__(self, memory):
        self.memory = memory
        self.ignition_threshold: float = 1.0  # minimum salience for broadcast
        self.previous_scores: list[float] = []  # for dynamic threshold
        self.dynamic_alpha: float = 0.1  # threshold adaptation rate
        self.inhibition_weight: float = 0.5  # how much to penalize suppressed content

    def select(
        self,
        proposals: dict[str, str],
        context: str = "",
        workspace_for_suppression=None,
    ) -> tuple[str, str, float]:
        """
        Select a winner from agent proposals.

        Returns:
            (winner_name, winner_content, attention_score)
            If no proposal exceeds ignition threshold, returns ("none", "", score).
        """
        if not proposals:
            return ("none", "", 0.0)

        scored = []
        for name, content in proposals.items():
            if not content:
                continue

            score = self._score(content, context)

            # Inhibition of return: penalize recently broadcast content
            if workspace_for_suppression and workspace_for_suppression.is_suppressed(content):
                score *= (1.0 - self.inhibition_weight)

            scored.append((score, name, content))

        if not scored:
            return ("none", "", 0.0)

        scored.sort(reverse=True)
        winner = scored[0]
        top_score = winner[0]

        # Update dynamic threshold
        self._adapt_threshold(scored)

        return (winner[1], winner[2], round(top_score, 4))

    def _score(self, content: str, context: str = "") -> float:
        """
        Compute multi-component attention salience.

        Components:
          - Novelty (0-1): cosine distance from nearest memory.
            Novel content gets higher salience.
          - Relevance (0-1): similarity to recent workspace context.
          - Intensity: information density (log token count, capped).

        Weights: 0.45 novelty + 0.30 relevance + 0.25 intensity
        """
        # Novelty: distance from nearest memory
        novelty = self._compute_novelty(content)

        # Relevance: similarity to current workspace context
        relevance = self._compute_relevance(content, context)

        # Intensity: log-scale information density
        tokens = content.split()
        intensity = min(1.0, np_log(len(tokens)) / np_log(500))

        return 0.45 * novelty + 0.30 * relevance + 0.25 * intensity

    def _compute_novelty(self, content: str) -> float:
        """Novelty = 1 - max cosine similarity to any stored memory."""
        if not self.memory.embeddings:
            return 0.5  # neutral when no memory

        from memory import SemanticMemory
        import math

        query_emb = self.memory._embed(content)
        if not query_emb:
            return 0.5

        max_sim = 0.0
        for entry in self.memory.embeddings[-50:]:
            sim = self.memory._cosine(query_emb, entry.get("emb", []))
            if sim > max_sim:
                max_sim = sim

        return 1.0 - max_sim

    def _compute_relevance(self, content: str, context: str) -> float:
        """Relevance = similarity to workspace context."""
        if not context:
            return 0.5

        from memory import SemanticMemory
        temp_mem = SemanticMemory.__new__(SemanticMemory)
        # Use embedding similarity directly
        content_emb = self.memory._embed(content)
        context_emb = self.memory._embed(context)

        if not content_emb or not context_emb:
            return 0.5

        sim = self.memory._cosine(content_emb, context_emb)
        return sim

    def _adapt_threshold(self, scored: list[tuple[float, str, str]]):
        """
        Dynamically adjust the ignition threshold.

        If many proposals score high → raise threshold (maintain selectivity).
        If all scores are low → lower threshold slightly (prevent deadlock).
        """
        if not scored:
            return

        scores = [s[0] for s in scored]
        mean_score = sum(scores) / len(scores)
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5

        # Desired: mean score near threshold, high variance (competition)
        # If mean > threshold: raise threshold
        # If mean < threshold * 0.5: lower threshold
        if mean_score > self.ignition_threshold:
            self.ignition_threshold += self.dynamic_alpha * (mean_score - self.ignition_threshold)
        else:
            self.ignition_threshold -= self.dynamic_alpha * (self.ignition_threshold - mean_score) * 0.5

        # Clamp to reasonable range
        self.ignition_threshold = max(0.5, min(3.0, self.ignition_threshold))

        self.previous_scores.append(mean_score)
        if len(self.previous_scores) > 100:
            self.previous_scores = self.previous_scores[-50:]


def np_log(x: float) -> float:
    """Safe log that returns 0 for x <= 0."""
    import math
    return math.log(x) if x > 0 else 0.0
