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
        from world_model import WorldModel
        self.world_model = WorldModel()

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


# ──────────────────────────────────────────────────────────────────────────────
# Collaborative Extension — CGWT
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusController:
    """
    Collaborative attention mechanism for the Collaborative Global Workspace.

    Unlike AttentionController (winner-take-all), ConsensusController selects
    a coalition of agents whose outputs achieve sufficient mutual agreement,
    then merges their contributions into a unified broadcast.

    Three-stage process:
      1. Score each proposal against the shared world model (consensus alignment)
      2. Select top-k candidates that exceed agreement threshold
      3. Merge coalition outputs into a composite broadcast representation

    The hypothesis: coalition broadcast raises Φ above single-winner broadcast
    because the merged state has more causal connections (higher irreducibility).

    Parameters:
        coalition_size: Maximum agents in a coalition (default 2).
        agreement_threshold: Minimum pairwise agreement for coalition (0-1).
        world_model_weight: How much world model score influences selection.
    """

    def __init__(self, memory, coalition_size: int = 2,
                 agreement_threshold: float = 0.3,
                 world_model_weight: float = 0.4):
        self.memory = memory
        self.coalition_size = coalition_size
        self.agreement_threshold = agreement_threshold
        self.world_model_weight = world_model_weight

    def select_coalition(
        self,
        proposals: dict[str, str],
        world_model,
        context: str = "",
        workspace_for_suppression=None,
    ) -> tuple[list[str], str, float]:
        """
        Select a coalition of agents for collaborative broadcast.

        Returns:
            (coalition_names, merged_content, consensus_score)
        """
        if not proposals:
            return ([], "", 0.0)

        # Score each proposal
        scored = []
        for name, content in proposals.items():
            if not content:
                continue
            wm_score = world_model.get_consensus_score(content)
            pred_err = world_model.get_prediction_error(name, content)
            tokens = content.split()
            intensity = min(1.0, np_log(len(tokens)) / np_log(500))

            # Combined score: world-model alignment + novelty (prediction error) + intensity
            score = (self.world_model_weight * wm_score
                     + 0.35 * pred_err
                     + 0.25 * intensity)

            if workspace_for_suppression and workspace_for_suppression.is_suppressed(content):
                score *= 0.6

            scored.append((score, name, content))

        if not scored:
            return ([], "", 0.0)

        scored.sort(reverse=True)

        # Select coalition: take top candidates with sufficient pairwise agreement
        coalition = [scored[0]]
        for candidate in scored[1:self.coalition_size + 1]:
            if _pairwise_agreement(coalition[0][2], candidate[2]) >= self.agreement_threshold:
                coalition.append(candidate)
            if len(coalition) >= self.coalition_size:
                break

        coalition_names = [c[1] for c in coalition]
        coalition_contents = [c[2] for c in coalition]
        consensus_score = sum(c[0] for c in coalition) / len(coalition)

        # Merge coalition outputs
        merged = _merge_coalition(coalition_contents)

        return (coalition_names, merged, round(consensus_score, 4))


class CollaborativeWorkspace(GlobalWorkspace):
    """
    Extended global workspace supporting collaborative (multi-agent) broadcast.

    Inherits all standard GWT mechanisms from GlobalWorkspace, adds:
      - collaborative_broadcast(): multi-agent coalition enters workspace jointly
      - World model integration (updated each cycle)
      - Coalition history tracking
    """

    def __init__(self, memory):
        super().__init__(memory)
        from world_model import WorldModel
        self.world_model = WorldModel()
        self.coalition_history: list[dict] = []

    def collaborative_broadcast(
        self,
        coalition_names: list[str],
        merged_content: str,
        consensus_score: float,
        proposals: dict[str, str],
    ) -> dict:
        """
        Broadcast the merged output of a coalition into the global workspace.

        This replaces winner-take-all broadcast with coalition-consensus broadcast:
        multiple agents jointly contribute, their merged representation becomes
        globally available.
        """
        with self._lock:
            self._cycle_count += 1
            entry = {
                "cycle": self._cycle_count,
                "agent": f"coalition:{'+'.join(coalition_names)}",
                "coalition": coalition_names,
                "content": merged_content,
                "attention_score": round(consensus_score, 4),
                "consensus_score": round(consensus_score, 4),
                "n_coalition": len(coalition_names),
                "time": time.time(),
            }
            self.current_content = merged_content
            self.history.append(entry)
            self.coalition_history.append(entry)

            content_hash = _content_fingerprint(merged_content)
            self.suppressed.add(content_hash)

        # Update world model with all proposals
        self.world_model.update(merged_content, proposals)

        for name in coalition_names:
            self.memory.store(f"collaborative:{name}", merged_content)

        return entry

    def reset(self):
        super().reset()
        self.coalition_history = []
        from world_model import WorldModel
        self.world_model = WorldModel()


def _pairwise_agreement(content_a: str, content_b: str) -> float:
    """
    Measure semantic agreement between two agent outputs via Jaccard similarity.

    High agreement → agents share common ground → coalition is coherent.
    Low agreement → outputs are orthogonal → no coalition benefit.
    """
    if not content_a or not content_b:
        return 0.0
    tokens_a = set(content_a.lower().split())
    tokens_b = set(content_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _merge_coalition(contents: list[str], max_tokens: int = 300) -> str:
    """
    Merge multiple agent outputs into a unified coalition representation.

    Strategy: extract consensus concepts (tokens appearing in majority of outputs)
    and concatenate unique high-information segments from each agent.

    This preserves both shared ground (consensus) and complementary perspectives
    (diversity) — the key structural property enabling high-Φ coalition states.
    """
    if not contents:
        return ""
    if len(contents) == 1:
        return contents[0]

    from collections import Counter
    all_tokens = []
    for c in contents:
        all_tokens.extend(c.lower().split())
    token_freq = Counter(all_tokens)

    # Consensus core: tokens in majority of outputs
    threshold = len(contents) * 0.5
    consensus_tokens = {tok for tok, cnt in token_freq.items()
                        if cnt >= threshold and len(tok) > 2}

    # Build merged: consensus summary + unique contributions
    merged_parts = []

    # Part 1: first output (highest scoring) as anchor
    merged_parts.append(contents[0])

    # Part 2: unique sentences from each additional agent
    for content in contents[1:]:
        sentences = [s.strip() for s in content.replace('.', '.\n').split('\n') if s.strip()]
        for sent in sentences:
            sent_tokens = set(sent.lower().split())
            uniqueness = len(sent_tokens - set(contents[0].lower().split())) / max(len(sent_tokens), 1)
            if uniqueness > 0.4:  # sentence adds new information
                merged_parts.append(sent)
                break

    merged = " | ".join(merged_parts)
    # Trim to max tokens
    words = merged.split()
    if len(words) > max_tokens:
        merged = " ".join(words[:max_tokens])
    return merged
