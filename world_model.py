"""
Collaborative World Model — shared predictive representation built jointly by all processors.

In the Collaborative Global Workspace Theory (CGWT), processors do not merely compete
for broadcast access. They collectively maintain a world model: a shared, evolving
representation of the stimulus environment that serves as common ground for
consensus-building before broadcast.

Theoretical grounding:
  - Classic GWT (Baars 1988): global workspace as shared "blackboard"
  - Predictive coding (Friston 2010): the brain as a hierarchical prediction machine
  - CGWT extension (this work): the workspace also hosts a collaboratively updated
    world model that reduces prediction error across all processors simultaneously.

The world model records:
  1. Stimulus history (what has been observed)
  2. Prediction residuals per processor (who was surprised, and by how much)
  3. Consensus map (which concepts all processors agree on)
  4. Shared bag-of-words representation (averaged across all processor proposals)
"""

import threading
import math
from collections import Counter, defaultdict
from typing import Optional


class WorldModel:
    """
    A shared predictive model maintained collaboratively by all workspace processors.

    Unlike the SemanticMemory (which stores broadcast history), WorldModel
    tracks the collective epistemic state of the processor coalition: what they
    jointly predict, where they disagree, and how surprised they are.

    This is the key structural difference from standard GWT:
      GWT     -> single winner's representation enters workspace
      CGWT    -> coalition's consensus representation enters workspace,
                grounded in shared world model

    On the first cycle (cold start), all processors have consensus_score=0.5 and
    prediction_error=0.5 — there is no prior to differentiate them. The selection
    defaults to intensity-based ranking. This is expected behaviour (no prior =
    equal weighting) but means early cycles have higher noise. After ~3 cycles
    the world model accumulates enough history for meaningful differentiation.
    """

    def __init__(self, consensus_threshold_ratio: float = 0.6,
                 max_stimulus_history: int = 50, max_prediction_history: int = 20,
                 prune_interval: int = 10):
        self.stimulus_history: list[str] = []
        self.processor_predictions: dict[str, list[str]] = defaultdict(list)
        self.consensus_concepts: Counter = Counter()
        self.disagreement_map: dict[str, float] = {}
        self._world_representation: list[float] = []
        self._cycle: int = 0
        self._lock = threading.Lock()

        self.consensus_threshold_ratio = consensus_threshold_ratio
        self.max_stimulus_history = max_stimulus_history
        self.max_prediction_history = max_prediction_history
        self.prune_interval = prune_interval

    def update(self, stimulus: str, proposals: dict[str, str]) -> dict:
        """
        Update the world model after processors have processed a stimulus.

        Args:
            stimulus: The current stimulus.
            proposals: {processor_name: processor_output} from all processors.

        Returns:
            World model update summary (used in consensus scoring).
        """
        with self._lock:
            self._cycle += 1
            self.stimulus_history.append(stimulus)
            if len(self.stimulus_history) > self.max_stimulus_history:
                self.stimulus_history = self.stimulus_history[-self.max_stimulus_history:]

            # Record each processor's prediction
            for name, content in proposals.items():
                self.processor_predictions[name].append(content)
                if len(self.processor_predictions[name]) > self.max_prediction_history:
                    self.processor_predictions[name] = self.processor_predictions[name][-self.max_prediction_history:]

            # Build consensus concept map: tokens that appear in most proposals
            all_tokens = []
            per_processor_tokens = {}
            for name, content in proposals.items():
                tokens = set(content.lower().split())
                per_processor_tokens[name] = tokens
                all_tokens.extend(tokens)

            token_counts = Counter(all_tokens)
            n_processors = max(len(proposals), 1)

            # Concepts endorsed by majority of processors
            consensus_threshold = n_processors * self.consensus_threshold_ratio
            new_consensus = {tok for tok, cnt in token_counts.items()
                             if cnt >= consensus_threshold and len(tok) > 3}
            self.consensus_concepts.update(new_consensus)

            # Prune stale concepts periodically to prevent unbounded growth
            if self._cycle % self.prune_interval == 0:
                stale = [tok for tok, cnt in self.consensus_concepts.items() if cnt < 3]
                for tok in stale:
                    del self.consensus_concepts[tok]

            # Disagreement: tokens unique to one processor (high idiosyncrasy)
            for name, tokens in per_processor_tokens.items():
                others = set()
                for other_name, other_tokens in per_processor_tokens.items():
                    if other_name != name:
                        others |= other_tokens
                unique = tokens - others
                self.disagreement_map[name] = len(unique) / max(len(tokens), 1)

            # Update shared representation (bag-of-words TF average)
            self._world_representation = _compute_shared_representation(
                [content for content in proposals.values() if content]
            )

            return {
                "cycle": self._cycle,
                "consensus_concepts": len(new_consensus),
                "mean_disagreement": (
                    sum(self.disagreement_map.values()) / n_processors
                    if self.disagreement_map else 0.0
                ),
            }

    def get_consensus_score(self, content: str) -> float:
        """
        Score how well a piece of content aligns with the shared world model.

        High consensus score -> content resonates with the collective representation.
        Used by ConsensusController to select collaborative broadcast candidates.
        """
        with self._lock:
            if not self.consensus_concepts or not content:
                return 0.5

            tokens = set(content.lower().split())
            # Counter.keys() is a dict_keys view — set intersection is efficient
            relevant = tokens & self.consensus_concepts.keys()
            weighted_hits = sum(self.consensus_concepts[tok] for tok in relevant)
            top_n = min(len(tokens), len(self.consensus_concepts))
            max_possible = sum(
                sorted(self.consensus_concepts.values(), reverse=True)[:top_n]
            )
            if max_possible == 0:
                return 0.5
            return min(1.0, weighted_hits / max_possible)

    def get_prediction_error(self, processor_name: str, content: str) -> float:
        """
        Estimate prediction error for a processor's output relative to its history.

        High error -> processor was surprised -> content is novel -> higher information value.
        Mirrors the free-energy principle: conscious access is triggered by prediction errors.
        """
        with self._lock:
            history = self.processor_predictions.get(processor_name, [])
            if not history:
                return 0.5

            current_tokens = set(content.lower().split())
            recent = history[-5:]
            historical_tokens = set()
            for h in recent:
                historical_tokens |= set(h.lower().split())

            overlap = len(current_tokens & historical_tokens) / max(len(current_tokens), 1)
            return 1.0 - overlap  # high overlap = low prediction error

    @property
    def world_representation(self) -> list[float]:
        """Lazy access to shared bag-of-words representation."""
        return self._world_representation

    def summary(self) -> dict:
        with self._lock:
            return {
                "cycle": self._cycle,
                "stimuli_seen": len(self.stimulus_history),
                "top_consensus_concepts": [
                    tok for tok, _ in self.consensus_concepts.most_common(10)
                ],
                "processor_disagreement": dict(self.disagreement_map),
                "world_representation_dim": len(self._world_representation),
            }


def _compute_shared_representation(texts: list[str], vocab_size: int = 200) -> list[float]:
    """
    Compute an averaged token-frequency representation across multiple texts.
    Uses bag-of-words TF averaging — a lightweight shared representation, not a
    semantic embedding (cf. SemanticMemory which uses Ollama nomic-embed-text).
    """
    if not texts:
        return []

    all_tokens = []
    for t in texts:
        all_tokens.extend(t.lower().split())
    vocab = [tok for tok, _ in Counter(all_tokens).most_common(vocab_size)]
    if not vocab:
        return []

    embeddings = []
    for text in texts:
        counts = Counter(text.lower().split())
        total = sum(counts.values()) or 1
        vec = [counts.get(tok, 0) / total for tok in vocab]
        embeddings.append(vec)

    avg = [sum(emb[i] for emb in embeddings) / len(embeddings)
           for i in range(len(vocab))]
    return avg
