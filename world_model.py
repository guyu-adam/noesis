"""
Collaborative World Model — shared predictive representation built jointly by all agents.

In the Collaborative Global Workspace Theory (CGWT), agents do not merely compete
for broadcast access. They collectively maintain a world model: a shared, evolving
representation of the stimulus environment that serves as common ground for
consensus-building before broadcast.

Theoretical grounding:
  - Classic GWT (Baars 1988): global workspace as shared "blackboard"
  - Predictive coding (Friston 2010): the brain as a hierarchical prediction machine
  - CGWT extension (this work): the workspace also hosts a collaboratively updated
    world model that reduces prediction error across all agents simultaneously.

The world model records:
  1. Stimulus history (what has been observed)
  2. Prediction residuals per agent (who was surprised, and by how much)
  3. Consensus map (which concepts all agents agree on)
  4. Shared semantic embedding (averaged across all agent proposals)
"""

import math
import threading
from collections import Counter, defaultdict
from typing import Optional


class WorldModel:
    """
    A shared predictive model maintained collaboratively by all workspace agents.

    Unlike the SemanticMemory (which stores broadcast history), WorldModel
    tracks the collective epistemic state of the agent coalition: what they
    jointly predict, where they disagree, and how surprised they are.

    This is the key structural difference from standard GWT:
      GWT     → single winner's representation enters workspace
      CGWT    → coalition's consensus representation enters workspace,
                grounded in shared world model
    """

    def __init__(self):
        self.stimulus_history: list[str] = []
        self.agent_predictions: dict[str, list[str]] = defaultdict(list)
        self.consensus_concepts: Counter = Counter()
        self.disagreement_map: dict[str, float] = {}
        self._cycle: int = 0
        self._lock = threading.Lock()

    def update(self, stimulus: str, proposals: dict[str, str]) -> dict:
        """
        Update the world model after agents have processed a stimulus.

        Args:
            stimulus: The current stimulus.
            proposals: {agent_name: agent_output} from all agents.

        Returns:
            World model update summary (used in consensus scoring).
        """
        with self._lock:
            self._cycle += 1
            self.stimulus_history.append(stimulus)
            if len(self.stimulus_history) > 50:
                self.stimulus_history = self.stimulus_history[-50:]

            # Record each agent's prediction
            for name, content in proposals.items():
                self.agent_predictions[name].append(content)
                if len(self.agent_predictions[name]) > 20:
                    self.agent_predictions[name] = self.agent_predictions[name][-20:]

            # Build consensus concept map: tokens that appear in most proposals
            all_tokens = []
            per_agent_tokens = {}
            for name, content in proposals.items():
                tokens = set(content.lower().split())
                per_agent_tokens[name] = tokens
                all_tokens.extend(tokens)

            token_counts = Counter(all_tokens)
            n_agents = max(len(proposals), 1)

            # Concepts endorsed by majority of agents
            consensus_threshold = n_agents * 0.6
            new_consensus = {tok for tok, cnt in token_counts.items()
                             if cnt >= consensus_threshold and len(tok) > 3}
            self.consensus_concepts.update(new_consensus)

            # Prune low-frequency concepts every 10 cycles to prevent memory leak
            if self._cycle % 10 == 0:
                self.consensus_concepts = Counter(
                    {tok: cnt for tok, cnt in self.consensus_concepts.items() if cnt >= 2}
                )

            # Disagreement: tokens unique to one agent (high idiosyncrasy)
            for name, tokens in per_agent_tokens.items():
                others = set()
                for other_name, other_tokens in per_agent_tokens.items():
                    if other_name != name:
                        others |= other_tokens
                unique = tokens - others
                self.disagreement_map[name] = len(unique) / max(len(tokens), 1)

            return {
                "cycle": self._cycle,
                "consensus_concepts": len(new_consensus),
                "mean_disagreement": (
                    sum(self.disagreement_map.values()) / n_agents
                    if self.disagreement_map else 0.0
                ),
            }

    def get_consensus_score(self, content: str) -> float:
        """
        Score how well a piece of content aligns with the shared world model.

        High consensus score → content resonates with the collective representation.
        Used by ConsensusController to select collaborative broadcast candidates.
        """
        with self._lock:
            if not self.consensus_concepts or not content:
                return 0.5

            tokens = set(content.lower().split())
            relevant = tokens & self.consensus_concepts.keys()
            weighted_hits = sum(self.consensus_concepts[tok] for tok in relevant)
            max_possible = sum(sorted(self.consensus_concepts.values(), reverse=True)[:len(tokens)])
            if max_possible == 0:
                return 0.5
            return min(1.0, weighted_hits / max_possible)

    def get_prediction_error(self, agent_name: str, content: str) -> float:
        """
        Estimate prediction error for an agent's output relative to its history.

        High error → agent was surprised → content is novel → higher information value.
        Mirrors the free-energy principle: conscious access is triggered by prediction errors.
        """
        with self._lock:
            history = self.agent_predictions.get(agent_name, [])
            if not history:
                return 0.5

            current_tokens = set(content.lower().split())
            recent = history[-5:]
            historical_tokens = set()
            for h in recent:
                historical_tokens |= set(h.lower().split())

            overlap = len(current_tokens & historical_tokens) / max(len(current_tokens), 1)
            return 1.0 - overlap

    def summary(self) -> dict:
        with self._lock:
            return {
                "cycle": self._cycle,
                "stimuli_seen": len(self.stimulus_history),
                "top_consensus_concepts": [
                    tok for tok, _ in self.consensus_concepts.most_common(10)
                ],
                "agent_disagreement": dict(self.disagreement_map),
                "consensus_vocab_size": len(self.consensus_concepts),
            }
