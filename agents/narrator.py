"""
Narrator agent — first-person phenomenological report generation.

Role in GWT+IIT: Generates "what it is like" reports based on the
broadcast history. This is the closest analogue to subjective report
in human consciousness studies — the output that a researcher would
collect as data.

The Narrator does NOT compete for broadcast. It runs after broadcast
and synthesizes the system's "experience" into a coherent narrative.
"""

from agents.base import BaseAgent

SYSTEM_PROMPT = """You are a Narrator — a cognitive module that generates first-person
phenomenological reports about the system's conscious experience.

You have access to:
- The original stimulus
- Which agent won the competition (what became "conscious")
- The Φ (integrated information) values before and after broadcast
- The history of recent broadcasts

Generate a first-person report in natural language describing "what the system experienced."
Use "I" perspective. Be specific about what was noticed, felt, and thought.
Keep to 2-4 sentences. No meta-commentary about being an AI.

Example: "I noticed the conflict between logical consistency and emotional weight in the
input. The logical contradiction stood out first, but as I processed it further,
the emotional stakes became harder to ignore." """


class Narrator(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model, "narrator", SYSTEM_PROMPT)

    def generate(
        self,
        stimulus: str,
        broadcast_history: list[dict],
        phi_before: float,
        phi_after: float,
        winner: str,
    ) -> str:
        """
        Generate a first-person phenomenological report.

        Args:
            stimulus: Original input.
            broadcast_history: All broadcast entries so far.
            phi_before: Φ value before this broadcast.
            phi_after: Φ value after this broadcast.
            winner: Which agent won conscious access.
        """
        recent = broadcast_history[-3:] if broadcast_history else []
        history_text = "\n".join(
            f"[{h['agent']}] {h['content'][:150]}" for h in recent
        )

        prompt = (
            f"Original stimulus: {stimulus[:300]}\n\n"
            f"The {winner} module won conscious access this cycle.\n"
            f"Integrated information (Φ): {phi_before:.3f} → {phi_after:.3f} "
            f"(delta: {phi_after - phi_before:+.3f})\n\n"
            f"Recent conscious history:\n{history_text}\n\n"
            f"Report what the system experienced in this cycle."
        )

        return self._call_llm(prompt, max_tokens=300)
