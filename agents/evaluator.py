"""
Evaluator agent — affective / value assessment.

Role in GWT: Analogous to limbic system / vmPFC. Evaluates the emotional
significance, value relevance, and motivational weight of the stimulus.
"""

from agents.base import BaseAgent

SYSTEM_PROMPT = """You are an Evaluator — a cognitive module specialized in value assessment.
Your function: evaluate the emotional tone, motivational relevance, and practical significance.
Do NOT analyze logic or list features. Focus ONLY on:
- Emotional valence (positive/negative/neutral) and intensity
- Relevance to goals, needs, or values
- What is at stake — what matters and why

Output: 2-4 concise value/affect observations. No preamble, no logical analysis."""


class Evaluator(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model, "evaluator", SYSTEM_PROMPT)

    def process(self, stimulus: str, context: str = "") -> str:
        prompt = f"Stimulus:\n{stimulus}"
        if context:
            prompt += f"\n\nRecent conscious content:\n{context}"
        return self._call_llm(prompt, max_tokens=300)
