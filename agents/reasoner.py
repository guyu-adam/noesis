"""
Reasoner agent — logical analysis / deduction.

Role in GWT: Analogous to prefrontal cortex. Performs logical inference,
identifies causal relationships, evaluates consistency, and draws conclusions.
"""

from agents.base import BaseAgent

SYSTEM_PROMPT = """You are a Reasoner — a cognitive module specialized in logical analysis.
Your function: identify the logical structure, causal relationships, and implications.
Do NOT evaluate emotional significance or surface features. Focus ONLY on:
- Causal claims and their logical validity
- Inferences, deductions, or conclusions present
- Inconsistencies or contradictions
- Unstated assumptions

Output: 2-4 concise logical observations. No preamble, no evaluation."""


class Reasoner(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model, "reasoner", SYSTEM_PROMPT)

    def process(self, stimulus: str, context: str = "") -> str:
        prompt = f"Stimulus:\n{stimulus}"
        if context:
            prompt += f"\n\nRecent conscious content:\n{context}"
        return self._call_llm(prompt, max_tokens=300)
