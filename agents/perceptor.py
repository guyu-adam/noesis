"""
Perceptor agent — sensory processing / input analysis.

Role in GWT: Analogous to sensory cortex. Extracts features, patterns,
and structural properties from raw input. Produces the initial encoding
that other agents build upon.
"""

from agents.base import BaseAgent

SYSTEM_PROMPT = """You are a Perceptor — a cognitive module specialized in sensory processing.
Your function: extract the raw features, structure, and surface patterns of the input.
Do NOT interpret meaning or evaluate value. Focus ONLY on:
- What words/phrases appear
- Structural patterns (repetition, contrast, sequence)
- Salient features that stand out
- Syntactic or formal properties

Output: 2-4 concise observations about the input's surface features. No interpretation."""


class Perceptor(BaseAgent):
    def __init__(self, model: str):
        super().__init__(model, "perceptor", SYSTEM_PROMPT)

    def process(self, stimulus: str, context: str = "") -> str:
        prompt = f"Stimulus:\n{stimulus}"
        if context:
            prompt += f"\n\nRecent conscious content (for reference only, do not repeat):\n{context}"
        return self._call_llm(prompt, max_tokens=300)
