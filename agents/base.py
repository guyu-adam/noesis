"""
Abstract base agent — all specialized agents inherit from this.

Each agent:
  1. Receives a stimulus + workspace context
  2. Generates a proposal via local LLM (Ollama)
  3. Returns the proposal for attention competition
"""

from abc import ABC, abstractmethod
import requests as req

OLLAMA_GENERATE = "http://localhost:11434/api/generate"


class BaseAgent(ABC):
    """Abstract agent interface for Noesis cognitive modules."""

    def __init__(self, model: str, role: str, system_prompt: str):
        self.model = model
        self.role = role
        self.system_prompt = system_prompt

    @abstractmethod
    def process(self, stimulus: str, context: str = "") -> str:
        """
        Process a stimulus and return a proposal for the global workspace.

        This is the core cognitive operation — the agent interprets the
        stimulus through its specialized lens and produces a candidate
        for conscious broadcast.

        Args:
            stimulus: The input text to process.
            context: Current global workspace context (past broadcasts).

        Returns:
            A proposal string — the agent's interpretation/output.
        """
        ...

    def _call_llm(self, user_prompt: str, max_tokens: int = 400) -> str:
        """
        Call Ollama with the agent's system prompt + user input.

        Uses the raw /api/generate endpoint with chatml format for
        maximum control over the output format.
        """
        raw_prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            resp = req.post(OLLAMA_GENERATE, json={
                "model": self.model,
                "prompt": raw_prompt,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
                "stream": False,
                "raw": True,
            }, timeout=120)
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[{self.role} LLM error: {e}]"

    def __repr__(self):
        return f"<{self.role} agent | {self.model}>"
