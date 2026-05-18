"""
Abstract base agent — all specialized agents inherit from this.

Each agent:
  1. Maintains internal state across cycles (short-term memory)
  2. Receives a stimulus + workspace context
  3. Generates a proposal via local LLM (Ollama)
  4. Returns the proposal for attention competition

The internal state allows agents to develop "priming" — prior proposals
influence future ones, creating the causal structure that IIT measures.
"""

import time
import threading
from abc import ABC, abstractmethod
import requests as req

OLLAMA_GENERATE = "http://localhost:11434/api/generate"


class BaseAgent(ABC):
    """Abstract agent interface for Noesis cognitive modules."""

    def __init__(self, model: str, role: str, system_prompt: str):
        self.model = model
        self.role = role
        self.system_prompt = system_prompt
        self.internal_state: list[str] = []  # short-term memory of own outputs
        self.max_internal_state: int = 5
        self._lock = threading.Lock()
        self.cycle_count: int = 0

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

    def _remember(self, output: str):
        """Store output in internal state for future priming."""
        with self._lock:
            self.internal_state.append(output)
            if len(self.internal_state) > self.max_internal_state:
                self.internal_state = self.internal_state[-self.max_internal_state:]
            self.cycle_count += 1

    def _internal_context(self) -> str:
        """Build internal state context string for prompt augmentation."""
        with self._lock:
            if not self.internal_state:
                return ""
            return "Your recent outputs:\n" + "\n".join(
                f"  - {s[:150]}" for s in self.internal_state[-3:]
            )

    def _call_llm(self, user_prompt: str, max_tokens: int = 400) -> str:
        """
        Call Ollama with the agent's system prompt + internal state + user input.
        """
        internal = self._internal_context()
        full_prompt = user_prompt
        if internal:
            full_prompt = f"{internal}\n\n{user_prompt}"

        raw_prompt = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
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
            output = resp.json().get("response", "").strip()
            self._remember(output)
            return output
        except Exception as e:
            return f"[{self.role} LLM error: {e}]"

    def reset_state(self):
        """Clear internal state for a fresh experiment."""
        with self._lock:
            self.internal_state = []
            self.cycle_count = 0

    def __repr__(self):
        return f"<{self.role} agent | {self.model} | {self.cycle_count} cycles>"
