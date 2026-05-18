"""
Noesis agents — specialized cognitive modules.

Each agent is an LLM instance (Ollama) with a distinct cognitive role.
They process the same stimulus in parallel, producing competing proposals
for conscious access (global workspace broadcast).

Agent roles map to GWT's "specialized processors":
  - Perceptor  → sensory processing / input parsing
  - Reasoner   → logical analysis / deduction
  - Evaluator  → affective/value assessment
  - Narrator   → first-person phenomenological report generation
"""

from agents.base import BaseAgent

__all__ = ["BaseAgent"]
