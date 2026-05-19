"""
Noesis agents — specialized cognitive modules.

Two agent backends:

  LLM agents (noesis-llm branch):
    - BaseAgent (agents/base.py) — Ollama-backed, text proposals
    - Perceptor, Reasoner, Evaluator, Narrator — prompt-specialized LLM roles
    - Φ proxy from token-distribution MI (iit.py)

  Neural agents (main branch):
    - NeuralAgent (agents/neural_base.py) — small RNN, activation proposals
    - NeuralPerceptor, NeuralReasoner, NeuralEvaluator — connectivity-specialized
    - Φ from neural activation TPM (neural_iit.py)

Agent roles map to GWT's "specialized processors":
  - Perceptor  → sensory processing / input analysis
  - Reasoner   → logical analysis / deduction
  - Evaluator  → affective/value assessment
  - Narrator   → first-person phenomenological report generation
"""

from agents.base import BaseAgent
from agents.neural_base import NeuralAgent

__all__ = ["BaseAgent", "NeuralAgent"]
