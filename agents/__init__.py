"""
Noesis agents — specialized cognitive modules.

Two agent backends:

  LLM agents (noesis-llm branch):
    - BaseAgent (agents/base.py) — Ollama-backed, text proposals
    - Perceptor, Reasoner, Evaluator, Narrator — prompt-specialized LLM roles

  Neural agents (main branch):
    - NeuralAgent (agents/neural_base.py) — RNN with GPU acceleration
    - NeuralPerceptor, NeuralReasoner, NeuralEvaluator — core agents
    - NeuralIntegrator, NeuralPredictor — extended agents
    - Φ from neural activation TPM (neural_iit.py)

Agent roles:
  - Perceptor   → sensory processing / input analysis (near-diagonal W_rec)
  - Reasoner    → logical analysis / deduction (chain W_rec)
  - Evaluator   → affective/value assessment (bistable W_rec)
  - Integrator  → holistic pattern detection (small-world W_rec)
  - Predictor   → temporal anticipation (forward-skewed W_rec)
  - Narrator    → first-person phenomenological report generation
"""

from agents.base import BaseAgent
from agents.neural_base import NeuralAgent

__all__ = ["BaseAgent", "NeuralAgent"]
