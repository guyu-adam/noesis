"""
Noesis processors — specialized cognitive modules.

Two backends:

  LLM agents (noesis-llm branch):
    - BaseAgent (agents/base.py) — Ollama-backed, text proposals
    - Perceptor, Reasoner, Evaluator, Narrator — prompt-specialized LLM roles

  Neural processors (main branch):
    - NeuralProcessor (agents/neural_base.py) — RNN with GPU acceleration
    - NeuralPerceptor, NeuralReasoner, NeuralEvaluator — core processors
    - NeuralIntegrator, NeuralPredictor — extended processors
    - Phi from neural activation TPM (neural_iit.py)

Processor roles (aligned with GWT "specialized processors"):
  - Perceptor   → sensory processing / input analysis (near-diagonal W_rec)
  - Reasoner    → logical analysis / deduction (chain W_rec)
  - Evaluator   → affective/value assessment (bistable W_rec)
  - Integrator  → holistic pattern detection (small-world W_rec)
  - Predictor   → temporal anticipation (forward-skewed W_rec)
  - Narrator    → first-person phenomenological report generation
"""

from agents.base import BaseAgent
from agents.neural_base import NeuralProcessor

__all__ = ["BaseAgent", "NeuralProcessor"]
