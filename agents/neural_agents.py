"""
Specialized neural agents — differentiated by recurrent connectivity patterns.

Each agent type has a distinct W_rec initialization, producing different
dynamical regimes (different causal structures). This is analogous to how
cortical areas have similar microcircuitry but different connectivity.

Agent types and their dynamical signatures:

    Perceptor  — High dimensional, rapidly decorrelating.
                 Extracts features. W_rec: sparse, near-diagonal (local processing).

    Reasoner   — Structured sequential dynamics.
                 Chain-like W_rec for stepwise computation.

    Evaluator  — Bistable attractor dynamics.
                 W_rec with positive self-loops + lateral inhibition (decision-making).

The key design principle: specialization emerges from connectivity, not from
prompts. This makes the causal structure real and measurable via Φ.
"""

import numpy as np
from agents.neural_base import NeuralAgent


class NeuralPerceptor(NeuralAgent):
    """
    Sensory processing agent — high-dimensional, rapidly decorrelating dynamics.

    W_rec: near-diagonal with local lateral connections. This produces
    fast decorrelation — each neuron processes a narrow feature band,
    maximizing differentiation (high information, low initial integration).
    """

    def __init__(self, n_neurons: int = 32, n_input: int = 16, seed: int = 100):
        super().__init__("perceptor", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Near-diagonal W_rec: local processing, fast decorrelation."""
        W = np.zeros((self.n_neurons, self.n_neurons))
        # Strong self-connections
        np.fill_diagonal(W, rng.uniform(0.3, 0.7, self.n_neurons))
        # Weak local lateral connections (nearest-neighbor ring)
        for i in range(self.n_neurons):
            W[i, (i - 1) % self.n_neurons] = rng.uniform(-0.1, 0.1)
            W[i, (i + 1) % self.n_neurons] = rng.uniform(-0.1, 0.1)
        # Scale to stable regime
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0.95:
            W = W * (0.9 / eig_max)
        return W


class NeuralReasoner(NeuralAgent):
    """
    Logical reasoning agent — structured sequential dynamics.

    W_rec: chain-like (feedforward bias in the recurrent matrix).
    Information flows through a sequence of "processing stages,"
    producing stepwise transformations characteristic of logical inference.
    """

    def __init__(self, n_neurons: int = 32, n_input: int = 16, seed: int = 200):
        super().__init__("reasoner", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Chain-structured W_rec: sequential processing stages."""
        W = np.zeros((self.n_neurons, self.n_neurons))
        # Forward chain: neuron i receives from i-1, i-2
        for i in range(1, self.n_neurons):
            W[i, i - 1] = rng.uniform(0.2, 0.6)           # direct predecessor
            if i >= 2:
                W[i, i - 2] = rng.uniform(0.05, 0.25)     # skip connection
        # Weak self-feedback
        np.fill_diagonal(W, rng.uniform(0.1, 0.3, self.n_neurons))
        # Enforce spectral radius < 1
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0.95:
            W = W * (0.9 / eig_max)
        return W


class NeuralEvaluator(NeuralAgent):
    """
    Affective/value evaluation agent — bistable attractor dynamics.

    W_rec: positive self-loops + lateral inhibition. Creates two
    attractor basins (positive/negative evaluation), implementing a
    simple decision-making circuit characteristic of value judgment.
    """

    def __init__(self, n_neurons: int = 32, n_input: int = 16, seed: int = 300):
        super().__init__("evaluator", n_neurons, n_input, seed=seed)
        # Affective bias: slight positive/negative shift in bias terms
        self.b = rng = np.random.RandomState(seed + 1)
        self.b = rng.uniform(-0.15, 0.15, n_neurons)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Bistable W_rec: self-excitation + lateral inhibition."""
        W = np.zeros((self.n_neurons, self.n_neurons))
        # Two pools: first half "positive", second half "negative"
        half = self.n_neurons // 2
        # Within-pool excitation
        W[:half, :half] = rng.uniform(0.1, 0.4, (half, half))
        W[half:, half:] = rng.uniform(0.1, 0.4, (half, half))
        # Cross-pool inhibition
        W[:half, half:] = rng.uniform(-0.3, -0.05, (half, half))
        W[half:, :half] = rng.uniform(-0.3, -0.05, (half, half))
        # Strong self-excitation
        np.fill_diagonal(W, rng.uniform(0.4, 0.8, self.n_neurons))
        # Stabilize
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0.95:
            W = W * (0.9 / eig_max)
        return W


class NeuralNarrator:
    """
    Phenomenological report generator for neural agents.

    Unlike the LLM-based Narrator (which generates text via Ollama),
    NeuralNarrator produces structured summaries of the system's state
    from activation patterns — describing what the system "experiences"
    in terms of its own dynamical regime.
    """

    def __init__(self):
        self.cycle_count: int = 0

    def generate(self, workspace_activation: np.ndarray,
                 phi_before: float, phi_after: float,
                 winner: str, broadcasted: bool) -> str:
        """
        Generate a phenomenological report from neural activation patterns.

        The report describes:
          - Global workspace state (sparsity, energy, dominant mode)
          - Φ transition (before → after broadcast)
          - Winner and broadcast status
        """
        w = workspace_activation
        n_neurons = len(w)
        active = int(np.sum(np.abs(w) > 0.5))
        sparsity = 1.0 - active / n_neurons
        energy = float(np.sum(w ** 2))
        dominant_dim = int(np.argmax(np.abs(w)))

        phi_delta = phi_after - phi_before
        if broadcasted:
            access = "broadcasted" if phi_delta > 0.01 else "weakly broadcast"
        else:
            access = "suppressed"

        regime = (
            "high-integration" if phi_after > 0.5 and sparsity < 0.7
            else "differentiated" if sparsity > 0.7
            else "low-activity"
        )

        return (
            f"[NeuralNarrator cycle {self.cycle_count}] "
            f"Workspace: {active}/{n_neurons} active (sparsity={sparsity:.2f}), "
            f"energy={energy:.2f}, dominant dim={dominant_dim}. "
            f"Φ: {phi_before:.4f}→{phi_after:.4f} (Δ={phi_delta:+.4f}). "
            f"Winner: {winner} ({access}). "
            f"Regime: {regime}."
        )

    def reset_state(self):
        self.cycle_count = 0
