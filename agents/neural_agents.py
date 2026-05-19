"""
Specialized neural processors — differentiated by recurrent connectivity patterns.

Each processor type has a distinct W_rec initialization, producing different
dynamical regimes (different causal structures). This is analogous to how
cortical areas have similar microcircuitry but different connectivity.

Processor types and their dynamical signatures:

    Perceptor   — Sparse near-diagonal W_rec. Fast decorrelation, feature extraction.
    Reasoner    — Chain-structured W_rec. Sequential processing stages.
    Evaluator   — Bistable attractor W_rec. Two-pool decision-making dynamics.
    Integrator  — Small-world W_rec. Holistic integration, long-range coupling.
    Predictor   — Forward-skewed W_rec. Anticipatory dynamics, temporal prediction.

The key design principle: specialization emerges from connectivity, not from
prompts. This makes the causal structure real and measurable via Φ.

All initializations are N-independent and scale to arbitrary neuron counts.
"""

import os
import numpy as np
from agents.neural_base import NeuralProcessor


# ── Helper ──────────────────────────────────────────────────────────────

def _stabilize(W: np.ndarray, target_radius: float = 0.9) -> np.ndarray:
    """Scale W so its spectral radius is at most target_radius (echo-state)."""
    eig_max = np.max(np.abs(np.linalg.eigvals(W)))
    if eig_max > target_radius:
        W = W * (target_radius / eig_max)
    return W


# ── Agent classes ────────────────────────────────────────────────────────

class NeuralPerceptor(NeuralProcessor):
    """
    Sensory processing — high-dimensional, rapidly decorrelating dynamics.

    W_rec: sparse near-diagonal with local lateral connections.
    Fast decorrelation — each neuron processes a narrow feature band,
    maximizing differentiation.

    Scales naturally: for N neurons, each neuron connects to ~3-5 neighbors.
    """

    def __init__(self, n_neurons: int = None, n_input: int = None, seed: int = 100):
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        super().__init__("perceptor", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)

        # Self-connections (strong)
        np.fill_diagonal(W, rng.uniform(0.3, 0.7, N).astype(np.float32))

        # Local lateral connections (radius ~3 neurons in each direction)
        radius = max(2, N // 64)
        for i in range(N):
            for offset in range(1, radius + 1):
                j_left = (i - offset) % N
                j_right = (i + offset) % N
                w = rng.uniform(-0.15, 0.15) / offset  # strength decays with distance
                W[i, j_left] = w
                W[i, j_right] = w

        return _stabilize(W)


class NeuralReasoner(NeuralProcessor):
    """
    Logical reasoning — structured sequential dynamics.

    W_rec: chain-like (feedforward bias in the recurrent matrix).
    Information flows through processing stages, producing stepwise
    transformations characteristic of logical inference.

    For N neurons, forms a directed acyclic backbone with skip connections.
    """

    def __init__(self, n_neurons: int = None, n_input: int = None, seed: int = 200):
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        super().__init__("reasoner", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)

        # Forward chain: neuron i receives from recent predecessors
        # The chain depth scales with N
        chain_depth = max(2, N // 32)
        for i in range(1, N):
            # Direct predecessor(s) — strong
            for d in range(1, min(chain_depth, i + 1)):
                strength = 0.5 / d
                W[i, i - d] = rng.uniform(0.1, strength)

            # Skip connections (every ~N/8 steps)
            skip_step = max(1, N // 8)
            if i >= skip_step:
                W[i, i - skip_step] = rng.uniform(0.02, 0.1)

        # Weak self-feedback for temporal smoothing
        np.fill_diagonal(W, rng.uniform(0.05, 0.2, N).astype(np.float32))

        return _stabilize(W)


class NeuralEvaluator(NeuralProcessor):
    """
    Affective/value evaluation — bistable attractor dynamics.

    W_rec: two-pool structure with within-pool excitation and cross-pool
    inhibition. Creates attractor basins (positive/negative evaluation),
    implementing a decision-making circuit characteristic of value judgment.

    Pool sizes are proportional to N, maintaining the bistable ratio.
    """

    def __init__(self, n_neurons: int = None, n_input: int = None, seed: int = 300):
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        super().__init__("evaluator", n_neurons, n_input, seed=seed)
        # Affective bias
        bias_rng = np.random.RandomState(seed + 1)
        self.b = bias_rng.uniform(-0.15, 0.15, n_neurons).astype(np.float32)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)
        half = N // 2

        # Within-pool excitation (dense, positive)
        W[:half, :half] = rng.uniform(0.05, 0.3, (half, half)).astype(np.float32)
        W[half:, half:] = rng.uniform(0.05, 0.3, (half, half)).astype(np.float32)

        # Cross-pool inhibition (negative)
        W[:half, half:] = rng.uniform(-0.25, -0.03, (half, half)).astype(np.float32)
        W[half:, :half] = rng.uniform(-0.25, -0.03, (half, half)).astype(np.float32)

        # Strong self-excitation for hysteresis (bistability)
        np.fill_diagonal(W, rng.uniform(0.4, 0.8, N).astype(np.float32))

        return _stabilize(W)


class NeuralIntegrator(NeuralProcessor):
    """
    Holistic integration — small-world connectivity.

    W_rec: Watts-Strogatz small-world topology. Combines local clustering
    (high within-module integration) with long-range shortcuts (low path
    length). This creates rich, non-local dynamics that are well-suited
    for detecting global patterns across diverse processor outputs.

    Particularly relevant for measuring Φ: small-world networks show
    higher integration-differentiation balance than purely local or random.
    """

    def __init__(self, n_neurons: int = None, n_input: int = None, seed: int = 400):
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        super().__init__("integrator", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        N = self.n_neurons
        k = max(4, N // 16)  # each neuron connects to ~k neighbors (ring)
        p_rewire = 0.1        # 10% of connections rewired as long-range shortcuts

        # Start with ring lattice: each neuron connects to k nearest neighbors
        W = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for offset in range(1, k // 2 + 1):
                j = (i + offset) % N
                w = rng.uniform(0.1, 0.5) * (1.0 - offset / (k / 2 + 1))
                W[i, j] = w
                W[j, i] = w  # symmetric for undirected base

        # Rewire: with probability p_rewire, replace a local edge with a long-range one
        for i in range(N):
            for offset in range(1, k // 2 + 1):
                j = (i + offset) % N
                if rng.random() < p_rewire:
                    # Remove local edge
                    W[i, j] = 0
                    W[j, i] = 0
                    # Add long-range edge to random distant neuron
                    far = rng.randint(0, N - 1)
                    while abs(far - i) <= k or far == i:
                        far = rng.randint(0, N - 1)
                    W[i, far] = rng.uniform(0.1, 0.4)
                    W[far, i] = rng.uniform(0.1, 0.4)

        return _stabilize(W)


class NeuralPredictor(NeuralProcessor):
    """
    Anticipatory prediction — forward-skewed recurrent dynamics.

    W_rec: upper-triangular dominant (forward-skewed). Creates dynamics
    where information preferentially flows from lower-index to higher-index
    neurons. This implements a simple temporal prediction hierarchy:
    early neurons encode current state, later neurons encode predicted future.

    The prediction error (mismatch between predicted and actual next state)
    drives learning and contributes to the world model's predictive coding.
    """

    def __init__(self, n_neurons: int = None, n_input: int = None, seed: int = 500):
        n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        super().__init__("predictor", n_neurons, n_input, seed=seed)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)

        # Forward-skewed: upper triangle is dense, lower triangle is sparse
        for i in range(N):
            for j in range(N):
                if j > i:
                    # Upper triangular — future-directed connections
                    # Strength decays with distance
                    dist = j - i
                    scale = N / 4
                    W[i, j] = rng.randn() * np.exp(-dist / scale) * 0.3
                elif j < i:
                    # Lower triangular — sparse feedback (10% density)
                    if rng.random() < 0.1:
                        W[i, j] = rng.uniform(-0.1, 0.1)

        # Self-connections for temporal integration
        np.fill_diagonal(W, rng.uniform(0.2, 0.5, N).astype(np.float32))

        return _stabilize(W)


class NeuralNarrator:
    """
    Phenomenological report generator for neural processors.

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
          - Φ transition (before -> after broadcast)
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
            f"Phi: {phi_before:.4f}->{phi_after:.4f} (Delta={phi_delta:+.4f}). "
            f"Winner: {winner} ({access}). "
            f"Regime: {regime}."
        )

    def reset_state(self):
        self.cycle_count = 0
