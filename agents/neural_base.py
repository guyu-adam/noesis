"""
Neural agent base class — small recurrent neural network.

Unlike the LLM-based BaseAgent (which calls Ollama for text generation),
NeuralAgent uses a small RNN with real causal structure. Each agent is
a dynamical system with 32 recurrently connected neurons. The agent's
"proposal" is its hidden-state activation pattern after processing a stimulus.

This is the foundation for computing genuine Φ (integrated information)
from neural state transition matrices — not token-distribution proxies.

Architecture:
    h_t = tanh(W_in · s + W_rec · h_{t-1} + b)
    output = h_T  (final hidden state after T unroll steps)

Where:
    s ∈ R^{16}   — stimulus embedding vector
    h ∈ R^{32}   — hidden state (also the "proposal" output)
    W_in  ∈ R^{32×16}  — input weights
    W_rec ∈ R^{32×32}  — recurrent weights
    b ∈ R^{32}   — bias
"""

import numpy as np
import threading
from abc import ABC, abstractmethod


class NeuralAgent(ABC):
    """
    Abstract neural agent — small RNN with recurrent causal structure.

    Each agent maintains:
      - Fixed recurrent weights (initialized once, evolve through Hebbian learning)
      - Internal state history (for building causal TPMs)
      - Short-term activation memory (primes future processing)

    The recurrent weight matrix W_rec defines the agent's causal structure —
    this is what Φ (effective information) measures. Different agent types
    have different W_rec initialization patterns, producing different
    cognitive "styles" (specialization).

    Parameters:
        n_neurons: Number of recurrent neurons (default 32).
        n_input: Stimulus embedding dimension (default 16).
        n_unroll: Number of recurrent unroll steps per stimulus (default 10).
        noise_std: Gaussian noise added to hidden state per step (default 0.01).
    """

    def __init__(self, role: str, n_neurons: int = 32, n_input: int = 16,
                 n_unroll: int = 10, noise_std: float = 0.01, seed: int = None):
        self.role = role
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.n_unroll = n_unroll
        self.noise_std = noise_std

        rng = np.random.RandomState(seed)
        self.W_in = self._init_input_weights(rng)
        self.W_rec = self._init_recurrent_weights(rng)
        self.b = np.zeros(n_neurons)

        # State
        self.hidden = np.zeros(n_neurons)            # current hidden state
        self.activation_history: list[np.ndarray] = []  # past hidden states (for TPM)
        self.max_history: int = 200
        self.cycle_count: int = 0
        self.proposal: np.ndarray = np.zeros(n_neurons)
        self._lock = threading.Lock()

    # ── Subclass hooks for specialized initialization ──────────────────────

    def _init_input_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Override in subclass for specialized input connectivity."""
        return rng.randn(self.n_neurons, self.n_input) * 0.1

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Override in subclass for specialized recurrent connectivity.

        Default: sparse random with spectral radius < 1 (stable dynamics).
        """
        W = rng.randn(self.n_neurons, self.n_neurons) * 0.1
        # Enforce sparsity: keep only 20% of connections
        mask = rng.rand(self.n_neurons, self.n_neurons) < 0.2
        W = W * mask
        # Scale to ensure spectral radius < 1 (echo-state property)
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0.95:
            W = W * (0.9 / eig_max)
        return W

    # ── Core processing ────────────────────────────────────────────────────

    def process(self, stimulus_vec: np.ndarray, context_vec: np.ndarray = None) -> np.ndarray:
        """
        Process a stimulus vector through the recurrent network.

        Args:
            stimulus_vec: Shape (n_input,). Embedding of the current stimulus.
            context_vec: Shape (n_neurons,). Global workspace broadcast from
                         previous cycle (zero if first cycle).

        Returns:
            Proposal activation vector, shape (n_neurons,).
        """
        if stimulus_vec.shape != (self.n_input,):
            raise ValueError(
                f"Expected stimulus shape ({self.n_input},), got {stimulus_vec.shape}"
            )

        with self._lock:
            h = self.hidden.copy()

            # Inject context from global workspace (if available)
            if context_vec is not None and np.any(context_vec):
                h = 0.7 * h + 0.3 * context_vec

            # Recurrent unrolling
            for _ in range(self.n_unroll):
                recurrent_input = self.W_rec @ h
                external_input = self.W_in @ stimulus_vec
                h_raw = recurrent_input + external_input + self.b
                # Gaussian noise for stochasticity
                h_raw += np.random.randn(self.n_neurons) * self.noise_std
                h = np.tanh(h_raw)

            self.hidden = h
            self.proposal = h

            # Store in activation history (discretized for TPM)
            self.activation_history.append(h.copy())
            if len(self.activation_history) > self.max_history:
                self.activation_history = self.activation_history[-self.max_history:]

            self.cycle_count += 1

        return h

    # ── State management ───────────────────────────────────────────────────

    def read_proposal(self) -> np.ndarray:
        """Return current proposal activation (not text, but a neural pattern)."""
        return self.proposal.copy()

    def read_proposal_text(self, vocab_map: dict = None) -> str:
        """
        Convert neural activation to human-readable summary.

        Without a vocab_map, returns statistical descriptors of the activation pattern.
        This is used by the Narrator to generate phenomenological reports.
        """
        p = self.proposal
        active_neurons = int(np.sum(np.abs(p) > 0.5))
        sparsity = 1.0 - active_neurons / self.n_neurons
        mean_act = float(np.mean(np.abs(p)))
        return (f"[{self.role}] {active_neurons}/{self.n_neurons} active, "
                f"sparsity={sparsity:.2f}, mean|act|={mean_act:.3f}")

    def get_activation_history(self, n: int = None) -> list[np.ndarray]:
        """Return recent activation history for Φ computation."""
        with self._lock:
            if n is None:
                return list(self.activation_history)
            return self.activation_history[-n:]

    def reset_state(self):
        """Reset internal state for a fresh experiment."""
        with self._lock:
            self.hidden = np.zeros(self.n_neurons)
            self.proposal = np.zeros(self.n_neurons)
            self.activation_history = []
            self.cycle_count = 0

    def __repr__(self):
        return (f"<NeuralAgent {self.role} | {self.n_neurons} neurons | "
                f"{self.cycle_count} cycles | spectral_radius={self._spectral_radius():.3f}>")

    def _spectral_radius(self) -> float:
        """Spectral radius of W_rec — controls dynamical stability."""
        try:
            return float(np.max(np.abs(np.linalg.eigvals(self.W_rec))))
        except Exception:
            return float('nan')


# ── Stimulus encoder ────────────────────────────────────────────────────────

def encode_stimulus(text: str, dim: int = 16, seed: int = 42) -> np.ndarray:
    """
    Encode a text stimulus into a fixed-dim vector for neural agents.

    Uses deterministic hashing + random projection. Same text always
    produces the same vector (reproducible). Different texts produce
    approximately orthogonal vectors.

    This is deliberately simple — the point is NOT semantic richness,
    it's to provide a controlled input to study causal structure.
    """
    rng = np.random.RandomState(seed)
    # Hash-based embedding
    tokens = text.lower().split()
    vec = np.zeros(dim)
    for i, tok in enumerate(tokens):
        tok_hash = hash(tok) % 100000
        local_rng = np.random.RandomState(tok_hash)
        vec += local_rng.randn(dim) * 0.1 / (i + 1)
    # Normalize to unit norm
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    return vec
