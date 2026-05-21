"""
Neural processor base class — recurrent neural network with optional GPU acceleration.

Each processor is a small RNN with real causal structure — a dynamical system
of recurrently connected neurons. The processor's "proposal" is its hidden-state
activation pattern after processing a stimulus. Uses "processor" to align
with GWT's original "specialized processor" terminology (Baars, Dehaene).

This is the foundation for computing genuine Φ (integrated information)
from neural state transition matrices — not token-distribution proxies.

Architecture:
    h_t = tanh(W_in · s + W_rec · h_{t-1} + b)
    output = h_T  (final hidden state after T unroll steps)

Where:
    s ∈ R^{n_input}   — stimulus embedding vector
    h ∈ R^{n_neurons} — hidden state (also the "proposal" output)
    W_in  ∈ R^{n_neurons × n_input}  — input weights
    W_rec ∈ R^{n_neurons × n_neurons} — recurrent weights
    b ∈ R^{n_neurons} — bias

Hardware scaling (auto-detected):
    5060Ti 16GB → 256 neurons/processor default
    <6GB VRAM   → 64 neurons/processor
    No GPU      → CPU fallback via numpy
"""

import os
import threading
import numpy as np
from abc import ABC, abstractmethod

# ── Optional GPU backend ────────────────────────────────────────────────
try:
    import cupy as xp
    _GPU = True
except ImportError:
    xp = np
    _GPU = False


def _array(arr, dtype="float32"):
    """Move array to GPU if available, else keep on CPU."""
    a = np.asarray(arr, dtype=dtype)
    if _GPU:
        if hasattr(a, 'get'):
            return xp.asarray(a, dtype=dtype)
        return xp.asarray(a, dtype=dtype)
    return a


def _to_numpy(arr) -> np.ndarray:
    """Ensure array is on CPU as numpy."""
    if hasattr(arr, 'get'):
        return np.asarray(arr.get(), dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


class NeuralProcessor(ABC):
    """
    Abstract neural processor — RNN with recurrent causal structure.

    Each processor maintains:
      - Fixed recurrent weights (initialized once)
      - Internal state history (for building causal TPMs)
      - Short-term activation memory (primes future processing)

    The recurrent weight matrix W_rec defines the processor's causal structure —
    this is what Φ (effective information) measures.

    Parameters:
        n_neurons: Number of recurrent neurons (default from config, usually 256).
        n_input: Stimulus embedding dimension (default 32).
        n_unroll: Number of recurrent unroll steps per stimulus (default 25).
        noise_std: Gaussian noise added to hidden state per step.
    """

    def __init__(self, role: str, n_neurons: int = None, n_input: int = None,
                 n_unroll: int = None, noise_std: float = None, seed: int = None):
        # Read from env or config, with sensible defaults
        self.role = role
        self.n_neurons = n_neurons or int(os.environ.get("NOESIS_N_NEURONS", "256"))
        self.n_input = n_input or int(os.environ.get("NOESIS_N_INPUT", "32"))
        self.n_unroll = n_unroll or int(os.environ.get("NOESIS_N_UNROLL", "25"))
        self.noise_std = noise_std if noise_std is not None else float(os.environ.get("NOESIS_NOISE", "0.01"))

        rng = np.random.RandomState(seed)
        self.W_in = _array(self._init_input_weights(rng))
        self.W_rec = _array(self._init_recurrent_weights(rng))
        self.b = _array(np.zeros(self.n_neurons))

        # State (keep on CPU for history, move to GPU during processing)
        self.hidden = np.zeros(self.n_neurons, dtype=np.float32)
        self.activation_history: list[np.ndarray] = []
        self.max_history: int = int(os.environ.get("NOESIS_MAX_HISTORY", "500"))
        self.cycle_count: int = 0
        self.proposal = np.zeros(self.n_neurons, dtype=np.float32)
        self._lock = threading.Lock()

    # ── Subclass hooks ────────────────────────────────────────────────────

    def _init_input_weights(self, rng: np.random.RandomState) -> np.ndarray:
        return (rng.randn(self.n_neurons, self.n_input) * 0.1).astype(np.float32)

    def _init_recurrent_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Default: sparse random with spectral radius < 1 (stable dynamics).

        Subclasses override this to create specialized connectivity patterns.
        """
        W = rng.randn(self.n_neurons, self.n_neurons).astype(np.float32) * 0.1
        mask = rng.rand(self.n_neurons, self.n_neurons) < 0.15
        W = W * mask
        # Enforce echo-state property: spectral radius < 1
        eig_max = np.max(np.abs(np.linalg.eigvals(W)))
        if eig_max > 0.95:
            W = W * (0.9 / eig_max)
        return W

    # ── Core processing (GPU-accelerated when available) ─────────────────

    def process(self, stimulus_vec: np.ndarray, context_vec: np.ndarray = None) -> np.ndarray:
        """
        Process a stimulus vector through the recurrent network.

        Uses GPU for matrix multiplication when CuPy is available.
        Falls back to CPU numpy otherwise.

        Args:
            stimulus_vec: Shape (n_input,). Embedding of the current stimulus.
            context_vec: Shape (n_neurons,). Global workspace broadcast from
                         previous cycle (zero if first cycle).

        Returns:
            Proposal activation vector, shape (n_neurons,).
        """
        sv = np.asarray(stimulus_vec, dtype=np.float32).flatten()
        if sv.shape[0] != self.n_input:
            raise ValueError(
                f"Expected stimulus shape ({self.n_input},), got {sv.shape}"
            )

        with self._lock:
            h = self.hidden.copy().astype(np.float32)

            # Inject context from global workspace
            if context_vec is not None and np.any(context_vec):
                cv = np.asarray(context_vec, dtype=np.float32).flatten()[:self.n_neurons]
                if len(cv) < self.n_neurons:
                    cv = np.pad(cv, (0, self.n_neurons - len(cv)))
                # Strong context injection: broadcast causally shapes processor dynamics.
                # Previous weight 0.3 was too weak — broadcast was "write-only",
                # recording history without measurably affecting processor states.
                alpha = float(os.environ.get("NOESIS_CTX_WEIGHT", "0.6"))
                h = (1.0 - alpha) * h + alpha * cv

            # Move to GPU for accelerated recurrence
            h_dev = _array(h)
            s_dev = _array(sv)
            noise = np.random.randn(self.n_unroll, self.n_neurons).astype(np.float32) * self.noise_std

            for step in range(self.n_unroll):
                recurrent = xp.dot(self.W_rec, h_dev)
                external = xp.dot(self.W_in, s_dev)
                h_raw = recurrent + external + self.b + _array(noise[step])
                h_dev = xp.tanh(h_raw)

            # Move result back to CPU for storage
            h = _to_numpy(h_dev)

            self.hidden = h
            self.proposal = h

            self.activation_history.append(h.copy())
            if len(self.activation_history) > self.max_history:
                self.activation_history = self.activation_history[-self.max_history:]

            self.cycle_count += 1

        return h

    def process_batch(self, stimuli: list[np.ndarray],
                      contexts: list[np.ndarray] = None) -> list[np.ndarray]:
        """
        Process multiple stimuli efficiently. Context injection is serial
        (each depends on prior broadcast), but matrix ops are on GPU.

        Returns list of proposal vectors, one per stimulus.
        """
        results = []
        ctx = None
        for i, stim in enumerate(stimuli):
            if contexts and i < len(contexts):
                ctx = contexts[i]
            results.append(self.process(stim, ctx))
        return results

    # ── State management ─────────────────────────────────────────────────

    def read_proposal(self) -> np.ndarray:
        return self.proposal.copy()

    def read_proposal_text(self) -> str:
        """Statistical summary of neural activation pattern."""
        p = self.proposal
        active = int(np.sum(np.abs(p) > 0.3))
        sparsity = 1.0 - active / self.n_neurons
        mean_act = float(np.mean(np.abs(p)))
        std_act = float(np.std(p))
        return (f"[{self.role}] {active}/{self.n_neurons} active "
                f"(sp={sparsity:.2f} μ|a|={mean_act:.3f} σ={std_act:.3f})")

    def get_activation_history(self, n: int = None) -> list[np.ndarray]:
        with self._lock:
            if n is None:
                return list(self.activation_history)
            return self.activation_history[-n:]

    def reset_state(self):
        with self._lock:
            self.hidden = np.zeros(self.n_neurons, dtype=np.float32)
            self.proposal = np.zeros(self.n_neurons, dtype=np.float32)
            self.activation_history = []
            self.cycle_count = 0

    def __repr__(self):
        sr = self._spectral_radius()
        backend = "GPU" if _GPU else "CPU"
        return (f"<NeuralProcessor {self.role} | {self.n_neurons} neurons "
                f"[{backend}] | ρ(W_rec)={sr:.3f} | {self.cycle_count} cycles>")

    def _spectral_radius(self) -> float:
        try:
            W = _to_numpy(self.W_rec)
            return float(np.max(np.abs(np.linalg.eigvals(W))))
        except Exception:
            return float('nan')


# ── Stimulus encoder ──────────────────────────────────────────────────

def encode_stimulus(text: str, dim: int = None, seed: int = 42) -> np.ndarray:
    """
    Encode a text stimulus into a fixed-dim vector for neural processors.

    Uses deterministic hashing + random projection. Same text always
    produces the same vector (reproducible). Different texts produce
    approximately orthogonal vectors.

    This is deliberately simple — the point is NOT semantic richness,
    it's to provide a controlled input to study causal structure.
    """
    dim = dim or int(os.environ.get("NOESIS_N_INPUT", "32"))
    rng = np.random.RandomState(seed)
    tokens = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    import hashlib
    for i, tok in enumerate(tokens):
        tok_hash = int(hashlib.md5(str(tok).encode()).hexdigest(), 16) % 100000
        local_rng = np.random.RandomState(tok_hash)
        vec += local_rng.randn(dim).astype(np.float32) * 0.1 / (i + 1)
    norm = np.linalg.norm(vec)
    if norm > 1e-10:
        vec = vec / norm
    return vec
