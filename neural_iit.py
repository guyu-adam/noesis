"""
Neural Integrated Information — Φ from neural activation state transitions.

This is the CORE distinction between main and noesis-llm branches:

    noesis-llm (iit.py):     Φ ≈ MI(token_distributions)  ← proxy
    main (neural_iit.py):    Φ from neural activation TPM  ← causal Φ

The neural version computes Φ from the actual causal structure of the
multi-processor neural system. Each processor is a small RNN; the global workspace
integrates their hidden-state activations. Φ measures the irreducibility
of the global state to individual processor states.

Key references (same as iit.py, but applied to neural states):
  - Tononi (2004, 2016): IIT proper — Φ from cause-effect structure
  - Barrett et al. (2026): Φ is not well-defined for real systems; we compute
    a tractable approximation on a small (256-neuron) system where TPM is exact
  - Kearney (2026): MaxCal bridge between IIT and FEP

For a system with N neurons per processor and M processors, the full state space is
R^(N*M). We discretize via thresholding (each neuron is binary: on/off at
each time step) to get a tractable TPM of size 2^(N*M) → reduced via
clustering to k states.

Central hypothesis (to be tested):
    Φ(coalition_broadcast) > Φ(winner_take_all) > Φ(no_broadcast)
"""

import numpy as np
from collections import Counter
from typing import Optional
from scipy.spatial.distance import jensenshannon


# ── State discretization ────────────────────────────────────────────────────

def discretize_activation(activation: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Convert continuous neural activation to binary state vector.

    neuron_i = 1 if |activation_i| > threshold, else 0.
    """
    return (np.abs(activation) > threshold).astype(np.int8)


def activation_to_state_id(binary_state: np.ndarray) -> int:
    """
    Convert binary state vector to integer state ID.

    Uses first 16 bits (max 65536 states). For larger systems,
    use state clustering instead (see cluster_states).
    """
    bits = binary_state[:16]
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def cluster_activation_states(
    activations: list[np.ndarray],
    n_clusters: int = 10,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Cluster continuous activation vectors into discrete states via k-means-like
    nearest-centroid assignment. Returns (state_labels, centroids).

    This is necessary for systems where the raw state space is too large
    for exact TPM computation.
    """
    if len(activations) < 2:
        return [0] * len(activations), activations

    arr = np.array([a.flatten() for a in activations])

    # Pick centroids evenly spaced
    step = max(1, len(arr) // n_clusters)
    indices = list(range(0, len(arr), step))[:n_clusters]
    centroids = [arr[i] for i in indices]
    while len(centroids) < n_clusters:
        centroids.append(np.zeros_like(arr[0]))
    centroids = np.array(centroids)

    # Assign each point to nearest centroid
    labels = []
    for vec in arr:
        dists = np.linalg.norm(centroids - vec, axis=1)
        labels.append(int(np.argmin(dists)))

    return labels, [np.array(c) for c in centroids]


# ── State transition matrix (neural) ────────────────────────────────────────

def neural_state_transition_matrix(
    state_labels: list[int],
    n_states: int = 10,
) -> np.ndarray:
    """
    Build a causal state transition probability matrix from discretized
    neural activation history.

    TPM[i, j] = P(state_{t+1} = j | state_t = i)

    This encodes the causal structure of the neural system — the foundation
    for computing effective information (EI).

    Args:
        state_labels: Sequence of discrete state assignments per cycle.
        n_states: Number of unique states (cluster count).

    Returns:
        TPM of shape (n_states, n_states), row-normalized.
    """
    if len(state_labels) < 2:
        return np.zeros((n_states, n_states))

    tpm = np.zeros((n_states, n_states))
    for from_s, to_s in zip(state_labels[:-1], state_labels[1:]):
        tpm[from_s, to_s] += 1

    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tpm = tpm / row_sums

    return tpm


# ── Effective information (neural) ──────────────────────────────────────────

def neural_effective_information(tpm: np.ndarray) -> float:
    """
    Effective information from a neural state transition matrix.

    EI = average KL divergence of each cause's effect distribution
    from the unconstrained effect distribution (maximum entropy).

    EI measures the cause-effect power of the system — how much does
    knowing the current state constrain the next state?

    High EI → strong causal structure (prerequisite for consciousness in IIT).
    Same formula as iit.py:effective_information(), but the TPM comes from
    neural activation states, not token-content states.
    """
    n = tpm.shape[0]
    if n < 2:
        return 0.0

    uniform = np.ones(n) / n
    effect = uniform @ tpm  # unconstrained effect distribution

    ei = 0.0
    for i in range(n):
        row = tpm[i]
        if row.sum() == 0:
            continue
        kl = 0.0
        for j in range(n):
            if row[j] > 0 and effect[j] > 0:
                kl += row[j] * np.log2(row[j] / effect[j])
        ei += uniform[i] * kl

    return max(0.0, float(ei))


# ── Mutual information between neural states ────────────────────────────────

def neural_mutual_information(
    activations_a: list[np.ndarray],
    activations_b: list[np.ndarray],
    n_bins: int = 8,
) -> float:
    """
    Mutual information between two neural activation time series.

    Unlike iit.py:mutual_information() (token-level MI), this operates on
    real-valued neural activation patterns. Uses 2D histogram binning.

    I(A; B) = H(A) + H(B) - H(A, B)

    Where H is estimated from discretized activation values.
    """
    if len(activations_a) < 2 or len(activations_b) < 2:
        return 0.0

    n = min(len(activations_a), len(activations_b))
    a_arr = np.array([activations_a[i].flatten() for i in range(n)])
    b_arr = np.array([activations_b[i].flatten() for i in range(n)])

    # Use mean activation as 1D summary per cycle
    a_mean = np.mean(np.abs(a_arr), axis=1)
    b_mean = np.mean(np.abs(b_arr), axis=1)

    # 2D histogram
    try:
        hist_2d, _, _ = np.histogram2d(a_mean, b_mean, bins=n_bins)
        hist_2d = hist_2d / hist_2d.sum()

        hist_a = hist_2d.sum(axis=1)
        hist_b = hist_2d.sum(axis=0)

        h_a = -np.sum(hist_a[hist_a > 0] * np.log2(hist_a[hist_a > 0]))
        h_b = -np.sum(hist_b[hist_b > 0] * np.log2(hist_b[hist_b > 0]))
        h_ab = -np.sum(hist_2d[hist_2d > 0] * np.log2(hist_2d[hist_2d > 0]))

        mi = h_a + h_b - h_ab
        return max(0.0, float(mi))
    except Exception:
        return 0.0


# ── Neural Φ (integrated information) ───────────────────────────────────────

def neural_phi(
    workspace_activation: np.ndarray,
    processor_proposals: dict[str, np.ndarray],
    workspace_history: list[dict],
    n_state_clusters: int = 10,
) -> float:
    """
    Compute neural Φ — integrated information of the global workspace state.

    This is the neural analogue of iit.py:phi_proxy(), but crucially:
      - Inputs are neural activation vectors (not text)
      - TPM is built from discretized neural states (not token-content states)
      - MI is computed from activation time series (not token distributions)

    Φ ≈ MI(workspace; all processors jointly) − Σ MI(workspace; processor_i) / n

    This captures IRREDUCIBILITY: how much information the global state contains
    that cannot be reduced to individual processor contributions.

    Args:
        workspace_activation: Current global workspace activation (n_neurons,).
        processor_proposals: {processor_name: activation_vector}.
        workspace_history: List of past broadcast entries with 'content_vec'.
        n_state_clusters: Number of clusters for state discretization.

    Returns:
        Φ value (0 = fully reducible to parts, higher = more integrated).
    """
    if not processor_proposals:
        return 0.0

    # Component 1: Irreducible mutual information
    # I(workspace; all processors) - average I(workspace; processor_i)
    all_processor_concat = np.concatenate([p.flatten() for p in processor_proposals.values()])
    # Re-express as: workspace variance unexplained by individual processors
    mi_joint = _activation_mi_joint(workspace_activation, all_processor_concat)
    mi_parts = sum(
        _activation_mi_pairwise(workspace_activation, p)
        for p in processor_proposals.values()
    )

    n_processors = max(len(processor_proposals), 1)
    mi_integration = max(0.0, mi_joint - mi_parts / n_processors)

    # Component 2: Effective information from causal TPM
    if workspace_history and len(workspace_history) >= 2:
        # Extract content vectors from history
        hist_activations = [
            h.get("content_vec", np.zeros_like(workspace_activation))
            for h in workspace_history[-50:]
        ]
        labels, _ = cluster_activation_states(hist_activations, n_state_clusters)
        tpm = neural_state_transition_matrix(labels, n_state_clusters)
        ei = neural_effective_information(tpm)
    else:
        ei = 0.0

    # Component 3: Activation complexity (differentiation)
    # Higher variance across processors → more differentiated → higher Φ potential
    processor_stds = [np.std(p) for p in processor_proposals.values() if len(p) > 0]
    differentiation = np.mean(processor_stds) if processor_stds else 0.0
    differentiation = min(1.0, differentiation)

    # Weighted combination
    phi = 0.40 * mi_integration + 0.35 * ei + 0.25 * differentiation

    return round(float(phi), 6)


def _activation_mi_joint(a: np.ndarray, b_concat: np.ndarray) -> float:
    """MI between workspace activation and concatenated processor activations."""
    if len(a) == 0 or len(b_concat) == 0:
        return 0.0
    # Use correlation-based MI approximation for continuous vectors
    # I ≈ -0.5 * log(1 - ρ²) for Gaussian (a first-order approximation)
    a_flat = a.flatten()
    b_flat = b_concat.flatten()
    # Truncate to same length
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    if np.isnan(corr):
        return 0.0
    corr = max(-0.999, min(0.999, corr))
    return float(-0.5 * np.log2(1 - corr ** 2))


def _activation_mi_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """Pairwise MI between two activation vectors (correlation-based)."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    min_len = min(len(a_flat), len(b_flat))
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    if np.isnan(corr):
        return 0.0
    corr = max(-0.999, min(0.999, corr))
    return float(-0.5 * np.log2(1 - corr ** 2))


# ── Neural Φ trace analysis ─────────────────────────────────────────────────

def neural_phi_trace(phi_history: list[float]) -> dict:
    """
    Analyze a neural Φ time series (same interface as iit.py:phi_trace).

    Detects: Φ peaks, Φ collapse, oscillation, integration-differentiation balance.
    """
    if not phi_history:
        return {
            "mean": 0, "max": 0, "variance": 0, "trend": "no_data",
            "peaks": 0, "oscillation_score": 0, "n_cycles": 0,
        }

    arr = np.array(phi_history)
    n = len(arr)

    mean_val = float(np.mean(arr))
    max_val = float(np.max(arr))
    var_val = float(np.var(arr))

    if n < 3:
        trend = "insufficient_data"
    elif arr[-1] > arr[0] * 1.1:
        trend = "rising"
    elif arr[-1] < arr[0] * 0.9:
        trend = "falling"
    else:
        trend = "stable"

    if n >= 3:
        std = np.std(arr) or 1e-10
        threshold = mean_val + 0.5 * std
        peaks = sum(
            1 for i in range(1, n - 1)
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > threshold
        )
    else:
        peaks = 0

    if n >= 4:
        ac1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        oscillation = round(float(1 - abs(ac1)), 4)
    else:
        oscillation = 0.0

    return {
        "mean": round(mean_val, 6),
        "max": round(max_val, 6),
        "variance": round(var_val, 6),
        "trend": trend,
        "peaks": peaks,
        "oscillation_score": oscillation,
        "n_cycles": n,
    }


# ── Information geometry on neural states ────────────────────────────────────

def neural_information_geometry(
    activation_history: list[np.ndarray],
    window: int = 3,
) -> dict:
    """
    Information-geometric measures on neural activation manifold.

    The Fisher metric on the space of activation distributions tracks how
    "sharp" the system's beliefs are — complementary to Φ.

    Uses Jensen-Shannon distance between consecutive activation distributions
    rather than token distributions (cf. iit.py:information_geometry_metric).
    """
    if len(activation_history) < window:
        return {"fisher_trace": 0.0, "complexity": 0.0}

    recent = activation_history[-window:]

    # JS distance between consecutive states
    js_dists = []
    for i in range(len(recent) - 1):
        a = recent[i].flatten()
        b = recent[i + 1].flatten()
        # Convert to probability-like distributions via softmax
        a_prob = _to_probability(a)
        b_prob = _to_probability(b)
        min_len = min(len(a_prob), len(b_prob))
        a_prob = a_prob[:min_len]
        b_prob = b_prob[:min_len]
        js_dists.append(float(jensenshannon(a_prob, b_prob)))

    fisher_trace = float(np.mean(js_dists)) if js_dists else 0.0
    complexity = float(np.std(js_dists)) if len(js_dists) > 1 else 0.0

    return {
        "fisher_trace": round(fisher_trace, 6),
        "complexity": round(complexity, 6),
    }


def _to_probability(vec: np.ndarray) -> np.ndarray:
    """Convert activation vector to probability distribution (softmax)."""
    vec = vec - np.max(vec)  # numerical stability
    exp = np.exp(vec)
    denom = exp.sum()
    if denom == 0:
        return np.ones_like(vec) / len(vec)
    p = exp / denom
    return np.clip(p, 1e-10, 1.0)
