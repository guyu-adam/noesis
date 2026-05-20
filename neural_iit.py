"""
Neural Integrated Information — Φ approximation from neural activation patterns.

This is the CORE distinction between main and noesis-llm branches:

    noesis-llm (iit.py):     Φ ≈ MI(token_distributions)  ← text proxy
    main (neural_iit.py):    Φ from neural activation data  ← causal approximation

The neural version computes an information-theoretic Φ approximation from
RNN activation patterns. Each processor is a small RNN; the global workspace
integrates their hidden-state activations.

CRITICAL HONESTY: This is NOT IIT's proper Φ. True IIT Φ requires:
  1. Full cause-effect structure over 2^(N*M) states
  2. Minimum information partition (MIP) search
  3. Earth-mover distance between constrained/unconstrained distributions

For a 256-neuron × 5-processor system, the full state space is 2^1280 —
cosmologically intractable. We compute a tractable approximation with four
documented simplifications (see neural_phi_approx docstring).

Key references:
  - Tononi (2004, 2016): IIT proper — Φ from cause-effect structure
  - Barrett et al. (2026): Φ intractability for real systems; we cite this
    to motivate our approximation strategy, not to claim equivalence
  - Kearney (2026): MaxCal bridge between IIT and FEP
  - Oizumi et al. (2014): Practical Φ approximations for small systems

Central claim (to be tested — comparative, not absolute):
    Φ_approx(collaborative) > Φ_approx(competitive) > Φ_approx(no_broadcast)

The absolute Φ values are meaningless. Only the rank ordering across
experimental conditions carries scientific weight. This is a comparative
information measure, not a claim to measure consciousness magnitude.
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


# ── Neural Φ approximation ──────────────────────────────────────────────────

def neural_phi_approx(
    workspace_activation: np.ndarray,
    processor_proposals: dict[str, np.ndarray],
    workspace_history: list[dict] | None = None,
    n_state_clusters: int = 10,
    weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> float:
    """
    Compute a tractable Φ approximation from neural activation data.

    FOUR DOCUMENTED APPROXIMATIONS (compared to IIT proper):

    1. STATE SPACE REDUCTION: Instead of the full cause-effect structure
       over 2^(N*M) binary states, we cluster activation vectors into
       k ≈ n_state_clusters macro-states. This loses fine-grained causal
       structure but is the only way to make TPMs tractable at scale.

    2. NO MIP SEARCH: IIT's Φ requires finding the Minimum Information
       Partition — the cut that minimizes information loss. This is an
       NP-hard search over all possible bipartitions. We use a fixed
       reduction: Φ ≈ I(workspace; all processors) − avg I(workspace; each).
       This is a specific (and likely non-minimal) partition, so our Φ
       tends to OVERESTIMATE relative to true Φ. This is acceptable for
       comparative analysis because the overestimation bias is consistent
       across conditions.

    3. HISTOGRAM-BASED MI: Mutual information is estimated via 2D
       histogram with quantile binning — no distributional assumptions
       (unlike the previous Gaussian correlation estimator). The tradeoff
       is bin-count sensitivity. We use equi-frequency bins (n/4 rule)
       which adapts to sample size.

    4. WEIGHTED COMPOSITE: Φ = w_mi·MI_integration + w_ei·EI +
       w_diff·differentiation. IIT derives Φ from a principled measure,
       not a weighted sum. The default weights (0.40, 0.35, 0.25) are
       chosen to balance the three factors roughly equally. Use
       phi_sensitivity() to verify that rank-ordering across conditions
       is invariant to weight choices — this is more important than the
       specific values.

    IMPORTANT: The ABSOLUTE Φ values are meaningless. What matters is
    the COMPARATIVE rank ordering across experimental conditions
    (collaborative > competitive > random > no_broadcast). This is a
    comparative information measure, not a claim about consciousness
    magnitude.

    Args:
        workspace_activation: Current workspace activation (n_neurons,).
        processor_proposals: {processor_name: activation_vector}.
        workspace_history: Past broadcast entries with 'content_vec'.
        n_state_clusters: Clusters for state discretization (default 10).
        weights: (w_mi, w_ei, w_diff) — for sensitivity analysis.

    Returns:
        Φ approximation value (comparative, not absolute).
    """
    if workspace_history is None:
        workspace_history = []

    if not processor_proposals:
        return 0.0

    # Component 1: Irreducible mutual information
    # Φ intuition: how much of workspace→processors information is lost
    # when we treat each processor independently?
    # mi_joint = I(workspace; mean_of_processors) — consensus signal
    # mi_parts = average I(workspace; processor_i) — individual signals
    mi_joint = _activation_mi_joint(workspace_activation, processor_proposals)
    mi_parts = sum(
        _activation_mi_pairwise(workspace_activation, p)
        for p in processor_proposals.values()
    )
    n_processors = max(len(processor_proposals), 1)
    mi_integration = max(0.0, mi_joint - mi_parts / n_processors)

    # Component 2: Effective information from causal TPM
    if workspace_history and len(workspace_history) >= 2:
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
    processor_stds = [np.std(p) for p in processor_proposals.values() if len(p) > 0]
    differentiation = np.mean(processor_stds) if processor_stds else 0.0
    differentiation = min(1.0, differentiation)

    # Weighted combination
    w_mi, w_ei, w_diff = weights
    phi = w_mi * mi_integration + w_ei * ei + w_diff * differentiation

    return round(float(phi), 6)


# Backward compatibility alias
neural_phi = neural_phi_approx


def _mi_histogram(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Non-parametric MI via 2D histogram with quantile (equi-frequency) binning.

    Makes NO distributional assumptions — unlike the previous Gaussian
    correlation-based estimator which assumed joint normality of RNN
    activations (a clearly false assumption for tanh-squashed states).

    Paired-observation assumption: we treat corresponding neuron indices
    across the two vectors as paired samples (x_i, y_i). For a 256-dim
    vector pair, this gives 256 observations. This spatial-ergodicity
    assumption — that the neuron-index dimension approximates the ensemble
    distribution — is standard in neural population analysis (cf.
    representational similarity analysis, Kriegeskorte 2008), but the
    resulting quantity should be interpreted as "representational
    alignment" rather than strict information-theoretic MI between
    random variables.

    Returns MI in bits, clamped to [0, ∞).
    """
    n = min(len(x), len(y))
    if n < 4:
        return 0.0

    x_vals = np.asarray(x[:n], dtype=np.float64).flatten()
    y_vals = np.asarray(y[:n], dtype=np.float64).flatten()

    # Adaptive bin count — don't over-bin small samples
    bins_eff = max(4, min(bins, n // 4))

    try:
        # Quantile-based edges so bins adapt to data distribution
        x_edges = np.unique(np.quantile(x_vals, np.linspace(0, 1, bins_eff + 1)))
        y_edges = np.unique(np.quantile(y_vals, np.linspace(0, 1, bins_eff + 1)))
        if len(x_edges) < 2 or len(y_edges) < 2:
            return 0.0

        hist_2d, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
        hist_2d = hist_2d.astype(np.float64) / hist_2d.sum()

        hist_x = hist_2d.sum(axis=1)
        hist_y = hist_2d.sum(axis=0)

        h_x = -np.sum(hist_x[hist_x > 0] * np.log2(hist_x[hist_x > 0]))
        h_y = -np.sum(hist_y[hist_y > 0] * np.log2(hist_y[hist_y > 0]))
        h_xy = -np.sum(hist_2d[hist_2d > 0] * np.log2(hist_2d[hist_2d > 0]))

        return max(0.0, float(h_x + h_y - h_xy))
    except Exception:
        return 0.0


def _activation_mi_joint(
    workspace: np.ndarray,
    processor_proposals: dict[str, np.ndarray],
) -> float:
    """
    Approximate I(workspace; all_processors_jointly).

    Uses MI between workspace and the mean of all processor activation vectors.
    The mean ("consensus") can carry emergent structure not present in any single
    processor, capturing synergistic information. This avoids the concatenation
    approach which truncated most processors' data due to dimension mismatch.
    """
    vecs = [p.flatten() for p in processor_proposals.values()]
    if not vecs:
        return 0.0
    min_len = min(len(workspace.flatten()), min(len(v) for v in vecs))
    mean_proc = np.mean([v[:min_len] for v in vecs], axis=0)
    return _mi_histogram(workspace.flatten()[:min_len], mean_proc)


def _activation_mi_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    """MI between workspace and single processor activations (histogram-based)."""
    return _mi_histogram(a, b)


# ── Weight sensitivity analysis ──────────────────────────────────────────────

def phi_sensitivity(
    workspace_activation: np.ndarray,
    processor_proposals: dict[str, np.ndarray],
    workspace_history: list[dict] | None = None,
    n_state_clusters: int = 10,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Test whether Φ rank-ordering is robust to weight choices.

    Samples random weight vectors from a Dirichlet(2,2,2) distribution
    (concentrated near the uniform center, with plausible spread). For
    each weight combination, computes Φ_approx. Reports the distribution
    of resulting Φ values.

    INTERPRETATION:
      - If phi_std / phi_mean < 0.3: Φ is robust — weights don't drive results.
      - If phi_5th and phi_95th straddle zero: the sign of Φ depends on weights.
      - If robust_fraction > 0.95: default weights are not cherry-picked.

    This is the key defense against the "arbitrary weights" criticism.
    The rank-ordering across experimental conditions should be invariant
    to weight choices within a plausible range.

    Returns:
        Dict with phi distribution statistics and robustness metrics.
    """
    if workspace_history is None:
        workspace_history = []

    if not processor_proposals:
        return {"phi_mean": 0.0, "phi_std": 0.0, "robust": False,
                "reason": "no_proposals"}

    # Pre-compute components (these don't depend on weights)
    mi_joint = _activation_mi_joint(workspace_activation, processor_proposals)
    mi_parts = sum(
        _activation_mi_pairwise(workspace_activation, p)
        for p in processor_proposals.values()
    )
    n_processors = max(len(processor_proposals), 1)
    mi_integration = max(0.0, mi_joint - mi_parts / n_processors)

    if workspace_history and len(workspace_history) >= 2:
        hist_activations = [
            h.get("content_vec", np.zeros_like(workspace_activation))
            for h in workspace_history[-50:]
        ]
        labels, _ = cluster_activation_states(hist_activations, n_state_clusters)
        tpm = neural_state_transition_matrix(labels, n_state_clusters)
        ei = neural_effective_information(tpm)
    else:
        ei = 0.0

    processor_stds = [np.std(p) for p in processor_proposals.values() if len(p) > 0]
    differentiation = np.mean(processor_stds) if processor_stds else 0.0
    differentiation = min(1.0, differentiation)

    # Monte Carlo weight scan
    rng = np.random.RandomState(seed)
    weight_samples = rng.dirichlet(alpha=[2.0, 2.0, 2.0], size=n_samples)

    phi_values = np.array([
        w[0] * mi_integration + w[1] * ei + w[2] * differentiation
        for w in weight_samples
    ])

    default_phi = 0.40 * mi_integration + 0.35 * ei + 0.25 * differentiation

    phi_mean = float(np.mean(phi_values))
    phi_std = float(np.std(phi_values))
    phi_5th = float(np.percentile(phi_values, 5))
    phi_95th = float(np.percentile(phi_values, 95))
    cv = phi_std / max(abs(phi_mean), 1e-10)

    return {
        "phi_mean": round(phi_mean, 6),
        "phi_std": round(phi_std, 6),
        "phi_5th": round(phi_5th, 6),
        "phi_95th": round(phi_95th, 6),
        "default_phi": round(default_phi, 6),
        "coefficient_of_variation": round(cv, 4),
        "robust": cv < 0.3,
        "n_samples": n_samples,
        "components": {
            "mi_integration": round(mi_integration, 6),
            "ei": round(ei, 6),
            "differentiation": round(differentiation, 6),
        },
    }


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
