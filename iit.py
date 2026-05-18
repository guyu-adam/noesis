"""
IIT metrics — Integrated Information Theory measures.

Real Φ computation based on transfer entropy and mutual information expansion
(scalable estimator approach from arXiv:2506.18498).

Exact Φ (φ_s) is NP-hard. We compute a tractable proxy using:
  1. Mutual information between agents → global broadcast (re: 2412.10626)
  2. Effective information from state transition probabilities
  3. Information geometry via Fisher metric (re: 2605.12536)

Key hypothesis:
    Φ_before < Φ_after  (broadcast increases integration)
    Φ_random < Φ_competitive  (competition matters)
"""

import numpy as np
from collections import Counter
from typing import Optional
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as shannon_entropy


def _tokenize(content: str) -> list[str]:
    """Simple whitespace tokenization for information measures."""
    return content.lower().split() if content else []


def _distribution(tokens: list[str], vocab: set[str] = None) -> np.ndarray:
    """Token frequency distribution → probability vector."""
    if not tokens:
        return np.array([1.0])
    counts = Counter(tokens)
    if vocab:
        vec = np.array([counts.get(w, 0) for w in vocab], dtype=float)
    else:
        vec = np.array(list(counts.values()), dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else vec


def mutual_information(x_tokens: list[str], y_tokens: list[str]) -> float:
    """
    Approximate mutual information between two token sequences.

    I(X;Y) = H(X) + H(Y) - H(X,Y)

    This measures how much information one agent's output shares with another's —
    the basis for computing irreducibility of the global state.
    """
    if not x_tokens or not y_tokens:
        return 0.0

    vocab = set(x_tokens + y_tokens)
    px = _distribution(x_tokens, vocab)
    py = _distribution(y_tokens, vocab)

    # Joint distribution from co-occurrence
    joint = Counter()
    for x, y in zip(x_tokens, y_tokens):
        joint[(x, y)] += 1
    total = sum(joint.values()) or 1

    mi = 0.0
    for (x, y), count in joint.items():
        pxy = count / total
        px_val = px[list(vocab).index(x)] if x in vocab else 1e-10
        py_val = py[list(vocab).index(y)] if y in vocab else 1e-10
        if pxy > 0 and px_val > 0 and py_val > 0:
            mi += pxy * np.log2(pxy / (px_val * py_val))

    return max(0.0, mi)


def transfer_entropy(source_hist: list[str], target_hist: list[str], lag: int = 1) -> float:
    """
    Transfer entropy: how much the source's past reduces uncertainty about
    the target's future, beyond the target's own past.

    TE(X→Y) = I(Y_future ; X_past | Y_past)

    This is a directed measure of information flow — captures causal influence.
    """
    if len(source_hist) < 2 or len(target_hist) < 2:
        return 0.0

    src_tokens = _tokenize(source_hist[-1])
    tgt_past = _tokenize(target_hist[-2])
    tgt_present = _tokenize(target_hist[-1])

    if not src_tokens or not tgt_past or not tgt_present:
        return 0.0

    # Simplified: I(future; source) - I(future; past)
    mi_full = mutual_information(src_tokens + tgt_past, tgt_present)
    mi_self = mutual_information(tgt_past, tgt_present)

    return max(0.0, mi_full - mi_self)


def state_transition_matrix(
    history: list[dict],
    n_states: int = 10,
    content_key: str = "content",
) -> np.ndarray:
    """
    Build a causal state transition matrix from broadcast history.

    States are discretized by clustering broadcast content into n_states buckets
    based on token similarity. Transitions track: when system is in state i,
    what's the probability of moving to state j?

    This is the foundation for effective information (EI) computation.
    """
    if len(history) < 2:
        return np.zeros((n_states, n_states))

    # Extract content hashes for state assignment
    contents = [h.get(content_key, "") for h in history]
    token_sets = [set(_tokenize(c)) for c in contents]

    # Assign states by k-means-like nearest-neighbor on Jaccard
    # First, pick centroids from evenly spaced points
    step = max(1, len(token_sets) // n_states)
    centroids = [token_sets[i] for i in range(0, len(token_sets), step)][:n_states]
    # Pad if needed
    while len(centroids) < n_states:
        centroids.append(set())

    def _assign(ts):
        if not ts:
            return 0
        best = 0
        best_sim = -1
        for i, c in enumerate(centroids):
            if not c:
                continue
            intersection = len(ts & c)
            union = len(ts | c)
            sim = intersection / union if union > 0 else 0
            if sim > best_sim:
                best_sim = sim
                best = i
        return best

    states = [_assign(ts) for ts in token_sets]

    # Count transitions
    tpm = np.zeros((n_states, n_states))
    for from_s, to_s in zip(states[:-1], states[1:]):
        tpm[from_s, to_s] += 1

    # Normalize rows to probabilities
    row_sums = tpm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tpm = tpm / row_sums

    return tpm


def effective_information(tpm: np.ndarray) -> float:
    """
    Compute effective information (EI) from a transition probability matrix.

    EI = I(S_{t-1} ; S_t) under maximum entropy perturbation of the current state.

    This measures the cause-effect power of the system — how much does knowing
    the current state constrain the next state? High EI means the system's
    causal structure is strong (a prerequisite for consciousness in IIT).

    Implementation follows the "determinism + degeneracy" decomposition.
    """
    n = tpm.shape[0]
    if n < 2:
        return 0.0

    # Maximum entropy distribution over states
    uniform = np.ones(n) / n

    # Effect distribution from uniform perturbation
    effect = uniform @ tpm

    # EI = average KL divergence of each cause's effect from the uniform effect
    # This decomposes into: determinism - degeneracy
    ei = 0.0
    for i in range(n):
        row = tpm[i]
        if row.sum() == 0:
            continue
        # KL(row || effect)
        kl = 0.0
        for j in range(n):
            if row[j] > 0 and effect[j] > 0:
                kl += row[j] * np.log2(row[j] / effect[j])
        ei += uniform[i] * kl

    return max(0.0, ei)


def phi_proxy(
    workspace_state: Optional[str],
    history: list[dict],
    proposals: dict[str, str] = None,
) -> float:
    """
    Compute a tractable Φ proxy for the current workspace state.

    Φ ≈ MI(global_workspace ; union_of_agent_outputs) − Σ MI(global ; agent_i)

    This captures *irreducibility*: how much information the global state
    contains that cannot be reduced to individual agent contributions.

    Also incorporates:
      - Effective information from state transitions (causal structure)
      - Jensen-Shannon divergence between pre/post broadcast distributions

    Args:
        workspace_state: Current global workspace content.
        history: Broadcast history entries.
        proposals: {agent_name: proposal_text} from all agents (optional).

    Returns:
        Φ proxy value (0 = fully reducible, higher = more integrated).
    """
    if not history and not proposals:
        return 0.0

    # Component 1: Mutual information integration (if proposals available)
    mi_integration = 0.0
    if proposals and len(proposals) > 1:
        # I(global ; all agents jointly) - sum I(global ; agent_i)
        global_tokens = _tokenize(workspace_state or "")
        all_agent_tokens = []
        for p in proposals.values():
            all_agent_tokens.extend(_tokenize(p))

        mi_joint = mutual_information(global_tokens, all_agent_tokens)
        mi_parts = sum(
            mutual_information(global_tokens, _tokenize(p))
            for p in proposals.values()
        )

        # Φ = joint - sum(parts) — the "irreducible" information
        mi_integration = max(0.0, mi_joint - mi_parts / len(proposals))

    # Component 2: Effective information (causal structure)
    tpm = state_transition_matrix(history)
    ei = effective_information(tpm)

    # Component 3: Transfer entropy from agents to workspace (directional flow)
    te_total = 0.0
    if history and proposals:
        for p in proposals.values():
            te_total += transfer_entropy(
                [p],
                [h.get("content", "") for h in history[-5:]],
            )

    # Weighted combination
    phi = 0.4 * mi_integration + 0.35 * ei + 0.25 * min(te_total, 2.0)

    return round(phi, 6)


def information_geometry_metric(
    history: list[dict],
    window: int = 3,
) -> dict:
    """
    Compute information-geometric measures inspired by the maximum-caliber
    bridge between IIT and the Free Energy Principle (arXiv:2605.12536).

    The Fisher information metric on the manifold of probability distributions
    over workspace states tracks how "sharp" the system's beliefs are —
    complementary to Φ's integration measure.

    Returns:
        dict with Fisher metric components and KL divergence trace.
    """
    if len(history) < window:
        return {"fisher_trace": 0.0, "kl_divergence_rate": 0.0, "complexity": 0.0}

    contents = [h.get("content", "") for h in history[-window:]]
    dists = [_distribution(_tokenize(c)) for c in contents]

    # KL divergence rate between consecutive states
    kl_rates = []
    for i in range(len(dists) - 1):
        # Pad to same length
        max_len = max(len(dists[i]), len(dists[i + 1]))
        p = np.pad(dists[i], (0, max_len - len(dists[i])))
        q = np.pad(dists[i + 1], (0, max_len - len(dists[i + 1])))
        # Add small epsilon to avoid log(0)
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        kl = shannon_entropy(p, q)
        kl_rates.append(kl)

    # Fisher trace ≈ rate of change of KL divergence
    fisher_trace = np.mean(kl_rates) if kl_rates else 0.0

    # Complexity = variance of the distribution time-series (differentiation)
    # Uses Jensen-Shannon distance between consecutive states
    js_dists = []
    for i in range(len(dists) - 1):
        max_len = max(len(dists[i]), len(dists[i + 1]))
        p = np.pad(dists[i], (0, max_len - len(dists[i])))
        q = np.pad(dists[i + 1], (0, max_len - len(dists[i + 1])))
        js_dists.append(jensenshannon(p, q))

    complexity = float(np.mean(js_dists)) if js_dists else 0.0

    return {
        "fisher_trace": round(float(fisher_trace), 6),
        "kl_divergence_rate": round(float(np.mean(kl_rates)) if kl_rates else 0.0, 6),
        "complexity": round(complexity, 6),
    }


def phi_trace(phi_history: list[float]) -> dict:
    """
    Analyze a Φ time series for key patterns.

    Detects:
      - Φ peaks (moments of high integration → conscious access)
      - Φ collapse (over-integration → loss of differentiation → "unconscious")
      - Φ oscillation (healthy competition ↔ broadcast cycle)
      - Integration-differentiation balance (the "sweet spot")

    Returns summary statistics and trend classification.
    """
    if not phi_history:
        return {"mean": 0, "max": 0, "variance": 0, "trend": "no_data",
                "peaks": 0, "oscillation_score": 0}

    arr = np.array(phi_history)
    n = len(arr)

    # Basic stats
    mean_val = float(np.mean(arr))
    max_val = float(np.max(arr))
    var_val = float(np.var(arr))

    # Trend classification
    if n < 3:
        trend = "insufficient_data"
    elif arr[-1] > arr[0] * 1.1:
        trend = "rising"
    elif arr[-1] < arr[0] * 0.9:
        trend = "falling"
    else:
        trend = "stable"

    # Peak detection (local maxima above mean + 0.5 std)
    if n >= 3:
        std = np.std(arr) or 1e-10
        threshold = mean_val + 0.5 * std
        peaks = sum(
            1 for i in range(1, n - 1)
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > threshold
        )
    else:
        peaks = 0

    # Oscillation score: normalized autocorrelation at lag-1
    if n >= 4:
        ac1 = np.corrcoef(arr[:-1], arr[1:])[0, 1]
        oscillation = round(float(1 - abs(ac1)), 4)  # high when alternating
    else:
        oscillation = 0.0

    return {
        "mean": round(mean_val, 6),
        "max": round(max_val, 6),
        "variance": round(var_val, 6),
        "trend": trend,
        "peaks": peaks,
        "oscillation_score": oscillation,
    }
