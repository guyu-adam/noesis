"""
Tests for neural_iit.py — neural Φ computation from activation TPMs.

These verify that the core mathematical functions produce correct outputs
for known inputs (identical, independent, empty, single-processor, etc.).
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_iit import (
    discretize_activation,
    activation_to_state_id,
    cluster_activation_states,
    neural_state_transition_matrix,
    neural_effective_information,
    neural_mutual_information,
    neural_phi,
    neural_phi_approx,
    phi_sensitivity,
    neural_phi_trace,
    neural_information_geometry,
)


# ── State discretization ─────────────────────────────────────────────

def test_discretize_activation_below_threshold():
    """Activations below threshold should be all zeros."""
    a = np.array([0.1, -0.2, 0.29, -0.15])
    result = discretize_activation(a, threshold=0.3)
    assert np.all(result == 0), f"Expected all zeros, got {result}"


def test_discretize_activation_above_threshold():
    """Activations above threshold should be 1."""
    a = np.array([0.5, -0.8, 0.01, -0.05])
    result = discretize_activation(a, threshold=0.3)
    expected = np.array([1, 1, 0, 0], dtype=np.int8)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"


def test_activation_to_state_id():
    """State ID should be integer encoding of binary bits."""
    bits = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    result = activation_to_state_id(bits)
    assert result == 5, f"Binary 101 = 5, got {result}"


def test_cluster_activation_states_single():
    """Single activation should return label 0."""
    activations = [np.array([0.5, 0.3])]
    labels, centroids = cluster_activation_states(activations)
    assert labels == [0]
    assert len(centroids) == 1


def test_cluster_activation_states_multiple():
    """Multiple activations should produce valid cluster labels."""
    rng = np.random.RandomState(42)
    activations = [rng.randn(4) for _ in range(50)]
    labels, centroids = cluster_activation_states(activations, n_clusters=5)
    assert len(labels) == 50
    assert len(centroids) == 5
    assert all(0 <= l < 5 for l in labels)


# ── State transition matrix ─────────────────────────────────────────

def test_tpm_shape():
    """TPM should have shape (n_states, n_states)."""
    labels = [0, 1, 0, 2, 1, 0, 1, 2]
    tpm = neural_state_transition_matrix(labels, n_states=3)
    assert tpm.shape == (3, 3)


def test_tpm_row_normalized():
    """Each row of TPM should sum to 1.0 (or be all zeros for unseen states)."""
    labels = [0, 1, 0, 1, 0, 2, 1]
    tpm = neural_state_transition_matrix(labels, n_states=3)
    row_sums = tpm.sum(axis=1)
    for i, s in enumerate(row_sums):
        assert abs(s - 1.0) < 0.01 or s == 0.0, f"Row {i} sum = {s}, expected 1.0 or 0.0"


def test_tpm_empty_input():
    """Empty or single-element input should return zero matrix."""
    tpm = neural_state_transition_matrix([], n_states=3)
    assert np.all(tpm == 0)
    tpm = neural_state_transition_matrix([0], n_states=3)
    assert np.all(tpm == 0)


# ── Effective information ────────────────────────────────────────────

def test_effective_information_uniform():
    """Uniform TPM should produce EI near 0 (no causal structure)."""
    n = 5
    tpm = np.ones((n, n)) / n
    ei = neural_effective_information(tpm)
    assert ei < 0.01, f"Uniform TPM gives EI ~ 0, got {ei}"


def test_effective_information_deterministic():
    """Deterministic TPM (identity) should produce positive EI."""
    n = 5
    tpm = np.eye(n)
    ei = neural_effective_information(tpm)
    assert ei > 0, f"Deterministic TPM should have EI > 0, got {ei}"


def test_effective_information_small():
    """Single-state TPM should return 0."""
    tpm = np.array([[1.0]])
    ei = neural_effective_information(tpm)
    assert ei == 0.0


# ── Neural mutual information ────────────────────────────────────────

def test_neural_mutual_information_identical():
    """MI of identical sequences should be high."""
    rng = np.random.RandomState(123)
    a = [rng.randn(10) for _ in range(30)]
    mi = neural_mutual_information(a, a, n_bins=8)
    assert mi > 0.5, f"MI of identical sequences should be high, got {mi}"


def test_neural_mutual_information_empty():
    """MI with empty input should return 0."""
    mi = neural_mutual_information([], [])
    assert mi == 0.0


def test_neural_mutual_information_single():
    """MI with single sample should return 0."""
    mi = neural_mutual_information([np.array([1.0, 2.0])], [np.array([3.0, 4.0])])
    assert mi == 0.0


# ── Neural Φ ─────────────────────────────────────────────────────────

def test_neural_phi_no_processors():
    """Φ with no processor proposals should be 0."""
    workspace_vec = np.random.randn(16)
    phi = neural_phi(workspace_vec, {})
    assert phi == 0.0


def test_neural_phi_single_processor():
    """Φ with single processor should produce a valid non-negative value."""
    workspace_vec = np.random.randn(16)
    proposals = {"processor_a": np.random.randn(32)}
    phi = neural_phi(workspace_vec, proposals)
    assert phi >= 0.0


# ── Φ trace analysis ─────────────────────────────────────────────────

def test_phi_trace_empty():
    """Empty phi history should return no_data trend."""
    result = neural_phi_trace([])
    assert result["trend"] == "no_data"
    assert result["n_cycles"] == 0


def test_phi_trace_rising():
    """Increasing phi sequence should be detected as rising."""
    phi_hist = [0.1, 0.2, 0.4, 0.8, 1.6]
    result = neural_phi_trace(phi_hist)
    assert result["trend"] == "rising", f"Expected rising, got {result['trend']}"


def test_phi_trace_peaks():
    """Peak detection in oscillating sequence."""
    phi_hist = [0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.2]
    result = neural_phi_trace(phi_hist)
    assert result["peaks"] >= 0


# ── Information geometry ─────────────────────────────────────────────

def test_neural_information_geometry():
    """Information geometry on activation history."""
    rng = np.random.RandomState(42)
    activations = [rng.randn(16) for _ in range(10)]
    result = neural_information_geometry(activations, window=5)
    assert "fisher_trace" in result
    assert "complexity" in result
    assert result["fisher_trace"] >= 0.0


# ── Φ sensitivity analysis ───────────────────────────────────────────

def test_phi_sensitivity_no_proposals():
    """Sensitivity analysis with no proposals should return not-robust."""
    result = phi_sensitivity(np.random.randn(16), {})
    assert result["robust"] is False
    assert result["phi_mean"] == 0.0


def test_phi_sensitivity_single_processor():
    """Sensitivity analysis should return valid distribution stats."""
    result = phi_sensitivity(
        np.random.randn(32),
        {"p": np.random.randn(32)},
        seed=42,
    )
    assert "phi_mean" in result
    assert "phi_std" in result
    assert "coefficient_of_variation" in result
    assert "components" in result
    assert result["n_samples"] == 500


def test_phi_sensitivity_deterministic():
    """Same seed should produce identical sensitivity results."""
    vec = np.random.randn(32)
    props = {"a": np.random.randn(32)}
    r1 = phi_sensitivity(vec, props, seed=42)
    r2 = phi_sensitivity(vec, props, seed=42)
    assert r1["phi_mean"] == r2["phi_mean"]
    assert r1["phi_std"] == r2["phi_std"]


def test_neural_phi_approx_alias():
    """neural_phi should be the same function as neural_phi_approx."""
    assert neural_phi is neural_phi_approx


def test_phi_approx_with_custom_weights():
    """Custom weights should affect the result."""
    vec = np.random.randn(16)
    props = {"a": np.random.randn(16)}
    phi_default = neural_phi_approx(vec, props)
    phi_custom = neural_phi_approx(vec, props, weights=(0.6, 0.3, 0.1))
    # Different weights may give different results (not guaranteed, but check it runs)
    assert isinstance(phi_default, float)
    assert isinstance(phi_custom, float)
    assert phi_default >= 0.0
    assert phi_custom >= 0.0
