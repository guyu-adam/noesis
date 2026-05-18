"""
IIT metrics — Integrated Information Theory measures.

Φ (phi) quantifies how much information a system generates as a whole
that cannot be reduced to the sum of its parts. Computing exact Φ is
NP-hard for real systems. This module implements tractable proxies.

Key concept for Noesis:
    We measure a Φ proxy *at each broadcast cycle* to test whether
    the GWT mechanism (competition → selection → broadcast) produces
    measurable increases in integrated information.

The hypothesis predicts:
    Φ_before < Φ_after  (broadcast increases integration)
    Φ_random_selection < Φ_competitive_selection  (competition matters)
"""

import numpy as np
from typing import Optional


def state_transition_matrix(history: list[dict], n_states: int = 10) -> np.ndarray:
    """
    Build a simplified causal state transition matrix from broadcast history.

    Each broadcast entry is treated as a system state. The transition
    matrix encodes how often state A leads to state B across cycles.
    This is the foundation for computing effective information (EI) and Φ.

    Args:
        history: Broadcast history entries.
        n_states: Number of discrete states to bucket transitions into.

    Returns:
        (n_states × n_states) transition probability matrix.
    """
    return np.zeros((n_states, n_states))  # placeholder


def effective_information(tpm: np.ndarray) -> float:
    """
    Compute effective information (EI) from a transition probability matrix.

    EI measures how much the current state constrains the next state —
    the "cause-effect power" of the system. High EI means the system's
    past strongly determines its future (a prerequisite for consciousness in IIT).

    Args:
        tpm: Transition probability matrix.

    Returns:
        EI value in bits.
    """
    return 0.0  # placeholder


def phi_proxy(workspace_state: Optional[str], history: list[dict]) -> float:
    """
    Compute a tractable Φ proxy for the current workspace state.

    Since exact Φ is computationally intractable, we compute a proxy
    based on mutual information between agents' proposals and the
    global broadcast — quantifying the "irreducibility" of the
    global state to individual agent states.

    Args:
        workspace_state: Current content of the global workspace.
        history: Broadcast history.

    Returns:
        Φ proxy value (0 = fully reducible, higher = more integrated).
    """
    return 0.0  # placeholder


def phi_trace(phi_history: list[float]) -> dict:
    """
    Analyze a Φ time series for key patterns.

    Returns summary statistics useful for detecting:
      - Φ peaks (moments of high integration)
      - Φ collapse (over-integration / loss of differentiation)
      - Φ oscillation (healthy competition ↔ broadcast cycle)
    """
    if not phi_history:
        return {"mean": 0, "max": 0, "variance": 0, "trend": "no_data"}

    arr = np.array(phi_history)
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "variance": float(np.var(arr)),
        "trend": "rising" if len(arr) > 1 and arr[-1] > arr[0] else "falling",
    }
