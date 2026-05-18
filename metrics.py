"""
Complementary consciousness metrics beyond Φ.

A complete assessment of a system's consciousness requires:

  Φ (integration)          — How unified is the system? (from iit.py)
  C (complexity/differentiation) — How rich are its states?
  D (decodability)         — Can internal states be read from outside?
  S (self-model)           — Does it have a model of itself?

Together these form the "consciousness profile" of the system at each cycle.

References:
  - arXiv:2512.19155 — consciousness marker framework
  - arXiv:2605.12536 — information geometry bridge
  - arXiv:2412.10626 — Shannon vs integrated information
"""

import math
import numpy as np
from collections import Counter


def lz_complexity(text: str) -> float:
    """
    Lempel-Ziv complexity estimate — measures the richness/diversity of output.

    Higher LZ = more differentiated states = more information in the system.
    Too low → "unconscious" (uniform, stereotyped). Too high → "noise".

    This is complementary to Φ: Φ measures integration, LZ measures differentiation.
    The "sweet spot" for consciousness is high Φ + moderate LZ.
    """
    if not text:
        return 0.0

    # Use token-level LZ approximation
    tokens = text.lower().split()
    if len(tokens) < 3:
        return 0.0

    # Build dictionary of seen substrings (LZ78-style)
    seen = set()
    complexity = 0
    current = ""

    for token in tokens:
        candidate = f"{current} {token}".strip()
        if candidate not in seen:
            seen.add(candidate)
            complexity += 1
            current = ""
        else:
            current = candidate

    # Normalize by text length
    return round(complexity / len(tokens), 6)


def semantic_entropy(text: str) -> float:
    """
    Token distribution entropy — Shannon entropy of the token frequency distribution.

    Higher entropy = more diverse vocabulary usage = more information processing.
    """
    tokens = text.lower().split()
    if len(tokens) < 2:
        return 0.0

    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]

    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    # Normalize by max possible entropy (log2 of unique tokens)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    return round(entropy / max_entropy, 6)


def integration_differentiation_balance(phi: float, complexity: float) -> dict:
    """
    Compute the balance between integration (Φ) and differentiation (complexity).

    In IIT, consciousness requires both:
      - High Φ: the system is unified
      - High information: the system is differentiated

    The balance score = 1 - |Φ_norm - C_norm| — peaks when both are high and balanced.

    Returns:
        balance_score (0-1), regime classification, and raw values.
    """
    # Normalize to [0, 1] range (assuming typical values)
    phi_norm = min(1.0, max(0.0, phi / 2.0))
    comp_norm = min(1.0, complexity)

    balance = 1.0 - abs(phi_norm - comp_norm)

    if phi > 0.5 and complexity > 0.3:
        regime = "conscious"  # high integration + high differentiation
    elif phi > 0.5 and complexity <= 0.3:
        regime = "unified_but_sterile"  # integrated but not rich
    elif phi <= 0.5 and complexity > 0.3:
        regime = "fragmented"  # rich but not integrated
    else:
        regime = "unconscious"  # neither integrated nor rich

    return {
        "balance_score": round(balance, 4),
        "phi_normalized": round(phi_norm, 4),
        "complexity_normalized": round(comp_norm, 4),
        "regime": regime,
    }


def consciousness_profile(
    phi: float,
    complexity: float,
    fisher_trace: float,
    broadcasted: bool,
    attention_score: float,
) -> dict:
    """
    Full consciousness profile for a single cycle.

    Combines all available metrics into a single diagnostic assessment.

    This is the output a researcher would use to:
      1. Track consciousness markers over time
      2. Compare conditions (competitive vs random)
      3. Identify the "conscious access" signature
    """
    balance = integration_differentiation_balance(phi, complexity)

    # Broadcast-aware: conscious access should correspond to high phi + broadcast
    if broadcasted:
        conscious_access_confidence = min(1.0, (phi * 0.5 + balance["balance_score"] * 0.3
                                                 + attention_score * 0.2))
    else:
        conscious_access_confidence = phi * 0.3

    return {
        "phi": phi,
        "complexity": complexity,
        "fisher_trace": fisher_trace,
        "balance_score": balance["balance_score"],
        "regime": balance["regime"],
        "conscious_access_confidence": round(conscious_access_confidence, 4),
        "broadcasted": broadcasted,
        "attention_score": attention_score,
    }
