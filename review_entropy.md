# Referee Report: "What Coalition Broadcast Reveals About Causal Integration"

**Manuscript:** CGWT: Computational Constraints on Coalition Broadcast and Causal Integrated Information in a Recurrent Neural Processor System
**Journal:** Entropy (MDPI)
**Date:** 2026-05-21

---

## Summary

This paper presents a computational framework (Noesis) that integrates Global Workspace Theory (GWT) with Integrated Information Theory (IIT) by implementing a system of five specialized RNN processors connected to a global workspace. The authors compute a tractable $\Phi$ approximation from neural state transition probability matrices across nine experimental modes (900 cycles) and report four findings: broadcast is necessary for high $\Phi$, processor differentiation is a prerequisite, multi-processor architecture introduces an integration bottleneck, and the CGWT coalition hypothesis is not supported. The paper's intellectual honesty---reporting null results with mechanistic explanations, documenting all approximation simplifications, and openly stating the limitations---is commendable and unfortunately rare in computational consciousness science.

However, the core experiment may not have tested what the paper claims it tested. Two technical issues require attention before the findings can be interpreted as reported.

---

## Major Issues

### 1. The "collaborative" mode does not actually form coalitions

This is the most critical problem. The paper claims that "collaborative" and "hybrid" modes execute coalition consensus broadcast---merging outputs from multiple processors into a unified workspace representation. However, inspection of the raw per-cycle JSONL data reveals that **n_merged is always 1** across all 100 collaborative cycles:

```
"coalition": ["reasoner"],  "n_merged": 1
"coalition": ["predictor"], "n_merged": 1
"coalition": ["integrator"], "n_merged": 1
```

No cycle ever shows two or more processors in a coalition. This means the collaborative and hybrid modes are functionally identical to competitive mode (single-winner selection), just with different labeling. Consequently:

- **Finding 4 ("CGWT's core prediction is not supported") is invalid.** It does not represent a falsification of the CGWT hypothesis; it represents a case where the hypothesis was never actually tested because the coalition formation mechanism never produced multi-member coalitions.
- The explanation given for the null result (attention weight collapse, cosine similarity ~0.96) is describing a symptom rather than the root cause. Even when processors produce similar outputs, a coalition should still *form*---the question is whether the merged output differs from single-winner. But no merge ever happened.
- The adaptive gate analysis (100% competitive fallback) similarly reflects the absence of coalition formation rather than a validated gating mechanism.

The relevant code path (`workspace.py:387-393`, `Select_coalition`; `neural_iit.py:952-958`, `merge_coalition`) needs to be examined to determine why coalition size never exceeds 1. The fix may be in the experiment runner rather than these functions. After fixing, all nine modes should be re-run, as the collaborative and hybrid results will change fundamentally.

### 2. EI accumulation artifact in the $\Phi$ computation creates artificially large effect sizes

The per-cycle $\Phi$ decomposition data reveals a clear linear accumulation pattern in $\Phi_{\text{within}}$ (effective information from TPM). For competitive mode, cycles 1–5:

| Cycle | $\Phi_{\text{within}}$ | $\Delta$ |
|-------|------------------------|----------|
| 1 | 0.000 | — |
| 2 | 0.624 | +0.624 |
| 3 | 0.957 | +0.332 |
| 4 | 1.289 | +0.332 |
| 5 | 1.621 | +0.332 |

The step size of ~0.332 per cycle matches `neural_iit.py:neural_effective_information` computing EI on a TPM that gains one additional transition observation per cycle. With $n_{\text{clusters}} = 50$ and only 5 cycles per stimulus, the TPM remains extremely sparse, and the EI computation becomes sensitive to the number of observations rather than the causal structure of the system.

This artifact directly inflates **Finding 1 (broadcast necessity, $d = 7.43$)**:
- No-broadcast: $\Phi_{\text{within}} = 0$ because there is no broadcast history from which to build a TPM. This is true by construction.
- All broadcast modes: $\Phi_{\text{within}} > 0$ because workspace history exists, and EI grows with each cycle.
- The 12–17× difference between broadcast and no-broadcast is therefore partly a **circular result**: the metric requires broadcast history to return non-zero values, and then it is used to demonstrate that broadcast is necessary.

The paper acknowledges this in Section 4.1 ("$\Phi_{\text{within}}$ drops to zero in no-broadcast because there is no broadcast history") but treats it as an interpretive note rather than a confound that threatens the validity of the finding. The fix should use a **fixed-length sliding window** for TPM construction so that EI is computed from a constant number of observations across all conditions and cycles.

### 3. Violation of independence assumptions in statistical tests

The paper treats all 900 observation cycles as independent, conducting ANCOVA with $F(8, 890)$ and Mann–Whitney $U$ tests. However:
- Five consecutive cycles within the same stimulus form a time series with strong positive autocorrelation ($\Phi$ accumulates monotonically within each stimulus, as shown above).
- Each cycle's $\Phi_{\text{after}}$ becomes the next cycle's $\Phi_{\text{before}}$.
- The effective sample size is closer to 180 independent sequences (20 stimuli $\times$ 9 modes), each of length 5, not 900 independent observations.

The reported $p$-values and effect sizes are therefore anti-conservative. A mixed-effects model with stimulus as a random effect, or cluster-robust standard errors grouped by stimulus, would be more appropriate.

---

## Minor Issues

4. **Stimulus encoding.** Twenty consciousness-related questions are encoded into 32-dimensional vectors via deterministic hashing with random projection. This is an unusual choice. A 32-dimensional hash of a natural language question discards almost all semantic content. Whether the RNN processors' outputs are driven by the stimulus or by their own recurrence is unclear. At minimum, the paper should discuss this design choice and consider an ablation where stimulus vectors are replaced with random noise to measure the stimulus-dependence of $\Phi$.

5. **Cycle count per stimulus is too low.** With only 5 cycles per stimulus, $\Phi$ never reaches a steady state (as the accumulation data show). Extending to 10–15 cycles per stimulus would allow distinguishing transient from stable effects.

6. **Calibration over-clustering is acknowledged but not fixed.** The paper reports calibration anchor values (Section 3.1) with $n_{\text{clusters}} = 30$ and $n_{\text{history}} = 25$ but notes this creates an over-clustering artifact. Providing corrected calibration values (using $n_{\text{clusters}} \leq 5$) would be straightforward and improve reader confidence.

7. **Acknowledgments reference a "three-round review process"** despite the manuscript appearing to be a preprint. If this refers to informal pre-submission feedback, the wording should be clarified to avoid implying prior formal peer review.

---

## Strengths

Despite the issues above, this paper has several genuine merits that should be preserved in revision:

1. **The $\Phi$ decomposition ($\Phi_{\text{within}}$ / $\Phi_{\text{between}}$) is a real methodological contribution** (`neural_iit.py:588–679`, `phi_decomposed`). Separating processor-internal causal structure from broadcast-mediated integration provides a diagnostic toolkit applicable beyond this specific study. The fact that it *revealed* the EI accumulation artifact is itself evidence of its utility.

2. **Intellectual honesty.** The paper reports null results transparently, documents all four approximations from proper IIT, states that absolute $\Phi$ values are meaningless, and acknowledges when a finding is circular. This sets a standard that computational consciousness papers should follow.

3. **The MaxCal connection to Kearney (2026)** (Equation 11) provides a principled bridge between the implementation and the theoretical literature. Operationalizing the prediction error penalty term in future work is a well-motivated direction.

4. **The differentiation gate (adaptive mode)** is a good conceptual contribution—even though it triggered 100% competitive fallback, the mechanism itself is sound and the diagnostic value of the attention weight standard deviation is a transferable idea.

5. **Reproducibility.** The code is open-source (MIT), the specific commit and experiment configuration are archived, and the raw JSONL data is available. This is exemplary.

---

## Recommendation

**Major Revision.**

The paper's intellectual approach—diagnostic methodology, decomposition of $\Phi$, honest null-result reporting—is valuable and should be published. However, the current version has two technical issues that prevent the reported findings from being interpreted as written: the coalition mechanism doesn't form coalitions, and the EI accumulation artifact inflates the key effect sizes. Both are fixable, and the revised paper would likely be stronger than the current version.

After revision, I would be happy to re-review.

---

## Specific Required Changes

1. Fix the coalition formation pipeline so that multi-processor coalitions are actually created in collaborative/hybrid modes. Re-run all nine modes and update all tables, figures, and findings accordingly.
2. Implement fixed-length sliding windows for TPM-based EI computation to eliminate the accumulation artifact. Report whether Finding 1 survives this correction.
3. Replace or supplement the current independent-observation statistics with mixed-effects models or clustered standard errors that account for within-stimulus serial correlation.
4. Revise the interpretation of Finding 4 to reflect whether the null result persists after coalition formation is fixed.
5. Recompute calibration anchors with non-overclustered parameters ($n_{\text{clusters}} \leq 5$).
