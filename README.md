# Noesis

**A computational framework for investigating consciousness through causal GWT–IIT integration.**

> **Branch**: `main` — Neural processor backend (small RNNs, causal TPM-based Φ)
> **Target**: Entropy (MDPI) / PLOS Computational Biology
> For the LLM-based version, see [`noesis-llm`](https://github.com/guyu-adam/noesis/tree/noesis-llm) branch.

---

## Motivation

Two theories dominate the scientific study of consciousness:

| | Global Workspace Theory (GWT) | Integrated Information Theory (IIT) |
|---|---|---|
| **Core claim** | Consciousness is *global broadcast* — information that wins access to a shared workspace | Consciousness is *integrated information* (Φ) — irreducible to the sum of parts |
| **Strength** | Explains the **function** of consciousness | Explains the **phenomenology** of consciousness |
| **Weakness** | Doesn't explain why broadcast *feels like anything* | Φ is computationally intractable for real systems |

**Central question**: Are GWT and IIT contradictory, or complementary? Can we build a system where GWT-like broadcast mechanisms *produce* IIT-measurable high-Φ states?

---

## Approach (main branch)

Unlike typical consciousness papers that argue from philosophy or neuroimaging, Noesis **builds a minimal computational system** where:

1. **Processors are small recurrent neural networks** (256 neurons each). Each processor has real causal structure via its recurrent weight matrix W_rec. This makes Φ measurable from neural activation state transition matrices — not token-distribution proxies.

2. **The GWT broadcast mechanism** (competition + attention + global workspace) is implemented as a dynamical system that the processors participate in cycle by cycle.

3. **Φ is computed from the causal TPM** of the system's neural states — effective information, mutual information between activation patterns, and irreducibility of the global state to individual processor states.

4. **CGWT (Collaborative GWT)** extends winner-take-all broadcast to coalition consensus broadcast, hypothesized to produce higher Φ by preserving both consensus ground and complementary diversity.

**Key distinction from the LLM version (noesis-llm branch):**

| | main (neural) | noesis-llm |
|---|---|---|
| Agent implementation | Small RNN (32 neurons) | Ollama qwen3:4b |
| Proposals | Activation vectors | Text strings |
| Φ computation | Neural activation TPM | Token-distribution MI |
| Causal structure | Real (W_rec connectivity) | Proxy (text similarity) |
| Relationship to IIT | Direct (causal Φ) | Indirect (proxy Φ) |
| Target venue | Entropy / PLOS Comp Bio | JAAMAS |

---

## Architecture

```
                     ┌──────────────────┐
                     │    Attention      │
                     │    Controller     │  ← selects winner via salience
                     └────────┬─────────┘
                              │ broadcast
              ┌───────────────┼───────────────┐
              │       Global Workspace        │  ← shared state, Φ measured here
              └───────────────┬───────────────┘
         ┌────────────────────┼────────────────────┐
    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌───────┴──────┐
    │Perceptor│  │Reasoner │  │Evaluator│  │   Narrator   │
    │ (input  │  │ (logic) │  │ (value) │  │ (first-person│
    │  parse) │  │         │  │         │  │   reports)   │
    └─────────┘  └─────────┘  └─────────┘  └──────────────┘
         │              │            │              │
    ┌────┴──────────────┴────────────┴──────────────┴────┐
    │              Shared Stimulus / Environment          │
    └────────────────────────────────────────────────────┘
```

Each processor is a small RNN with specialized recurrent connectivity:
- **Perceptor**: near-diagonal W_rec → fast decorrelation, feature extraction
- **Reasoner**: chain-structured W_rec → sequential processing stages
- **Evaluator**: bistable W_rec → attractor dynamics, value judgment

---

## Project structure

```
noesis/
├── noesis.py              # Flask API (LLM + Neural endpoints)
├── workspace.py           # Global workspace + attention + CGWT
├── iit.py                 # Φ from token MI (LLM mode)
├── neural_iit.py          # Φ from neural TPM (neural mode)
├── experiment.py          # Experiment runner (both modes)
├── metrics.py             # Consciousness profile metrics
├── world_model.py         # CGWT shared world model
├── memory.py              # Semantic memory (embeddings)
├── agents/
│   ├── __init__.py
│   ├── base.py            # LLM agent base (Ollama)
│   ├── perceptor.py       # LLM agents (prompt-based)
│   ├── reasoner.py
│   ├── evaluator.py
│   ├── narrator.py
│   ├── neural_base.py     # Neural processor base (RNN)
│   └── neural_agents.py   # Specialized neural processors
├── tests/
│   ├── test_neural_iit.py   # Neural Φ computation tests
│   └── test_world_model.py  # World model tests
├── experiments/           # Auto-saved experiment data (JSONL)
├── requirements.txt
└── .gitignore
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python noesis.py
```

### Neural experiment (main branch)

```bash
# Single cycle
curl -X POST http://localhost:7860/neural/run \
  -H 'Content-Type: application/json' \
  -d '{"stimulus": "What is consciousness?", "mode": "collaborative"}'

# Multi-mode comparison
curl -X POST http://localhost:7860/neural/compare \
  -H 'Content-Type: application/json' \
  -d '{"stimuli": ["What is consciousness?", "Explain pain", "Define self-awareness"], "modes": ["competitive", "random", "no_broadcast", "collaborative"], "cycles_per": 3}'

# Get status
curl http://localhost:7860/status
```

### Available modes

| Mode | Description |
|------|-------------|
| `competitive` | Standard GWT winner-take-all (baseline) |
| `random` | Random winner selection (control) |
| `no_broadcast` | No broadcast, processors process independently (control) |
| `single_agent` | Only one processor active (control) |
| `collaborative` | CGWT coalition consensus broadcast |
| `hybrid` | Competitive narrowing → top-2 coalition merge |

### Running tests

```bash
pip install pytest numpy scipy
python -m pytest tests/ -v
```

---

## Data persistence

Experiment results are automatically saved to `experiments/<date>/<mode>.jsonl` each cycle.
Each record includes: timestamp, cycle_id, phi values, proposals, winner, coalition, and more.

---

## Status

**Neural framework** in place. Architecture: workspace + attention controller + consensus controller + 5 specialized RNN processors + world model + neural IIT module.

**Completed:**
- Small RNN processors with specialized recurrent connectivity
- Neural Φ computation from activation TPMs
- Collaborative workspace with coalition broadcast
- World model with consensus scoring and prediction error tracking
- Dual-backend API (LLM + neural endpoints)
- Experiment data auto-persistence
- Unit tests for core Φ math and world model

**Next:**
- Run experiments, validate Φ computation
- Weight sensitivity analysis
- Scale to 5 processors with ablation experiments
- Write paper for Entropy submission

---

## Research questions

1. Does Φ (from neural TPM) peak at the moment of coalition broadcast?
2. Does coalition consensus produce higher Φ than winner-take-all?
3. Does the competitive attention mechanism increase Φ compared to random selection?
4. Can we identify a "sweet spot" where integration and differentiation are balanced?

---

## License

MIT
