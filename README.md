# Noesis

**A computational framework for investigating consciousness through causal GWT–IIT integration.**

> **Branch**: `main` — Neural agent backend (small RNNs, causal TPM-based Φ)
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

1. **Agents are small recurrent neural networks** (32 neurons each), not LLMs. Each agent has real causal structure via its recurrent weight matrix W_rec. This makes Φ measurable from neural activation state transition matrices — not token-distribution proxies.

2. **The GWT broadcast mechanism** (competition + attention + global workspace) is implemented as a dynamical system that the agents participate in cycle by cycle.

3. **Φ is computed from the causal TPM** of the system's neural states — effective information, mutual information between activation patterns, and irreducibility of the global state to individual agent states.

4. **CGWT (Collaborative GWT)** extends winner-take-all broadcast to coalition consensus broadcast, hypothesized to produce higher Φ by preserving both consensus ground and complementary diversity.

**Key distinction from the LLM version (noesis-llm branch):**

| | main (neural) | noesis-llm |
|---|---|---|
| Agent implementation | Small RNN (32 neurons) | Ollama qwen3:4b |
| Proposals | Activation vectors | Text strings |
| Φ computation | Neural activation TPM | Token-distribution MI |
| Causal structure | Real (W_rec connectivity) | Proxy (text similarity) |
| Relationship to IIT | Direct (causal Φ) | Indirect (proxy Φ) |
| Target venue | Entropy / PLOS Comp Bio | AAMAS / JAIR |

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

Each agent is a small RNN with specialized recurrent connectivity:
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
│   ├── neural_base.py     # Neural agent base (RNN)
│   └── neural_agents.py   # Specialized neural agents
├── cgwt_paper.tex         # CGWT paper draft (Entropy format)
├── 审稿意见20260519.md     # Internal review document
├── requirements.txt
└── .gitignore
```

---

## Quick start

```bash
pip install -r requirements.txt
# Start Ollama (for LLM endpoints only)
ollama serve
# Run server
python noesis.py
```

### LLM experiment (noesis-llm branch style)
```bash
curl -X POST http://localhost:7860/experiment/run \
  -H 'Content-Type: application/json' \
  -d '{"stimulus": "What is consciousness?", "mode": "competitive"}'
```

### Neural experiment (main branch style)
```bash
curl -X POST http://localhost:7860/neural/run \
  -H 'Content-Type: application/json' \
  -d '{"stimulus": "What is consciousness?", "mode": "collaborative"}'
```

---

## Status

**main branch**: Neural agent framework in place. Architecture: workspace + attention controller + consensus controller + 3 specialized RNN agents + neural IIT module. Next: run experiments, validate Φ computation, write paper.

**noesis-llm branch**: LLM-based implementation functional. See branch README.

---

## Research questions

1. Does Φ (from neural TPM) peak at the moment of coalition broadcast?
2. Does coalition consensus produce higher Φ than winner-take-all?
3. Does the competitive attention mechanism increase Φ compared to random selection?
4. Can we identify a "sweet spot" where integration and differentiation are balanced?

---

## License

MIT
