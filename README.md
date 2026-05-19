# Noesis-LLM

**Consciousness-inspired multi-agent broadcast comparison using LLM agents.**

> **Branch**: `noesis-llm` — LLM agent backend (Ollama qwen3:4b, token-distribution Φ proxy)
> For the neural RNN version, see [`main`](https://github.com/guyu-adam/noesis) branch.

---

## What this branch is

This is a **multi-agent coordination** research project. We test whether different broadcast mechanisms in a multi-agent language system produce different levels of information integration.

The architecture is **inspired by** Global Workspace Theory (GWT) and Integrated Information Theory (IIT), but the research question is about **multi-agent system design**, not consciousness science. We are not claiming to measure consciousness. We are comparing broadcast strategies.

---

## Research question

> In a multi-LLM system, does coalition consensus broadcast produce higher information integration than winner-take-all broadcast?

We test this across 5 modes:
- **collaborative**: CGWT coalition consensus broadcast
- **hybrid**: competitive narrow-to-2 + collaborative merge
- **competitive**: standard GWT winner-take-all
- **random**: random winner selection
- **no-broadcast**: no global sharing

---

## Architecture

```
                     ┌──────────────────┐
                     │    Attention      │
                     │    Controller     │
                     └────────┬─────────┘
                              │ broadcast
              ┌───────────────┼───────────────┐
              │       Global Workspace        │
              └───────────────┬───────────────┘
         ┌────────────────────┼────────────────────┐
    ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌───────┴──────┐
    │Perceptor│  │Reasoner │  │Evaluator│  │   Narrator   │
    │ (Ollama)│  │ (Ollama)│  │ (Ollama)│  │   (Ollama)   │
    └─────────┘  └─────────┘  └─────────┘  └──────────────┘
```

Each agent is a qwen3:4b instance with a specialized prompt. They process the same text stimulus in parallel and produce competing text proposals. The attention/consensus controller selects what gets broadcast.

---

## Key distinction from `main` branch

| | noesis-llm (this branch) | main |
|---|---|---|
| Agent | Ollama qwen3:4b | Small RNN (32 neurons) |
| Proposals | Text strings | Activation vectors |
| Φ computation | Token-distribution MI | Neural activation TPM |
| Research focus | Multi-agent coordination | Consciousness science |
| Target venue | AAMAS / JAIR | Entropy / PLOS Comp Bio |

---

## Quick start

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python noesis.py
```

```bash
curl -X POST http://localhost:7860/experiment/run \
  -H 'Content-Type: application/json' \
  -d '{"stimulus": "How does attention shape awareness?", "mode": "competitive"}'

curl -X POST http://localhost:7860/experiment/compare \
  -H 'Content-Type: application/json' \
  -d '{"stimuli": ["What is consciousness?", "How does memory work?"], "modes": ["competitive", "collaborative"]}'
```

---

## Project structure

```
noesis (noesis-llm branch)/
├── noesis.py            # Flask API (LLM endpoints only)
├── workspace.py         # Global workspace + attention + CGWT
├── iit.py               # Φ from token-distribution MI
├── experiment.py        # Experiment runner (LLM mode)
├── metrics.py           # Complexity, entropy, consciousness profile
├── world_model.py       # CGWT shared world model
├── memory.py            # Semantic memory (Ollama embeddings)
├── agents/
│   ├── base.py          # LLM agent base (Ollama API)
│   ├── perceptor.py     # Sensory processing LLM agent
│   ├── reasoner.py      # Logical analysis LLM agent
│   ├── evaluator.py     # Value/affect LLM agent
│   └── narrator.py      # Phenomenological report LLM agent
├── cgwt_paper.tex       # Paper draft
├── 审稿意见20260519.md   # Full review (both branches)
└── requirements.txt
```

---

## Status

LLM-based implementation functional. Next: fix Φ formula consistency, run real experiments, reframe paper as multi-agent coordination.

---

## License

MIT
