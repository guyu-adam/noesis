# Noesis

**A computational framework for investigating consciousness through the lens of GWT–IIT integration.**

---

## Motivation

Two theories dominate the scientific study of consciousness, and they rarely speak to each other:

| | Global Workspace Theory (GWT) | Integrated Information Theory (IIT) |
|---|---|---|
| **Core claim** | Consciousness is *global broadcast* — information that wins access to a shared workspace and becomes available to all specialized modules | Consciousness is *integrated information* (Φ) — a system is conscious to the extent that its whole generates information irreducible to the sum of its parts |
| **Strength** | Explains the **function** of consciousness: attention selection, serial bottleneck, reportability, cognitive control | Explains the **phenomenology** of consciousness: unity, richness, why some states feel like something and others don't |
| **Weakness** | Does not explain why global availability *feels like anything* — the "hard problem" gap | Φ is computationally intractable for real systems; does not explain the cognitive architecture that produces it |
| **Key metaphor** | A stage with a spotlight — many actors (specialized processors) compete, only one performs at a time | A photodiode has Φ=0, a complex network has Φ>0 — consciousness is a structural property of causal interaction |

**The central question**: Are GWT and IIT contradictory, or are they describing the same phenomenon from different levels of analysis?

---

## Hypothesis

> **GWT provides the *mechanism* that generates high-Φ states. IIT provides the *metric* that quantifies the outcome. They are complementary descriptions of a single underlying process: consciousness as competitive integration.**

Specifically:

1. **Competition precedes integration.** Multiple specialized agents (analogous to cortical modules) process the same stimulus independently, generating competing interpretations. This competition raises the system's *effective information* — the system is in a more uncertain state before resolution.

2. **Broadcast creates irreducibility.** When one agent's output wins the competition and is globally broadcast, the system transitions to a causally integrated state — the whole now constrains the parts in a way that cannot be decomposed.

3. **Φ peaks at the broadcast moment.** The integrated information of the system should peak immediately after global broadcast, when the causal structure is maximally unified. Before broadcast, the system is differentiated (high *effective* information but low integration). After broadcast, it is integrated (high Φ but reduced differentiation). Consciousness is the transition between these two regimes.

4. **Attention bottleneck is a Φ-maximizing mechanism.** The serial, competitive nature of GWT is not a design limitation — it is what allows the system to generate high-Φ states from local computation. Parallel broadcast would saturate integration; serial selection maintains differentiation.

---

## Architecture

Noesis implements this hypothesis as a multi-agent system:

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

### Key components

- **Specialized Agents** — Each agent is a local LLM (Ollama) with a distinct cognitive role. They process the same stimulus in parallel and produce competing "proposals" for conscious access.
- **Attention Controller** — Computes salience for each proposal (novelty, relevance, emotional weight) and selects a single winner per cycle.
- **Global Workspace** — The winner's output is broadcast here, visible to all agents in subsequent cycles. This is where Φ is computed.
- **Narrator** — Generates first-person phenomenological reports based on the broadcast history.
- **IIT Metrics Module** — Computes Φ-like measures from the causal state transition matrix of the global workspace over time.

---

## Research questions

1. Does Φ (or a tractable proxy) peak at the moment of global broadcast?
2. Does the competitive attention mechanism increase Φ compared to random/no selection?
3. Do phenomenological reports from a GWT+IIT hybrid better match human consciousness data than either theory alone?
4. Can we identify a "sweet spot" where Φ is maximized — enough competition to be differentiated, enough broadcast to be integrated?

---

## Project structure

```
noesis/
├── noesis.py            # Main server — Flask API for experiment orchestration
├── workspace.py         # Global workspace + attention controller
├── iit.py               # Φ computation (starts as tractable proxy)
├── experiment.py        # Experiment runner — stimuli, data collection, analysis
├── agents/
│   ├── __init__.py
│   ├── base.py          # Abstract agent interface
│   ├── perceptor.py     # Input processing agent
│   ├── reasoner.py      # Logical reasoning agent
│   ├── evaluator.py     # Affective/value evaluation agent
│   └── narrator.py      # Phenomenological report generation
├── memory.py            # Semantic memory with embeddings
├── client.py            # Client for external agents (Claude Code, etc.)
├── requirements.txt
└── .gitignore
```

---

## Status

**Phase 1 — Framework** (current): Architecture design, component stubs, experiment protocol.

**Phase 2 — Single-agent baseline**: Verify that the agent system generates coherent proposals before adding competition.

**Phase 3 — Competition + broadcast**: Implement attention controller and global workspace. Run simple stimulus→broadcast→report cycles.

**Phase 4 — Φ measurement**: Implement tractable Φ proxy. Measure integration across broadcast cycles.

**Phase 5 — Publication experiments**: Run the experiments described in the research questions.

---

## Requirements

- Python 3.10+
- Ollama running locally
- `flask requests numpy rich scipy`

---

## License

MIT
