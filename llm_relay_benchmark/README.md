# Bandwidth-Limited Relay Benchmark

This benchmark is a separate, prospective complement to the lossless
distributed-evidence experiment in `llm_task_benchmark`. It asks whether fixed
communication protocols provide different access to deterministic evidence when
every agent may broadcast only one card per synchronous round.

Eight agents receive one distinct exclusion card for an eight-label task. A
card excludes two false labels. No set of one, two or three cards identifies
the answer, while some four-card sets do. The benchmark uses two rounds, five
protocols and a deterministic relay rule: every agent broadcasts the first
card it has not sent before, or repeats its first card when no new card is
available. Model-written prose is never relayed.

The protocol is frozen before requests:

```bash
python3 llm_relay_benchmark/run_limited_relay_benchmark.py --mode design
export LLM_GATEWAY_API_KEY="..."
export LLM_GATEWAY_API_URL="https://gateway.example/v1/chat/completions"
python3 llm_relay_benchmark/run_limited_relay_benchmark.py --mode pilot
python3 llm_relay_benchmark/run_limited_relay_benchmark.py --mode full
```

`--mode design` writes the task manifest, pilot protocol, held-out protocol and
deterministic access diagnostics without calling a model. The four pilot tasks
are reserved for model and logging checks. The held-out run uses 16
answer-balanced tasks, two fixed model IDs, temperature zero and 512 generated
tokens.

The primary endpoint is final-round group plurality accuracy with ties counted
as incorrect. Secondary endpoints include individual accuracy, the fraction of
agents with sufficient evidence, accuracy conditional on sufficient evidence,
Brier score, and API or parse failure rates. This task tests bounded evidence
relay under its specified protocols; it does not identify a topology-only
causal effect or a general multi-agent performance advantage.
