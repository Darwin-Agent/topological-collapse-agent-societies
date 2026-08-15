# Matched One-Round Relay Ablation

This directory contains a post hoc mechanism ablation of the frozen
two-round relay benchmark in `llm_relay_benchmark`. It reuses the same 16
previously analysed task cells, model labels, agent count, evidence cards
and five protocol families. The only planned change is the relay budget: one
synchronous card-broadcast round instead of two.

The ablation is not an independent held-out replication because the task
manifest had already been analysed in the two-round experiment. It is reported
only as a matched capacity check. It does not isolate topology from group size,
centralisation or delivery reach.

Run the design freeze before calls:

```bash
python3 llm_relay_round1_ablation/run_round1_ablation.py --mode design
```

Run the 160 matched cells after setting `LLM_GATEWAY_API_KEY` and `LLM_GATEWAY_API_URL`:

```bash
export LLM_GATEWAY_API_KEY="..."
export LLM_GATEWAY_API_URL="https://gateway.example/v1/chat/completions"
python3 llm_relay_round1_ablation/run_round1_ablation.py --mode full
```

The runner writes the frozen protocol, deterministic access diagnostics,
per-cell raw records and a summary. It reads no credentials from repository
files.
