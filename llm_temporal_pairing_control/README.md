# Temporal Pairing Relay Control

This prospective control holds fixed eight agents, four dyads per round, dyad
size, one group membership and one card broadcast per agent per round, the
four-round relay budget, task construction and model IDs. It compares a
repeated task-specific dyadic matching with four rotating matchings from the
same task-specific round-robin factorization.

The responding model sees the same neutral protocol label in both conditions.
Cards are transmitted deterministically; model-written text is never relayed.
The control evaluates repeated versus rotating dyadic contacts and their
resulting card reach under this bounded task. It does not establish a general
topology effect or a partner-novelty effect independent of information access.

Run the no-API design freeze first:

```bash
python3 llm_temporal_pairing_control/run_temporal_pairing_control.py --mode design
```

After setting `LLM_GATEWAY_API_KEY` and `LLM_GATEWAY_API_URL`, run the reserved engineering
pilot, then the 16-task held-out protocol:

```bash
export LLM_GATEWAY_API_KEY="..."
export LLM_GATEWAY_API_URL="https://gateway.example/v1/chat/completions"
python3 llm_temporal_pairing_control/run_temporal_pairing_control.py --mode pilot
python3 llm_temporal_pairing_control/run_temporal_pairing_control.py --mode full
```

The runner writes a new task manifest, frozen pilot and full protocols,
schedule and access diagnostics, per-cell raw records, and a held-out summary.

In environments with short process limits, the frozen held-out task IDs can be
run as resumable batches. Run a final `--mode full` without `--task-ids` after
all batches are present to write the complete summary:

```bash
python3 llm_temporal_pairing_control/run_temporal_pairing_control.py \
  --mode full --task-ids T05 T06
```
