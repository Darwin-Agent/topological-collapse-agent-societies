# Distributed Evidence Benchmark

This project-contained benchmark measures a narrow task-level capability:
whether a group of LLM agents can integrate distributed, deterministic evidence
under different message-passing protocols.

Each of eight agents receives one exclusion card for an eight-label task. Every
card rules out three incorrect labels; no one or two cards identify the answer,
while all eight cards jointly do. At each synchronous round, a protocol
determines which peers receive each agent's accumulated cards. Card delivery is
lossless and logged, so the analysis separates protocol-dependent information
access from LLM reasoning over sufficient evidence.

The frozen task manifest is stored in `data/task_manifest.json`. Each API call
records its exact system and user prompts, raw response, parsed answer,
confidence, usage metadata and safe response headers. Model identifiers do not
establish an immutable provider-side deployment revision.

The pre-specified primary endpoint is run-level plurality accuracy, with ties
counted as incorrect. Secondary endpoints are individual accuracy, the fraction
of agents receiving enough cards for a unique answer, accuracy conditional on
that sufficient evidence, Brier score and failure rates.

## Run

```bash
export LLM_GATEWAY_API_KEY="..."
export LLM_GATEWAY_API_URL="https://gateway.example/v1/chat/completions"
python3 llm_task_benchmark/run_distributed_evidence_benchmark.py --mode pilot
python3 llm_task_benchmark/run_distributed_evidence_benchmark.py --mode full
```

The pilot uses four engineering tasks (`T01`--`T04`) to check API behaviour and
logging. The full protocol uses the 16 held-out tasks (`T05`--`T20`), balanced
with two instances of each answer label, two configured model IDs, five
conditions and three synchronous rounds. Runs are resumable at the
model-task-condition-seed level. The models were selected by a preflight using
the actual task prompt: a candidate must return a final parseable JSON decision
within the configured 512-token budget.

The script preserves separate frozen `results/protocol_pilot.json` and
`results/protocol_full.json` files, complete raw cell artefacts,
`results/summary.json`, and a descriptive figure. These results should be
included in the manuscript only after the completed records, error rate and
condition-level patterns have been reviewed.
