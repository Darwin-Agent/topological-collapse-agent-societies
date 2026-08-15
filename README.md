# Topological Collapse in AI Agent Societies

Code and compact reproducibility artifacts associated with the research project:

> **Topological collapse of higher-order interactions bottlenecks collective intelligence in AI agent societies**

The repository name, `topological-collapse-agent-societies`, is a concise,
searchable form of the paper title.

## Scope

This public release combines the core analysis and experiment code with later
reproducibility controls developed for the same project:

- hypergraph construction and higher-order topology metrics;
- Hyperedge Irreducibility Score (HIS) and topology comparisons;
- higher-order contagion and public-goods-game simulations;
- controlled topology experiments;
- multi-agent LLM experiment runners;
- frozen distributed-evidence and relay benchmark summaries;
- compact figure inputs, renderers, and generated figures.

The release is intentionally compact. Raw social-platform records, scraped
profiles, databases, per-call model responses, request headers, API logs, and
credentials are not included.

## Repository Layout

| Path | Contents |
| --- | --- |
| `src/analysis/` | Hypergraph construction, topology metrics, statistics, and plots |
| `src/models/` | Pairwise and higher-order contagion models |
| `src/experiments/` | ABM, topology, AgentPanel, and LLM experiment runners |
| `controlled_topology_experiment/` | Deterministic degree-matched topology control |
| `llm_*_benchmark/` | Frozen benchmark designs and aggregate results |
| `figure_source/` | Figure renderers and compact, hashed figure inputs |
| `results/` | Aggregate numerical results and selected figures |
| `images/` | Main reproducibility figures |
| `scripts/verify_release.py` | Offline privacy, secret, integrity, and syntax audit |
| `tests/` | Offline unit and smoke tests |

## Installation

Python 3.10 or newer is required. The tested release environment uses Python
3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The equivalent requirements-only installation is:

```bash
python -m pip install -r requirements.txt
```

## Verify the Release

The following commands do not access the network or call an LLM:

```bash
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

The verifier checks:

- prohibited secret and credential patterns;
- private absolute paths, internal hostnames, and personal identifiers;
- prohibited files such as `.env`, databases, private keys, and raw traces;
- Python syntax;
- the SHA-256 manifest for compact figure inputs;
- expected frozen benchmark cell counts.

## Reproduce Figures

Install the rendering dependencies and use only the checked-in compact inputs:

```bash
python -m pip install -r figure_source/requirements.txt
python scripts/verify_release.py
python figure_source/build_revision_figures.py
python controlled_topology_experiment/run_topology_control.py
python figure_source/fig7_distributed_evidence.py
python figure_source/fig8_limited_relay.py
python figure_source/fig9_round1_ablation.py
python figure_source/fig12_temporal_pairing_control.py
python figure_source/fig10_raw_snapshot_check.py
python figure_source/fig11_raw_rewire_null.py
```

These commands write regenerated files to `images/`. They do not issue API
requests. Optional raw-data reconstruction helpers are documented in
`figure_source/README.md`.

## Run the Core Simulations

A full ABM run:

```bash
python -m src.experiments.run_abm_experiment
```

A fast local smoke run:

```bash
python - <<'PY'
from src.experiments.abm_pgg import Condition, run_simulation

result = run_simulation(
    Condition.C,
    n_agents=30,
    n_rounds=20,
    seed_fraction=0.10,
    seed=42,
)
print(result.final_cooperation, result.final_norm_adoption)
PY
```

To reconstruct the observational analysis, download the public source datasets
under their original licenses and then run:

```bash
python -m src.data.download_hf --dataset all
python -m src.data.download_human_baselines
python -m src.analysis.run_study1 --mode quick
```

See `DATA_AVAILABILITY.md` before redistributing any downloaded records.

## LLM Experiments

No credential or private endpoint is stored in this repository. Copy the
template locally and fill only the providers you use:

```bash
cp .env.example .env
```

The gateway-based AgentPanel and frozen benchmark runners require:

```bash
export LLM_GATEWAY_API_KEY="..."
export LLM_GATEWAY_API_URL="https://gateway.example/v1/chat/completions"
export LLM_GATEWAY_API_BASE_URL="https://gateway.example/v1"
export LLM_GATEWAY_MODEL="your-model-id"
```

Provider-specific runners may instead use `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`. Keep `.env` untracked. The frozen
model identifiers may need adaptation for a different OpenAI-compatible
endpoint.

## Privacy and Release Boundary

This release includes aggregate metrics, compact frozen inputs, task manifests,
and selected figures. It excludes:

- `.env` files and all credentials;
- raw Parquet datasets and scraped databases;
- account, operator, profile, and agent-ID lists;
- raw model prompts and responses;
- safe-response headers, request IDs, and gateway logs;
- caches, virtual environments, archives, and internal documents.

The exclusion is deliberate: aggregate artifacts are sufficient for the
included figure and consistency checks without publishing unnecessary
identifiers or operational metadata.

## License

Code in this repository is released under the MIT License. Third-party datasets
and model services remain subject to their own terms. Generated results and
figures should be cited with the associated paper.
