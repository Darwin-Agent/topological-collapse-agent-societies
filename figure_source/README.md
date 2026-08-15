# Figure Reproduction

Run these commands from the manuscript root to regenerate the figures currently
reported in the paper. They write PDF and PNG outputs to `images/`.

The tested rendering environment is Python 3.12.13:

```bash
python3 -m pip install -r figure_source/requirements.txt
```

```bash
# Fig. 1, Fig. 2, Fig. 3, Extended Data Fig. 1 and Extended Data Fig. 2
python3 figure_source/build_revision_figures.py

# Fig. 4: deterministic degree-matched synthetic topology control
python3 controlled_topology_experiment/run_topology_control.py

# Extended Data Fig. 3, Fig. 5, Extended Data Figs. 4--7
python3 figure_source/fig7_distributed_evidence.py
python3 figure_source/fig8_limited_relay.py
python3 figure_source/fig9_round1_ablation.py
python3 figure_source/fig12_temporal_pairing_control.py
python3 figure_source/fig10_raw_snapshot_check.py
python3 figure_source/fig11_raw_rewire_null.py

# Check compact artifact hashes, frozen counts, syntax, and release privacy
python3 scripts/verify_release.py
```

These figure commands do not issue model requests. They read the compact
archives in `figure_source/data` or the checked-in benchmark summaries and raw
records. The topology-control command deterministically rebuilds its synthetic
experiment from its declared master seed. The requirements file covers figure
rendering only; the benchmark runners have their own API execution requirements.
The release verifier is read-only and uses only the Python standard library.
It checks `figure_source/data/manifest.json`, a versioned SHA-256 inventory of
the compact inputs used for Figs. 1--3 and Extended Data Figs. 1--2 and 5--7.
It also scans for credentials, private paths, internal hostnames, prohibited
raw files, syntax errors, and unexpected frozen benchmark cell counts.

For Fig. 2, the verifier additionally rejects synthetic fallback reference
datasets. The comparison is limited to the Moltbook sample, six named
SocioPatterns temporal-contact traces and the declared unique Enron
dyadic-tie construction. It checks the six raw fingerprints, 300-s
trace-specific aggregation rule, retained dimensions and frozen displayed
metrics before the figure is accepted.

## Clean-build audit

Run the following from a fresh clone or an extracted `git archive`, rather
than from a directory containing existing LaTeX auxiliary files:

```bash
python3 scripts/verify_release.py
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The first command validates all compact figure inputs and frozen experiment
summaries. The second command rebuilds the bibliography, cross-references and
the complete manuscript. Neither command issues model or network requests.

All current manuscript figure renderers embed TrueType fonts in PDFs and set
stable PDF/PNG metadata. In the tested environment, repeating the commands
against the checked-in inputs produces byte-identical figure files. To verify
this locally, save hashes after one complete render, rerun the commands above,
then check the saved hashes:

```bash
sha256sum images/fig1_overview.pdf images/fig1_overview.png \
  images/fig2_collapse.pdf images/fig2_collapse.png \
  images/fig4_abm.pdf images/fig4_abm.png \
  images/fig5_degree_matched_topology_control.pdf images/fig5_degree_matched_topology_control.png \
  images/fig6_protocol_outcomes.pdf images/fig6_protocol_outcomes.png \
  images/fig7_distributed_evidence.pdf images/fig7_distributed_evidence.png \
  images/fig8_limited_relay.pdf images/fig8_limited_relay.png \
  images/fig9_round1_ablation.pdf images/fig9_round1_ablation.png \
  images/fig_temporal_pairing_control.pdf images/fig_temporal_pairing_control.png \
  images/fig_snapshot_consistency.pdf images/fig_snapshot_consistency.png \
  images/fig_raw_rewire_null.pdf images/fig_raw_rewire_null.png \
  images/fig_temporal_evolution.pdf images/fig_temporal_evolution.png > /tmp/nmi-figure-hashes
```

```bash
sha256sum -c /tmp/nmi-figure-hashes
```

Do not use the `run_*benchmark.py` scripts merely to reproduce the manuscript
figures: their `full` modes can resume or initiate API calls. The benchmark
README files describe those execution protocols separately.

## Optional Raw-File Checks

Extended Data Fig. 6 is a frozen directional consistency check, not a
re-estimate of the primary observation. To reconstruct it from the local
raw-file pair without using an API, use the project virtual environment:

```bash
python figure_source/rebuild_raw_snapshot_check.py \
  --posts data/raw/moltbook_hf/lnajt/posts.parquet \
  --comments data/raw/moltbook_hf/moltnet/data/v2026-02-28/comments.parquet \
  --output /tmp/raw_snapshot_check.json
```

The raw files originate from distinct public releases and are paired by post
identifier; this command therefore checks descriptive direction only. It does
not reproduce or replace the archived Fig. 2 input.

Extended Data Fig. 7 applies a fixed-length, node-hyperdegree- and
thread-size-preserving switching ensemble to the same alternative raw-file
pairing. It rejects duplicate participant--thread memberships, but does not
claim uniform sampling from a configuration-model distribution:

```bash
python figure_source/rebuild_raw_rewire_null.py \
  --posts data/raw/moltbook_hf/lnajt/posts.parquet \
  --comments data/raw/moltbook_hf/moltnet/data/v2026-02-28/comments.parquet \
  --output /tmp/raw_rewire_null.json
```

This command runs the 80-chain primary ensemble with three accepted swaps per
incidence, plus independent 20-chain diagnostics at one and five swaps per
incidence. The latter inspect sensitivity to the declared chain length only;
they do not demonstrate mixing or uniform configuration-model sampling. The
command does not use an API and does not reproduce or replace the archived
primary Fig. 2 input.

## Optional SocioPatterns Rebuild

The frozen six-trace SocioPatterns archive can be recreated from the locally
retained raw contact files with the same non-overlapping 300-s construction:

```bash
python figure_source/rebuild_sociopatterns_references.py \
  --output /tmp/fig2_sociopatterns_references.json
```

The helper uses no network or API request. It fixes every raw SHA-256
fingerprint in the generated output; the shared topology sampler sorts
candidates before drawing so a fixed seed is process-stable.

## Inputs and Outputs

| Manuscript item | Reproducible input | Renderer | Output |
| --- | --- | --- | --- |
| Fig. 1 | `fig1_overview_data.json` | `build_revision_figures.py` | `fig1_overview.pdf` |
| Fig. 2 | `fig2_observational_data.json` and `fig2_sociopatterns_references.json` | `build_revision_figures.py` | `fig2_collapse.pdf` |
| Fig. 3 | `fig4_abm_data.json` | `build_revision_figures.py` | `fig4_abm.pdf` |
| Fig. 4 | Deterministic synthetic construction | `run_topology_control.py` | `fig5_degree_matched_topology_control.pdf` |
| Fig. 5 | `llm_relay_benchmark/results/summary.json` | `fig8_limited_relay.py` | `fig8_limited_relay.pdf` |
| Extended Data Fig. 1 | `fig3_temporal_data.json` | `build_revision_figures.py` | `fig_temporal_evolution.pdf` |
| Extended Data Fig. 2 | `fig6_agentpanel_data.json` | `build_revision_figures.py` | `fig6_protocol_outcomes.pdf` |
| Extended Data Fig. 3 | `llm_task_benchmark/results/summary.json` | `fig7_distributed_evidence.py` | `fig7_distributed_evidence.pdf` |
| Extended Data Fig. 4 | One- and two-round benchmark summaries | `fig9_round1_ablation.py` | `fig9_round1_ablation.pdf` |
| Extended Data Fig. 5 | `temporal_pairing_control_summary.json` | `fig12_temporal_pairing_control.py` | `fig_temporal_pairing_control.pdf` |
| Extended Data Fig. 6 | `raw_snapshot_check.json` | `fig10_raw_snapshot_check.py` | `fig_snapshot_consistency.pdf` |
| Extended Data Fig. 7 | `raw_rewire_null.json` | `fig11_raw_rewire_null.py` | `fig_raw_rewire_null.pdf` |
