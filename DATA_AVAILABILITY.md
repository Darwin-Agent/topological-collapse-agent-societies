# Data Availability and Privacy Boundary

## Included

The repository contains:

- aggregate topology and experiment summaries;
- deterministic synthetic-control outputs;
- frozen task manifests and protocol descriptions;
- compact figure inputs with a SHA-256 inventory;
- selected publication figures.

These files are sufficient for the included release audit and figure
rendering.

## Excluded

The public release does not contain:

- raw Moltbook posts, comments, profiles, or account identifiers;
- raw SocioPatterns or Enron source files;
- SQLite databases;
- scraped agent/operator lists;
- raw LLM prompts, responses, or per-request traces;
- response headers, request IDs, tokens, or gateway logs.

The raw records are unnecessary for the compact checks and could carry
identifiers or service metadata. They should be distributed only through an
appropriate data-access process and under the original source licenses.

## Public Source Acquisition

Download helpers are available in `src/data/`. They write into `data/raw/`,
which is ignored by Git. Verify the source terms before downloading,
processing, or redistributing any dataset.

## Artifact Provenance

`figure_source/data/manifest.json` records SHA-256 hashes for the compact inputs
used by the figure renderers. Run `python scripts/verify_release.py` to validate
the inventory.
