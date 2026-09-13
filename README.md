# archive_pipeline

Turns a continuous seismic archive into model-ready artifacts: noise baselines,
per-event signal-to-noise, arrival-anchored windows, encoded tensors,
continuous detector scores and two-station coincidence tables.

It does this in **one decode pass per chunk, for every station at once**.

```bash
uv sync
uv run apipe inventory          # what data exists, and which pairs are testable
uv run apipe plan --all         # what a batch would do
uv run apipe run --all          # do it; safe to interrupt and re-run
```

## Why it exists

The tools this replaces each globbed the archive, unzipped, and decoded
independently — three full passes per station before anything was produced, and
a fourth sampled pass for the noise baseline. They all consumed the same
decoded stream. Here the stream is decoded once and fanned out to every
product.

Decoding was also slower than it needed to be. 89% of the cost of a chunk was
ObsPy's `Stream.merge()`, which builds masked arrays over every trace so that
`split()` can immediately cut them apart again. `archive/segments.py` joins
sorted traces directly instead: bit-identical output, 60× faster, and the
pathological chunks that took 2215 s no longer exist.

## Layout

| package | what it owns |
|---|---|
| `archive/` | zip → decoded → contiguous conditioned arrays |
| `inventory/` | which data exists, reconciled across campaign ledgers and disk |
| `products/` | baselines, SNR, windows, detector scores, coincidence |
| `encode/` | windows → tensors, the one definition of the preprocessing |
| `batch/` | the one-pass driver, and its resume logic |

Paths live in `archive_pipeline.toml` so no command carries them.
