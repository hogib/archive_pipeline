# Usage

```bash
uv sync
uv run apipe                     # the listing
```

Paths live in `archive_pipeline.toml` and resolve against that file, not the
working directory, so commands behave the same wherever they are run from. Any
field is overridable per invocation with the matching flag.

```toml
archive  = "../tdvms/afad_raw"
stations = "../cnn_earthquake/catalogs/istasyon_katalog.csv"
catalog  = "../cnn_earthquake/catalogs/catalog_current.csv"
out      = "out"
ledgers  = ["../tdvms/tdvms_ledger.jsonl", "..."]
```

## Look first

```bash
apipe inventory                  # what exists, and which pairs are testable
apipe inventory --no-pairs       # station table only
apipe plan --all                 # what a batch would do
```

`inventory` also reports what it *skipped* — in-flight downloads, duplicate
redeliveries — so a station whose chunk count looks short says why.

## Build

```bash
apipe baseline --all             # minutes; scanning cannot start without it
apipe run --all                  # the pass
```

`run` takes the same selection flags as `plan`, so check the plan and then run
exactly it.

```bash
apipe run --station SEMS --station KAND
apipe run --all --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm
apipe run --all --no-range       # scores only
apipe run --station SEMS --limit-chunks 1     # smoke test
```

**Smoke-test a new station with `--limit-chunks 1` before committing hours.**
One chunk is about a minute and catches a missing baseline, a station absent
from the catalogue, or an arm spec that resolves to the wrong checkpoints.

Interrupting is safe. Re-running resumes at the chunk it stopped on.

## Migrate

```bash
apipe adopt --from ../cnn_earthquake --also ../cascade_impl --dry-run
apipe adopt --from ../cnn_earthquake --also ../cascade_impl
```

`--also` exists because one station's noise baseline had been written into a
different repository.

## Conditioning

`--fs`, `--freqmin` and `--freqmax` must match the baseline the scan is fed.
`run` checks this and refuses a mismatch, for baselines that record it.
