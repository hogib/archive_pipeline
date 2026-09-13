# archive_pipeline documentation

Technical reference for the processing in `src/archive_pipeline`. The
top-level [`README.md`](../README.md) is the short version; these pages are the
detail behind it.

| page | what it covers |
|---|---|
| [decoding.md](decoding.md) | zip to contiguous arrays, and why ObsPy's merge is not used |
| [inventory.md](inventory.md) | four ledgers, one archive, which pairs are testable |
| [products.md](products.md) | baselines, per-event SNR, detector scores, coincidence |
| [batch.md](batch.md) | the one-pass driver, the output layout, resuming |
| [usage.md](usage.md) | the CLI, every command, worked runs |
| [equivalence.md](equivalence.md) | **what was checked against the tooling this replaces** |

## Orientation

The archive is 216 chunks across 13 stations, about 133 GB packed, each chunk a
zip holding roughly three weeks of continuous three-component record at 100 Hz.
Out of it this project builds the inputs a detector and a magnitude regressor
need, and the tables a two-station coincidence test is read from.

### Why the repo exists

Two reasons, both measured rather than assumed.

**One decode, not four.** The tools this replaces — `sk station-range`,
`sk cut-events`, `sk falsealarm baseline`, `sk falsealarm scan` — each globbed
the archive, unzipped to a temporary directory, and decoded with ObsPy,
independently. They all consumed the same decoded stream. On the archive as it
stands that is 1004 chunk decodes where 180 would do.

**Decoding was slower than the work in it.** On `GCAM_2024-11-27` (0.51 GB
packed, 1160 traces):

| stage | seconds | share |
|---|---|---|
| unzip | 2.9 | 5% |
| ObsPy mseed read | 3.2 | 6% |
| ObsPy `merge()` + `split()` | 48.5 | **89%** |

`merge` builds masked arrays spanning every trace so `split` can immediately
cut them apart again. Nothing downstream wants interpolated samples across a
gap, so the entire round trip is overhead. Joining sorted traces directly gives
the same segments, bit for bit, in 0.33 s. The repo this came from records
individual chunks taking **2215 s** in that call; those stop existing.

### What it does not do

No training, no model selection, no reporting. A trained detector is loaded and
run, because scoring a continuous record is a data product, but the architecture
here is inference-only — see [products.md](products.md).
