# The noise baseline defect

A detector window is standardised as `(x - mu) / sigma` in the station's own
long-term noise units before it reaches the network. `sigma` is therefore not
a preprocessing detail — it is part of the model's input contract, and getting
it wrong rescales every input the detector ever sees.

It was wrong, at most stations, by a factor of five to fourteen.

## The estimator is right; the population is not

The training pipeline builds its baseline by pooling sum and sum-of-squares
across **curated noise-only files** — windows selected for containing no
event. The continuous scanner needs the same number for a record that has no
such curation, and it used the identical estimator over the **whole
continuous record**.

The tooling this replaces addressed the difference explicitly, and argued it
away:

> A continuous record also contains the events; a continuous record's
> earthquakes occupy a vanishing fraction of a station-year, so the effect on
> sigma is far below the precision that matters — and it biases sigma UP,
> making the detector marginally more conservative.

The reasoning is sound and the conclusion is wrong, because it is about the
wrong thing. Earthquakes are indeed a vanishing fraction of a station-year.
Loud *non-earthquake* periods are not: storms, cultural noise bursts,
instrument glitches, gain changes, and whatever a station does on a bad day.

## Why a pooled variance cannot survive them

A pooled variance is a mean of squares, so it is an RMS, and an RMS is
dominated by its largest term. Take `N` pieces of which one has `k` times the
typical amplitude:

    sigma_pooled^2  ~  ( k^2 + (N-1) ) / N  *  sigma_typical^2

At `N = 6` and `k = 30`, that is `sigma_pooled ≈ 12 x sigma_typical`. One bad
piece in six sets the answer, and the other five contribute nothing
measurable.

The baseline samples six chunks spread across the archive. So whether sigma
describes the station or describes its worst day is decided by which six
chunks the sampler happened to land on.

## What it measures instead

At BAND:

| measurement | sigma_Z |
|---|---|
| individual chunks | 133, 149, 155, 226, 329 |
| pooled over 6 of the 30 chunks now held | 235 |
| pooled over 6 of the 19 chunks held earlier | **1743** |
| within one quiet chunk: pooled / trimmed / median piece | 155.5 / 90.9 / 82.4 |

Re-sampling moves it by a factor of ten. That is not a statistic of a station.

The last row shows the same failure inside a single chunk, at smaller scale:
even a chunk with no outlier day has hours loud enough to put the pooled value
70% above the trimmed one.

**This is not an artefact of the reimplementation.** Run on one identical
chunk, the old and new implementations agree exactly — `sigma_Z 155.478`,
`mu +7.573e-08`, same sample counts. The estimator was faithfully reproduced;
the estimator is the problem.

## What it did to the scores

Baseline sigma tracks how compressed a station's score distribution is:

| station | sigma_Z | windows over 0.5 | distinct scores |
|---|---|---|---|
| KIRK | 100 | 8.8% | — |
| GCAM | — | 10.6% | 294,292 / 300,638 |
| ELBA | — | 39% | 297,610 / 302,400 |
| CMH | 1608 | 93.4% | — |
| MANT | — | 94.8% | 248,631 / 300,053 |
| BAND | 1743 | 96.1% | — |
| KAND | 1162 | 97.4% | — |
| SEMS | 1263 | 99.7% | 132,039 / 302,383 |
| KURT | 2490 | 99.9% | — |

The two stations whose baselines happened to be sane — KIRK and GCAM — are the
two with healthy score distributions. Everything else is pinned near the top of
the range.

Rescoring one BAND chunk with the same detector and the same weights, changing
only the standardisation:

| | median | p99 | p99.9 | over 0.5 | distinct |
|---|---|---|---|---|---|
| pooled sigma (1743) | 0.8201 | 0.8370 | 0.8580 | 99.3% | 242,242 / 302,054 |
| trimmed sigma (124) | 0.1114 | 0.4670 | 0.8229 | 0.9% | 285,382 / 302,054 |

The usable dynamic range — p50 to p99.9 — widens from 0.038 to 0.71, a factor
of 19, and 43,000 windows that had been tied become distinguishable. The
corrected distribution matches the shape of the two stations that were never
affected.

The mechanism is ordinary: fed inputs 14 times smaller than anything in its
training distribution, the network stops discriminating and emits something
close to a constant.

## It does not merely compress; it inverts

Compression alone would be survivable, since alarm thresholds are quantiles.
Scoring one chunk four ways with identical weights and measuring how well each
separates windows containing a catalogued arrival from background windows:

| events | pooled | trimmed | constant | per-window |
|---|---|---|---|---|
| SNR ≥ 10 | **0.3330** | 0.8487 | 0.8627 | 0.7932 |
| SNR ≥ 3 | **0.3533** | 0.7818 | 0.8098 | 0.7551 |
| all catalogued | 0.4510 | 0.5432 | 0.5548 | 0.5457 |

0.333 is below chance: under the deployed scale the detector scores clearly
recorded earthquakes *lower* than background. That is not a loss of resolution
and no threshold recovers it.

The first attempt at this measurement labelled every catalogued event a
positive and returned 0.45–0.49 for all four schemes, which looked like
"nothing distinguishes them". Only 27.5% of catalogued events near MANT reach
SNR 3 and 9.4% reach SNR 10; the rest leave no visible trace, so that label
measures the catalogue's reach rather than the detector. The bottom row above
is that measurement, kept as the reminder.

Do not over-read the gaps among the three well-scaled schemes: at SNR ≥ 10 the
comparison rests on 18 events and 77 positive windows. What the table supports
is the separation between the deployed scale and the rest.

## Scale-free standardisation

`--standardize perwindow` divides each window by its own mean and standard
deviation, so for any gain `a > 0` and offset `b`, mapping `x -> a*x + b`
leaves the input unchanged. Station gain, site noise, instrument replacement
and mid-archive gain changes all cancel identically; there is no per-station
quantity to estimate, nothing to recompute as the archive grows, and a station
with no history works immediately. PhaseNet and EQTransformer both do this.

It costs absolute amplitude, so the detector must discriminate on shape alone.
That it scores slightly below the station-scaled schemes above is expected
rather than damning: those weights were trained under a station scale, so a
per-window input is mismatched to them by construction. The comparison that
settles it is a detector trained under it.

KURT is the case no station-level scale handles. Its per-chunk sigma_Z runs
2.9, 2.9, 4.0, 9.2, 48.4, 61.8, 89.7, 496.5 — a factor of 170, not monotone in
time. Trimming fixes the estimator, not a station whose noise floor genuinely
moves; per-window is the only one of the four schemes that is correct for it.

## The fix

`accumulate` keeps per-piece statistics and reports three numbers:

- `sigma` — pooled, as training computes it. Kept because it is what training
  computed, and because dropping it would hide the problem rather than record
  it.
- `sigma_trimmed` — pooled over the quietest 90% of hour-long pieces. This is
  the closer analogue of a corpus of files selected for being quiet, which is
  what the training baseline was actually built from. **This is what the scan
  uses.**
- `sigma_median` — the median piece's sigma, reported as a check: when it sits
  far below the pooled value, the pooled value is being set by a tail.

`build` warns when pooled exceeds twice trimmed, and the file records the trim
and the names of the chunks sampled, so a later reader can tell what a baseline
was measured over. `sigma_for` falls back to `sigma` for baselines written
before the trimmed statistic existed, so adopted files still load rather than
failing.

The 10% trim is not tuned. It is enough to drop a storm or a glitch day without
touching the bulk, and the untrimmed value is always reported beside it.

## What it means for the findings

Every continuous detector score this project has produced — including the three
pair tables in the pilot report — was computed against a pooled baseline. The
correction applies to all of them, and the affected results are being recomputed
into a separate `6s-trim` arm so the two can be compared directly rather than
the correction being asserted.

What is **not** affected:

- **Per-event SNR** (`range.csv`). It is a ratio of RMS in two windows of the
  same trace and never touches the baseline.
- **Cut windows** and everything downstream of them, including the magnitude
  regression results. Windows are cut from raw samples; the encoder computes
  its own standardisation from its own noise corpus.
- **The decoder, the detector weights, and the coincidence arithmetic**, all of
  which are checked separately in [equivalence.md](equivalence.md).

What **is** affected is every continuous score, and the inversion above means
the effect cannot be bounded by the "thresholds are quantiles" argument. A
purely monotone compression would cancel out of the coincidence tables
entirely; an inverted ranking does not.

The reported recall deserves specific suspicion. A recall of 0.73 was reported
at one MANT operating point, which cannot be reconciled with an AUC of 0.33
unless most of those apparent detections are a saturated score crossing a
threshold rather than responding to signal.

Two saturated streams alarming on near-ties are also close to random with
respect to each other, which is indistinguishable in the table from two
stations whose false alarms are genuinely independent.

That matters most for the pairs that showed **no** excess. Provisional
measurements before the correction:

| pair | km | excess at 10/day |
|---|---|---|
| MANT–DEMI | 62.9 | 57.5x |
| KIRK–VIZE | 65.7 | 9.6x |
| SEMS–KAND | 45.4 | 6.2x |
| BAND–CMH | 39.8 | 5.7x |
| MANT–GCAM | 144.0 | 2.2x |
| ELBA–VIZE | 68.8 | 0.8x |
| DEMI–GCAM | 195.8 | 0.4x |
| SEMS–KURT | 54.5 | 0.0x |

SEMS–KURT and ELBA–VIZE involve the most saturated stations in the set, so
their near-independence is exactly the reading the defect would manufacture.
Read none of this column until the rescan lands.
