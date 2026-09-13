# Equivalence

Nothing here is trusted because it looks right. Each claim below is a check
that fails loudly.

## The decoder against ObsPy

`tests/test_segments.py` builds streams covering the shapes the archive has and
asserts segment-for-segment, sample-for-sample equality with
`Stream.merge(method=1, fill_value=None).split()`: single traces, contiguous
joins, gaps, out-of-order input, duplicate traces, a 300-fragment stream with
scattered gaps, and both overlap rules.

On a real chunk (`GCAM_2024-11-27`, 1160 traces):

| component | segments | samples | |
|---|---|---|---|
| Z | 205 | 105,499,600 | identical |
| N | 199 | 105,489,525 | identical |
| E | 203 | 105,493,105 | identical |

0.33 s against ObsPy's 41.2 s.

## The detector against its checkpoints

`tests/test_detector.py` loads the project's nine real checkpoints with
`load_state_dict(strict=True)` across all three branches. Strict is the whole
point: it fails on any parameter missing, extra or wrongly shaped, where
anything looser would let a transcription error through as scores that look
plausible.

It also asserts that an unnarrowed checkpoint directory **refuses** to
ensemble — `trained_model_branch1d_asinh` holds three distinct runs, and
averaging across them is not an ensemble but a mixture of models answering
different questions — and that `cnn` and `cnn-lstm` never overlap, since one
name is a prefix of the other.

## The products against the tooling they replace

Run on chunks the old tools had already produced:

| check | result |
|---|---|
| MANT_2024-05-01 scores | 300,053 windows both; timestamps identical; max \|dp\| 1.6e-4; alarms over 0.5 identical at 284,365 |
| GCAM_2024-05-01 scores | 300,638 windows both; timestamps identical; max \|dp\| 1.6e-4; alarms over 0.5 identical at 31,730 |
| GCAM per-event SNR | 670 of 670 events matched; correlation 1.000000; median relative difference 2.9e-4; worst 1.1% on an SNR-20 event; **zero events cross the SNR ≥ 3 cut** |

The residual is float32 and cuDNN nondeterminism, not a difference in what is
computed. Distances agree to 4e-13 km.

## Where that residual matters

1.6e-4 is small against a probability but not against the *spread* of these
scores, which differs enormously by station:

| station | median | p90 | p99.9 | distinct values |
|---|---|---|---|---|
| GCAM | 0.131 | 0.520 | 0.854 | 294,292 / 300,638 |
| ELBA | 0.608 | 0.801 | 0.882 | 297,610 / 302,400 |
| MANT | 0.816 | 0.830 | 0.837 | 248,631 / 300,053 |
| SEMS | 0.835 | 0.837 | 0.839 | 132,039 / 302,383 |
| DEMI | 0.836 | 0.838 | 0.848 | 108,296 / 302,004 |

At SEMS and DEMI, 99.9% of windows fall in a band 0.004 wide, and a third of
window scores are ties. Budget-calibrated thresholds land inside that band, so
the share of alarm decisions that sit within reproduction noise of their own
threshold is worth knowing:

| budget | MANT: alarms within ±1.6e-4 of threshold | DEMI |
|---|---|---|
| 100/day | 18.5% | 13.9% |
| 30/day | 15.7% | 3.2% |
| 10/day | 1.0% | 1.3% |
| 3/day | 2.9% | 0.0% |
| 1/day | 5.8% | 5.1% |
| 0.1/day | 29.6% | 20.0% |

**The headline operating points are safe.** The 10/day and 3/day rows — which
carry the 57.5× and 177× excess-coincidence figures — are stable to within a
few percent of their alarms. The 0.1/day row is not: it rests on roughly 27
alarms at MANT, a fifth to a third of which are within numerical noise of the
cut, and it should be read as an order of magnitude rather than a number.

The loose 100/day and 30/day rows are also soft, for the same reason in reverse:
the threshold sits at the mode of a compressed distribution where a great many
windows are nearly tied.
