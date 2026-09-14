# Products

Everything below is fed from one decode of the chunk it came from.

## Noise baseline — `baseline.json`

A window reaching the detector is `(x - mu) / sigma` in the station's own
long-term noise units, so `mu` and `sigma` are part of the model's input
contract, not a preprocessing detail. A baseline built at one passband is not
the standardization a scan at another passband needs, and nothing downstream
would notice the mismatch — so the conditioning travels inside the file, and
`apipe run` refuses a baseline whose band does not match the scan it is about
to feed. Baselines adopted from the older tooling carry no such block; they
were all built at the 1–45 Hz default, but since the file does not say so,
`load` returns `None` rather than assuming it.

Six chunks spread evenly across the archive, not taken from the front, so a
station that was noisy for its first month is not defined by it.

Two departures from building this on curated noise files, both stated rather
than hidden. A continuous record also contains the events — but earthquakes
occupy a vanishing fraction of a station-year, and the effect biases sigma
**up**, making the detector marginally more conservative. And segments are cut
into hour-long pieces before cleaning, because conditioning a 21-day trace
whole is neither affordable nor faithful to a pipeline that cleans short files;
this biases nothing, since a 5% taper attenuates the same *fraction* of any
piece.

## Per-event SNR — `range.csv`

For every catalogued event within 500 km, predict the P arrival with iasp91,
measure RMS in `[P-1, P+12]` against `[P-60, P-10]`, record the ratio. No
detector is involved: this measures the data.

A window is refused unless it lies wholly inside one contiguous segment. A
partial window says nothing about the station's reach, so it is dropped — and
counted. This table's row count is the denominator of every "fraction of events
reaching SNR 3" figure, so events vanishing silently would move those
percentages with nothing to show for it.

Written one part per chunk, then concatenated, so an interrupted station
resumes rather than restarting.

## Detector scores — `scores/<arm>/<chunk>.npz`

An **arm** is a window geometry plus a set of weights, given as
`NAME:WINDOW_SECONDS:CKPT_DIR:BRANCH[:STEP_SECONDS]`. Step defaults to window,
giving disjoint windows — so an alarm count is also a count of independent
decisions. Several arms window the same decoded segments, which is why running
two costs well under twice one.

Each `.npz` holds `t` (epoch, one per window) and `p` (ensemble-mean
probability).

### The model is transcribed, not imported

Every continuous arm in this project is `--channels 1d`, whose state dict is 42
tensors: one waveform branch, a projection, two fusion scalars, a head.
Depending on the training package for that would pull in six architectures and
their training machinery, into a project whose job is data.

So `products/detector.py` is that one configuration, written out, with
parameter names matching exactly. It is not trusted on inspection — see
[equivalence.md](equivalence.md).

The fusion scalars are kept even though a single-branch model has nothing to
fuse. The optimizer settled `w1` near but not at 1.0, and dropping it would
shift every logit.

### Refusing to guess a checkpoint

A checkpoint directory rarely holds one arm: `trained_model_branch1d_asinh`
holds nine spanning `lstm`, `cnn` and `cnn-lstm`. A bare glob averages across
all three, which is not an ensemble. `find_checkpoints` matches loosely — the
run tag grew over time, so a strict pattern rejects older checkpoints — and
then checks the *result*, requiring exactly one run identity or raising with
the candidates named.

Anchoring is load-bearing separately: `cnn` is a prefix of `cnn-lstm`, so the
trailing underscore is what keeps the two apart.

## Cut windows — `windows/<W>s/{eq,noise}/`

An event window starting `pre` seconds before the predicted P, and a paired
noise window from `noise_offset` seconds earlier. Every requested length is cut
in the same pass, and an event is kept only if **every** length is clean — so
the length series covers identical events, and a difference between lengths
cannot come from a difference in which events survived.

A window is refused unless all three components cover it inside one contiguous
segment. Nothing is padded across a gap.

### The filename is load-bearing

Windows are named `event_<id>_raw.mseed`, exactly. The encoder parses an event
id with `^(?:noise_)?event_(.+?)_raw$` and the capture is **non-greedy**, so a
station infix — `event_627233_MANT_raw` — parses as the event id
`627233_MANT`, which matches no catalogue row. Every window would lose its
magnitude label, silently, and the dataset would come out empty with no error
anywhere.

The station goes in the path instead, where it distinguishes two stations' cuts
without breaking the consumer. `tests/test_windows.py` pins this.

## Coincidence — `pairs/<A>-<B>/<arm>.csv`

Requiring two stations to agree within a travel-time window, priced against
what chance agreement would give at the same alarm budget. Read
[inventory.md](inventory.md) first: the test is bounded by joint coverage, and
[equivalence.md](equivalence.md) for which budget rows are stable.
