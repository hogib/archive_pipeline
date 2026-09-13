# Decoding

`archive/` is the only layer that touches ObsPy. Everything above it consumes
`(t0, data)` segments.

## The chunk

A TDVMS campaign delivers one zip per request, holding a single mseed of about
three weeks of continuous three-component record. Chunks are named
`<STATION>_<YYYY-MM-DD>.zip`, one directory per station.

`find_chunks` matches that name exactly, which is load-bearing in two ways. A
pull writing into the same tree leaves `incoming.*.zip.part` files, and SEMS
carries nine `SEMS_<date>.dup<id>.zip` redeliveries that are byte-identical to
chunks already held. Globbing `*.zip` picks up both. The redeliveries are the
dangerous case: they decode fine, score fine, and silently count nine spans
twice in every per-day denominator downstream. `apipe inventory` reports what it
skipped and why, so a station whose count drops by nine says so.

## Segments, not a merged stream

A **segment** is a `(t0, data)` pair where `data` is uninterrupted at the
nominal sampling rate. Windows are only ever built inside one segment, so a
window can never straddle a data gap. That guarantee is the reason the
segmentation is kept rather than the record being flattened.

`component_segments` sorts a component's traces by start time, groups them into
runs of continuous coverage, and paints each run into one array.

### Matching `merge(method=1)` takes two rules

Painting in start-time order is not enough on its own, because ObsPy resolves
an overlap two different ways depending on its shape:

- A trace **extending past** what is already covered wins the overlap, on the
  reasoning that it carries the more correct time value.
- A trace **wholly contained** in what is already covered is discarded whole.

The first implementation here appended only the non-overlapping tail, keeping
the earlier samples. It matched ObsPy on every real chunk tested, because every
overlap in this archive is an identical redelivery — and it was wrong, by 500
samples of 1500, the moment the overlap carried different data. The second
implementation painted in both cases, which is right for the first rule and
wrong for the second.

Both shapes are now pinned against ObsPy in `tests/test_segments.py`, along with
contiguity, gaps, out-of-order input, duplicate traces, and a 300-fragment
stream with scattered gaps — the shape real chunks have.

## Conditioning

`archive/clean.py` holds the preprocessing definition once: detrend linear,
detrend constant, 5% Hann taper each end, 4th-order Butterworth bandpass, in
that order. Tapering before detrending leaves a step at the window edge that
the filter rings on.

Before this project the definition existed twice — once in the dataset encoder
and once transcribed by hand into the continuous scanner — with a `verify`
command whose job was to catch them drifting apart. Both call sites import from
here now and the drift is not expressible.

Signal-to-noise measurement is the one deliberate exception: it uses a 2–20 Hz
band rather than the detector's 1–45 Hz, because a ratio meant to measure
regional P energy should not fold in high-frequency site noise. That is stated
at the call site, not left to be discovered.
