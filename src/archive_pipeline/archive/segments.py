"""Contiguous per-component segments from a decoded mseed stream.

This replaces ObsPy's `Stream.merge(method=1, fill_value=None).split()`, which
is the single most expensive operation in the whole pipeline. On a 21-day TDVMS
chunk (GCAM_2024-11-27, 0.51 GB packed, 1160 traces):

    unzip                        2.9 s
    obspy read                   3.2 s
    obspy merge + split         48.5 s      <- 89% of the chunk
    this module                  0.84 s     <- same output, bit for bit

The repo this was extracted from records chunks taking **2215 s** in that call
(MANT_2025-02-19), which is fragmentation cost and nothing else.

Why ObsPy is slow here is not mysterious: `merge` builds masked arrays over the
union of every trace, then `split` walks the mask to cut them apart again. The
mask is the only thing it is used for -- no caller here wants interpolated
samples across a gap -- so the round trip through it is pure overhead. Joining
sorted traces directly is O(n log n) and touches each sample once.

The equivalence is exercised by `tests/test_segments.py` against ObsPy on
synthetic streams covering the cases that actually occur in the archive:
contiguity, gaps, overlaps, single traces and out-of-order traces.

**What a "segment" is.** A `(t0, data)` pair where `data` is uninterrupted at
the nominal sampling rate and `t0` is the epoch time of its first sample.
Windows are only ever built inside one segment, so a window can never straddle
a data gap -- that guarantee is the reason the segmentation is kept at all
rather than the record being flattened into one array.
"""
import numpy as np

# Component roles, in the order the training encoder stacks them. Taking the
# first three channels alphabetically would grab ['1','2','E'] at a station
# with mixed sensor codes -- two horizontals and no vertical.
COMPONENT_ROLES = (("Z",), ("N", "1"), ("E", "2"))

# Instrument bands, most preferred first. A station can deliver more than one
# sensor: KURT carries a broadband (HH) and an accelerometer (HN) whose
# recorded intervals overlap for 14.5 days of a 21-day chunk. They are
# different instruments with different sensitivities, and selecting on the
# component letter alone silently merges them into one stream -- which is what
# a station whose "noise floor moves by a factor of 170" turns out to be.
#
# Broadband first because that is what the detector was trained on. The choice
# is made once per chunk and applied to all three components, so the three
# never come from different sensors.
BAND_PREFERENCE = ("HH", "BH", "EH", "SH", "HN", "BN", "EN")


def band_of(code):
    """The instrument band of a channel code: 'HHZ' -> 'HH'."""
    return code[:-1].upper()


def role_of(code):
    """The component role of a channel code: 'HHZ' -> 'Z'. Idempotent."""
    return code[-1].upper()


def pick_components(stream):
    """The three full channel codes to use, in Z/N/E role order.

    One instrument band is chosen for the whole chunk -- the most preferred
    band that can supply all three roles -- so the three components always
    come from the same sensor.

    Args:
        stream: A decoded ObsPy `Stream`.

    Returns:
        A list of three channel codes such as `['HHZ', 'HHN', 'HHE']`, or None
        when no single band covers all three roles. None means the chunk
        cannot be scored and callers skip it rather than mixing sensors.
    """
    have = {tr.stats.channel.upper() for tr in stream}
    bands = {band_of(c) for c in have}
    for band in [b for b in BAND_PREFERENCE if b in bands] + sorted(
            bands - set(BAND_PREFERENCE)):
        out = []
        for role in COMPONENT_ROLES:
            match = next((band + c for c in role if band + c in have), None)
            if match is None:
                break
            out.append(match)
        if len(out) == 3:
            return out
    return None


def component_segments(stream, comp, fs, dtype=np.float64):
    """Contiguous `(t0, data)` segments for one component.

    Traces are grouped into runs of continuous coverage -- a run extends while
    the next trace begins no later than half a sample after the current end --
    and each run is painted into one array in start-time order.

    Painting, rather than trimming and concatenating, is what makes this a
    drop-in for `merge(method=1)`, which resolves an overlap two different ways
    depending on its shape:

    - A trace that **extends past** what is already covered wins the overlap,
      on the reasoning that it carries the more correct time value. Appending
      only its non-overlapping tail would instead keep the earlier samples --
      agreeing everywhere the overlap is an identical redelivery, as all of
      them are in the archive this was built against, and diverging silently
      where it is not.
    - A trace **wholly contained** in what is already covered is discarded
      whole, not painted into the middle.

    Both rules are asserted against ObsPy in `tests/test_segments.py`; the
    second was found only by testing the case, having first been implemented
    the other way.

    Args:
        stream: A decoded ObsPy `Stream`, not merged.
        comp: A full channel code such as `'HHZ'`, which selects exactly that
            instrument, or a bare role letter such as `'Z'`, which selects
            every band -- kept only for reading data written before bands were
            distinguished, and never what `pick_components` returns.
        fs: Nominal sampling rate in Hz. Traces recorded at another rate are
            resampled to it, as ObsPy's path did.
        dtype: Output sample dtype.

    Returns:
        List of `(t0_epoch, data)` tuples, ordered in time, each contiguous.
    """
    traces = []
    want = comp.upper()
    exact = len(want) > 1          # a full code selects one instrument
    for tr in stream:
        code = tr.stats.channel.upper()
        if (code != want) if exact else (role_of(code) != want):
            continue
        if abs(tr.stats.sampling_rate - fs) > 1e-6:
            tr = tr.copy()
            tr.resample(fs)
        traces.append((tr.stats.starttime.timestamp, tr.data))
    if not traces:
        return []
    traces.sort(key=lambda t: t[0])

    dt = 1.0 / fs
    tol = 0.5 * dt
    segs = []
    for run in _runs(traces, dt, tol):
        t0 = run[0][0]
        if len(run) == 1:
            segs.append((t0, np.asarray(run[0][1], dtype=dtype)))
            continue
        total = max(int(round((s - t0) / dt)) + len(d) for s, d in run)
        buf = np.empty(total, dtype=dtype)
        covered = 0
        for s, d in run:
            lo = int(round((s - t0) / dt))
            if lo + len(d) <= covered:
                continue                    # contained: discarded, not painted
            buf[lo:lo + len(d)] = d
            covered = max(covered, lo + len(d))
        segs.append((t0, buf))
    return segs


def _runs(traces, dt, tol):
    """Splits start-sorted traces into runs of continuous coverage.

    A run ends where the next trace begins more than half a sample after the
    furthest point covered so far. `end` tracks the maximum rather than the
    last trace's end, so a short trace nested inside a long one does not look
    like a gap.
    """
    out = []
    cur = [traces[0]]
    end = traces[0][0] + len(traces[0][1]) * dt
    for start, data in traces[1:]:
        if start <= end + tol:
            cur.append((start, data))
            end = max(end, start + len(data) * dt)
        else:
            out.append(cur)
            cur, end = [(start, data)], start + len(data) * dt
    out.append(cur)
    return out


def make_windows(data, offset, n_windows, win, step):
    """An `(n_windows, win)` strided view over `data`, without copying."""
    sub = data[offset:offset + (n_windows - 1) * step + win]
    return np.lib.stride_tricks.as_strided(
        sub, shape=(n_windows, win),
        strides=(sub.strides[0] * step, sub.strides[0]), writeable=False)
