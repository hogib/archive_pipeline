"""Cutting arrival-anchored windows out of the continuous record.

These are the training and evaluation windows a magnitude regressor is built
from: an event window starting `pre` seconds before the predicted P, and a
paired noise window from `noise_offset` seconds earlier.

Every requested length is cut in the same pass. Decoding a chunk costs tens of
seconds where cutting costs milliseconds, so a second pass would pay the whole
read again for nothing.
"""
import numpy as np
from obspy import Stream, Trace, UTCDateTime

DEFAULT_PRE = 2.0
DEFAULT_NOISE_OFFSET = 300.0


def cut(segments, t0, n, fs):
    """An `(n, 3)` array at epoch `t0`, or None if any component is short.

    A window is returned only when all three components cover it inside a
    single contiguous segment. Anything spanning a gap is refused rather than
    padded, so no window is ever built across missing samples.
    """
    out = np.empty((n, 3), dtype=np.float64)
    for k in range(3):
        for s0, data in segments[k]:
            i = int(round((t0 - s0) * fs))
            if 0 <= i and i + n <= len(data):
                out[:, k] = data[i:i + n]
                break
        else:
            return None
    return out


def write_mseed(arr, comps, t0, station, path, fs):
    """Writes one `(n, 3)` window as a three-trace mseed file."""
    traces = []
    for k, code in enumerate(comps):
        tr = Trace(arr[:, k].astype(np.int32))
        tr.stats.network = "TU"
        tr.stats.station = station
        # The real channel code, not a hard-coded band. A window cut from an
        # accelerometer must not claim to be broadband.
        tr.stats.channel = code if len(code) > 1 else f"HH{code}"
        tr.stats.sampling_rate = fs
        tr.stats.starttime = UTCDateTime(t0)
        traces.append(tr)
    Stream(traces).write(str(path), format="MSEED")


def event_tag(event_id):
    """The filename stem a window must carry, and why it is exactly this.

    The dataset encoder parses an event id with `^(?:noise_)?event_(.+?)_raw$`,
    and the capture is non-greedy. An extra station infix -- `event_627227_MANT_raw`
    -- therefore parses as the event id `627227_MANT`, which matches no
    catalogue row. Every window would lose its magnitude label, silently, and
    the dataset would come out empty with no error anywhere.

    The station belongs in the path instead, where it distinguishes two
    stations' cuts without breaking the consumer.
    """
    return f"event_{int(event_id)}"


def cut_chunk(segments, comps, station, catalog, lengths, dirs, fs=100.0,
              pre=DEFAULT_PRE, noise_offset=DEFAULT_NOISE_OFFSET):
    """Cuts every usable event in one decoded chunk, at every length.

    Args:
        segments: Per-component `(t0, data)` lists, in Z/N/E order.
        comps: The three component codes.
        station: Station code, written into each trace header.
        catalog: Events with `EventID`, `Magnitude`, `p_epoch` and `cut_epoch`.
        lengths: Window lengths in seconds.
        dirs: Mapping of length to `(eq_dir, noise_dir)`.
        fs: Sampling rate in Hz.
        pre: Seconds before the anchor the window starts.
        noise_offset: Seconds before P the paired noise window is taken.

    Returns:
        Tuple of `(kept, skipped, per_magnitude_band)`.
    """
    if not segments[0]:
        return 0, 0, {}
    lo = min(s0 for seg in segments for s0, _ in seg)
    hi = max(s0 + len(d) / fs for seg in segments for s0, d in seg)
    sub = catalog[(catalog.p_epoch >= lo + noise_offset)
                  & (catalog.p_epoch <= hi - 60)]

    kept = skipped = 0
    bands = {}
    for ev in sub.itertuples():
        # An event is kept only if EVERY requested length is clean, so the
        # length series covers identical events and a difference between
        # lengths cannot come from a difference in which events survived.
        cuts = {}
        for w in lengths:
            n = int(round(w * fs))
            sig = cut(segments, ev.cut_epoch - pre, n, fs)
            noise = cut(segments, ev.p_epoch - noise_offset, n, fs)
            if sig is None or noise is None:
                cuts = None
                break
            cuts[w] = (sig, noise)
        if cuts is None:
            skipped += 1
            continue
        tag = event_tag(ev.EventID)
        for w, (sig, noise) in cuts.items():
            eq, nz = dirs[w]
            write_mseed(sig, comps, ev.cut_epoch - pre, station,
                        eq / f"{tag}_raw.mseed", fs)
            write_mseed(noise, comps, ev.p_epoch - noise_offset, station,
                        nz / f"noise_{tag}_raw.mseed", fs)
        band = int(min(ev.Magnitude, 6.0) * 2) / 2.0
        bands[band] = bands.get(band, 0) + 1
        kept += 1
    return kept, skipped, bands
