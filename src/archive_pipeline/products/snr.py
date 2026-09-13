"""How far out a station actually recovers catalogued events.

Waveform training corpora in this project were only ever requested within
~56 km of an epicentre, so nothing in them supports a statement about
attenuation with distance. Continuous data removes that ceiling: every
catalogued event inside the recorded span has a waveform at the station,
whether or not anyone picked it.

For each catalogued event the P arrival is predicted with iasp91, RMS in a
signal window is measured against a pre-arrival noise window, and the ratio is
recorded. No detector is involved -- this measures the data.

The conditioning here is deliberately **not** the detector's. Signal-to-noise
is measured in a 2-20 Hz band, where regional P energy sits, rather than the
1-45 Hz the network is fed; widening it would fold in high-frequency site noise
that the ratio is meant to be measured against.
"""
import numpy as np
import pandas as pd
from scipy import signal as sp

from archive_pipeline.arrivals import ArrivalTimes
from archive_pipeline.inventory.stations import haversine_km

NOISE_WIN = (-60.0, -10.0)    # seconds relative to predicted P
SIGNAL_WIN = (-1.0, 12.0)     # a little before, to tolerate model error
DEFAULT_BAND = (2.0, 20.0)


def load_catalog(path, slat, slon, max_distance=500.0, min_magnitude=0.0):
    """Catalogued events near one station, with epicentral distance attached.

    Args:
        path: Event catalogue CSV.
        slat: Station latitude.
        slon: Station longitude.
        max_distance: Drop events beyond this, in km.
        min_magnitude: Drop events below this.

    Returns:
        DataFrame with `t` (UTC timestamp) and `dist` (km) columns added.
    """
    cat = pd.read_csv(path, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S",
                              errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat["dist"] = haversine_km(slat, slon, cat.Latitude.values,
                               cat.Longitude.values)
    return cat[(cat.dist <= max_distance)
               & (cat.Magnitude >= min_magnitude)].copy()


def condition(segments, fs, band=DEFAULT_BAND):
    """Demeans and bandpasses each segment, returning new `(t0, data)` pairs."""
    b, a = sp.butter(4, list(band), btype="bandpass", fs=fs)
    return [(t0, sp.filtfilt(b, a, sp.detrend(d, type="constant")))
            for t0, d in segments]


def _slice(segments, fs, start, stop):
    """Samples between two epoch times, only if wholly inside one segment.

    Returns None when the window straddles a gap or an edge. That is the point
    of keeping the segmentation: a partial window says nothing about the
    station's reach, so it is refused rather than measured short.
    """
    want = int(round((stop - start) * fs))
    for t0, data in segments:
        lo = int(round((start - t0) * fs))
        if lo < 0 or lo + want > len(data):
            continue
        return data[lo:lo + want]
    return None


def measure(segments, fs, catalog, taup=None, band=DEFAULT_BAND):
    """Per-event signal-to-noise for every catalogued event in this record.

    Args:
        segments: Vertical-component `(t0, data)` segments, unconditioned.
        fs: Sampling rate in Hz.
        catalog: Output of `load_catalog`, already restricted to the span.
        taup: An `ArrivalTimes`, shared across chunks so its cache is reused.
        band: Bandpass corners in Hz.

    Returns:
        Tuple of `(rows, dropped)` -- a list of dicts and a count of events
        skipped because a window straddled a gap. The count is returned rather
        than swallowed: this table's row count is the denominator of every
        "fraction of events reaching SNR 3" figure, so events vanishing here
        would move those percentages with nothing to show for it.
    """
    taup = taup or ArrivalTimes(grid_km=5.0)
    segs = condition(segments, fs, band)
    rows, dropped = [], 0
    for ev in catalog.itertuples():
        depth = float(ev.Depth) if pd.notna(ev.Depth) else 10.0
        tt = taup.travel(ev.dist, depth)
        if tt is None:
            continue
        p = ev.t.timestamp() + tt
        noise = _slice(segs, fs, p + NOISE_WIN[0], p + NOISE_WIN[1])
        sig = _slice(segs, fs, p + SIGNAL_WIN[0], p + SIGNAL_WIN[1])
        if noise is None or sig is None:
            dropped += 1
            continue
        n_rms = float(np.sqrt(np.mean(noise ** 2)))
        if n_rms <= 0:
            dropped += 1
            continue
        rows.append({"event_id": ev.EventID, "time": ev.t, "mag": ev.Magnitude,
                     "dist_km": ev.dist, "depth": ev.Depth,
                     "snr": float(np.sqrt(np.mean(sig ** 2))) / n_rms,
                     "location": ev.Location})
    return rows, dropped


def span_of(segments):
    """First and last epoch time covered, or None for an empty record."""
    if not segments:
        return None
    return segments[0][0], segments[-1][0] + len(segments[-1][1]) / 100.0


def concatenate(parts, dest):
    """Merges per-chunk SNR tables into the station's one table.

    Args:
        parts: Per-chunk CSV paths.
        dest: Where the concatenated table goes.

    Returns:
        The concatenated DataFrame.
    """
    frames = [pd.read_csv(p) for p in sorted(parts) if p.stat().st_size]
    df = (pd.concat(frames, ignore_index=True) if frames
          else pd.DataFrame(columns=["event_id", "time", "mag", "dist_km",
                                     "depth", "snr", "location"]))
    df = df.drop_duplicates(subset="event_id").sort_values("time")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df
