"""Per-station noise statistics: the standardization the detector expects.

A window reaching the network is `(x - mu) / sigma` in the station's own
long-term noise units, so `mu` and `sigma` are part of the model's input
contract. A baseline built at one passband is not the standardization a scan
at another passband needs, and nothing downstream would notice the mismatch --
which is why the conditioning parameters travel with the baseline file.

Two deliberate departures from building this on curated noise files:

- A continuous record also contains the events. Earthquakes occupy a vanishing
  fraction of a station-year, so the effect on sigma is far below the precision
  that matters, and it biases sigma **up**, making the detector marginally more
  conservative.
- Segments are cut into pieces before cleaning, because conditioning a 21-day
  trace whole is neither affordable in memory nor faithful to a pipeline that
  cleans short files. This does not bias anything: the 5% Hann taper always
  attenuates the same *fraction* of any piece, so its effect on sigma is the
  same at any piece length.
"""
import json
import math
import time

import numpy as np

from archive_pipeline.archive import (clean_block, component_segments,
                                      pick_components, read_chunk, taper_vector)

# Enough chunks to average over seasons and instrument changes, few enough that
# a baseline is minutes rather than a full pass. Spread evenly, not taken from
# the front, so a station that was noisy for its first month is not defined by it.
DEFAULT_SAMPLE_CHUNKS = 6
DEFAULT_PIECE_SECONDS = 3600.0


def accumulate(chunks, fs=100.0, freqmin=1.0, freqmax=45.0,
               piece_seconds=DEFAULT_PIECE_SECONDS, log=print):
    """Streams sums and sums of squares per component over sampled chunks.

    Args:
        chunks: Chunk paths to read.
        fs: Sampling rate in Hz.
        freqmin: Bandpass low corner in Hz.
        freqmax: Bandpass high corner in Hz.
        piece_seconds: Length of the pieces each segment is cleaned in.
        log: Where progress goes.

    Returns:
        Dict of component to `{"mu", "sigma", "n_samples"}`.
    """
    accum = {}
    for path in chunks:
        t = time.time()
        stream = read_chunk(path)
        comps = pick_components(stream)
        if comps is None:
            log(f"  {path.stem}: incomplete components, skipped")
            continue
        piece = int(round(piece_seconds * fs))
        for comp in comps:
            for _, data in component_segments(stream, comp, fs):
                for lo in range(0, len(data), piece):
                    part = data[lo:lo + piece]
                    if len(part) < fs * 10:
                        continue
                    c = clean_block(part[None, :], fs, freqmin, freqmax,
                                    taper_vector(len(part)))[0]
                    s, ss, n = accum.get(comp, (0.0, 0.0, 0))
                    accum[comp] = (s + float(c.sum()),
                                   ss + float((c ** 2).sum()), n + c.size)
        del stream
        log(f"  {path.stem}: done in {time.time() - t:.0f}s")

    out = {}
    for comp, (s, ss, n) in accum.items():
        mu = s / n
        out[comp] = {"mu": mu,
                     "sigma": math.sqrt(max(ss / n - mu ** 2, 0.0)),
                     "n_samples": n}
    return out


def build(chunks, dest, fs=100.0, freqmin=1.0, freqmax=45.0,
          sample_chunks=DEFAULT_SAMPLE_CHUNKS,
          piece_seconds=DEFAULT_PIECE_SECONDS, log=print):
    """Builds one station's baseline and writes it.

    Args:
        chunks: Every chunk of the station, in time order.
        dest: Where to write the JSON.
        fs: Sampling rate in Hz.
        freqmin: Bandpass low corner in Hz.
        freqmax: Bandpass high corner in Hz.
        sample_chunks: How many chunks to read, spread evenly.
        piece_seconds: Cleaning piece length in seconds.
        log: Where progress goes.

    Returns:
        The written dict, or None when no chunk yielded usable components.
    """
    take = np.linspace(0, len(chunks) - 1, min(sample_chunks, len(chunks)))
    picked = [chunks[int(round(i))] for i in take]
    log(f"[baseline] {len(picked)} of {len(chunks)} chunks, "
        f"{freqmin:g}-{freqmax:g} Hz")
    stats = accumulate(picked, fs, freqmin, freqmax, piece_seconds, log)
    if not stats:
        log("[baseline] no usable components; nothing written")
        return None
    for comp, v in stats.items():
        log(f"  {comp}: mu={v['mu']:+.4g}  sigma={v['sigma']:.6g}  "
            f"({v['n_samples'] / fs / 3600:.1f} h)")
    # The conditioning travels with the statistics: a baseline is only valid
    # for a scan that filters the same way.
    stats["_conditioning"] = {"fs": fs, "freqmin": freqmin, "freqmax": freqmax,
                              "piece_seconds": piece_seconds,
                              "chunks": [p.name for p in picked]}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(stats, indent=2) + "\n")
    log(f"[baseline] wrote {dest}")
    return stats


def load(path):
    """Reads a baseline, tolerating both the current and the older format.

    Returns:
        Tuple of `(per-component stats, conditioning dict or None)`. Baselines
        written by the tooling this replaces carry no conditioning block; they
        were all built at the 1-45 Hz default, but that is not recorded in the
        file, so None is returned rather than assumed.
    """
    data = json.loads(path.read_text())
    conditioning = data.pop("_conditioning", None)
    return data, conditioning
