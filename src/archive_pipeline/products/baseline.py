"""Per-station noise statistics: the standardization the detector expects.

A window reaching the network is `(x - mu) / sigma` in the station's own
long-term noise units, so `mu` and `sigma` are part of the model's input
contract. A baseline built at one passband is not the standardization a scan
at another passband needs, and nothing downstream would notice the mismatch --
which is why the conditioning parameters travel with the baseline file.

## Which population sigma is estimated over

The training pipeline pools sum and sum-of-squares over **curated noise-only
files**. This uses the identical estimator, but over the **whole continuous
record** -- and that is not the same population. The tooling this replaces
argued the difference was negligible because earthquakes occupy a vanishing
fraction of a station-year. Earthquakes do; loud *non*-earthquake periods do
not, and they dominate a pooled variance:

    BAND, per chunk        sigma_Z 133, 155, 226, ...
    BAND, pooled over a
    6-chunk sample         sigma_Z 1743

A pooled variance is an RMS, so one chunk ten times louder than typical
contributes a hundred times the variance and sets the answer on its own.
Sampling six different chunks moves sigma by a factor of ten, which is not a
statistic of the station.

So both are computed. `sigma` is the pooled value, kept because it is what
training computed. `sigma_trimmed` pools only the quietest
`1 - trim` of hour-long pieces, which is the closer analogue of a corpus of
files selected for being quiet, and is what `sigma_for` returns by default.

Segments are cut into pieces before cleaning because conditioning a 21-day
trace whole is neither affordable in memory nor faithful to a pipeline that
cleans short files. This biases nothing on its own: the 5% Hann taper always
attenuates the same *fraction* of any piece, so its effect is the same at any
piece length. It is also what makes the trimmed statistic available.
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
# Share of the loudest pieces excluded from the trimmed statistic. 0.1 is
# enough to drop a storm or a glitch day without touching the bulk; it is not
# tuned, and the untrimmed value is always reported beside it.
DEFAULT_TRIM = 0.1


def accumulate(chunks, fs=100.0, freqmin=1.0, freqmax=45.0,
               piece_seconds=DEFAULT_PIECE_SECONDS, trim=DEFAULT_TRIM,
               log=print):
    """Streams sums and sums of squares per component over sampled chunks.

    Args:
        chunks: Chunk paths to read.
        fs: Sampling rate in Hz.
        freqmin: Bandpass low corner in Hz.
        freqmax: Bandpass high corner in Hz.
        piece_seconds: Length of the pieces each segment is cleaned in.
        trim: Share of the loudest pieces excluded from `sigma_trimmed`.
        log: Where progress goes.

    Returns:
        Dict of component to `{"mu", "sigma", "sigma_trimmed",
        "sigma_median", "n_samples", "n_pieces"}`.
    """
    accum, pieces = {}, {}
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
                    ps, pss, pn = float(c.sum()), float((c ** 2).sum()), c.size
                    s, ss, n = accum.get(comp, (0.0, 0.0, 0))
                    accum[comp] = (s + ps, ss + pss, n + pn)
                    pieces.setdefault(comp, []).append((ps, pss, pn))
        del stream
        log(f"  {path.stem}: done in {time.time() - t:.0f}s")

    out = {}
    for comp, (s, ss, n) in accum.items():
        mu = s / n
        per = np.array([math.sqrt(max(pss / pn - (ps / pn) ** 2, 0.0))
                        for ps, pss, pn in pieces[comp]])
        keep = per <= np.quantile(per, 1.0 - trim) if len(per) > 1 else per
        quiet = [t for t, k in zip(pieces[comp], keep) if k]
        qs = sum(x[0] for x in quiet)
        qss = sum(x[1] for x in quiet)
        qn = sum(x[2] for x in quiet)
        qmu = qs / qn if qn else mu
        out[comp] = {"mu": mu,
                     "sigma": math.sqrt(max(ss / n - mu ** 2, 0.0)),
                     "sigma_trimmed": math.sqrt(max(qss / qn - qmu ** 2, 0.0))
                     if qn else 0.0,
                     "sigma_median": float(np.median(per)),
                     "n_samples": n, "n_pieces": len(per)}
    return out


def sigma_for(stats, trimmed=True):
    """The sigma to standardize with.

    Args:
        stats: One component's entry from a baseline file.
        trimmed: Use the trimmed statistic where the file has one. Baselines
            written before it existed carry only `sigma`, and fall back to it.

    Returns:
        Positive float.
    """
    key = "sigma_trimmed" if trimmed else "sigma"
    return max(float(stats.get(key, stats["sigma"])), 1e-12)


def build(chunks, dest, fs=100.0, freqmin=1.0, freqmax=45.0,
          sample_chunks=DEFAULT_SAMPLE_CHUNKS,
          piece_seconds=DEFAULT_PIECE_SECONDS, trim=DEFAULT_TRIM, log=print):
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
    stats = accumulate(picked, fs, freqmin, freqmax, piece_seconds, trim, log)
    if not stats:
        log("[baseline] no usable components; nothing written")
        return None
    for comp, v in stats.items():
        log(f"  {comp}: mu={v['mu']:+.4g}  sigma={v['sigma']:.6g}  "
            f"trimmed={v['sigma_trimmed']:.6g}  "
            f"median-piece={v['sigma_median']:.6g}  "
            f"({v['n_samples'] / fs / 3600:.1f} h, {v['n_pieces']} pieces)")
        if v["sigma"] > 2 * v["sigma_trimmed"]:
            log(f"    ^ pooled sigma is {v['sigma'] / v['sigma_trimmed']:.1f}x "
                f"the trimmed one: a few loud pieces are setting it")
    # The conditioning travels with the statistics: a baseline is only valid
    # for a scan that filters the same way.
    stats["_conditioning"] = {"fs": fs, "freqmin": freqmin, "freqmax": freqmax,
                              "piece_seconds": piece_seconds, "trim": trim,
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
