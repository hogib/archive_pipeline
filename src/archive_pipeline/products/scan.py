"""Scoring every window of a continuous record with a trained detector.

One decode feeds every arm. Reading and decoding a chunk is a fixed cost that
should not be paid per detector, so an arm is a window geometry plus a set of
weights, and several of them window the same segments independently.

Windows are disjoint by default -- step equals window -- so an alarm count is
also a count of independent decisions. A window is only ever built inside one
contiguous segment, so none straddles a data gap.
"""
import dataclasses
import time

import numpy as np

from archive_pipeline.archive import (clean_block, clip_spans, common_spans,
                                      make_windows, taper_vector)
from archive_pipeline.products.detector import load_ensemble, score_block

DEFAULT_BATCH = 1024
DEFAULT_BLOCK_WINDOWS = 20000


@dataclasses.dataclass
class Arm:
    """One detector's window geometry and weights.

    Built from `NAME:WINDOW_SECONDS:CKPT_DIR:BRANCH[:STEP_SECONDS]`.
    """
    name: str
    window_seconds: float
    ckpt_dir: str
    branch: str
    step_seconds: float
    fs: float = 100.0
    models: list = None

    @classmethod
    def parse(cls, spec, fs=100.0):
        """Builds an arm from its spec string, without loading weights.

        Raises:
            ValueError: The spec has the wrong number of fields.
        """
        parts = spec.split(":")
        if len(parts) not in (4, 5):
            raise ValueError(
                f"--arm wants NAME:WINDOW:CKPT_DIR:BRANCH[:STEP], got {spec!r}")
        return cls(name=parts[0], window_seconds=float(parts[1]),
                   ckpt_dir=parts[2], branch=parts[3],
                   step_seconds=float(parts[4]) if len(parts) == 5
                   else float(parts[1]), fs=fs)

    @property
    def win(self):
        return int(round(self.window_seconds * self.fs))

    @property
    def step(self):
        return int(round(self.step_seconds * self.fs))

    def load(self, device, hidden=48, fusion_dim=96):
        """Loads this arm's checkpoint ensemble onto `device`."""
        self.models = load_ensemble(self.ckpt_dir, self.branch, device,
                                    hidden=hidden, fusion_dim=fusion_dim)
        return self


def score_chunk(arm, seg_lists, comps, baseline, device, fs=100.0,
                freqmin=1.0, freqmax=45.0, batch_size=DEFAULT_BATCH,
                block_windows=DEFAULT_BLOCK_WINDOWS, near=None, pool=None):
    """Scores every window of one chunk with one arm.

    Args:
        arm: A loaded `Arm`.
        seg_lists: Per-component segment lists, in Z/N/E order.
        comps: The three component codes, matching `seg_lists`.
        baseline: Per-component `{"mu", "sigma"}` from the station's baseline.
        device: Torch device.
        fs: Sampling rate in Hz.
        freqmin: Bandpass low corner in Hz.
        freqmax: Bandpass high corner in Hz.
        batch_size: Windows per forward pass.
        block_windows: Windows conditioned per vectorized block. Batches are
            filled *across* span boundaries: a chunk where the station drops
            out hundreds of times yields many tiny spans, and conditioning one
            block per span paid full overhead on each.
        near: Optional intervals to restrict scoring to.
        pool: Optional thread pool for the filtering. scipy's detrend and
            filtfilt release the GIL, so threads give real parallelism here.

    Returns:
        Tuple of `(t, p)` float arrays, or `(None, None)` when the chunk has no
        unbroken three-component span long enough for one window.
    """
    win, step, taper = arm.win, arm.step, taper_vector(arm.win)
    spans = clip_spans(common_spans(seg_lists, fs, win), near, fs, win, step)

    times, views_of, nwin_of = [], {}, {}
    for i, (t0, where, n_samp) in enumerate(spans):
        n_win = (n_samp - win) // step + 1
        if n_win < 1:
            continue
        views_of[i] = [make_windows(seg_lists[k][sj][1], off, n_win, win, step)
                       for k, (sj, off) in enumerate(where)]
        nwin_of[i] = n_win
        times.append(t0 + (np.arange(n_win) * step) / fs)
    if not times:
        return None, None

    mus = np.array([baseline[c]["mu"] for c in comps])
    sigmas = np.array([max(baseline[c]["sigma"], 1e-12) for c in comps])
    probs = []

    def flush(pending, total):
        """Conditions, standardizes and scores one cross-span batch."""
        blk = np.empty((total, win, 3), dtype=np.float32)
        tasks, dest = [], 0
        for si, a, b in pending:
            tasks.append((si, a, b, dest))
            dest += b - a

        def fill(task):
            sj, a, b, d = task
            for k in range(3):
                c = clean_block(np.array(views_of[sj][k][a:b]), fs,
                                freqmin, freqmax, taper)
                blk[d:d + (b - a), :, k] = (c - mus[k]) / sigmas[k]

        list(map(fill, tasks) if pool is None else pool.map(fill, tasks))
        for lo in range(0, total, batch_size):
            probs.append(score_block(arm.models, blk[lo:min(lo + batch_size, total)],
                                     device))

    pending, total = [], 0
    for si in sorted(views_of):
        lo = 0
        while lo < nwin_of[si]:
            take = min(block_windows - total, nwin_of[si] - lo)
            pending.append((si, lo, lo + take))
            total += take
            lo += take
            if total >= block_windows:
                flush(pending, total)
                pending, total = [], 0
    if total:
        flush(pending, total)

    return np.concatenate(times), np.concatenate(probs).astype(np.float32)


def write_scores(dest, t, p):
    """Writes one chunk's scores, creating the arm directory."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, t=t, p=p)
