"""Turning scored windows into declarations, and matching them across stations."""
import glob

import numpy as np


def load_scores(pattern):
    """Every scored window from one glob, sorted by time.

    Raises:
        FileNotFoundError: Nothing matched, named so a typo in a station code
            does not read as a station with no alarms.
    """
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no score files matched {pattern}")
    t = np.concatenate([np.load(f)["t"] for f in files])
    p = np.concatenate([np.load(f)["p"] for f in files])
    order = np.argsort(t)
    return t[order], p[order]


def declarations(t, p, thr, cluster_seconds):
    """Alarms above `thr`, collapsed to one declaration per burst.

    A noise burst spanning ten windows is one declaration, not ten.

    Returns:
        Tuple of `(time, score)` at each burst's peak.
    """
    hit = np.flatnonzero(p > thr)
    if not len(hit):
        return np.empty(0), np.empty(0)
    cuts = np.flatnonzero(np.diff(t[hit]) > cluster_seconds)
    return (lambda peak: (t[peak], p[peak]))(
        [g[np.argmax(p[g])] for g in np.split(hit, cuts + 1)])


def confirmed(ta, tb, window):
    """Mask over A's declarations: does B declare within +/- `window`?

    Both arrays are sorted, so this is two binary searches per declaration
    rather than a cross product -- the alarm lists run to tens of thousands at
    a loose threshold.
    """
    if not len(ta) or not len(tb):
        return np.zeros(len(ta), dtype=bool)
    lo = np.searchsorted(tb, ta - window, side="left")
    hi = np.searchsorted(tb, ta + window, side="right")
    return hi > lo


def background_mask(t, cat, win_s, guard_pre=10.0, guard_post=60.0):
    """Which scored windows are explained by a catalogued event.

    The `- win_s` on the lower edge makes this an overlap test rather than a
    start-time test: a window beginning before the guard still reaches into it.

    Returns:
        Boolean array, True where the window overlaps some event's guard.
    """
    explained = np.zeros(len(t), dtype=bool)
    for a, b in zip(cat.p_epoch.values - guard_pre - win_s,
                    cat.p_epoch.values + guard_post):
        explained[np.searchsorted(t, a):np.searchsorted(t, b, side="right")] = True
    return explained
