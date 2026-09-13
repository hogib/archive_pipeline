"""The preprocessing definition, held in exactly one place.

Every product in this project -- noise baselines, cut training windows, encoded
tensors, continuous detector scores -- must condition samples identically, or
a detector trained on one is being run on another. Before this project the
definition existed twice: once in the dataset encoder and once, transcribed by
hand, in the continuous scanner, with a `verify` command whose job was to catch
them drifting apart. Both call sites now import from here and the drift is not
expressible.

The operation is: detrend linear, detrend constant, 5% Hann taper at each end,
4th-order Butterworth bandpass, in that order. Order matters -- tapering before
detrending leaves a step at the window edge that the filter rings on.
"""
import numpy as np
from scipy import signal


def taper_vector(n):
    """The 5% Hann cosine taper applied to each end of a window of length `n`."""
    t = np.ones(n)
    k = int(n * 0.05)
    if k > 0:
        w = signal.windows.hann(k * 2)
        t[:k] = w[:k]
        t[-k:] = w[k:]
    return t


def clean_block(x, fs, freqmin, freqmax, taper):
    """Conditions an `(n, win)` block of windows at once.

    Identical in operation and order to `clean_window`; scipy is given an axis
    instead of being called once per row, which is the whole reason a block
    form exists.

    Args:
        x: `(n, win)` array of raw samples. Not modified in place.
        fs: Sampling rate in Hz.
        freqmin: Bandpass low corner in Hz.
        freqmax: Bandpass high corner in Hz, clamped below Nyquist.
        taper: Taper of length `win`, from `taper_vector`.

    Returns:
        `(n, win)` array of conditioned samples.
    """
    x = signal.detrend(x, type="linear", axis=-1)
    x = signal.detrend(x, type="constant", axis=-1)
    x = x * taper
    b, a = _bandpass(fs, freqmin, freqmax)
    return x if b is None else signal.filtfilt(b, a, x, axis=-1)


def clean_window(x, fs, freqmin, freqmax):
    """Conditions a single 1-D window.

    The reference form. `clean_block` must agree with this row for row; that
    equivalence is asserted in `tests/test_clean.py`.
    """
    x = signal.detrend(np.asarray(x, dtype=np.float64), type="linear")
    x = signal.detrend(x, type="constant")
    x = x * taper_vector(len(x))
    b, a = _bandpass(fs, freqmin, freqmax)
    return x if b is None else signal.filtfilt(b, a, x)


def _bandpass(fs, freqmin, freqmax):
    """Butterworth coefficients, or `(None, None)` when the band is degenerate.

    A band whose high corner has been clamped to below its low corner cannot be
    filtered; the samples pass through untouched rather than the call failing,
    which is what happens at a station recorded below 2*freqmax.
    """
    nyquist = fs / 2.0
    high = freqmax if nyquist > freqmax else nyquist - 1.0
    if high <= freqmin:
        return None, None
    return signal.butter(4, [freqmin, high], btype="bandpass", fs=fs)
