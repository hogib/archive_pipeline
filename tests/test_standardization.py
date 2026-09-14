"""The standardisation schemes, and the invariance that is the point of one of them.

These run the real `score_chunk` path with a stub network, because the claim
worth testing is a property of the pipeline's arithmetic and not of any
trained weights: per-window standardisation must be invariant under a change
of gain, and the station-scaled schemes must not be.
"""
import numpy as np
import pytest
import torch

from archive_pipeline.products import scan as sc


class MeanModel(torch.nn.Module):
    """Returns the mean absolute value of each window, as a logit.

    Monotone in amplitude, so any change in the standardised input shows up in
    the output. Stands in for a detector without needing one.
    """

    def forward(self, x):
        return x.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(1)


def segments(n=6000, fs=100.0, seed=0, gain=1.0, offset=0.0):
    """Three components of one contiguous segment."""
    rng = np.random.default_rng(seed)
    return [[(0.0, rng.standard_normal(n) * 100.0 * gain + offset)]
            for _ in range(3)]


def run(segs, standardize, baseline=None, win=6.0):
    arm = sc.Arm(name="t", window_seconds=win, ckpt_dir="", branch="cnn-lstm",
                 step_seconds=win, fs=100.0, models=[MeanModel().eval()])
    comps = ["Z", "N", "E"]
    base = baseline or {c: {"mu": 0.0, "sigma": 100.0, "sigma_trimmed": 100.0}
                        for c in comps}
    return sc.score_chunk(arm, segs, comps, base, torch.device("cpu"),
                          standardize=standardize)


def test_perwindow_is_invariant_to_gain():
    """The property the scheme exists for: x -> a*x leaves the input unchanged."""
    _, a = run(segments(gain=1.0), "perwindow")
    _, b = run(segments(gain=25.0), "perwindow")
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_perwindow_is_invariant_to_offset():
    """And to a DC offset, since the window's own mean is removed."""
    _, a = run(segments(offset=0.0), "perwindow")
    _, b = run(segments(offset=5000.0), "perwindow")
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_station_scale_is_not_invariant_to_gain():
    """The contrast: a station-scaled input tracks the gain, which is the defect."""
    _, a = run(segments(gain=1.0), "trimmed")
    _, b = run(segments(gain=25.0), "trimmed")
    assert not np.allclose(a, b, rtol=1e-3)


def test_trimmed_and_pooled_differ_when_the_baseline_does():
    base = {c: {"mu": 0.0, "sigma": 1000.0, "sigma_trimmed": 100.0}
            for c in ("Z", "N", "E")}
    _, a = run(segments(), "trimmed", base)
    _, b = run(segments(), "pooled", base)
    assert not np.allclose(a, b, rtol=1e-3)


def test_constant_ignores_the_baseline_entirely():
    lo = {c: {"mu": 0.0, "sigma": 1.0, "sigma_trimmed": 1.0} for c in "ZNE"}
    hi = {c: {"mu": 0.0, "sigma": 1e6, "sigma_trimmed": 1e6} for c in "ZNE"}
    _, a = run(segments(), "constant", lo)
    _, b = run(segments(), "constant", hi)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="standardize must be one of"):
        run(segments(), "whatever")


def test_window_geometry_is_unaffected_by_the_scheme():
    """Same windows, same times; only their contents are scaled differently."""
    ta, _ = run(segments(), "perwindow")
    tb, _ = run(segments(), "trimmed")
    np.testing.assert_array_equal(ta, tb)
