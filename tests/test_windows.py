"""Window cutting: the geometry, and the filename rule that silently empties a dataset."""
import numpy as np
import pytest

from archive_pipeline.products.windows import cut, event_tag

FS = 100.0


def seg(t0, n, value=1.0):
    return [(t0, np.full(n, float(value)))]


def three(t0, n):
    return [seg(t0, n, k + 1) for k in range(3)]


def test_cut_returns_all_three_components():
    got = cut(three(0.0, 1000), 1.0, 600, FS)
    assert got.shape == (600, 3)
    assert (got[:, 0] == 1.0).all() and (got[:, 2] == 3.0).all()


def test_cut_refuses_to_straddle_a_gap():
    """Two segments with a hole between them must not be spliced."""
    segs = [[(0.0, np.ones(500)), (10.0, np.ones(500))] for _ in range(3)]
    assert cut(segs, 4.0, 600, FS) is None


def test_cut_refuses_past_the_end():
    assert cut(three(0.0, 500), 0.0, 600, FS) is None


def test_cut_refuses_before_the_start():
    assert cut(three(100.0, 1000), 99.0, 600, FS) is None


def test_cut_refuses_when_one_component_is_short():
    """All three must cover it; a window built from two is not a window."""
    segs = [seg(0.0, 1000), seg(0.0, 1000), seg(0.0, 300)]
    assert cut(segs, 1.0, 600, FS) is None


@pytest.mark.parametrize("event_id", [627233, "627233", 1])
def test_event_tag_carries_no_station_infix(event_id):
    """The encoder's parse is `^(?:noise_)?event_(.+?)_raw$`, non-greedy.

    A station infix -- `event_627233_MANT_raw` -- therefore captures
    "627233_MANT", which matches no catalogue row: every window loses its
    magnitude label and the dataset comes out empty with no error anywhere.
    """
    import re
    pat = re.compile(r"^(?:noise_)?event_(.+?)_raw$")
    for name in (f"{event_tag(event_id)}_raw", f"noise_{event_tag(event_id)}_raw"):
        m = pat.match(name)
        assert m is not None
        assert m.group(1).isdigit(), f"{name} parses to {m.group(1)!r}"
