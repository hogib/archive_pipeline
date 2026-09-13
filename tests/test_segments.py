"""`component_segments` must agree with ObsPy's merge/split, not merely resemble it.

The fast path exists only because it is a drop-in replacement. If it ever
disagrees, every detector score in the project is computed on samples the
training pipeline would not have produced, and nothing downstream would notice.
So the reference is ObsPy itself, on streams built to contain the cases the
archive actually has.
"""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from archive_pipeline.archive import component_segments, pick_components

FS = 100.0
T0 = UTCDateTime("2024-05-01T00:00:00")


def trace(start_offset, n, comp="Z", seed=None, fs=FS):
    """One trace of `n` samples starting `start_offset` seconds after T0."""
    rng = np.random.default_rng(seed if seed is not None else n + int(start_offset))
    tr = Trace(rng.standard_normal(n).astype(np.float64))
    tr.stats.sampling_rate = fs
    tr.stats.starttime = T0 + start_offset
    tr.stats.channel = "HN" + comp
    tr.stats.station = "TEST"
    return tr


def obspy_segments(stream, comp, fs=FS):
    """The reference: what merge/split produced before this module existed."""
    st = stream.copy()
    st.merge(method=1, fill_value=None)
    st = st.split()
    segs = [(tr.stats.starttime.timestamp, np.asarray(tr.data, dtype=np.float64))
            for tr in st if tr.stats.channel[-1].upper() == comp]
    segs.sort(key=lambda s: s[0])
    return segs


def assert_same(stream, comp="Z"):
    """Asserts the fast path and ObsPy produce identical segments."""
    fast = component_segments(stream, comp, FS)
    slow = obspy_segments(stream, comp)
    assert len(fast) == len(slow), f"{len(fast)} segments vs {len(slow)}"
    for (ta, da), (tb, db) in zip(fast, slow):
        assert ta == pytest.approx(tb, abs=1e-9)
        assert len(da) == len(db)
        np.testing.assert_array_equal(da, db)
    return fast


def test_single_trace():
    assert len(assert_same(Stream([trace(0, 1000)]))) == 1


def test_contiguous_traces_join():
    # 1000 samples at 100 Hz is exactly 10 s, so the second starts on the
    # sample after the first ends.
    segs = assert_same(Stream([trace(0, 1000), trace(10.0, 1000)]))
    assert len(segs) == 1
    assert len(segs[0][1]) == 2000


def test_gap_splits():
    segs = assert_same(Stream([trace(0, 1000), trace(30.0, 1000)]))
    assert len(segs) == 2


def test_out_of_order_input():
    assert_same(Stream([trace(30.0, 1000), trace(0, 1000), trace(10.0, 1000)]))


def test_many_fragments():
    """The archive's real shape: hundreds of short traces with scattered gaps."""
    rng = np.random.default_rng(0)
    trs, t = [], 0.0
    for i in range(300):
        n = int(rng.integers(200, 2000))
        trs.append(trace(t, n, seed=i))
        t += n / FS + (0.0 if rng.random() < 0.7 else float(rng.integers(1, 50)))
    rng.shuffle(trs)
    segs = assert_same(Stream(trs))
    assert len(segs) > 1


def test_other_components_ignored():
    st = Stream([trace(0, 1000, "Z"), trace(0, 500, "N"), trace(0, 700, "E")])
    assert len(component_segments(st, "Z", FS)[0][1]) == 1000
    assert len(component_segments(st, "N", FS)[0][1]) == 500
    assert component_segments(st, "Q", FS) == []


def test_missing_component_returns_none():
    """A chunk missing a role is skipped, never silently given two horizontals."""
    assert pick_components(Stream([trace(0, 10, "N"), trace(0, 10, "E")])) is None
    assert pick_components(Stream([trace(0, 10, "Z"), trace(0, 10, "1"),
                                   trace(0, 10, "2")])) == ["Z", "1", "2"]


def test_duplicate_trace_is_dropped():
    """A redelivered trace covering held samples must not extend the segment."""
    a = trace(0, 1000)
    segs = component_segments(Stream([a, a.copy()]), "Z", FS)
    assert len(segs) == 1
    assert len(segs[0][1]) == 1000


def test_conflicting_overlap_matches_obspy():
    """`merge(method=1)` resolves an overlap in favour of the later trace.

    Trimming the later trace's front instead would agree on every identical
    redelivery in the archive and disagree here, on 500 of 1500 samples.
    """
    a, b = trace(0, 1000), trace(5.0, 1000)
    a.data[:] = 1.0
    b.data[:] = 2.0
    seg = assert_same(Stream([a, b]))[0][1]
    assert len(seg) == 1500
    assert seg[499] == 1.0 and seg[500] == 2.0


def test_contained_overlap_is_discarded():
    """A trace nested inside coverage already held is dropped, not painted.

    The opposite of the rule for a trailing overlap, and the reason this file
    tests both shapes: the first implementation painted here too, and matched
    ObsPy on every real chunk anyway.
    """
    a, b = trace(0, 2000), trace(5.0, 500)
    a.data[:] = 1.0
    b.data[:] = 2.0
    seg = assert_same(Stream([a, b]))[0][1]
    assert len(seg) == 2000
    assert (seg == 1.0).all()


def test_overlap_does_not_break_a_run():
    """Coverage held past a nested trace must not be mistaken for a gap."""
    a, b = trace(0, 2000), trace(1.0, 100)
    assert len(assert_same(Stream([a, b]))) == 1
