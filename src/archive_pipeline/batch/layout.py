"""Where every product lives, and how to tell whether it is already there.

Resumability is not a convenience here. A full pass over this archive is
hours, the machine also runs downloads, and the batch will be interrupted.
Every product is therefore keyed to the chunk it came from, so an interrupted
run resumes at chunk granularity rather than restarting a station.

    out/
      stations/<STN>/
        baseline.json              whole-archive, sampled; scanning needs it
        range/<chunk>.csv          per-event SNR, one part per chunk
        range.csv                  the parts concatenated
        scores/<arm>/<chunk>.npz   detector scores
        windows/<W>s/eq|noise/     cut windows
        windows/<W>s/.chunks/<chunk>   marker: this chunk has been cut
      pairs/<A>-<B>/<arm>.csv
      datasets/<name>/

The marker directory exists because cutting a chunk emits a variable number of
window files, sometimes zero -- a quiet station over three weeks is a real and
correct outcome -- so "no files" cannot distinguish done from not started.
"""
import pathlib


class Layout:
    """Resolves product paths under one output root."""

    def __init__(self, out):
        self.out = pathlib.Path(out)

    # -- per station -----------------------------------------------------
    def station(self, stn):
        return self.out / "stations" / stn

    def baseline(self, stn):
        return self.station(stn) / "baseline.json"

    def range_csv(self, stn):
        return self.station(stn) / "range.csv"

    def range_dir(self, stn):
        return self.station(stn) / "range"

    def range_part(self, stn, chunk):
        return self.range_dir(stn) / f"{chunk}.csv"

    def scores(self, stn, arm):
        return self.station(stn) / "scores" / arm

    def score_file(self, stn, arm, chunk):
        return self.scores(stn, arm) / f"{chunk}.npz"

    def windows(self, stn, seconds):
        return self.station(stn) / "windows" / f"{seconds}s"

    def window_marker(self, stn, seconds, chunk):
        return self.windows(stn, seconds) / ".chunks" / chunk

    # -- per pair --------------------------------------------------------
    def pair(self, a, b, arm):
        lo, hi = sorted((a, b))
        return self.out / "pairs" / f"{lo}-{hi}" / f"{arm}.csv"

    def dataset(self, name):
        return self.out / "datasets" / name

    # -- completion ------------------------------------------------------
    def has_range(self, stn, chunks):
        """Whether per-event SNR is complete for this station.

        A whole-archive `range.csv` with no per-chunk parts counts as complete:
        that is what `apipe adopt` leaves behind for a station processed before
        this project, and recomputing it would cost a full pass for nothing.
        """
        if self.range_csv(stn).exists() and not self.range_dir(stn).exists():
            return True
        return all(self.range_part(stn, c).exists() for c in chunks) and bool(chunks)

    def scored(self, stn, arm, chunks):
        """Chunks of this station not yet scored by this arm."""
        return [c for c in chunks if not self.score_file(stn, arm, c).exists()]

    def uncut(self, stn, seconds, chunks):
        """Chunks of this station not yet cut at this window length.

        Windows present without a marker directory came from `apipe adopt` and
        count as complete, for the same reason an adopted `range.csv` does.
        """
        d = self.windows(stn, seconds)
        if d.is_dir() and not (d / ".chunks").exists() and any(d.iterdir()):
            return []
        return [c for c in chunks if not self.window_marker(stn, seconds, c).exists()]
