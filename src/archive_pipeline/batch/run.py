"""The one-pass driver.

For each chunk that needs anything, decode once and hand the same segments to
every product that wants them. This is the whole point of the project: the
tools it replaces each globbed the archive and decoded it independently, so a
station cost one full pass per product rather than one pass.

Ordering is by station and then by chunk, and every product is written as soon
as it is computed. An interrupted run therefore loses at most one chunk of
work, which matters because a full pass is hours on a machine that is also
downloading.
"""
import concurrent.futures
import time

import pandas as pd
import torch

from archive_pipeline.archive import (component_segments, pick_components,
                                      read_chunk)
from archive_pipeline.arrivals import ArrivalTimes
from archive_pipeline.products import baseline as bl
from archive_pipeline.products import snr, windows
from archive_pipeline.products.association import load_snr, predicted_arrivals
from archive_pipeline.products.scan import score_chunk, write_scores


class StationRunner:
    """Processes one station's outstanding chunks in a single pass."""

    def __init__(self, station, chunks, layout, cfg, arms, device,
                 fs=100.0, freqmin=1.0, freqmax=45.0, workers=6,
                 want_range=True, standardize="trimmed", lengths=(), snr_min=3.0,
                 pre=windows.DEFAULT_PRE,
                 noise_offset=windows.DEFAULT_NOISE_OFFSET, log=print):
        """Prepares a station for processing.

        Args:
            station: Station code.
            chunks: Every chunk path of this station, in time order.
            layout: A `Layout`.
            cfg: Resolved `Config`, for the station and event catalogues.
            arms: Loaded `Arm`s to score with.
            device: Torch device.
            fs: Sampling rate in Hz.
            freqmin: Detector bandpass low corner in Hz.
            freqmax: Detector bandpass high corner in Hz.
            workers: Filter threads.
            want_range: Whether to measure per-event signal-to-noise.
            standardize: How windows are put into detector units.
            lengths: Window lengths to cut, in seconds. Empty cuts nothing.
            snr_min: Events below this measured SNR are not cut.
            pre: Seconds before the anchor a window starts.
            noise_offset: Seconds before P the paired noise window is taken.
            log: Where progress goes.
        """
        self.station, self.chunks, self.lay = station, chunks, layout
        self.cfg, self.arms, self.device = cfg, arms, device
        self.fs, self.freqmin, self.freqmax = fs, freqmin, freqmax
        self.want_range, self.log = want_range, log
        self.standardize = standardize
        self.lengths = sorted(lengths)
        self.snr_min, self.pre, self.noise_offset = snr_min, pre, noise_offset
        self.cut_catalog, self.dirs = None, {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.taup = ArrivalTimes(grid_km=5.0)
        self.baseline = None
        self.catalog = None

    def prepare(self):
        """Loads the baseline and catalogue this station needs.

        Returns:
            True when the station can be processed. A station with no noise
            baseline cannot be scanned at all -- the standardization is part of
            the detector's input contract -- so it is reported and skipped
            rather than scanned against some other station's statistics.
        """
        if self.standardize == "perwindow":
            # Scale-free: there is no station statistic to load, which is the
            # entire point of it.
            self.baseline = {}
            if self.want_range:
                self._load_catalog()
            if self.lengths:
                self._prepare_cutting()
            return True
        path = self.lay.baseline(self.station)
        if not path.exists():
            self.log(f"[{self.station}] no baseline at {path}; "
                     f"run `apipe baseline --station {self.station}` first")
            return False
        self.baseline, cond = bl.load(path)
        if cond and (cond["freqmin"], cond["freqmax"]) != (self.freqmin, self.freqmax):
            self.log(f"[{self.station}] baseline was built at "
                     f"{cond['freqmin']:g}-{cond['freqmax']:g} Hz but this scan "
                     f"is {self.freqmin:g}-{self.freqmax:g} Hz; refusing")
            return False
        if self.want_range:
            self._load_catalog()
        if self.lengths:
            self._prepare_cutting()
        return True

    def _load_catalog(self):
        """Loads the event catalogue this station's SNR pass needs."""
        coords = pd.read_csv(self.cfg.stations, encoding="utf-8-sig")
        coords.columns = [c.strip() for c in coords.columns]
        row = coords[coords.Code == self.station]
        if row.empty:
            self.log(f"[{self.station}] not in the station catalogue; "
                     f"skipping per-event SNR")
            self.want_range = False
            return
        self.catalog = snr.load_catalog(self.cfg.catalog,
                                        float(row.iloc[0].Latitude),
                                        float(row.iloc[0].Longitude))

    def _prepare_cutting(self):
        """Loads the anchored catalogue windows are cut against.

        Cutting is filtered by measured signal-to-noise, and that table is
        itself a product of this pass -- so a station seeing its first pass
        cannot cut in it. Rather than cut unfiltered and discard later, which
        at this station's recovery rate would write roughly eight times the
        files, cutting is deferred and the user told to run again. A second
        pass is still half of what the tooling this replaces needed, and a
        station that already has `range.csv` cuts in its first.
        """
        range_csv = self.lay.range_csv(self.station)
        if not range_csv.exists():
            self.log(f"[{self.station}] no range.csv yet, so windows cannot be "
                     f"filtered by SNR; cutting deferred. Re-run `apipe run` "
                     f"once this pass finishes.")
            self.lengths = []
            return
        cat, _ = predicted_arrivals(self.station, self.cfg.stations,
                                    self.cfg.catalog, taup=self.taup)
        catalogued = len(cat)
        measured = load_snr(range_csv)
        cat = cat.merge(measured, left_on="EventID", right_on="event_id",
                        how="left")
        cat = cat[cat.snr >= self.snr_min]
        cat["cut_epoch"] = cat.p_epoch
        self.cut_catalog = cat.sort_values("p_epoch").reset_index(drop=True)
        # Three denominators, all different and all easy to confuse: events in
        # the catalogue near the station, events whose signal-to-noise was
        # actually measurable, and events clearing the cut.
        self.log(f"[{self.station}] {len(cat):,} events reach SNR "
                 f"{self.snr_min:g}, of {len(measured):,} measured and "
                 f"{catalogued:,} catalogued within range; cutting "
                 + ", ".join(f"{w:g}s [P-{self.pre:g}, P+{w - self.pre:g}]"
                             for w in self.lengths))
        for w in self.lengths:
            eq = self.lay.windows(self.station, int(w)) / "eq"
            nz = self.lay.windows(self.station, int(w)) / "noise"
            eq.mkdir(parents=True, exist_ok=True)
            nz.mkdir(parents=True, exist_ok=True)
            self.dirs[w] = (eq, nz)

    def todo(self, path):
        """What this chunk still needs.

        Returns:
            Tuple of (arms needing it, whether SNR is needed, lengths to cut).
        """
        stem = path.stem
        arms = [a for a in self.arms
                if not self.lay.score_file(self.station, a.name, stem).exists()]
        need_snr = (self.want_range
                    and not self.lay.range_part(self.station, stem).exists()
                    and not (self.lay.range_csv(self.station).exists()
                             and not self.lay.range_dir(self.station).exists()))
        cut = [w for w in self.lengths
               if not self.lay.window_marker(self.station, int(w), stem).exists()]
        return arms, need_snr, cut

    def process(self, path):
        """Decodes one chunk and writes every product still outstanding.

        Returns:
            Number of products written.
        """
        stem = path.stem
        arms, need_snr, cut_lengths = self.todo(path)
        if not arms and not need_snr and not cut_lengths:
            return 0

        t0 = time.time()
        # Announced before the read: on a fragmented chunk this is minutes of
        # a single opaque call, and a slow chunk should not look like a hang.
        self.log(f"  {stem}: reading ({path.stat().st_size / 1e6:.0f} MB)...")
        stream = read_chunk(path)
        comps = pick_components(stream)
        if comps is None:
            self.log(f"  {stem}: incomplete components, skipped")
            return 0
        missing = ([] if self.standardize == "perwindow"
                   else [c for c in comps if c not in self.baseline])
        if missing:
            self.log(f"  {stem}: no baseline for component(s) {missing}, skipped")
            return 0
        segs = [component_segments(stream, c, self.fs) for c in comps]
        del stream
        t_read = time.time() - t0

        written = 0
        for arm in arms:
            t = time.time()
            times, probs = score_chunk(arm, segs, comps, self.baseline,
                                       self.device, fs=self.fs,
                                       freqmin=self.freqmin,
                                       freqmax=self.freqmax, pool=self.pool,
                                       standardize=self.standardize)
            if times is None:
                self.log(f"  {stem} {arm.name}: no unbroken 3-component span")
                continue
            write_scores(self.lay.score_file(self.station, arm.name, stem),
                         times, probs)
            written += 1
            dt = time.time() - t
            self.log(f"  {stem} {arm.name:>8}: {len(times):>8,} windows "
                     f"({len(times) * arm.step_seconds / 86400:5.1f} d), "
                     f"{(probs > 0.5).mean() * 100:5.1f}% over 0.5, {dt:.0f}s "
                     f"({len(times) / dt:,.0f} win/s)")

        if need_snr:
            t = time.time()
            span = snr.span_of(segs[0])
            sub = self.catalog
            if span:
                sub = sub[(sub.t >= pd.Timestamp(span[0], unit="s"))
                          & (sub.t <= pd.Timestamp(span[1], unit="s"))]
            rows, dropped = snr.measure(segs[0], self.fs, sub, self.taup)
            dest = self.lay.range_part(self.station, stem)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(dest, index=False)
            written += 1
            self.log(f"  {stem} {'snr':>8}: {len(rows):>8,} of {len(sub):,} "
                     f"catalogued events measured"
                     + (f", {dropped} dropped at a gap" if dropped else "")
                     + f", {time.time() - t:.0f}s")

        if cut_lengths:
            t = time.time()
            kept, dropped, bands = windows.cut_chunk(
                segs, comps, self.station, self.cut_catalog, cut_lengths,
                self.dirs, fs=self.fs, pre=self.pre,
                noise_offset=self.noise_offset)
            for w in cut_lengths:
                marker = self.lay.window_marker(self.station, int(w), stem)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            written += len(cut_lengths)
            self.log(f"  {stem} {'cut':>8}: {kept:>8,} event(s) x "
                     f"{len(cut_lengths)} length(s)"
                     + (f", {dropped} at a gap or edge" if dropped else "")
                     + f", {time.time() - t:.0f}s")

        self.log(f"  {stem}: {written} product(s), {time.time() - t0:.0f}s "
                 f"(read {t_read:.0f}s)")
        return written

    def finish(self):
        """Rolls per-chunk SNR parts up into the station's one table."""
        if not self.want_range:
            return
        parts = sorted(self.lay.range_dir(self.station).glob("*.csv"))
        if not parts:
            return
        df = snr.concatenate(parts, self.lay.range_csv(self.station))
        if len(df):
            self.log(f"[{self.station}] {len(df):,} events measured, median "
                     f"SNR {df.snr.median():.2f}, "
                     f"{100 * (df.snr >= 3).mean():.1f}% reach SNR 3")


def run_stations(plan_rows, layout, cfg, arms, device, log=print, **kw):
    """Processes every station in the plan, one decode per chunk.

    Args:
        plan_rows: Rows from `commands.plan.build`.
        layout: A `Layout`.
        cfg: Resolved `Config`.
        arms: Loaded `Arm`s.
        device: Torch device.
        log: Where progress goes.
        **kw: Passed to `StationRunner`.

    Returns:
        Total number of products written.
    """
    total, started = 0, time.time()
    for row in plan_rows:
        station = row["station"]
        runner = StationRunner(station, row["paths"], layout, cfg, arms,
                               device, log=log, **kw)
        if not runner.prepare():
            continue
        pending = [p for p in row["paths"] if any(runner.todo(p))]
        if not pending:
            log(f"[{station}] nothing outstanding")
            continue
        log(f"[{station}] {len(pending)} of {len(row['paths'])} chunk(s) to decode")
        for i, path in enumerate(pending, 1):
            log(f"[{station}] {i}/{len(pending)}")
            total += runner.process(path)
        runner.finish()
    log(f"\n{total} product(s) written in {(time.time() - started) / 60:.1f} min")
    return total
