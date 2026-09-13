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
from archive_pipeline.products import snr
from archive_pipeline.products.scan import score_chunk, write_scores


class StationRunner:
    """Processes one station's outstanding chunks in a single pass."""

    def __init__(self, station, chunks, layout, cfg, arms, device,
                 fs=100.0, freqmin=1.0, freqmax=45.0, workers=6,
                 want_range=True, log=print):
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
            log: Where progress goes.
        """
        self.station, self.chunks, self.lay = station, chunks, layout
        self.cfg, self.arms, self.device = cfg, arms, device
        self.fs, self.freqmin, self.freqmax = fs, freqmin, freqmax
        self.want_range, self.log = want_range, log
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
            coords = pd.read_csv(self.cfg.stations, encoding="utf-8-sig")
            coords.columns = [c.strip() for c in coords.columns]
            row = coords[coords.Code == self.station]
            if row.empty:
                self.log(f"[{self.station}] not in the station catalogue; "
                         f"skipping per-event SNR")
                self.want_range = False
            else:
                self.catalog = snr.load_catalog(self.cfg.catalog,
                                                float(row.iloc[0].Latitude),
                                                float(row.iloc[0].Longitude))
        return True

    def todo(self, path):
        """What this chunk still needs.

        Returns:
            Tuple of (arms needing it, whether SNR is needed).
        """
        stem = path.stem
        arms = [a for a in self.arms
                if not self.lay.score_file(self.station, a.name, stem).exists()]
        need_snr = (self.want_range
                    and not self.lay.range_part(self.station, stem).exists()
                    and not (self.lay.range_csv(self.station).exists()
                             and not self.lay.range_dir(self.station).exists()))
        return arms, need_snr

    def process(self, path):
        """Decodes one chunk and writes every product still outstanding.

        Returns:
            Number of products written.
        """
        stem = path.stem
        arms, need_snr = self.todo(path)
        if not arms and not need_snr:
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
        missing = [c for c in comps if c not in self.baseline]
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
                                       freqmax=self.freqmax, pool=self.pool)
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
