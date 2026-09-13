"""The two tables that decide what is worth running.

Per station: how much record exists, and how much is still owed.
Per pair: how far apart, and over how many days both were recording.

The pair table is the one that matters for planning. A coincidence test is
bounded by joint coverage, not by either station's own archive, and joint
coverage is frequently a third of what the two stations have separately.
"""
import dataclasses
import datetime as dt
import pathlib

from archive_pipeline.archive import CHUNK_RE, find_chunks
from archive_pipeline.inventory.ledger import (FETCHED, PENDING,
                                               intersect_days, merge_spans,
                                               read_ledgers, span_days)
from archive_pipeline.inventory.stations import separation_km

# Below this, a coincidence test cannot resolve the tight alarm budgets: at one
# alarm per day per station the expected count of chance agreements over two
# months rounds to zero and every cell reads 0.000.
MIN_JOINT_DAYS = 60


@dataclasses.dataclass
class StationRow:
    """One station's holdings."""
    station: str
    chunks: int
    bytes: int
    fetched_days: int
    pending_days: int
    spans: list
    sources: set

    @property
    def gigabytes(self):
        return self.bytes / 1e9

    @property
    def first(self):
        return self.spans[0][0].date() if self.spans else None

    @property
    def last(self):
        return self.spans[-1][1].date() if self.spans else None


def survey(archive_root, ledger_paths, stations=None):
    """Reconciles ledgers against the archive directory.

    Args:
        archive_root: Directory of per-station chunk directories.
        ledger_paths: Iterable of campaign ledger JSONL paths.
        stations: Optional restriction to particular station codes.

    Returns:
        List of `StationRow`, ordered by chunk count descending.
    """
    spans, sources = read_ledgers(ledger_paths)
    on_disk = find_chunks(archive_root, stations)
    rows = []
    for station in sorted(set(spans) | set(on_disk)):
        if stations and station not in stations:
            continue
        chunks = on_disk.get(station, [])
        fetched = merge_spans([s for k in FETCHED for s in spans[station].get(k, [])])
        pending = merge_spans([s for k in PENDING for s in spans[station].get(k, [])])
        if not chunks and not fetched:
            continue
        rows.append(StationRow(
            station=station, chunks=len(chunks),
            bytes=sum(p.stat().st_size for p in chunks),
            fetched_days=span_days(fetched), pending_days=span_days(pending),
            spans=fetched, sources=sources.get(station, set())))
    rows.sort(key=lambda r: -r.chunks)
    return rows


def usable(rows, min_chunks=2):
    """Stations with enough record to be worth processing at all.

    A single chunk is three weeks, which is not enough to measure a false-alarm
    rate against and not enough to pair with anything.
    """
    return [r for r in rows if r.chunks >= min_chunks]


def pairs(rows, coords, min_joint_days=MIN_JOINT_DAYS):
    """Every station pair worth a coincidence test, nearest first.

    Args:
        rows: `StationRow` list, already filtered to usable stations.
        coords: Station code to `(lat, lon)`.
        min_joint_days: Pairs below this are dropped; see `MIN_JOINT_DAYS`.

    Returns:
        List of `(separation_km, joint_days, station_a, station_b)`, sorted by
        separation.
    """
    have = [r for r in rows if r.station in coords]
    out = []
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            joint = intersect_days(a.spans, b.spans)
            if joint < min_joint_days:
                continue
            out.append((separation_km(coords, a.station, b.station), joint,
                        a.station, b.station))
    out.sort()
    return out
