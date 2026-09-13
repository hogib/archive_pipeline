"""What has actually been fetched, across every campaign ledger.

A download campaign writes one JSONL ledger per run, and a long project
accumulates several: this work has four (`tdvms_ledger`, `afad_campaign_ledger`,
`demi_ledger`, `gcam_ledger`) whose station sets overlap and whose fetched
spans disagree, because the older ones stopped being updated when the newer one
took over. Reading any single ledger understates coverage.

Both sources of truth are used and reconciled. The ledgers say what was
requested and what came back; the archive directory says what is on disk right
now. A station is only usable to the extent both agree, and where they do not,
the disk wins -- a ledger row marked fetched whose zip was deleted is not data.

Ledgers are read append-only and never locked, so this is safe to run while a
pull is in flight.
"""
import collections
import datetime as dt
import json
import pathlib

# States that mean the bytes were delivered, versus still owed to us.
FETCHED = ("fetched",)
PENDING = ("pending", "claimed", "submitted")


def read_ledgers(paths):
    """Every ledger row, keyed by station and state.

    Args:
        paths: Iterable of ledger JSONL paths. Missing files are skipped, so a
            caller can list every ledger the project has ever had.

    Returns:
        Dict of station code to `{state: [(start, end), ...]}`, times as
        `datetime`, plus a parallel dict of station code to the set of ledger
        filenames it appeared in.
    """
    spans = collections.defaultdict(lambda: collections.defaultdict(list))
    sources = collections.defaultdict(set)
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A pull appending to this file can leave a half-written final
                # line. Skipping it is correct; the row arrives complete on the
                # next read.
                continue
            station = row.get("station")
            if not station:
                continue
            spans[station][row.get("state", "?")].append(
                (dt.datetime.fromisoformat(row["start"]),
                 dt.datetime.fromisoformat(row["end"])))
            sources[station].add(p.stem.replace("_ledger", ""))
    return spans, sources


def merge_spans(spans):
    """Collapses overlapping `(start, end)` pairs into disjoint ordered spans."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def span_days(spans):
    """Total days covered by a list of disjoint spans."""
    return sum((b - a).days for a, b in spans)


def intersect_days(a, b):
    """Days covered by both span lists.

    This is the number that decides whether a station pair can be tested for
    coincidence: alarms can only be required to agree over the interval both
    stations were recording.
    """
    total = 0
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                total += (hi - lo).days
    return total
