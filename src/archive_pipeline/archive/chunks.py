"""Locating and decoding the packed archive.

A TDVMS campaign delivers one zip per request, each holding a single mseed of
roughly three weeks of continuous three-component record. Chunks are named
`<STATION>_<YYYY-MM-DD>.zip` and live one directory per station.

Decoding is deliberately split from segmentation: `read_chunk` does the part
ObsPy is good at (parsing mseed) and `archive.segments` does the part it is
slow at. Nothing here calls `Stream.merge`.
"""
import pathlib
import re
import tempfile
import zipfile

from obspy import read

CHUNK_RE = re.compile(r"^(?P<station>[A-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})\.zip$")


def find_chunks(archive_root, stations=None):
    """Every complete chunk in the archive, grouped by station.

    In-flight downloads (`*.part`, `incoming.*`) and the bookkeeping
    directories a campaign leaves behind (`_nodata_notices`, `_stale`) are
    skipped, so this is safe to run while a pull is writing into the same tree.

    Args:
        archive_root: Directory holding one subdirectory per station.
        stations: Optional iterable restricting which stations are returned.

    Returns:
        Dict of station code to a time-sorted list of chunk `Path`s.
    """
    root = pathlib.Path(archive_root)
    want = set(stations) if stations else None
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("_"):
            continue
        if want is not None and d.name not in want:
            continue
        found = [p for p in d.iterdir()
                 if p.is_file() and CHUNK_RE.match(p.name)]
        if found:
            out[d.name] = sorted(found, key=lambda p: p.name)
    return out


def nonconforming(archive_root, stations=None):
    """Files in the archive that are not usable chunks, and why.

    Worth surfacing rather than silently skipping. The archive here carries
    nine `SEMS_<date>.dup<id>.zip` files, byte-identical redelivery of spans
    already held: excluding them is correct, and a station whose chunk count
    silently drops by nine is not.

    Args:
        archive_root: Directory holding one subdirectory per station.
        stations: Optional iterable restricting which stations are scanned.

    Returns:
        Dict of station code to a list of `(filename, reason)`, omitting
        stations with nothing to report.
    """
    root = pathlib.Path(archive_root)
    want = set(stations) if stations else None
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("_") or (want is not None and d.name not in want):
            continue
        bad = []
        for p in sorted(d.iterdir()):
            if not p.is_file() or CHUNK_RE.match(p.name):
                continue
            if ".part" in p.name or p.name.startswith("incoming."):
                bad.append((p.name, "download in flight"))
            elif ".dup" in p.name:
                bad.append((p.name, "duplicate delivery"))
            else:
                bad.append((p.name, "unrecognised name"))
        if bad:
            out[d.name] = bad
    return out


def chunk_date(path):
    """The date string embedded in a chunk filename."""
    m = CHUNK_RE.match(pathlib.Path(path).name)
    return m.group("date") if m else None


def read_chunk(zpath):
    """Decodes one archive chunk into an unmerged `Stream`.

    The zip is extracted to a temporary directory rather than read through a
    file object because ObsPy's mseed reader wants a real path for the formats
    it memory-maps.

    Args:
        zpath: Path to a chunk zip.

    Returns:
        An ObsPy `Stream` of raw traces, in file order and not merged. Pass it
        to `archive.segments.component_segments` to get usable arrays.
    """
    with zipfile.ZipFile(zpath) as zf, tempfile.TemporaryDirectory() as tmp:
        member = next(n for n in zf.namelist() if n.lower().endswith(".mseed"))
        zf.extract(member, tmp)
        return read(str(pathlib.Path(tmp) / member))
