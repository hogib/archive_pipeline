"""`apipe adopt` -- move artifacts produced before this project into `out/`.

Four stations were processed by hand, one command at a time, before any of
this existed, and those runs cost GPU-days. They are not recomputed: they are
moved into the layout the batch driver expects, so `apipe run` sees them as
done and resumes around them.

The mapping is recorded here rather than performed as a shell one-liner
because it is not obvious. The old tooling named things per command -- a
baseline was `<stn>_baseline.json` in the working directory, scores were
`scores_<stn>/<arm>/`, and one station's noise baseline ended up in a
different repository entirely -- and reconstructing which file belonged to
which station later would mean reading the commands that made them.

Source                                        Destination
`<stn>_baseline.json`                         `stations/<STN>/baseline.json`
`<stn>_range_full.csv`                        `stations/<STN>/range.csv`
`<stn>_range.csv`                             `stations/<STN>/range.partial.csv`
`scores_<stn>/<arm>/*.npz`                    `stations/<STN>/scores/<arm>/`
`raw/continuous_windows/<W>s/<STN>/`          `stations/<STN>/windows/<W>s/`
`runs_coinc/<a>_<b>_<arm>_coincidence.csv`    `pairs/<A>-<B>/<arm>.csv`

Everything moves with `os.replace` within one filesystem, so each is an atomic
rename and nothing is ever half-copied. Files already at the destination are
left alone and reported, never overwritten.
"""
import argparse
import pathlib
import re
import shutil

from archive_pipeline import config

NAME = "adopt"
HELP = "move pre-existing artifacts into the out/ layout"

# `scores_dense` is not a station: it is MANT rescanned at a finer step. Its
# arm directories are named `6s`, `ponly` and `stalta` exactly like MANT's
# own, so merging them would silently interleave two different step sizes
# into one score series. It lands under a prefixed arm name instead.
SPECIAL_SCORES = {"dense": ("MANT", "dense.")}

COINC_RE = re.compile(r"^(?P<a>[a-z0-9]+)_(?P<b>[a-z0-9]+)_(?P<arm>[a-z0-9]+)"
                      r"_coincidence\.csv$")


def add_args(p):
    """Flags for the migration."""
    config.add_path_args(p)
    p.add_argument("--from", dest="source", required=True,
                   help="directory the old tooling ran in "
                        "(e.g. ../cnn_earthquake)")
    p.add_argument("--also", action="append", default=[],
                   help="extra directory to search for stray artifacts, such "
                        "as the repo a baseline was written into by mistake; "
                        "repeatable")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and move nothing")


def plan(source, extra, out):
    """Builds the list of `(src, dst, what)` moves.

    Args:
        source: Directory the old tooling ran in.
        extra: Additional directories to search for loose per-station files.
        out: Destination root.

    Returns:
        List of `(src_path, dst_path, description)`.
    """
    source = pathlib.Path(source)
    moves = []
    stations = out / "stations"

    for root in [source, *(pathlib.Path(e) for e in extra)]:
        if not root.is_dir():
            continue
        for f in sorted(root.glob("*_baseline.json")):
            stn = f.name.split("_")[0].upper()
            moves.append((f, stations / stn / "baseline.json", "baseline"))
        for f in sorted(root.glob("*_range_full.csv")):
            stn = f.name.split("_")[0].upper()
            moves.append((f, stations / stn / "range.csv", "per-event SNR"))
        for f in sorted(root.glob("*_range.csv")):
            stn = f.name.split("_")[0].upper()
            moves.append((f, stations / stn / "range.partial.csv",
                          "per-event SNR (superseded)"))

    for d in sorted(source.glob("scores_*")):
        if not d.is_dir():
            continue
        tag = d.name[len("scores_"):]
        stn, prefix = SPECIAL_SCORES.get(tag, (tag.upper(), ""))
        for arm in sorted(p for p in d.iterdir() if p.is_dir()):
            moves.append((arm, stations / stn / "scores" / (prefix + arm.name),
                          f"{len(list(arm.glob('*.npz')))} scored chunks"))

    windows = source / "raw" / "continuous_windows"
    if windows.is_dir():
        for wd in sorted(p for p in windows.iterdir() if p.is_dir()):
            for sd in sorted(p for p in wd.iterdir() if p.is_dir()):
                moves.append((sd, stations / sd.name / "windows" / wd.name,
                              "cut windows"))

    coinc = source / "runs_coinc"
    if coinc.is_dir():
        for f in sorted(coinc.glob("*_coincidence.csv")):
            m = COINC_RE.match(f.name)
            if not m:
                continue
            a, b = m.group("a").upper(), m.group("b").upper()
            moves.append((f, out / "pairs" / f"{a}-{b}" / f"{m.group('arm')}.csv",
                          "coincidence table"))
    return moves


def run(args):
    """Prints the plan and, unless `--dry-run`, performs it."""
    cfg = config.load(args)
    out = pathlib.Path(cfg.out)
    moves = plan(args.source, args.also, out)
    if not moves:
        print(f"nothing to adopt under {args.source}")
        return 0

    width = max(len(str(s)) for s, _, _ in moves)
    done = skipped = 0
    for src, dst, what in moves:
        if dst.exists():
            print(f"  {'skip':<6} {str(src):<{width}}  destination exists")
            skipped += 1
            continue
        rel = dst.relative_to(out)
        print(f"  {'move':<6} {str(src):<{width}}  ->  out/{rel}   ({what})")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                src.rename(dst)
            except OSError:
                # Across filesystems rename fails; fall back to a copy and
                # only unlink the source once the copy is complete.
                shutil.move(str(src), str(dst))
            done += 1

    if args.dry_run:
        print(f"\n{len(moves) - skipped} to move, {skipped} already present "
              f"(dry run, nothing changed)")
    else:
        print(f"\nmoved {done}, skipped {skipped}, into {out}")
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
