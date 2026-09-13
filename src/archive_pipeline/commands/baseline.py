"""`apipe baseline` -- per-station noise statistics, needed before scanning.

    apipe baseline --all
    apipe baseline --station SEMS

Cheap: six chunks spread across the archive, not a full pass. Stations that
already have one are skipped.
"""
import argparse

from archive_pipeline import config
from archive_pipeline.batch import Layout
from archive_pipeline.commands import plan as plan_cmd
from archive_pipeline.products import baseline as bl

NAME = "baseline"
HELP = "per-station noise (mu, sigma); needed before scanning"


def add_args(p):
    """Flags for baseline building."""
    config.add_path_args(p)
    p.add_argument("--station", action="append",
                   help="restrict to these stations; repeatable")
    p.add_argument("--all", action="store_true", help="every usable station")
    p.add_argument("--force", action="store_true",
                   help="rebuild even where a baseline is present")
    p.add_argument("--sample-chunks", type=int, default=bl.DEFAULT_SAMPLE_CHUNKS,
                   help=f"chunks to read, spread evenly across the archive "
                        f"(default: {bl.DEFAULT_SAMPLE_CHUNKS})")
    p.add_argument("--piece-seconds", type=float, default=bl.DEFAULT_PIECE_SECONDS)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--freqmin", type=float, default=1.0)
    p.add_argument("--freqmax", type=float, default=45.0)


def run(args):
    """Builds a baseline for every selected station that lacks one."""
    cfg = config.load(args)
    cfg.require("archive")
    lay = Layout(cfg.out)
    selected = plan_cmd.select(cfg, args)
    if not selected:
        print("no stations selected; pass --all or --station CODE")
        return 1
    built = 0
    for station, paths in selected.items():
        dest = lay.baseline(station)
        if dest.exists() and not args.force:
            print(f"[{station}] baseline present, skipping")
            continue
        print(f"[{station}] {len(paths)} chunk(s) available")
        if bl.build(paths, dest, fs=args.fs, freqmin=args.freqmin,
                    freqmax=args.freqmax, sample_chunks=args.sample_chunks,
                    piece_seconds=args.piece_seconds):
            built += 1
    print(f"\n{built} baseline(s) built")
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
