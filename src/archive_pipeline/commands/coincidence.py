"""`apipe coincidence` -- require two stations to agree, and price what that costs.

    apipe coincidence --pair ELBA SEMS
    apipe coincidence --all                  # every testable pair
    apipe coincidence --all --max-separation 150

With `--all`, pairs come from the inventory: both stations scored, and enough
joint coverage that the tight alarm budgets can be resolved at all. Pairs
already measured are skipped.
"""
import argparse
import itertools

from archive_pipeline import config
from archive_pipeline.arrivals import ArrivalTimes
from archive_pipeline.batch import Layout
from archive_pipeline.inventory import (load_stations, pairs, survey, usable)
from archive_pipeline.products import coincidence as co

NAME = "coincidence"
HELP = "require two stations to agree, and price what that costs"


def add_args(p):
    """Flags for the coincidence tables."""
    config.add_path_args(p)
    p.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                   help="one station pair; repeatable")
    p.add_argument("--all", action="store_true",
                   help="every pair with both stations scored and enough "
                        "joint coverage")
    p.add_argument("--arm", default="6s",
                   help="which scored arm to read (default: 6s)")
    p.add_argument("--window-seconds", type=float, default=6.0,
                   help="the arm's window length; must be the same arm at "
                        "both stations or the streams are not comparable")
    p.add_argument("--min-joint-days", type=int, default=60)
    p.add_argument("--max-separation", type=float, default=None,
                   help="drop pairs further apart than this, in km")
    p.add_argument("--coincidence-seconds", type=float, default=None,
                   help="default: separation / Vp")
    p.add_argument("--vp", type=float, default=co.DEFAULT_VP)
    p.add_argument("--snr-min", type=float, default=3.0)
    p.add_argument("--force", action="store_true",
                   help="recompute pairs already measured")


def select(cfg, args, lay):
    """The station pairs to measure, nearest first.

    A pair is only offered when both stations have scores for the requested
    arm; the inventory knows what was fetched, not what was processed.
    """
    coords = load_stations(cfg.stations)
    if args.pair:
        return [(0.0, 0, a, b) for a, b in args.pair]
    if not args.all:
        return []
    rows = [r for r in usable(survey(cfg.archive, cfg.ledgers))
            if any(lay.scores(r.station, args.arm).glob("*.npz"))]
    out = pairs(rows, coords, args.min_joint_days)
    if args.max_separation:
        out = [p for p in out if p[0] <= args.max_separation]
    return out


def run(args):
    """Measures every selected pair."""
    cfg = config.load(args)
    cfg.require("archive", "stations", "catalog")
    lay = Layout(cfg.out)
    coords = load_stations(cfg.stations)
    todo = select(cfg, args, lay)
    if not todo:
        print("no pairs selected; pass --all or --pair A B")
        return 1

    taup = ArrivalTimes(grid_km=5.0)
    done = failed = 0
    for sep, joint, a, b in todo:
        dest = lay.pair(a, b, args.arm)
        if dest.exists() and not args.force:
            print(f"== {a}-{b}: already measured, skipping")
            continue
        for stn in (a, b):
            if not any(lay.scores(stn, args.arm).glob("*.npz")):
                print(f"== {a}-{b}: {stn} has no {args.arm} scores, skipping")
                break
        else:
            snr = {s: (lay.range_csv(s) if lay.range_csv(s).exists() else None)
                   for s in (a, b)}
            try:
                table = co.measure(
                    lay.scores(a, args.arm) / "*.npz", a,
                    lay.scores(b, args.arm) / "*.npz", b,
                    cfg.stations, cfg.catalog, args.window_seconds,
                    coords=coords, coincidence_seconds=args.coincidence_seconds,
                    vp=args.vp, snr_csv_a=snr[a], snr_csv_b=snr[b],
                    snr_min=args.snr_min, taup=taup)
            except (ValueError, FileNotFoundError) as e:
                print(f"== {a}-{b}: {e}")
                failed += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(dest, index=False)
            print(f"\n  wrote {dest}\n")
            done += 1
    print(f"{done} pair(s) measured" + (f", {failed} skipped" if failed else ""))
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
