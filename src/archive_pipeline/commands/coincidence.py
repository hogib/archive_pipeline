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
from archive_pipeline.archive import find_chunks
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
    p.add_argument("--station", action="append",
                   help="restrict --all to pairs among these stations; "
                        "repeatable")
    p.add_argument("--complete-only", action="store_true",
                   help="restrict --all to stations with nothing still owed "
                        "by the download campaign. A station whose archive is "
                        "still growing will give a different answer next week, "
                        "which is not a property of the station pair")
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
    p.add_argument("--allow-partial", action="store_true",
                   help="measure a pair even when a station has chunks on "
                        "disk that this arm has not scored. Off by default: "
                        "the result would be written, then skipped as "
                        "'already measured' on the next run, and silently "
                        "stand as the answer for a fraction of the record")


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
    rows = usable(survey(cfg.archive, cfg.ledgers))
    if args.station:
        want = set(args.station)
        rows = [r for r in rows if r.station in want]
    if args.complete_only:
        dropped = [r.station for r in rows if r.pending_days]
        rows = [r for r in rows if not r.pending_days]
        if dropped:
            print(f"still owed data, so excluded: {', '.join(sorted(dropped))}")
    unscored = [r.station for r in rows
                if not any(lay.scores(r.station, args.arm).glob("*.npz"))]
    if unscored:
        print(f"no {args.arm} scores, so excluded: {', '.join(sorted(unscored))}")
    rows = [r for r in rows if r.station not in set(unscored)]
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

    # Scored chunks against chunks on disk, per station. A pair computed from
    # a station mid-scan is not wrong so much as provisional, and the skip
    # rule would then freeze it as final.
    on_disk = {s: len(v) for s, v in find_chunks(cfg.archive).items()}
    scored = {s: len(list(lay.scores(s, args.arm).glob("*.npz")))
              for s in on_disk}
    partial = {s: (scored[s], on_disk[s]) for s in on_disk
               if 0 < scored[s] < on_disk[s]}

    taup = ArrivalTimes(grid_km=5.0)
    done = failed = skipped_partial = 0
    for sep, joint, a, b in todo:
        dest = lay.pair(a, b, args.arm)
        if dest.exists() and not args.force:
            print(f"== {a}-{b}: already measured, skipping")
            continue
        for stn in (a, b):
            if not scored.get(stn):
                print(f"== {a}-{b}: {stn} has no {args.arm} scores, skipping")
                break
            if stn in partial and not args.allow_partial:
                n, tot = partial[stn]
                print(f"== {a}-{b}: {stn} is only {n}/{tot} scored on "
                      f"{args.arm}; skipping (--allow-partial to override)")
                skipped_partial += 1
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
    print(f"{done} pair(s) measured"
          + (f", {failed} failed" if failed else "")
          + (f", {skipped_partial} waiting on a partial scan" if skipped_partial
             else ""))
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
