"""`apipe inventory` -- what has been fetched, and which pairs are testable."""
import argparse

from archive_pipeline import config
from archive_pipeline.archive import nonconforming
from archive_pipeline.inventory import load_stations, pairs, survey, usable

NAME = "inventory"
HELP = "what has been fetched, and which pairs are testable"


def add_args(p):
    """Flags for the inventory report."""
    config.add_path_args(p)
    p.add_argument("--station", action="append",
                   help="restrict to these stations; repeatable")
    p.add_argument("--min-joint-days", type=int, default=60,
                   help="drop pairs with less joint coverage than this "
                        "(default: 60, below which tight alarm budgets "
                        "cannot be resolved)")
    p.add_argument("--no-pairs", action="store_true",
                   help="station table only")


def run(args):
    """Prints the station table and, unless suppressed, the pair table."""
    cfg = config.load(args)
    cfg.require("archive")
    rows = survey(cfg.archive, cfg.ledgers, args.station)
    if not rows:
        print(f"no stations found under {cfg.archive}")
        return 1

    print(f"archive {cfg.archive}")
    print(f"ledgers {len(cfg.ledgers)}: "
          f"{', '.join(p.name for p in cfg.ledgers)}\n")
    print(f"{'stn':<6}{'chunks':>7}{'GB':>7}{'fetched_d':>11}{'pending_d':>11}"
          f"  {'coverage':<26} sources")
    for r in rows:
        span = f"{r.first}..{r.last}" if r.spans else "-"
        print(f"{r.station:<6}{r.chunks:>7}{r.gigabytes:>7.1f}"
              f"{r.fetched_days:>11}{r.pending_days:>11}  "
              f"{span + f' ({len(r.spans)} seg)':<26} "
              f"{','.join(sorted(r.sources))}")
    keep = usable(rows)
    print(f"\n{len(rows)} station(s), {sum(r.chunks for r in rows)} chunks, "
          f"{sum(r.bytes for r in rows) / 1e9:.0f} GB; "
          f"{len(keep)} with enough record to process")

    skipped = nonconforming(cfg.archive, args.station)
    if skipped:
        print("\nnot counted as chunks:")
        for station, items in skipped.items():
            why = {}
            for _, reason in items:
                why[reason] = why.get(reason, 0) + 1
            print(f"  {station:<6} " + ", ".join(f"{n} {r}" for r, n in why.items()))

    if args.no_pairs:
        return 0
    cfg.require("stations")
    coords = load_stations(cfg.stations)
    missing = [r.station for r in keep if r.station not in coords]
    table = pairs(keep, coords, args.min_joint_days)
    print(f"\n{len(table)} pair(s) with >= {args.min_joint_days} joint days\n")
    print(f"{'km':>7}{'joint_d':>9}  pair")
    for sep, joint, a, b in table:
        print(f"{sep:>7.1f}{joint:>9}  {a}-{b}")
    if missing:
        print(f"\nnot in the station catalogue, so unpaired: "
              f"{', '.join(missing)}")
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
