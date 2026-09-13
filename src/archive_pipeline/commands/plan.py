"""`apipe plan` -- what a batch would process, and what it would skip.

The point of the table is that it is per chunk, not per station. "MANT: done"
and "MANT: 34 of 36 chunks scored" lead to different decisions, and only the
second survives an interrupted run.
"""
import argparse

from archive_pipeline import config
from archive_pipeline.archive import chunk_date, find_chunks
from archive_pipeline.batch import Layout
from archive_pipeline.inventory import survey, usable

NAME = "plan"
HELP = "what a batch would process, and what it would skip"

DEFAULT_ARMS = ["6s"]
DEFAULT_WINDOWS = [6, 10, 20]


def add_args(p):
    """Flags for the plan, shared with `apipe run`."""
    config.add_path_args(p)
    p.add_argument("--station", action="append",
                   help="restrict to these stations; repeatable")
    p.add_argument("--all", action="store_true",
                   help="every station with enough record to process")
    p.add_argument("--arm", action="append",
                   help="detector arm NAME:WINDOW:CKPT_DIR:BRANCH[:STEP]; "
                        "repeatable. For planning only the NAME is read.")
    p.add_argument("--window-seconds", type=int, action="append",
                   help="window length to cut; repeatable "
                        f"(default: {' '.join(map(str, DEFAULT_WINDOWS))})")
    p.add_argument("--no-windows", action="store_true",
                   help="skip window cutting")
    p.add_argument("--no-range", action="store_true",
                   help="skip per-event SNR")


def select(cfg, args):
    """Stations to process, and their chunks.

    Returns:
        Dict of station code to a list of chunk stems, in time order.
    """
    wanted = args.station
    if not wanted and args.all:
        wanted = [r.station for r in usable(survey(cfg.archive, cfg.ledgers))]
    chunks = find_chunks(cfg.archive, wanted)
    return {s: [p.stem for p in ps] for s, ps in sorted(chunks.items())}


def arm_names(args):
    """Arm names from `--arm` specs, or the default single arm."""
    if not args.arm:
        return list(DEFAULT_ARMS)
    return [spec.split(":")[0] for spec in args.arm]


def build(cfg, args):
    """The work matrix.

    Returns:
        List of per-station dicts carrying the counts `apipe plan` prints and
        `apipe run` acts on.
    """
    lay = Layout(cfg.out)
    arms = arm_names(args)
    windows = [] if args.no_windows else (args.window_seconds or DEFAULT_WINDOWS)
    rows = []
    for stn, chunks in select(cfg, args).items():
        rows.append({
            "station": stn,
            "chunks": chunks,
            "baseline": lay.baseline(stn).exists(),
            "range": True if args.no_range else lay.has_range(stn, chunks),
            "scores": {a: lay.scored(stn, a, chunks) for a in arms},
            "windows": {w: lay.uncut(stn, w, chunks) for w in windows},
        })
    return rows, arms, windows


def decode_passes(row, arms, windows):
    """Chunks needing a decode at all, as a set of chunk stems.

    This is the number the whole project turns on: every product below is fed
    from one decode, so the cost of a station is the size of this union, not
    the sum of its products.
    """
    need = set()
    for pending in row["scores"].values():
        need |= set(pending)
    for pending in row["windows"].values():
        need |= set(pending)
    if not row["range"]:
        need |= set(row["chunks"])
    return need


def run(args):
    """Prints the work matrix."""
    cfg = config.load(args)
    cfg.require("archive")
    rows, arms, windows = build(cfg, args)
    if not rows:
        print("no stations selected; pass --all or --station CODE")
        return 1

    head = (f"{'stn':<6}{'chunks':>7}{'base':>6}{'range':>7}"
            + "".join(f"{'scan:' + a:>12}" for a in arms)
            + "".join(f"{'cut:' + str(w) + 's':>10}" for w in windows)
            + f"{'decode':>8}")
    print(head)
    print("-" * len(head))
    total_decode = 0
    for r in rows:
        n = len(r["chunks"])
        need = decode_passes(r, arms, windows)
        total_decode += len(need)
        line = (f"{r['station']:<6}{n:>7}"
                f"{('yes' if r['baseline'] else 'NO'):>6}"
                f"{('yes' if r['range'] else 'no'):>7}")
        for a in arms:
            done = n - len(r["scores"][a])
            line += f"{f'{done}/{n}':>12}"
        for w in windows:
            done = n - len(r["windows"][w])
            line += f"{f'{done}/{n}':>10}"
        line += f"{len(need):>8}"
        print(line)
    print("-" * len(head))
    print(f"{'':<6}{sum(len(r['chunks']) for r in rows):>7}"
          + " " * (13 + 12 * len(arms) + 10 * len(windows))
          + f"{total_decode:>8}")

    missing = [r["station"] for r in rows if not r["baseline"]]
    if missing:
        print(f"\nno noise baseline yet, so scanning cannot start: "
              f"{', '.join(missing)}")
        print("  apipe baseline --all")
    naive = sum(len(r["chunks"]) * (len(arms) + len(windows)
                                    + (0 if r["range"] else 1)) for r in rows)
    if total_decode:
        print(f"\n{total_decode} chunk decode(s) to do. The same work as "
              f"separate per-product passes would be {naive}.")
    else:
        print("\nnothing to do; every product is present.")
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
