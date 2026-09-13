"""`apipe run` -- one decode pass per chunk, every product, resumable.

    apipe run --all
    apipe run --station SEMS --station KAND
    apipe run --all --arm 6s:6.0:trained_model_branch1d_asinh:cnn-lstm

Interrupting is safe. Products are written as each is computed and skipped
when present, so re-running resumes at the chunk it stopped on.
"""
import argparse

import torch

from archive_pipeline import config
from archive_pipeline.batch import Layout
from archive_pipeline.batch.run import run_stations
from archive_pipeline.commands import plan as plan_cmd
from archive_pipeline.products.scan import Arm

NAME = "run"
HELP = "one decode pass per chunk, every product, resumable"

DEFAULT_ARM = "6s:6.0:trained_model_branch1d_asinh:cnn-lstm"


def add_args(p):
    """Flags for the batch run."""
    plan_cmd.add_args(p)
    g = p.add_argument_group("conditioning (must match the baseline)")
    g.add_argument("--fs", type=float, default=100.0)
    g.add_argument("--freqmin", type=float, default=1.0)
    g.add_argument("--freqmax", type=float, default=45.0)
    g = p.add_argument_group("compute")
    g.add_argument("--workers", type=int, default=6,
                   help="filter threads; scipy releases the GIL, so these "
                        "give real parallelism (default: 6)")
    g.add_argument("--device", default=None, help="cuda or cpu; default auto")
    g.add_argument("--limit-chunks", type=int, default=None,
                   help="stop each station after N chunks, for a smoke test")


def run(args):
    """Loads the arms and processes every selected station."""
    cfg = config.load(args)
    cfg.require("archive", "stations", "catalog")
    rows, _, _ = plan_cmd.build(cfg, args)
    if not rows:
        print("no stations selected; pass --all or --station CODE")
        return 1
    if args.limit_chunks:
        for r in rows:
            r["paths"] = r["paths"][:args.limit_chunks]
            r["chunks"] = r["chunks"][:args.limit_chunks]

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = args.arm or [DEFAULT_ARM]
    arms = []
    for spec in specs:
        arm = Arm.parse(spec, fs=args.fs).load(device)
        arms.append(arm)
        print(f"[arm {arm.name}] {arm.window_seconds:g}s window every "
              f"{arm.step_seconds:g}s ({arm.win} samples), "
              f"{len(arm.models)} seed(s) from {arm.ckpt_dir}")
    print(f"[run] device={device}, {len(rows)} station(s)\n")

    run_stations(rows, Layout(cfg.out), cfg, arms, device,
                 fs=args.fs, freqmin=args.freqmin, freqmax=args.freqmax,
                 workers=args.workers, want_range=not args.no_range)
    return 0


def main():
    """Standalone entry point."""
    p = argparse.ArgumentParser(prog=f"apipe {NAME}", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
