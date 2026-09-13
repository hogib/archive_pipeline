r"""`apipe` -- one front door for turning a seismic archive into artifacts.

    apipe                      the listing
    apipe inventory            what data exists and what is worth running
    apipe plan                 what a batch would do, without doing it
    apipe run --all            do it, resuming where it left off

Every command reads its paths from `archive_pipeline.toml`, so an invocation
carries only what is specific to it. Each command also runs standalone as
`python -m archive_pipeline.commands.<name>`.
"""
import argparse
import importlib
import sys

# name -> (module, one-line description), grouped below by what you are doing.
COMMANDS = {
    "inventory":   ("archive_pipeline.commands.inventory",
                    "what has been fetched, and which pairs are testable"),
    "plan":        ("archive_pipeline.commands.plan",
                    "what a batch would process, and what it would skip"),
    "run":         ("archive_pipeline.commands.run",
                    "one decode pass per chunk, every product, resumable"),
    "baseline":    ("archive_pipeline.commands.baseline",
                    "per-station noise (mu, sigma); needed before scanning"),
    "coincidence": ("archive_pipeline.commands.coincidence",
                    "require two stations to agree, and price what that costs"),
    "encode":      ("archive_pipeline.commands.encode",
                    "cut windows -> model-ready tensors"),
    "adopt":       ("archive_pipeline.commands.adopt",
                    "move artifacts made before this project into out/"),
}

GROUPS = [
    ("look", ["inventory", "plan"]),
    ("build", ["baseline", "run", "encode"]),
    ("migrate", ["adopt"]),
    ("evaluate", ["coincidence"]),
]


def usage():
    """Prints the grouped command listing."""
    print("apipe -- continuous seismic archive to model-ready artifacts\n")
    print("usage: apipe <command> [args...]     (each command has its own --help)\n")
    for group, names in GROUPS:
        print(f"  {group}")
        for n in names:
            print(f"    {n:<14} {COMMANDS[n][1]}")
        print()


def main(argv=None):
    """Dispatches to one command's `main()`, leaving its arguments untouched."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        usage()
        return 0
    name = argv[0]
    if name not in COMMANDS:
        near = [c for c in COMMANDS if c.startswith(name[:3])]
        print(f"apipe: unknown command {name!r}"
              + (f" -- did you mean {' or '.join(near)}?" if near else ""),
              file=sys.stderr)
        print("run `apipe` for the list", file=sys.stderr)
        return 2
    sys.argv = [f"apipe {name}"] + argv[1:]
    return importlib.import_module(COMMANDS[name][0]).main() or 0


def command_parser(module, description):
    """A parser for one command, used both by `apipe` and standalone."""
    p = argparse.ArgumentParser(prog=f"apipe {module.NAME}",
                                description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    module.add_args(p)
    return p


if __name__ == "__main__":
    sys.exit(main())
