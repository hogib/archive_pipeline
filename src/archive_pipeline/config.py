"""Where the data is.

Every command needs the same four paths -- the archive, the campaign ledgers,
the station catalogue and the event catalogue -- and typing them on each
invocation is how a run ends up pointed at the wrong catalogue without anyone
noticing. They are declared once in `archive_pipeline.toml`:

    archive   = "../tdvms/afad_raw"
    stations  = "../cnn_earthquake/catalogs/istasyon_katalog.csv"
    catalog   = "../cnn_earthquake/catalogs/catalog_current.csv"
    out       = "out"
    ledgers   = ["../tdvms/tdvms_ledger.jsonl",
                 "../cnn_earthquake/afad_campaign_ledger.jsonl"]

Relative paths resolve against the file's own directory, not the working
directory, so commands behave the same wherever they are run from. Any field
can still be overridden per invocation by the matching flag.
"""
import dataclasses
import pathlib
import tomllib

CONFIG_NAME = "archive_pipeline.toml"


@dataclasses.dataclass
class Config:
    """Resolved paths for one run."""
    archive: pathlib.Path
    stations: pathlib.Path
    catalog: pathlib.Path
    out: pathlib.Path
    ledgers: list
    source: pathlib.Path = None

    def require(self, *fields):
        """Fails loudly when a needed path is unset or missing.

        Raises:
            SystemExit: naming the field, the path and the config file it came
                from. A missing catalogue otherwise surfaces hours later as an
                empty association table.
        """
        for f in fields:
            v = getattr(self, f)
            if v is None:
                raise SystemExit(
                    f"{f!r} is not set; add it to {CONFIG_NAME} or pass --{f}")
            for p in (v if isinstance(v, list) else [v]):
                if f != "out" and not pathlib.Path(p).exists():
                    raise SystemExit(f"{f}: no such path: {p}"
                                     + (f"  (from {self.source})" if self.source else ""))


def find_config(start=None):
    """Nearest `archive_pipeline.toml`, searching upward from `start`."""
    here = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    for d in [here, *here.parents]:
        p = d / CONFIG_NAME
        if p.exists():
            return p
    return None


def load(args=None, start=None):
    """Builds a `Config` from the config file, overridden by CLI flags.

    Args:
        args: Parsed arguments, optionally carrying `archive`, `stations`,
            `catalog`, `out` and `ledger` attributes.
        start: Directory to search for the config file from.

    Returns:
        A `Config`. Paths from the file are resolved against its directory;
        paths from flags are resolved against the working directory, which is
        what someone typing a path expects.
    """
    path = find_config(start)
    data, base = {}, pathlib.Path.cwd()
    if path is not None:
        data = tomllib.loads(path.read_text())
        base = path.parent

    def pick(field, flag=None, default=None):
        override = getattr(args, flag or field, None) if args else None
        if override:
            return (pathlib.Path(override).resolve() if not isinstance(override, list)
                    else [pathlib.Path(o).resolve() for o in override])
        raw = data.get(field, default)
        if raw is None:
            return None
        if isinstance(raw, list):
            return [(base / r).resolve() for r in raw]
        return (base / raw).resolve()

    return Config(archive=pick("archive"), stations=pick("stations"),
                  catalog=pick("catalog"), out=pick("out", default="out"),
                  ledgers=pick("ledgers", "ledger") or [], source=path)


def add_path_args(p):
    """Adds the per-invocation overrides for the config fields."""
    g = p.add_argument_group("paths (override archive_pipeline.toml)")
    g.add_argument("--archive", help="directory of per-station chunk directories")
    g.add_argument("--stations", help="AFAD station catalogue CSV")
    g.add_argument("--catalog", help="event catalogue CSV")
    g.add_argument("--out", help="where products are written")
    g.add_argument("--ledger", action="append",
                   help="campaign ledger JSONL; repeatable")
