# The batch driver

## One pass

For each chunk that needs anything, decode once and hand the same segments to
every product that wants them. The plan is per chunk, not per station, because
"MANT: done" and "MANT: 34 of 36 chunks scored" lead to different decisions and
only the second survives an interruption.

```
apipe plan --all

stn    chunks  base  range     scan:6s    cut:6s   cut:10s   cut:20s  decode
----------------------------------------------------------------------------
BAND       18    NO     no        0/18      0/18      0/18      0/18      18
...
MANT       36   yes    yes       36/36     36/36     36/36     36/36       0
----------------------------------------------------------------------------
          216                                                            180

180 chunk decode(s) to do. The same work as separate per-product passes
would be 1004.
```

The last column is the number the project turns on: the union of chunks needing
*any* product, not the sum over products.

## Layout

```
out/
  stations/<STN>/
    baseline.json              whole-archive, sampled; scanning needs it
    range/<chunk>.csv          per-event SNR, one part per chunk
    range.csv                  the parts concatenated
    scores/<arm>/<chunk>.npz   detector scores
    windows/<W>s/eq|noise/     cut windows
    windows/<W>s/.chunks/<chunk>   marker: this chunk has been cut
  pairs/<A>-<B>/<arm>.csv
  datasets/<name>/
```

The marker directory exists because cutting a chunk emits a variable number of
window files, sometimes zero — a quiet station over three weeks is a real and
correct outcome — so "no files" cannot distinguish done from not started.

## Resuming

Every product is written the moment it is computed and skipped when present, so
an interrupted run loses at most one chunk. This matters: a full pass is hours
on a machine that is also downloading.

### Cutting needs the pass before it

Window cutting is filtered by measured signal-to-noise, and that table is
itself a product of the same pass. A station seeing its first pass therefore
cannot cut in it: `run` says so and defers. Re-running cuts.

The alternative — cut everything and filter at encode time — was rejected on
volume. At GCAM only 624 of 5,116 measured events clear SNR 3, so cutting
unfiltered would write roughly eight times the files to throw most away. Two
passes is still half of what the tooling this replaces needed, and a station
that already has `range.csv` cuts in its first.

Two forms of completion are recognised. A product keyed per chunk is done when
its file exists. A whole-archive product adopted from the older tooling — a
`range.csv` with no per-chunk parts, a `windows/<W>s/` with no marker
directory — is also done, because recomputing it would cost a full pass for
nothing.

## Adoption

`apipe adopt` moves artifacts made before this project into the layout above.
Four stations were processed by hand, one command at a time, and those runs
cost GPU-days.

The mapping is a command rather than a shell one-liner because it is not
obvious. The old tooling named things per command — a baseline was
`<stn>_baseline.json` in whatever the working directory happened to be, and one
station's ended up in a different repository entirely — so reconstructing which
file belonged to which station later would mean reading the commands that made
them.

Moves are `os.replace` within one filesystem: atomic renames, nothing ever
half-copied, and a destination that already exists is reported rather than
overwritten.
