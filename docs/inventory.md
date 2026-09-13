# Inventory

Two questions: what record exists, and which station pairs are worth testing.

## Four ledgers

A download campaign writes one JSONL ledger per run, and this project
accumulated four — `tdvms_ledger`, `afad_campaign_ledger`, `demi_ledger`,
`gcam_ledger`. Their station sets overlap and their fetched spans disagree,
because the older ones stopped being updated when a newer one took over. SEMS
reads 504 fetched days in one and 525 in another; DEMI and GCAM appear in
neither of the two largest. **Reading any single ledger understates coverage.**

All four are read and reconciled against the archive directory. Where they
disagree with disk, disk wins: a row marked fetched whose zip is gone is not
data. Ledgers are read append-only and never locked, so this is safe to run
while a pull is in flight — and a half-written final line is skipped rather
than crashing the read.

## What the archive holds

```
stn    chunks     GB  fetched_d  pending_d
MANT       36   27.3        747          0
DEMI       31   14.6        537          0
SEMS       25   11.4        525         42
KAND       21    9.8        441        180
ELBA       20   16.4        408         21
BAND       18   12.1        378        327
KIRK       18   10.3        378        285
CMH        16   11.3        336        348
KURT       11    5.5        231        390
VIZE       10    5.8        210        411
GCAM        9    6.9        189          0
ARNA        1    0.9         21         21
CATL        1    0.5         21         21
```

ARNA and CATL hold one chunk each — three weeks, which is not enough to measure
a false-alarm rate against and not enough to pair with anything. Eleven
stations are usable.

## Pairs

A coincidence test is bounded by **joint** coverage, not by either station's
own archive, and joint coverage is frequently a third of what the two have
separately. The pair table is therefore the one that decides what to run.

Below 60 joint days a test cannot resolve the tight alarm budgets: at one alarm
per day per station the expected count of chance agreements over two months
rounds to zero and every cell reads 0.000. That is the cutoff `pairs()` applies.

55 pairs clear it. The most valuable, nearest first:

| km | joint days | pair | |
|---|---|---|---|
| 39.8 | 273 | BAND–CMH | |
| 45.4 | 328 | SEMS–KAND | |
| 54.5 | 202 | SEMS–KURT | |
| 62.9 | 453 | MANT–DEMI | done |
| 65.7 | 168 | KIRK–VIZE | |
| 114.0 | 357 | SEMS–ELBA | fills the transition |
| 144.0 | 189 | MANT–GCAM | done |
| 195.8 | 63 | DEMI–GCAM | done |

The three already measured span 63–453 joint days at 63–196 km. What the set
lacks is short separations, which is why BAND–CMH and SEMS–KAND lead the queue.
