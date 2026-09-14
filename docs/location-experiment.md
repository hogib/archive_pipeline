# Proposal: locating events from three stations

*Status: proposed, not started. Nothing here has been run.*

## The idea

The cascade detects and estimates magnitude. It does not say **where**. Every
alarm is currently a time and a station, and the two-station coincidence work
already computes the one quantity a locator wants — the difference in arrival
time between stations. Turning that into an epicentre is a third stage, and
the data to attempt it already exists.

## Why three stations

An earthquake has four unknowns: latitude, longitude, depth and origin time.
Each P pick is one equation. Absolute arrival times therefore need four
stations. Using *differences* between stations eliminates the origin time,
which buys one unknown back: N stations give N−1 independent differences, so

    3 stations  ->  2 differences  ->  latitude and longitude, depth fixed
    4 stations  ->  3 differences  ->  latitude, longitude and depth

Three is the minimum that can produce an epicentre at all, which is why the
experiment is built on triplets.

**Three is also exactly determined, and that has a cost.** Two equations in
two unknowns leaves no residual, so the fit cannot report its own quality —
a wrong answer looks identical to a right one. Validation has to come from
outside, by comparing against the catalogue. Four stations would give one
redundant observation and a residual to read; that is the natural follow-up,
and §"If it works" says what it needs.

A second consequence of exact determination: two hyperbolae can intersect
twice. The solver must return both roots and choose, and the choice cannot be
made from the arrival times alone.

## The tension this experiment is really about

Ranking all 35 triplets of the seven complete stations by how many catalogued
events reach SNR 3 at all three, against the smallest interior angle of the
station triangle:

| triplet | events at all 3 | joint days | min angle | aperture |
|---|---|---|---|---|
| **DEMI–KAND–KIRK** | 560 | 306 | **50°** | 334 km |
| **KAND–KIRK–CMH** | 549 | 243 | **49°** | 269 km |
| MANT–BAND–KAND | 472 | 432 | 36° | 321 km |
| **MANT–DEMI–CMH** | **1,521** | 285 | **14°** | 177 km |
| MANT–DEMI–BAND | 1,460 | 390 | 9° | 211 km |
| DEMI–BAND–CMH | 1,121 | 285 | 11° | 155 km |
| **BAND–CMH–GCAM** | 36 | 105 | **0°** | 303 km |

**The event-rich triplets have the worst geometry.** That is not a
coincidence: stations were sited along the seismically active trend, so the
triplets that see the most earthquakes are the ones strung out in a line. A
near-collinear triangle constrains position well along its axis and hardly at
all across it — the error ellipse degenerates into a cigar — and at 0° there
is no solution perpendicular to the line at all.

So the experiment is not "can we locate earthquakes". It is **how much
geometry is worth against how much data**, which is a question the network's
own layout forces and which a single triplet could not answer.

## Design

Three triplets, chosen to span the tension rather than to win:

| role | triplet | why |
|---|---|---|
| **primary** | DEMI–KAND–KIRK | best geometry with a usable event count |
| **contrast** | MANT–DEMI–CMH | 2.7× the events, a third of the angle |
| **control** | BAND–CMH–GCAM | collinear; must fail, and in a predictable direction |

The control is the point. A locator that returns plausible-looking answers for
a collinear triplet is broken, and without a triplet that *must* fail there is
no way to notice.

## Method

1. **Association.** Take alarms from the `6s-trim` scores at each station and
   group them into events using the existing coincidence window
   (separation / Vp). This machinery exists.
2. **Arrival refinement.** A 6 s disjoint window locates P to ±3 s, which at
   6 km/s is ±18 km of slop — far too coarse. Re-score a dense grid around
   each alarm (`--near-csv` already supports exactly this: it restricts
   scoring to intervals around given times, so a dense rescan of alarm
   neighbourhoods costs minutes rather than days) and take the alarm peak, or
   fit an STA/LTA onset inside the window.
3. **Solve.** Grid search over latitude/longitude at fixed 10 km depth,
   minimising the misfit between observed and iasp91-predicted differential
   times. `ArrivalTimes` already provides the travel-time model, cached.
   Report both roots where two exist.
4. **Score against the catalogue.** Median and 90th-percentile epicentral
   error in km, stratified by whether the event falls inside or outside the
   station triangle.

## Floors

No location result is meaningful without them. In this project's convention,
each is reported beside the figure it had to clear:

- **network centroid** — predict the same point for every event. The number
  to beat before anything has been demonstrated.
- **nearest triggering station** — predict the location of whichever station
  saw it strongest. Surprisingly hard to beat at regional distance.
- **catalogue self-consistency** — AFAD's own locations carry error (median
  RMS residual 0.42 s), so there is a floor below which "disagreement with the
  catalogue" stops meaning "wrong".

## What would make it interesting

Not accuracy. A three-station regional location will be worse than AFAD's
network solution, which uses far more stations, and beating it is not the
goal. The interesting results are:

- **the geometry–data exchange rate** — how many extra events buy how many
  degrees of angle, measured rather than assumed
- **inside versus outside the triangle** — the degradation should be sharp,
  and quantifying where it becomes useless tells you where a sparse network
  can be trusted
- **whether detector alarm times are precise enough at all**, which is a
  property of the detector and worth knowing independently

## What could kill it

- **Alarm timing precision.** If refinement cannot get P below about ±0.5 s,
  differential times carry ~3 km of noise each and three-station geometry
  will not survive it. This is the most likely failure and should be measured
  **first**, before any solver is written.
- **Depth.** Fixing 10 km is wrong for events that are not at 10 km, and the
  error projects into the epicentre. Regional practice, but a real error term.
- **The two-root ambiguity** may not be resolvable without back-azimuth.

## If it works

Four-station subsets give a residual and therefore an uncertainty, and 2,067
catalogued events reach SNR 3 at four or more of the seven stations.

The larger prize is **single-station location**: P-wave polarisation on the
three components gives back-azimuth, and S−P gives distance, which together
locate an event from one station. 16,819 events are seen at exactly one
station and are currently unusable — an order of magnitude more than the
four-station set. That needs an S picker, which the cascade does not have.

## What exists already

- alarm times per station, at 7 stations over 189–747 days (`6s-trim`)
- per-event SNR, so events can be restricted to those actually recorded
- association within a travel-time window (`products/coincidence.py`)
- cached iasp91 travel times for P and S (`arrivals.py`)
- dense rescanning around given times (`scan`'s `near` argument)

What is missing is the solver and the arrival refinement.
