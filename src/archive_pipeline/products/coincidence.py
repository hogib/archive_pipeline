"""What requiring two stations to agree costs, and what it buys.

Single-station continuous detection is dominated by false alarms: at the
thresholds this detector needs to keep any recall, one station declares tens of
times a day. Requiring a second to agree within the time an event's P wave
could plausibly take to cross the pair is the standard network answer, and the
reduction it delivers is usually quoted from an independence assumption.

**That assumption is the thing worth measuring.** Two stations 130 km apart
share weather, share the regional noise field, and share whatever diurnal
cultural signal drives the day/night ratio. To the extent their false alarms
are common-mode the reduction is smaller than independence predicts, and no
amount of arithmetic can say by how much. So the measured joint rate is
reported against the independent prediction, and their ratio.

Three things this refuses to do:

**It scores only the span both stations recorded.** Coverage is intersected
first. Counting an unconfirmed alarm as suppressed while the other station was
off the air would read as a large gain and be nothing but missing data.

**It removes catalogued events from both streams.** A real earthquake is
detected at both stations by construction, so leaving events in makes every one
a guaranteed coincidence and the "excess" then measures how many events the
span contains rather than how much the two stations' false alarms agree.

**It asks recall only of events both stations actually recorded.** An event
below SNR at either station cannot be confirmed by a network rule, and charging
the rule for it measures the catalogue's reach, not the method.
"""
import numpy as np
import pandas as pd

from archive_pipeline.archive import (coverage_spans, in_spans,
                                      intersect_spans)
from archive_pipeline.inventory.stations import separation_km
from archive_pipeline.products.alarms import (background_mask, confirmed,
                                              declarations, load_scores)
from archive_pipeline.products.association import load_snr, predicted_arrivals

BUDGETS = (100.0, 30.0, 10.0, 3.0, 1.0, 0.1)
DEFAULT_VP = 6.0


def measure(scores_a, station_a, scores_b, station_b, stations_csv, catalog,
            window_seconds, coords=None, coincidence_seconds=None,
            vp=DEFAULT_VP, snr_csv_a=None, snr_csv_b=None, snr_min=3.0,
            max_distance=500.0, guard_pre=10.0, guard_post=60.0,
            signal_post=20.0, cluster_seconds=60.0, budgets=BUDGETS,
            taup=None, log=print):
    """The coincidence table for one station pair at one arm.

    Args:
        scores_a: Glob of station A's `.npz` score files.
        station_a: Station A's code.
        scores_b: Glob of station B's score files.
        station_b: Station B's code.
        stations_csv: Station table.
        catalog: Event catalogue CSV.
        window_seconds: The arm's window length. Must be the same arm at both
            stations or the two alarm streams are not comparable.
        coords: Station coordinates, to avoid re-reading the table.
        coincidence_seconds: How far apart two declarations may be and still
            count as one. Defaults to separation / `vp`, the largest P-arrival
            difference any event can produce at this pair; smaller loses real
            events on the line through both stations.
        vp: Crustal Vp for the default coincidence window.
        snr_csv_a: Per-event SNR table for A, enabling the recall column.
        snr_csv_b: The same for B.
        snr_min: Recall is asked only of events clearing this at both.
        max_distance: Catalogue radius in km.
        guard_pre: Seconds before P a catalogued event's guard starts.
        guard_post: Seconds after P it ends.
        signal_post: Seconds after P within which a detection counts.
        cluster_seconds: Alarms closer than this are one declaration.
        budgets: Alarms per day per station to threshold at.
        taup: An `ArrivalTimes` to share.
        log: Where the narrated table goes.

    Returns:
        DataFrame, one row per budget.

    Raises:
        ValueError: The two stations barely overlap.
    """
    ta_all, pa_all = load_scores(scores_a)
    tb_all, pb_all = load_scores(scores_b)
    step = float(np.median(np.diff(ta_all[:100000])))

    sep = separation_km(coords, station_a, station_b)
    w = coincidence_seconds if coincidence_seconds is not None else sep / vp
    log(f"{'=' * 78}\nTWO-STATION COINCIDENCE  --  {station_a} + {station_b}  "
        f"({window_seconds:g}s windows)\n{'=' * 78}")
    log(f"  separation {sep:.0f} km -> coincidence window +/-{w:.1f} s "
        + ("(default: separation / Vp " + format(vp, "g") + ")"
           if coincidence_seconds is None else "(given)"))

    spans = intersect_spans(coverage_spans(ta_all, step),
                            coverage_spans(tb_all, step))
    joint_s = sum(hi - lo for lo, hi in spans)
    days = joint_s / 86400.0
    ka, kb = in_spans(ta_all, spans), in_spans(tb_all, spans)
    ta, pa, tb, pb = ta_all[ka], pa_all[ka], tb_all[kb], pb_all[kb]
    log(f"  {station_a}: {len(ta_all) * step / 86400:.1f} d scored, "
        f"{station_b}: {len(tb_all) * step / 86400:.1f} d scored, "
        f"both at once: {days:.1f} d in {len(spans)} span(s)")
    if days < 1:
        raise ValueError(f"{station_a} and {station_b} overlap for "
                         f"{days:.2f} d; nothing to measure")

    # Keyed "a"/"b", not by station name: passing the same station twice is
    # the obvious self-test, and a name-keyed dict silently collapses to one
    # entry for it -- one station's background overwrites the other's and the
    # threshold table comes out empty.
    cats = {}
    for side, name, csv in (("a", station_a, snr_csv_a),
                            ("b", station_b, snr_csv_b)):
        cat, _ = predicted_arrivals(name, stations_csv, catalog, max_distance,
                                    taup=taup)
        # `in_spans`, not a comprehension over spans: a gap-split archive has
        # tens of thousands of them, and one Python-level pass per event over
        # all of them is hours rather than seconds.
        cat = cat[in_spans(cat.p_epoch.values, spans)].copy()
        if csv:
            cat = cat.merge(load_snr(csv), left_on="EventID",
                            right_on="event_id", how="left")
        else:
            cat["snr"] = np.nan
        cats[side] = cat.drop_duplicates(subset="EventID")

    both = cats["a"].merge(cats["b"][["EventID", "snr", "p_epoch"]],
                           on="EventID", suffixes=("_a", "_b"))
    good = both[(both.snr_a >= snr_min) & (both.snr_b >= snr_min)]
    log(f"  {len(both):,} catalogued event(s) in that span; "
        f"{len(good):,} reach SNR {snr_min:g} at BOTH stations")
    if len(good):
        dp = (good.p_epoch_b - good.p_epoch_a).abs()
        log(f"  their |P_A - P_B| spans {dp.min():.1f}..{dp.max():.1f} s "
            f"(median {dp.median():.1f}) -- the window must cover this")

    bg, unexplained = {}, {}
    for side, tt, pp in (("a", ta, pa), ("b", tb, pb)):
        explained = background_mask(tt, cats[side], window_seconds,
                                    guard_pre, guard_post)
        bg[side], unexplained[side] = pp[~explained], ~explained

    log("\n  Each station is thresholded to the SAME alarm budget, not the same")
    log("  threshold: their backgrounds differ and a shared number would not")
    log("  mean the same thing at both. Rates count UNEXPLAINED declarations")
    log("  only; windows overlapping a catalogued event's guard are removed.\n")
    log(f"  {'budget/day':>11}{'thr ' + station_a:>12}{'thr ' + station_b:>12}"
        f"{'A/day':>9}{'B/day':>9}{'2of2/day':>10}{'if indep':>10}"
        f"{'excess':>8}{'recall':>9}")

    rows = []
    for target in budgets:
        want = target * days
        if any(want >= len(bg[s]) for s in ("a", "b")):
            continue
        thr = {s: float(np.quantile(bg[s], 1.0 - want / len(bg[s])))
               for s in ("a", "b")}
        ua, ub = unexplained["a"], unexplained["b"]
        da, _ = declarations(ta[ua], pa[ua], thr["a"], cluster_seconds)
        db, _ = declarations(tb[ub], pb[ub], thr["b"], cluster_seconds)
        n_a, n_b = len(da), len(db)
        n_2 = int(confirmed(da, db, w).sum())
        ra, rb = n_a / joint_s, n_b / joint_s
        # The measured quantity is "A declarations having at least one B within
        # +/-w", so the prediction must be for that and not for the number of
        # coincident pairs: a Poisson B stream puts 1 - exp(-rb*2w) of them in
        # the window, which is below rb*2w whenever B is busy. The two agree to
        # 0.25% at 10 alarms/day and diverge by 10% at 200 -- so the distinction
        # matters exactly at the loose end, where the reduction looks best.
        indep = ra * (1.0 - np.exp(-rb * 2 * w)) * 86400
        rec = np.nan
        if len(good):
            fired = {}
            for side, tt, pp, col in (("a", ta, pa, "p_epoch_a"),
                                      ("b", tb, pb, "p_epoch_b")):
                fired[side] = np.array(
                    [(pp[np.searchsorted(tt, c - window_seconds):
                         np.searchsorted(tt, c + signal_post,
                                         side="right")] > thr[side]).any()
                     for c in good[col].values])
            rec = float((fired["a"] & fired["b"]).mean())
        excess = n_2 / days / indep if indep > 0 else np.nan
        rows.append({"budget_per_day": target, "separation_km": sep,
                     "joint_days": days, "coincidence_seconds": w,
                     "station_a": station_a, "station_b": station_b,
                     "thr_a": thr["a"], "thr_b": thr["b"],
                     "a_per_day": n_a / days, "b_per_day": n_b / days,
                     "both_per_day": n_2 / days, "independent_per_day": indep,
                     "excess_over_independent": excess, "recall_both": rec})
        log(f"  {target:>11.4g}{thr['a']:>12.4f}{thr['b']:>12.4f}"
            f"{n_a / days:>9.2f}{n_b / days:>9.2f}{n_2 / days:>10.3f}"
            f"{indep:>10.4f}{excess:>8.1f}x"
            + (f"{rec:>9.3f}" if rec == rec else f"{'-':>9}"))

    log("\n  `excess` is the measured two-station rate divided by what two")
    log("  independent alarm streams of the same rates would produce. 1.0x")
    log("  means their false alarms are independent and the textbook reduction")
    log("  holds; above 1.0x they share a cause and the network rule buys less")
    log("  than the arithmetic promises.")
    return pd.DataFrame(rows)
