"""Tying a station's record to the event catalogue.

Predicted arrivals and the measured-SNR table, shared by every product that
needs to know when an event should have been visible.
"""
import numpy as np
import pandas as pd

from archive_pipeline.arrivals import P_PHASES, S_PHASES, ArrivalTimes
from archive_pipeline.inventory.stations import haversine_km


def station_coords(stations_csv, station):
    """Latitude and longitude of one station.

    Raises:
        KeyError: The station is not in the table, named so the caller can say
            which one rather than failing on an empty frame later.
    """
    tab = pd.read_csv(stations_csv, encoding="utf-8-sig")
    tab.columns = [c.strip() for c in tab.columns]
    row = tab[tab.Code == station]
    if row.empty:
        raise KeyError(f"{station} is not in {stations_csv}")
    return float(row.iloc[0].Latitude), float(row.iloc[0].Longitude)


def predicted_arrivals(station, stations_csv, catalog, max_distance=500.0,
                       taup=None):
    """Catalogued events near the station, with predicted P and S arrivals.

    Args:
        station: Station code, matched against the table's `Code`.
        stations_csv: Station table with Code/Latitude/Longitude.
        catalog: Event catalogue CSV.
        max_distance: Events beyond this are dropped, in km.
        taup: An `ArrivalTimes` to share, so its cache survives across stations.

    Returns:
        Tuple of `(DataFrame with p_epoch/s_epoch/sp_seconds/dist, (lat, lon))`.
    """
    slat, slon = station_coords(stations_csv, station)
    cat = pd.read_csv(catalog, encoding="utf-8-sig")
    cat["t"] = pd.to_datetime(cat.Date, format="%d/%m/%Y %H:%M:%S",
                              errors="coerce")
    cat = cat.dropna(subset=["t"])
    cat["dist"] = haversine_km(slat, slon, cat.Latitude.values,
                               cat.Longitude.values)
    cat = cat[cat.dist <= max_distance].copy()

    taup = taup or ArrivalTimes(grid_km=5.0)
    depth = [d if pd.notna(d) else 10.0 for d in cat.Depth.values]
    cat["tt_p"] = [taup.travel(d, z, P_PHASES)
                   for d, z in zip(cat.dist.values, depth)]
    cat["tt_s"] = [taup.travel(d, z, S_PHASES)
                   for d, z in zip(cat.dist.values, depth)]
    cat = cat.dropna(subset=["tt_p"])
    origin = cat.t.map(lambda x: x.timestamp())
    cat["p_epoch"] = origin + cat.tt_p
    cat["s_epoch"] = origin + cat.tt_s
    cat["sp_seconds"] = cat.tt_s - cat.tt_p
    return cat.sort_values("p_epoch").reset_index(drop=True), (slat, slon)


def load_snr(path):
    """The measured-SNR table, one row per event, larger reading kept.

    The SNR table can carry an event twice when it falls in two overlapping
    chunks, and a left join on a non-unique key silently *expands* the frame it
    is joined into. That is not hypothetical: DEMI's table had 269 duplicated
    ids where MANT's and GCAM's had none, and the expansion desynchronised a
    probability column from the catalogue it was computed for.

    The larger reading is kept, because a duplicate is the same event seen from
    two chunks and the smaller is usually the one that fell near a chunk edge
    and was measured on a truncated window.
    """
    snr = pd.read_csv(path)[["event_id", "snr"]]
    return (snr.sort_values("snr", ascending=False)
            .drop_duplicates(subset="event_id"))
