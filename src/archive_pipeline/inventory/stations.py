"""Station coordinates and the distances between them."""
import csv

import numpy as np

EARTH_RADIUS_KM = 6371.0


def load_stations(path):
    """Station code to `(lat, lon)` from an AFAD station catalogue.

    Args:
        path: CSV with `Code`, `Latitude` and `Longitude` columns. Read as
            utf-8-sig because the distributed file carries a BOM, which
            otherwise makes the first column name unmatchable.

    Returns:
        Dict of station code to `(lat, lon)`.
    """
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("Code") or "").strip()
            if code:
                out[code] = (float(row["Latitude"]), float(row["Longitude"]))
    return out


def haversine_km(lat0, lon0, lats, lons):
    """Great-circle distance in km from one point to arrays of points."""
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    lats, lons = np.radians(np.asarray(lats)), np.radians(np.asarray(lons))
    d = (np.sin((lats - lat0) / 2) ** 2
         + np.cos(lat0) * np.cos(lats) * np.sin((lons - lon0) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(d))


def separation_km(coords, a, b):
    """Distance between two stations, by code."""
    (la, lo), (lb, lob) = coords[a], coords[b]
    return float(haversine_km(la, lo, [lb], [lob])[0])
