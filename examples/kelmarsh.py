"""Power curves, availability by IEC 61400-26 category and reliability of the Kelmarsh wind farm, 2016-2021.

Kelmarsh (UK) has six Senvion MM92 turbines of 2050 kW; Cubico Sustainable Investments published its 10-minute
SCADA data and status logs on Zenodo (record 16807551, CC BY 4.0). This script reads the yearly Parquet files
that the companion project wind-power-curve-kelmarsh builds from that record (``processed/scada_<year>.parquet``,
index turbine and time stamp; ``processed/status_<year>.parquet`` with ``turbine``, ``start``, ``end``,
``status`` and the operator's ``iec_category``).

It is an example of the library's API on real data, not a study: the operator's IEC categories are taken as
they are and a failure is any stop the log files as a forced outage.

Usage:
    python examples/kelmarsh.py --data <folder with processed/> [--out <folder for CSV results>]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from res_asset_kpi import intervals, wind

YEARS = range(2016, 2022)
COD = pd.Timestamp("2016-04-15", tz="UTC")       # commercial operation date of all six turbines
END = pd.Timestamp("2022-01-01", tz="UTC")
ELEVATION_M = 150.0                               # site elevation (the turbines stand at 125-157 m)
RATED_KW = 2050.0
CATEGORY = {                                      # operator's category -> category used by the library
    "Forced outage": "forced_outage",
    "Scheduled Maintenance": "scheduled_maintenance",
    "Out of Environmental Specification": "environmental",
    "Out of Electrical Specification": "electrical",
    "Requested Shutdown": "requested",
    "Technical Standby": "technical_standby",
}


def load_stops(data: Path) -> pd.DataFrame:
    s = pd.concat([pd.read_parquet(data / "processed" / f"status_{y}.parquet") for y in YEARS], ignore_index=True)
    stops = s[(s["status"] == "Stop") & s["end"].notna() & (s["end"] > s["start"])].copy()
    logging.info("%d stops kept, %d without an end or a positive duration dropped", len(stops),
                 int((s["status"] == "Stop").sum()) - len(stops))
    stops["turbine_id"] = "T" + stops["turbine"].astype(str)
    stops["category"] = stops["iec_category"].map(CATEGORY).fillna("forced_outage")
    stops["start"] = stops["start"].clip(lower=COD)
    return stops[stops["end"] > stops["start"]].reset_index(drop=True)


def availability_by_year(stops: pd.DataFrame) -> pd.DataFrame:
    """Time-based availability per turbine-year: overlapping stops attributed first-up, the rest is operation."""
    rows = []
    for tid, g in stops.groupby("turbine_id"):
        seg = intervals.first_up(g["start"], g["end"])
        seg["category"] = g["category"].to_numpy()[seg["owner"]]
        for year in YEARS:
            a, b = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), COD), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            hours = pd.Series(intervals.overlap_hours(seg["start"], seg["end"], a, b)).groupby(
                seg["category"].to_numpy()).sum()
            totals = pd.DataFrame({"category": [*hours.index, "full_performance"],
                                   "hours": [*hours.to_numpy(), (b - a).total_seconds() / 3600 - hours.sum()]})
            for contract in ("iec_operational", "oem_warranty"):
                r = wind.availability(totals, contract, allowance_h=wind.MAINTENANCE_ALLOWANCE_H[contract]
                                      * (b - a).total_seconds() / 3600 / 8760)
                rows.append({"turbine_id": tid, "year": year, "contract": contract, "time": r.time,
                             "unavailable_h": r.unavailable_h})
    return pd.DataFrame(rows)


def power_curves(data: Path, stops: pd.DataFrame, years: tuple[int, ...] = (2017, 2021)) -> pd.DataFrame:
    """Binned curves of normal operation (no stop logged in the record, positive power), density-normalised."""
    out = []
    for year in years:
        scada = pd.read_parquet(data / "processed" / f"scada_{year}.parquet").reset_index()
        scada["turbine_id"] = "T" + scada["turbine"].astype(str)
        for tid, g in scada.groupby("turbine_id"):
            st = stops[stops["turbine_id"] == tid]
            s, e = intervals.union(intervals.to_ns(st["start"]), intervals.to_ns(st["end"]))
            t0 = intervals.to_ns(g["timestamp"])
            t1 = t0 + 600 * 10**9
            k = np.searchsorted(s, t1, side="left") - 1
            stopped = (k >= 0) & (e[np.clip(k, 0, None)] > t0)
            # the SCADA "wind speed" changes definition in April 2018 (mean of both anemometers -> close to
            # sensor 2): rebuilding the mean keeps the years comparable (see wind-power-curve-kelmarsh)
            ws = g[["ws1", "ws2"]].mean(axis=1, skipna=False).fillna(g["ws"])
            rho = wind.air_density(g["temp_c"], elevation_m=ELEVATION_M)
            ws_n = wind.normalise_wind_speed(ws, rho)
            ok = ~stopped & (g["power_kw"] > 0).to_numpy() & np.isfinite(ws.to_numpy())
            curve = wind.bin_power_curve(ws_n[ok], g["power_kw"].to_numpy()[ok])
            out.append({"turbine_id": tid, "year": year, "records": int(ok.sum()),
                        "complete": wind.database_complete(curve, RATED_KW),
                        "power_at_8ms_kw": float(wind.curve_power(curve["ws"], curve["power_kw"], [8.0])[0]),
                        "aep_6_5ms_mwh": wind.aep_rayleigh(curve["ws"], curve["power_kw"], 6.5)})
    return pd.DataFrame(out)


def reliability(stops: pd.DataFrame) -> tuple[pd.DataFrame, wind.CrowAMSAA]:
    """MTBF and MTTR per turbine and the fleet's Crow-AMSAA fit; overlapping forced outages form one failure."""
    rows = []
    for tid, g in stops[stops["category"] == "forced_outage"].groupby("turbine_id"):
        s, e = intervals.union(intervals.to_ns(g["start"]), intervals.to_ns(g["end"]))
        rows.append(pd.DataFrame({"turbine_id": tid, "start": pd.to_datetime(s, utc=True),
                                  "duration_h": (e - s) / intervals.NS_PER_HOUR}))
    failures = pd.concat(rows, ignore_index=True)
    windows = pd.DataFrame({"turbine_id": sorted(stops["turbine_id"].unique()), "start": COD, "end": END})
    up = wind.operating_time(windows, stops)
    kpi = wind.mtbf_mttr(failures, up.groupby("turbine_id")["hours"].sum())
    ca = wind.crow_amsaa(failures, windows, origin=dict.fromkeys(windows["turbine_id"], COD))
    return kpi, ca


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder holding processed/ (Kelmarsh Parquet files)")
    ap.add_argument("--out", type=Path, help="write the tables as CSV here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    stops = load_stops(args.data)
    av = availability_by_year(stops)
    curves = power_curves(args.data, stops)
    kpi, ca = reliability(stops)
    print("Time availability per year (mean of the six turbines):")
    print(av.pivot_table(index="year", columns="contract", values="time").round(4), "\n")
    print("Power curves of normal operation:")
    print(curves.round(1).to_string(index=False), "\n")
    print("MTBF and MTTR per turbine (forced outages):")
    print(kpi[["turbine_id", "failures", "operating_h", "mtbf_h", "mtbf_lo_h", "mtbf_hi_h", "mttr_h",
               "ttr_median_h"]].round(2).to_string(index=False), "\n")
    print(f"Crow-AMSAA (fleet): beta {ca.beta:.3f} ({ca.beta_lo:.3f}-{ca.beta_hi:.3f}), {ca.failures} failures, "
          f"{ca.hours / 8766:.1f} turbine-years, p(beta = 1) {ca.p_trend:.2g}, verdict {ca.verdict}; "
          f"Laplace U {ca.laplace_u:.2f} (p {ca.p_laplace:.2g})")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        av.to_csv(args.out / "kelmarsh_availability.csv", index=False)
        curves.to_csv(args.out / "kelmarsh_power_curves.csv", index=False)
        kpi.to_csv(args.out / "kelmarsh_mtbf.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
