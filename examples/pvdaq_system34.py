"""PR, temperature-corrected and weather-corrected PR, availability and data quality of NREL PVDAQ system 34.

PVDAQ system 34 is a 146.64 kWp rooftop array in Las Vegas with one 135 kW inverter, published by NREL in the
PV Data Acquisition database (OEDI, CC BY 4.0). This script reads the 15-minute long table of 2016-2018 that the
companion project pv-budget-bridge builds from the public files (``long_34.parquet``: signals ``poa_irradiance``,
``ambient_temp_f``, ``module_temp_f``, ``wind_speed_mph`` of the met station and ``p_ac`` of the inverter, stamped
at the end of each interval in local standard time).

Usage:
    python examples/pvdaq_system34.py --data <folder with long_34.parquet> [--out <folder for CSV results>]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from res_asset_kpi import pv, quality, schema, timeseries

P_STC_KW = 146.64      # DC rating from the PVDAQ metadata
AC_KW = 135.0          # inverter rating
GAMMA = -0.0045        # power temperature coefficient of the polycrystalline modules, 1/K (typical value)
TZ = "Etc/GMT+8"       # the logger runs on local standard time all year

MAPPING = {
    "format": "long",
    "timezone": TZ,
    "label": "end",
    "step": "15min",
    "signals": {
        "poa_irradiance": {"signal": "poa", "unit": "W/m2"},
        "ambient_temp_f": {"signal": "t_amb", "unit": "degF"},
        "module_temp_f": {"signal": "t_mod", "unit": "degF"},
        "wind_speed_mph": {"signal": "wind_speed", "unit": "mph"},
    },
    "devices": {"MET": "PLANT"},
}


def load(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw = raw[raw["signal"].isin([*MAPPING["signals"], "p_ac"])]
    wide = schema.to_wide(schema.read(raw, MAPPING))
    frame = pd.DataFrame({sig: wide[(dev, sig)] for dev, sig in wide.columns})
    start = pd.Timestamp("2016-01-01", tz=TZ)  # the first hours of the file fall on the last day of 2015
    return frame[frame.index >= start]


def quality_report(frame: pd.DataFrame, step_h: float) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    six_hours = int(round(6 / step_h))
    flags = {
        "poa": quality.quality_flags(frame["poa"], "poa", stuck_run=six_hours, stuck_ignore=[0.0]),
        "t_amb": quality.quality_flags(frame["t_amb"], "t_amb", stuck_run=six_hours),
        "t_mod": quality.quality_flags(frame["t_mod"], "t_mod", stuck_run=six_hours),
        "wind_speed": quality.quality_flags(frame["wind_speed"], "wind_speed", stuck_run=six_hours,
                                            stuck_ignore=[0.0]),
        "p_ac": quality.quality_flags(frame["p_ac"], lower=-5.0, upper=1.1 * AC_KW),
    }
    table = pd.DataFrame({k: quality.summary(f) for k, f in flags.items()}).T
    table["usable_share"] = [quality.coverage(f) for f in flags.values()]
    return table, flags


def analyse(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    step_h = timeseries.interval_hours(frame.index)
    q_table, flags = quality_report(frame, step_h)
    clean = {k: quality.apply_flags(frame[k], f) for k, f in flags.items()}
    poa, p_ac = clean["poa"], clean["p_ac"].clip(lower=0.0)
    wind = clean["wind_speed"].fillna(1.0)
    t_cell = pd.Series(pv.cell_temperature_pvsyst(poa, clean["t_amb"], wind, u_c=21.0, u_v=1.35),
                       index=frame.index)
    t_typ = pv.typical_cell_temperature(poa, t_cell)
    months = timeseries.period_labels(frame.index, "M", TZ)
    years = timeseries.period_labels(frame.index, "Y", TZ)
    data = pd.DataFrame({"p_ac": p_ac, "poa": poa, "t_cell": t_cell})
    day = (poa > 50.0).to_numpy()
    pr_m = pv.pr_by_period(data, months, P_STC_KW, step_h, gamma_per_k=GAMMA, t_typ_c=t_typ, mask=day)
    pr_y = pv.pr_by_period(data, years, P_STC_KW, step_h, gamma_per_k=GAMMA, t_typ_c=t_typ, mask=day)

    # availability: the inverter counts as running when it delivers more than 2 % of its rating in daylight;
    # expected power for the energy basis from a simple temperature-corrected model, scaled to the
    # plant's median ratio of measured to modelled power in daylight
    model = P_STC_KW * poa / 1000.0 * (1.0 + GAMMA * (t_cell - 25.0))
    ratio = float(np.nanmedian((p_ac / model)[day & (poa > 400).to_numpy()]))
    expected = (model * ratio).clip(upper=AC_KW)
    running = pd.DataFrame({"INV1": np.where(p_ac.isna(), np.nan, (p_ac > 0.02 * AC_KW).astype(float))},
                           index=frame.index)
    av_m = pv.availability(running, step_h, poa_wm2=poa, expected_kw=expected.to_frame().to_numpy(),
                           periods=np.asarray(months))
    av_y = pv.availability(running, step_h, poa_wm2=poa, expected_kw=expected.to_frame().to_numpy(),
                           periods=np.asarray(years))
    cols = ["availability_time", "availability_energy", "lost_kwh"]
    monthly = pr_m[["energy_kwh", "insolation_kwh_m2", "pr", "pr_25", "pr_wc"]].join(av_m[cols])
    annual = pr_y[["energy_kwh", "insolation_kwh_m2", "pr", "pr_25", "pr_wc"]].join(av_y[cols])
    annual.attrs.update(t_typ=t_typ, model_ratio=ratio)
    return {"quality": q_table, "monthly": monthly, "annual": annual}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with long_34.parquet")
    ap.add_argument("--out", type=Path, help="write the tables as CSV here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = analyse(load(args.data / "long_34.parquet"))
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("Data quality (intervals per flag):\n", res["quality"].round(3), "\n")
    print(f"T_typ = {res['annual'].attrs['t_typ']:.1f} degC (irradiance-weighted cell temperature, 2016-2018)\n")
    print("Annual:\n", res["annual"].round(3), "\n")
    print("Monthly:\n", res["monthly"].round(3))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, table in res.items():
            table.to_csv(args.out / f"pvdaq34_{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
