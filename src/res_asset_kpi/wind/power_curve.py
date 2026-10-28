"""Power curves by the method of bins, following the description of IEC 61400-12-1, and AEP for a Rayleigh wind.

Written from public descriptions of the standard; not a certified implementation. The standard measures the
wind with a met mast in front of the turbine; SCADA-based analyses use the nacelle anemometer instead, so the
absolute curve carries that anemometer's bias, while comparisons between turbines and over time remain valid.

* Air density from temperature and pressure (dry air; pressure from the ISO standard atmosphere at the site
  elevation when it is not measured): ``rho = p / (R_d T)``.
* Normalisation of the wind speed to the reference density, for pitch-regulated turbines:
  ``V_n = V (rho / rho_0)^(1/3)``.
* Bins of 0.5 m/s centred on multiples of 0.5 m/s; mean wind speed and power per bin, the bin's statistical
  (category A) uncertainty ``s / sqrt(n)``; bins with fewer than three 10-minute records are dropped.
* AEP for a Rayleigh distribution of mean ``V_ave``:
  ``AEP = N_h sum (F(V_i) - F(V_(i-1))) (P_(i-1) + P_i) / 2`` with
  ``F(V) = 1 - exp(-pi / 4 (V / V_ave)^2)``, the curve held at its last value up to cut-out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "BIN_WIDTH",
    "RHO_0",
    "aep_rayleigh",
    "air_density",
    "bin_power_curve",
    "curve_power",
    "database_complete",
    "normalise_wind_speed",
    "standard_pressure",
]

RHO_0 = 1.225
"""Reference air density, kg/m3."""
R_DRY = 287.05
"""Gas constant of dry air, J/(kg K)."""
BIN_WIDTH = 0.5
"""Width of the wind speed bins, m/s."""
HOURS_PER_YEAR = 8760.0


def standard_pressure(elevation_m: ArrayLike) -> np.ndarray:
    """Air pressure of the ISO standard atmosphere at a height above sea level, Pa.

    Examples
    --------
    >>> round(float(standard_pressure(0.0)), 1), round(float(standard_pressure(500.0)), 0)
    (101325.0, 95461.0)
    """
    return 101325.0 * (1.0 - 2.25577e-5 * np.asarray(elevation_m, dtype=float)) ** 5.25588


def air_density(
    temp_c: ArrayLike, pressure_pa: ArrayLike | None = None, elevation_m: ArrayLike | None = None
) -> np.ndarray:
    """Dry-air density, kg/m3, from temperature and pressure (or the standard pressure at an elevation).

    Humidity, which the standard also corrects for, changes the density by well under 1 % at temperate
    temperatures and is ignored.

    Parameters
    ----------
    temp_c : array-like
        Air temperature, degC.
    pressure_pa : array-like, optional
        Air pressure, Pa.
    elevation_m : array-like, optional
        Height above sea level, used when the pressure is not given.

    Examples
    --------
    >>> round(float(air_density(15.0, 101325.0)), 4)
    1.225
    """
    if pressure_pa is None:
        if elevation_m is None:
            raise ValueError("give pressure_pa or elevation_m")
        pressure_pa = standard_pressure(elevation_m)
    return np.asarray(pressure_pa, dtype=float) / (R_DRY * (np.asarray(temp_c, dtype=float) + 273.15))


def normalise_wind_speed(wind_speed_ms: ArrayLike, rho: ArrayLike, rho_ref: float = RHO_0) -> np.ndarray:
    """Wind speed normalised to the reference density, ``V (rho / rho_ref)^(1/3)``.

    Examples
    --------
    >>> round(float(normalise_wind_speed(10.0, 1.15)), 4)
    9.7916
    """
    return np.asarray(wind_speed_ms, dtype=float) * (np.asarray(rho, dtype=float) / rho_ref) ** (1.0 / 3.0)


def bin_power_curve(
    wind_speed_ms: ArrayLike, power_kw: ArrayLike, bin_width: float = BIN_WIDTH, min_count: int = 3
) -> pd.DataFrame:
    """Power curve by the method of bins.

    Parameters
    ----------
    wind_speed_ms : array-like
        Normalised 10-minute mean wind speeds, m/s (normal operation only).
    power_kw : array-like
        10-minute mean power, kW.
    bin_width : float, default 0.5
        Bin width; bins are centred on its multiples.
    min_count : int, default 3
        Bins with fewer records are dropped (three 10-minute records, the minimum of the standard).

    Returns
    -------
    pandas.DataFrame
        Index: bin centre; columns ``ws`` (mean wind speed), ``power_kw`` (mean power), ``std_kw``, ``n`` and
        ``u_a_kw`` (statistical uncertainty of the bin mean, ``std / sqrt(n)``).

    Examples
    --------
    >>> c = bin_power_curve([4.9, 5.0, 5.1, 5.3, 7.0], [100.0, 110.0, 120.0, 130.0, 400.0])
    >>> c.index.tolist(), c["power_kw"].tolist(), c["n"].tolist()
    ([5.0], [110.0], [3])
    """
    df = pd.DataFrame({"ws": np.asarray(wind_speed_ms, dtype=float),
                       "power_kw": np.asarray(power_kw, dtype=float)}).dropna()
    df["bin"] = np.floor(df["ws"] / bin_width + 0.5) * bin_width
    g = df.groupby("bin")
    out = pd.DataFrame({"ws": g["ws"].mean(), "power_kw": g["power_kw"].mean(),
                        "std_kw": g["power_kw"].std(ddof=1), "n": g.size()})
    out["u_a_kw"] = out["std_kw"] / np.sqrt(out["n"])
    return out[out["n"] >= min_count]


def database_complete(curve: pd.DataFrame, rated_kw: float, cut_in_ms: float = 3.0, step_h: float = 1 / 6,
                      bin_width: float = BIN_WIDTH) -> bool:
    """Completeness of a binned data set in the sense of the standard.

    Every bin from 1 m/s below cut-in to 1.5 times the wind speed at 85 % of rated power is present, and the
    data cover at least 180 hours.

    Parameters
    ----------
    curve : pandas.DataFrame
        Output of [`bin_power_curve`][res_asset_kpi.wind.power_curve.bin_power_curve].
    rated_kw : float
        Rated power.
    cut_in_ms : float, default 3.0
        Cut-in wind speed.
    step_h : float, default 1/6
        Length of one record, hours.
    bin_width : float, default 0.5
        Bin width used for the curve.

    Examples
    --------
    >>> ws = np.repeat(np.arange(2.0, 14.01, 0.5), 60)
    >>> curve = bin_power_curve(ws, np.clip((ws - 3.0) * 300.0, 0.0, 2000.0))
    >>> database_complete(curve, rated_kw=2000.0)
    True
    """
    above = curve.index[curve["power_kw"] >= 0.85 * rated_kw]
    if len(above) == 0:
        return False
    needed = np.arange(cut_in_ms - 1.0, 1.5 * curve.loc[above[0], "ws"] + 1e-9, bin_width)
    needed = np.round(needed / bin_width) * bin_width
    return bool(set(needed) <= set(curve.index)) and float(curve["n"].sum()) * step_h >= 180.0


def curve_power(curve_ws: ArrayLike, curve_kw: ArrayLike, wind_speed_ms: ArrayLike, cut_out_ms: float = 25.0
                ) -> np.ndarray:
    """Power from a curve at any wind speed: linear in between, last value held up to cut-out, 0 outside.

    Examples
    --------
    >>> curve_power([3.0, 4.0, 5.0], [0.0, 100.0, 300.0], [2.0, 4.5, 20.0, 26.0]).tolist()
    [0.0, 200.0, 300.0, 0.0]
    """
    xs, ys = np.asarray(curve_ws, dtype=float), np.asarray(curve_kw, dtype=float)
    ws = np.asarray(wind_speed_ms, dtype=float)
    p = np.interp(ws, xs, ys, left=0.0, right=ys[-1])
    p[(ws > cut_out_ms) | (ws < xs[0])] = 0.0
    p[np.isnan(ws)] = np.nan
    return p


def aep_rayleigh(curve_ws: ArrayLike, curve_kw: ArrayLike, mean_ws: float, cut_out_ms: float = 25.0,
                 step: float = 0.1, availability: float = 1.0) -> float:
    """Annual energy production for a Rayleigh wind speed distribution, MWh.

    Parameters
    ----------
    curve_ws, curve_kw : array-like
        Power curve (m/s, kW), e.g. the ``ws`` and ``power_kw`` columns of
        [`bin_power_curve`][res_asset_kpi.wind.power_curve.bin_power_curve].
    mean_ws : float
        Annual mean wind speed at hub height, m/s.
    cut_out_ms : float, default 25.0
        The curve is extended with its last value up to cut-out ("AEP-extrapolated").
    step : float, default 0.1
        Integration step, m/s.
    availability : float, default 1.0
        Multiplier for an assumed availability.

    Examples
    --------
    A flat 1000 kW curve produces 8760 MWh less the hours above cut-out (exp(-pi/4 (25/7)^2) of the year):

    >>> round(aep_rayleigh([0.0, 30.0], [1000.0, 1000.0], 7.0), 1)
    8759.6
    """
    v = np.arange(0.0, cut_out_ms + step / 2.0, step)
    p = curve_power(curve_ws, curve_kw, v, cut_out_ms)
    cdf = 1.0 - np.exp(-np.pi / 4.0 * (v / mean_ws) ** 2)
    return float(availability * HOURS_PER_YEAR * np.sum(np.diff(cdf) * (p[1:] + p[:-1]) / 2.0) / 1000.0)
