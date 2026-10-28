"""Performance ratio (PR), temperature-corrected PR and weather-corrected PR, and two cell temperature models.

Written from public descriptions of IEC 61724-1 and of Dierauf et al. (2013), NREL/TP-5200-57991; this is not a
certified implementation of the standard. With ``E`` the AC energy, ``P_STC`` the DC rating at standard test
conditions, ``G`` the plane-of-array irradiance, ``G_STC = 1000 W/m2``, ``gamma`` the power temperature
coefficient (1/K, negative) and ``T_c`` the cell temperature, summed over the intervals ``i`` used:

```text
PR     = sum E_i / sum (P_STC G_i / G_STC) dt
PR_25  = sum E_i / sum (P_STC G_i / G_STC [1 + gamma (T_c,i - 25)]) dt
PR_wc  = sum E_i / sum (P_STC G_i / G_STC [1 + gamma (T_c,i - T_typ)]) dt
```

``T_typ`` is the irradiance-weighted mean cell temperature of a typical year (e.g. the budget's TMY), so that the
annual ``PR_wc`` stays close to the annual plain PR while the monthly values no longer swing with the seasons.

Intervals with a missing power or irradiance are left out of numerator and denominator alike. When cell
temperatures are given, intervals without one are left out of all three ratios, so that the three describe the
same intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "G_STC",
    "PRTerms",
    "cell_temperature_from_module",
    "cell_temperature_pvsyst",
    "performance_ratio",
    "pr_by_period",
    "pr_terms",
    "typical_cell_temperature",
]

G_STC = 1000.0
"""Irradiance at standard test conditions, W/m2."""


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _div(a: float, b: float) -> float:
    return a / b if np.isfinite(b) and b > 0 else float("nan")


@dataclass(frozen=True)
class PRTerms:
    """Sums behind the PR family over a set of intervals (all energies in kWh).

    Attributes
    ----------
    energy_kwh : float
        Measured AC energy.
    insolation_kwh_m2 : float
        Plane-of-array irradiation ``H``.
    reference_kwh : float
        Reference energy ``P_STC H / G_STC`` (the reference yield times the DC rating).
    reference_25_kwh : float
        Reference energy corrected to 25 degC cell temperature (NaN without temperatures).
    reference_wc_kwh : float
        Reference energy corrected to ``T_typ`` (NaN without temperatures or ``T_typ``).
    intervals : int
        Number of intervals used.
    """

    energy_kwh: float
    insolation_kwh_m2: float
    reference_kwh: float
    reference_25_kwh: float
    reference_wc_kwh: float
    intervals: int

    @property
    def pr(self) -> float:
        """Plain performance ratio."""
        return _div(self.energy_kwh, self.reference_kwh)

    @property
    def pr_temperature_corrected(self) -> float:
        """Performance ratio corrected to 25 degC cell temperature."""
        return _div(self.energy_kwh, self.reference_25_kwh)

    @property
    def pr_weather_corrected(self) -> float:
        """Weather-corrected performance ratio (cell temperature corrected to ``T_typ``)."""
        return _div(self.energy_kwh, self.reference_wc_kwh)

    def as_dict(self) -> dict[str, float]:
        """Sums and the three ratios as a flat dictionary."""
        return {
            "energy_kwh": self.energy_kwh,
            "insolation_kwh_m2": self.insolation_kwh_m2,
            "reference_kwh": self.reference_kwh,
            "pr": self.pr,
            "pr_25": self.pr_temperature_corrected,
            "pr_wc": self.pr_weather_corrected,
            "intervals": float(self.intervals),
        }


def pr_terms(
    p_ac_kw: ArrayLike,
    poa_wm2: ArrayLike,
    p_stc_kw: float,
    interval_h: float,
    *,
    t_cell_c: ArrayLike | None = None,
    gamma_per_k: float | None = None,
    t_typ_c: float | None = None,
    mask: ArrayLike | None = None,
) -> PRTerms:
    """Sums for the performance ratios over the intervals selected by ``mask``.

    Parameters
    ----------
    p_ac_kw : array-like
        Mean AC power per interval, kW (at the inverters or the meter, as the contract defines).
    poa_wm2 : array-like
        Mean plane-of-array irradiance per interval, W/m2.
    p_stc_kw : float
        DC rating at STC, kWp.
    interval_h : float
        Interval length, hours.
    t_cell_c : array-like, optional
        Cell temperature per interval, degC (see
        [`cell_temperature_from_module`][res_asset_kpi.pv.performance.cell_temperature_from_module] and
        [`cell_temperature_pvsyst`][res_asset_kpi.pv.performance.cell_temperature_pvsyst]). Needed for the
        corrected ratios.
    gamma_per_k : float, optional
        Power temperature coefficient, 1/K (e.g. ``-0.0037``). Needed for the corrected ratios.
    t_typ_c : float, optional
        Irradiance-weighted typical cell temperature
        ([`typical_cell_temperature`][res_asset_kpi.pv.performance.typical_cell_temperature]); needed for
        ``PR_wc``.
    mask : array-like of bool, optional
        Intervals to use, e.g. irradiance above a threshold and no excluded event.

    Returns
    -------
    PRTerms

    Examples
    --------
    One hour at 800 W/m2 and 50 degC cell temperature on a 100 kWp array delivering 68 kW:

    >>> t = pr_terms([68.0], [800.0], 100.0, 1.0, t_cell_c=[50.0], gamma_per_k=-0.004, t_typ_c=40.0)
    >>> round(t.pr, 4), round(t.pr_temperature_corrected, 4), round(t.pr_weather_corrected, 4)
    (0.85, 0.9444, 0.8854)
    """
    p, g = _arr(p_ac_kw), _arr(poa_wm2) / G_STC
    if p.shape != g.shape:
        raise ValueError("p_ac_kw and poa_wm2 must have the same length")
    use = np.ones(len(p), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    use = use & np.isfinite(p) & np.isfinite(g)
    tc = _arr(t_cell_c) if t_cell_c is not None and gamma_per_k is not None else None
    if tc is not None:
        if tc.shape != p.shape:
            raise ValueError("t_cell_c must have the same length as p_ac_kw")
        use &= np.isfinite(tc)
    e = float(p[use].sum() * interval_h)
    h = float(g[use].sum() * interval_h)
    ref = p_stc_kw * h
    ref_25 = ref_wc = float("nan")
    if tc is not None and gamma_per_k is not None:
        gu, tu, gamma = g[use], tc[use], float(gamma_per_k)
        ref_25 = float(p_stc_kw * np.sum(gu * (1.0 + gamma * (tu - 25.0))) * interval_h)
        if t_typ_c is not None and np.isfinite(t_typ_c):
            ref_wc = float(p_stc_kw * np.sum(gu * (1.0 + gamma * (tu - t_typ_c))) * interval_h)
    return PRTerms(e, h, float(ref), ref_25, ref_wc, int(use.sum()))


def performance_ratio(
    p_ac_kw: ArrayLike, poa_wm2: ArrayLike, p_stc_kw: float, interval_h: float, mask: ArrayLike | None = None
) -> float:
    """Plain performance ratio, ``sum E / sum (P_STC G / G_STC) dt``.

    Examples
    --------
    >>> performance_ratio([40.0, 80.0], [500.0, 1000.0], 100.0, 0.25)
    0.8
    """
    return pr_terms(p_ac_kw, poa_wm2, p_stc_kw, interval_h, mask=mask).pr


def typical_cell_temperature(poa_wm2: ArrayLike, t_cell_c: ArrayLike, mask: ArrayLike | None = None) -> float:
    """Irradiance-weighted mean cell temperature ``T_typ = sum G T_c / sum G`` (e.g. over a TMY).

    Examples
    --------
    >>> typical_cell_temperature([200.0, 800.0], [20.0, 45.0])
    40.0
    """
    g, t = _arr(poa_wm2), _arr(t_cell_c)
    use = np.isfinite(g) & np.isfinite(t)
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    return _div(float(np.sum(g[use] * t[use])), float(np.sum(g[use])))


def pr_by_period(
    frame: pd.DataFrame,
    periods: ArrayLike,
    p_stc_kw: float,
    interval_h: float,
    *,
    gamma_per_k: float | None = None,
    t_typ_c: float | None = None,
    mask: ArrayLike | None = None,
) -> pd.DataFrame:
    """PR, PR_25 and PR_wc per period label.

    Parameters
    ----------
    frame : pandas.DataFrame
        Columns ``p_ac`` (kW), ``poa`` (W/m2) and optionally ``t_cell`` (degC), one row per interval.
    periods : array-like
        Period label per row (see [`period_labels`][res_asset_kpi.timeseries.period_labels]).
    p_stc_kw, interval_h, gamma_per_k, t_typ_c, mask
        As in [`pr_terms`][res_asset_kpi.pv.performance.pr_terms].

    Returns
    -------
    pandas.DataFrame
        One row per period with the columns of [`PRTerms.as_dict`][res_asset_kpi.pv.performance.PRTerms.as_dict].
    """
    labels = np.asarray(periods)
    m = np.ones(len(frame), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    rows = {}
    for key in pd.unique(labels):
        sel = labels == key
        sub = frame[sel]
        rows[key] = pr_terms(
            sub["p_ac"], sub["poa"], p_stc_kw, interval_h,
            t_cell_c=sub.get("t_cell"),
            gamma_per_k=gamma_per_k, t_typ_c=t_typ_c, mask=m[sel],
        ).as_dict()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "period"
    return out


def cell_temperature_from_module(t_mod_c: ArrayLike, poa_wm2: ArrayLike, delta_t_c: float = 3.0) -> np.ndarray:
    """Cell temperature from a back-of-module sensor: ``T_c = T_m + G / 1000 * dT`` (King et al., 2004).

    Parameters
    ----------
    t_mod_c : array-like
        Back-of-module temperature, degC.
    poa_wm2 : array-like
        Plane-of-array irradiance, W/m2.
    delta_t_c : float, default 3.0
        Cell-to-back difference at 1000 W/m2 (3 K for glass/cell/polymer sheet modules on open racks in the
        Sandia array performance model).

    Examples
    --------
    >>> cell_temperature_from_module([40.0], [500.0]).tolist()
    [41.5]
    """
    return _arr(t_mod_c) + _arr(poa_wm2) / G_STC * delta_t_c


def cell_temperature_pvsyst(
    poa_wm2: ArrayLike,
    t_amb_c: ArrayLike,
    wind_speed_ms: ArrayLike = 1.0,
    u_c: float = 29.0,
    u_v: float = 0.0,
    absorptance: float = 0.9,
    efficiency: float = 0.1,
) -> np.ndarray:
    """Cell temperature with the PVsyst heat balance: ``T_c = T_a + alpha G (1 - eta) / (U_c + U_v v)``.

    Parameters
    ----------
    poa_wm2 : array-like
        Plane-of-array irradiance, W/m2.
    t_amb_c : array-like
        Ambient temperature, degC.
    wind_speed_ms : array-like, default 1.0
        Wind speed, m/s.
    u_c, u_v : float, default 29.0 and 0.0
        Constant and wind-dependent heat loss factors, W/(m2 K) and W s/(m3 K); 29 / 0 are the PVsyst
        defaults for free-standing arrays.
    absorptance, efficiency : float, default 0.9 and 0.1
        Absorbed share of the irradiance and module efficiency.

    Examples
    --------
    >>> cell_temperature_pvsyst([1000.0], [25.0]).round(3).tolist()
    [52.931]
    """
    g, ta, v = _arr(poa_wm2), _arr(t_amb_c), _arr(wind_speed_ms)
    return ta + absorptance * g * (1.0 - efficiency) / (u_c + u_v * v)
