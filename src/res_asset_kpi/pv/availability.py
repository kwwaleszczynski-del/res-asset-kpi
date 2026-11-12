"""Time-based and energy-based availability of a PV plant, weighted by capacity, with contractual exclusions.

Written from public descriptions of IEC 61724-1 and IEC TS 63019 and from typical O&M contract clauses; the
treatment of each event class is a parameter, because contracts differ. For units ``u`` with capacity weights
``w_u`` (summing to 1) and intervals ``i`` of length ``dt``:

```text
counted_i,u  = daylight_i  and  state known  and  not excluded  and  not planned-within-allowance
A_time       = sum w_u [running and counted] / sum w_u [counted]
A_energy     = 1 - sum E_exp,i,u [stopped and counted] / sum E_exp,i,u [counted]
```

* **daylight**: plane-of-array irradiance above a threshold (50 W/m2 by default);
* **excluded**: periods outside the operator's responsibility (grid outage, grid-operator instruction, force
  majeure), removed from numerator and denominator;
* **planned maintenance**: excluded until the capacity-weighted daylight hours of planned work in the period
  exceed an allowance, counted as downtime afterwards;
* intervals without data on a unit are left out (unknown state).

**Missing irradiance.** An interval without an irradiance value cannot be classed as day or night. By default it
is left out like an interval without state (``missing_poa="exclude"``) and its capacity-weighted hours are reported
in ``poa_missing_h``, so that a gap during a stop is visible instead of passing for night; ``"daylight"`` counts
it as daylight, the conservative reading for the operator.

The energy basis needs the expected power of each unit per interval (kW), e.g. from a performance model, after
soiling, snow and clipping, so that a stopped inverter loses what it would have produced. A counted interval
without expected power makes the energy availability of its period NaN (``expected_missing_h`` says how much is
missing) instead of counting as a loss of zero.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = ["availability"]

log = logging.getLogger(__name__)


def _matrix(x: pd.DataFrame | pd.Series | ArrayLike | None, shape: tuple[int, int], name: str) -> np.ndarray:
    """Boolean (time, unit) matrix from a frame, a per-interval series or None (all False)."""
    if x is None:
        return np.zeros(shape, dtype=bool)
    a = np.asarray(x, dtype=bool)
    if a.ndim == 1:
        if len(a) != shape[0]:
            raise ValueError(f"{name} must have one value per interval")
        return np.repeat(a[:, None], shape[1], axis=1)
    if a.shape != shape:
        raise ValueError(f"{name} must have the shape of running {shape}")
    return a


def availability(
    running: pd.DataFrame,
    interval_h: float,
    *,
    capacity: pd.Series | ArrayLike | None = None,
    poa_wm2: ArrayLike | None = None,
    daylight_wm2: float = 50.0,
    excluded: pd.DataFrame | pd.Series | ArrayLike | None = None,
    planned: pd.DataFrame | pd.Series | ArrayLike | None = None,
    planned_allowance_h: float = 0.0,
    expected_kw: pd.DataFrame | ArrayLike | None = None,
    periods: ArrayLike | None = None,
    missing_poa: Literal["exclude", "daylight"] = "exclude",
) -> pd.DataFrame:
    """Capacity-weighted time and energy availability per period.

    Parameters
    ----------
    running : pandas.DataFrame
        One column per unit (inverter), one row per interval: ``True`` running, ``False`` stopped, NaN no data.
    interval_h : float
        Interval length, hours.
    capacity : pandas.Series or array-like, optional
        Capacity of each unit (e.g. kWp DC), in the column order of ``running``; equal weights by default.
    poa_wm2 : array-like, optional
        Plane-of-array irradiance per interval; intervals at or below ``daylight_wm2`` are not counted. Without
        it every interval counts. Missing values are treated as ``missing_poa`` says.
    daylight_wm2 : float, default 50.0
        Daylight threshold, W/m2.
    excluded : DataFrame, Series or array-like of bool, optional
        Excluded intervals, per interval (whole plant) or per interval and unit.
    planned : DataFrame, Series or array-like of bool, optional
        Planned maintenance, per interval or per interval and unit.
    planned_allowance_h : float, default 0.0
        Capacity-weighted daylight hours of planned maintenance excluded per period; the rest counts.
    expected_kw : DataFrame or array-like, optional
        Expected power per interval and unit (kW) for the energy basis.
    periods : array-like, optional
        Period label per interval (see [`period_labels`][res_asset_kpi.timeseries.period_labels]); one period by
        default.
    missing_poa : {"exclude", "daylight"}, default "exclude"
        Intervals without irradiance: left out (and reported in ``poa_missing_h``) or counted as daylight.

    Returns
    -------
    pandas.DataFrame
        Per period: ``availability_time``, ``availability_energy`` (NaN without ``expected_kw``),
        ``counted_h`` and ``stopped_h`` (capacity-weighted hours), ``excluded_share`` (of the known daylight
        hours), ``planned_h``, ``planned_counted_h``, ``expected_kwh``, ``lost_kwh``, ``poa_missing_h``
        (capacity-weighted hours with a known unit state but no irradiance) and ``expected_missing_h``
        (capacity-weighted counted hours without expected power; the energy columns are NaN when it is above 0).

    Examples
    --------
    Two equal inverters over four daylight hours; inverter B is down for one hour:

    >>> running = pd.DataFrame({"A": [True] * 4, "B": [True, False, True, True]})
    >>> a = availability(running, 1.0, poa_wm2=[400.0] * 4, expected_kw=np.full((4, 2), 50.0))
    >>> a[["availability_time", "availability_energy", "lost_kwh"]].round(3)
            availability_time  availability_energy  lost_kwh
    period
    all                 0.875                0.875      50.0
    """
    state = running.to_numpy(dtype=float)
    shape = state.shape
    w = np.ones(shape[1]) if capacity is None else np.asarray(capacity, dtype=float)
    if w.shape != (shape[1],) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("capacity needs one non-negative value per unit with a positive sum")
    w = w / w.sum()
    if missing_poa not in ("exclude", "daylight"):
        raise ValueError("missing_poa must be 'exclude' or 'daylight'")
    poa = np.full(shape[0], np.inf) if poa_wm2 is None else np.asarray(poa_wm2, dtype=float)
    if poa.shape != (shape[0],):
        raise ValueError("poa_wm2 must have one value per interval")
    no_poa = np.isnan(poa)
    with np.errstate(invalid="ignore"):
        daylight = (poa > daylight_wm2) | (no_poa & (missing_poa == "daylight"))
    known = np.isfinite(state) & daylight[:, None]
    unknown_day = np.isfinite(state) & no_poa[:, None] & (missing_poa == "exclude")
    up_state = state == 1.0
    excl = _matrix(excluded, shape, "excluded")
    pm = _matrix(planned, shape, "planned") & known
    lim = None if expected_kw is None else np.asarray(expected_kw, dtype=float)
    if lim is not None and lim.shape != shape:
        raise ValueError("expected_kw must have the shape of running")
    labels = np.full(shape[0], "all", dtype=object) if periods is None else np.asarray(periods)
    if len(labels) != shape[0]:
        raise ValueError("periods must have one label per interval")
    rows = []
    for key in pd.unique(labels):
        pos = np.flatnonzero(labels == key)
        pm_h = (pm[pos].astype(float) @ w) * interval_h
        within = np.cumsum(pm_h) <= planned_allowance_h + 1e-9
        pm_ok = pm[pos] & within[:, None]
        counted = known[pos] & ~excl[pos] & ~pm_ok
        up = up_state[pos] & counted
        down = ~up_state[pos] & counted
        den = float((counted.astype(float) @ w).sum())
        num = float((up.astype(float) @ w).sum())
        known_w = float((known[pos].astype(float) @ w).sum())
        row = {
            "period": key,
            "availability_time": num / den if den > 0 else np.nan,
            "availability_energy": np.nan,
            "counted_h": den * interval_h,
            "stopped_h": float((down.astype(float) @ w).sum()) * interval_h,
            "excluded_share": 1.0 - den / known_w if known_w > 0 else np.nan,
            "planned_h": float(pm_h.sum()),
            "planned_counted_h": float(pm_h[~within].sum()),
            "expected_kwh": np.nan,
            "lost_kwh": np.nan,
            "poa_missing_h": float((unknown_day[pos].astype(float) @ w).sum()) * interval_h,
            "expected_missing_h": np.nan,
        }
        if lim is not None:
            no_exp = counted & ~np.isfinite(lim[pos])
            missing_h = float((no_exp.astype(float) @ w).sum()) * interval_h
            row["expected_missing_h"] = missing_h
            if missing_h == 0:
                e_exp = float(np.where(counted, lim[pos], 0.0).sum() * interval_h)
                e_lost = float(np.where(down, lim[pos], 0.0).sum() * interval_h)
                row.update(expected_kwh=e_exp, lost_kwh=e_lost,
                           availability_energy=1.0 - e_lost / e_exp if e_exp > 0 else np.nan)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("period")
    if out["poa_missing_h"].sum() > 0:
        log.warning("%.1f capacity-weighted hours without irradiance left out of the availability",
                    out["poa_missing_h"].sum())
    if (out["expected_missing_h"] > 0).any():
        log.warning("energy availability not computed in %d period(s): counted intervals without expected power",
                    int((out["expected_missing_h"] > 0).sum()))
    return out
