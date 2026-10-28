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

The energy basis needs the expected power of each unit per interval (kW), e.g. from a performance model, after
soiling, snow and clipping, so that a stopped inverter loses what it would have produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = ["availability"]


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
        it every interval counts.
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

    Returns
    -------
    pandas.DataFrame
        Per period: ``availability_time``, ``availability_energy`` (NaN without ``expected_kw``),
        ``counted_h`` and ``stopped_h`` (capacity-weighted hours), ``excluded_share`` (of the known daylight
        hours), ``planned_h``, ``planned_counted_h``, ``expected_kwh`` and ``lost_kwh``.

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
    daylight = (
        np.ones(shape[0], dtype=bool) if poa_wm2 is None else np.nan_to_num(np.asarray(poa_wm2, float)) > daylight_wm2
    )
    known = np.isfinite(state) & daylight[:, None]
    up_state = state == 1.0
    excl = _matrix(excluded, shape, "excluded")
    pm = _matrix(planned, shape, "planned") & known
    lim = None if expected_kw is None else np.nan_to_num(np.asarray(expected_kw, dtype=float))
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
        }
        if lim is not None:
            e_exp = float(np.where(counted, lim[pos], 0.0).sum() * interval_h)
            e_lost = float(np.where(down, lim[pos], 0.0).sum() * interval_h)
            row.update(expected_kwh=e_exp, lost_kwh=e_lost,
                       availability_energy=1.0 - e_lost / e_exp if e_exp > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")
