"""Contractual availability of a battery as typically defined in a long-term service agreement (LTSA).

Per interval ``i`` the measured availability ``a_i`` follows from the available power the EMS reports in each
direction ``d`` (discharge, charge) against the contracted power, and for the energy formula also from the
available energy:

```text
time:      a_i = 1 if P_avail,d / P_contract,d >= threshold in every direction, else 0
capacity:  a_i = min_d min(1, P_avail,d / P_contract,d)          (or the mean over the directions)
energy:    a_i = capacity term x min(1, E_avail / E_contract)
```

Events from the operator's log decide how an interval counts:

* ``deemed_available``: ``a_i = 1`` (e.g. an instruction of the grid operator, a grid outage, an owner request);
* ``excluded``: removed from numerator and denominator (e.g. force majeure, the capacity test itself);
* ``allowance``: excluded until the planned-maintenance allowance of the period is used up, counted afterwards;
* ``counted``: ``a_i`` as measured (forced outages; also intervals without an event).

The availability of a period is ``A = sum(a_i dt) / sum(dt)`` over the intervals not excluded. Intervals without data
and without an event are excluded, counted as available or as unavailable (``missing_data``). When events overlap, the
treatment listed first in [`PRIORITY`][res_asset_kpi.bess.availability.PRIORITY] wins. The allowance is given per year
and pro-rated to the length of the period.

Which categories exist and how each is treated is contract-specific: the example in
``res_asset_kpi/data/bess_ltsa.example.yaml`` is an illustration, not a reading of any LTSA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from res_asset_kpi import timeseries

__all__ = ["PRIORITY", "TREATMENTS", "interval_treatment", "ltsa_availability", "measured_availability"]

TREATMENTS = ("excluded", "deemed_available", "allowance", "counted")
"""Ways an event category can count (see the module description)."""
PRIORITY = TREATMENTS
"""Order in which overlapping treatments win (first wins)."""


def measured_availability(
    site: pd.DataFrame,
    *,
    power: float,
    charge_power: float | None = None,
    formula: Literal["time", "capacity", "energy"] = "time",
    threshold: float = 0.95,
    directions: Sequence[str] = ("discharge", "charge"),
    combine: Literal["min", "mean"] = "min",
    energy: float | None = None,
) -> pd.Series:
    """Measured availability ``a_i`` per interval (NaN where a needed signal is missing).

    Parameters
    ----------
    site : pandas.DataFrame
        Columns ``p_avail_dis`` and/or ``p_avail_ch`` (magnitudes), and ``e_avail`` for the energy formula.
    power : float
        Contracted discharge power.
    charge_power : float, optional
        Contracted charge power (default: ``power``).
    formula : {"time", "capacity", "energy"}, default "time"
    threshold : float, default 0.95
        Time formula: share of the contracted power needed to count as available.
    directions : sequence of {"discharge", "charge"}, default both
    combine : {"min", "mean"}, default "min"
        How the capacity formula combines the directions.
    energy : float, optional
        Contracted energy (energy formula).

    Examples
    --------
    >>> site = pd.DataFrame({"p_avail_dis": [50.0, 40.0, 50.0], "p_avail_ch": [50.0, 50.0, None]})
    >>> measured_availability(site, power=50.0).tolist()
    [1.0, 0.0, nan]
    >>> measured_availability(site, power=50.0, formula="capacity").tolist()
    [1.0, 0.8, nan]
    """
    ratios = []
    for d in directions:
        if d not in ("discharge", "charge"):
            raise ValueError(f"unknown direction {d!r}")
        col, ref = ("p_avail_dis", power) if d == "discharge" else ("p_avail_ch", charge_power or power)
        if col not in site:
            raise ValueError(f"availability in direction {d} needs the signal {col}")
        ratios.append((site[col].astype(float).abs() / ref).clip(upper=1.0))
    r = pd.concat(ratios, axis=1)
    nan = r.isna().any(axis=1)
    if formula == "time":
        out = (r >= threshold - 1e-12).all(axis=1).astype(float)
    elif formula in ("capacity", "energy"):
        out = r.min(axis=1) if combine == "min" else r.mean(axis=1)
        if formula == "energy":
            if "e_avail" not in site or energy is None:
                raise ValueError("the energy formula needs the signal e_avail and the contracted energy")
            e = (site["e_avail"].astype(float) / energy).clip(upper=1.0)
            nan |= e.isna()
            out = out * e
    else:
        raise ValueError(f"unknown formula {formula!r}")
    return out.where(~nan)


def interval_treatment(index: pd.DatetimeIndex, events: pd.DataFrame | None,
                       treatments: Mapping[str, str]) -> pd.DataFrame:
    """Event category and its treatment for every interval (by the interval's start time).

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Interval starts (UTC).
    events : pandas.DataFrame or None
        ``start``, ``end`` (time-zone aware) and ``category``.
    treatments : mapping
        Category -> one of [`TREATMENTS`][res_asset_kpi.bess.availability.TREATMENTS].

    Returns
    -------
    pandas.DataFrame
        ``category`` (missing without an event) and ``treatment`` (``counted`` without an event).
    """
    bad = sorted(set(treatments.values()) - set(TREATMENTS))
    if bad:
        raise ValueError(f"unknown treatments {bad}; use {list(TREATMENTS)}")
    cat = pd.Series(pd.NA, index=index, dtype="object")
    if events is not None and not events.empty:
        unknown = sorted(set(events["category"]) - set(treatments))
        if unknown:
            raise ValueError(f"events with categories not in the treatments: {unknown}")
        if (pd.to_datetime(events["end"]) <= pd.to_datetime(events["start"])).any():
            raise ValueError("events: end must be after start")
        rank = {t: i for i, t in enumerate(PRIORITY)}
        order = events.assign(r=events["category"].map(treatments).map(rank)).sort_values("r", ascending=False)
        for ev in order.itertuples():
            cat[(index >= ev.start) & (index < ev.end)] = ev.category
    treat = cat.map(lambda c: treatments.get(c, "counted") if isinstance(c, str) else "counted")
    return pd.DataFrame({"category": cat, "treatment": treat.astype(str)}, index=index)


def ltsa_availability(
    measured: pd.Series,
    *,
    events: pd.DataFrame | None = None,
    treatments: Mapping[str, str] | None = None,
    planned_allowance_h: float = 0.0,
    missing_data: Literal["excluded", "available", "unavailable"] = "excluded",
    periods: ArrayLike | None = None,
    guaranteed: float | None = None,
) -> pd.DataFrame:
    """Availability per period with event treatments, a planned-maintenance allowance and missing data.

    Parameters
    ----------
    measured : pandas.Series
        ``a_i`` per interval ([`measured_availability`][res_asset_kpi.bess.availability.measured_availability])
        on a regular UTC index.
    events : pandas.DataFrame, optional
        Operator's log: ``start``, ``end``, ``category``.
    treatments : mapping, optional
        Category -> treatment (see the module description).
    planned_allowance_h : float, default 0.0
        Planned maintenance hours per year that are excluded; pro-rated to each period's length.
    missing_data : {"excluded", "available", "unavailable"}, default "excluded"
        Treatment of counted intervals without a measurement.
    periods : array-like, optional
        Period label per interval (e.g. [`period_labels`][res_asset_kpi.timeseries.period_labels] or contract
        years); one period by default.
    guaranteed : float, optional
        Guaranteed availability, copied to the output for convenience.

    Returns
    -------
    pandas.DataFrame
        Per period: ``hours``, ``hours_excluded``, ``hours_deemed_available``, ``hours_planned_in_allowance``,
        ``hours_planned_counted``, ``hours_missing``, ``equivalent_unavailable_hours``, ``availability`` and
        ``guaranteed``.

    Examples
    --------
    Ten hours; a forced outage of two hours and a grid-operator instruction of one hour:

    >>> idx = pd.date_range("2024-05-01", periods=10, freq="1h", tz="UTC")
    >>> a = pd.Series([1, 1, 0, 0, 0, 1, 1, 1, 1, 1], index=idx, dtype=float)
    >>> ev = pd.DataFrame({"start": [idx[4]], "end": [idx[5]], "category": ["tso_instruction"]})
    >>> r = ltsa_availability(a, events=ev, treatments={"tso_instruction": "deemed_available"})
    >>> float(r["availability"].iloc[0])
    0.8
    """
    idx = pd.DatetimeIndex(measured.index)
    dt_h = timeseries.interval_hours(idx)
    iv = interval_treatment(idx, events, treatments or {})
    labels = np.full(len(idx), "all", dtype=object) if periods is None else np.asarray(periods)
    if len(labels) != len(idx):
        raise ValueError("periods must have one label per interval")
    meas_all = measured.to_numpy(dtype=float)
    treat_all = iv["treatment"].to_numpy(dtype=object)
    rows = []
    for key in pd.unique(labels):
        sel = labels == key
        dt = np.full(int(sel.sum()), dt_h)
        hours = float(dt.sum())
        treat, meas = treat_all[sel], meas_all[sel]
        allowance = planned_allowance_h * hours / 8760.0
        is_allow = treat == "allowance"
        used = np.cumsum(np.where(is_allow, dt, 0.0))
        allow_excluded = is_allow & (used <= allowance + 1e-9)
        allow_counted = is_allow & ~allow_excluded
        excluded = (treat == "excluded") | allow_excluded
        deemed = treat == "deemed_available"
        counted = ~excluded & ~deemed
        missing = counted & np.isnan(meas)
        value = np.where(deemed, 1.0, meas)
        if missing_data == "excluded":
            excluded = excluded | missing
        else:
            value = np.where(missing, 1.0 if missing_data == "available" else 0.0, value)
        base = ~excluded
        denom = float(dt[base].sum())
        num = float((np.nan_to_num(value[base]) * dt[base]).sum())
        rows.append({"period": key, "hours": hours, "hours_excluded": float(dt[excluded].sum()),
                     "hours_deemed_available": float(dt[deemed].sum()),
                     "hours_planned_in_allowance": float(dt[allow_excluded].sum()),
                     "hours_planned_counted": float(dt[allow_counted].sum()),
                     "hours_missing": float(dt[missing].sum()), "equivalent_unavailable_hours": denom - num,
                     "availability": num / denom if denom > 0 else np.nan,
                     "guaranteed": np.nan if guaranteed is None else guaranteed})
    return pd.DataFrame(rows).set_index("period")
