"""Time-based and energy-based availability of wind turbines from the information categories of IEC 61400-26.

Written from public descriptions of IEC 61400-26-1 (time-based) and IEC 61400-26-2 (production-based
availability); not a certified implementation. Every interval of every turbine carries one category; a
*contract* says for each category whether its time counts as available, unavailable or is excluded (removed
from numerator and denominator alike):

```text
A_time   = T_available / (T_available + T_unavailable)
A_energy = 1 - E_lost,unavailable / (E_actual + E_lost)          over the categories not excluded
```

A maintenance allowance (hours per turbine and year) moves scheduled maintenance from unavailable to excluded
until it is used up; its lost energy is excused in proportion.

The category list follows the top levels of the standard's information model; the standard's tree is finer
(e.g. "out of environmental specification" covers low wind, high wind, icing and temperature). The two contract
presets are illustrations, not readings of any agreement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "AVAILABLE",
    "CATEGORIES",
    "CONTRACTS",
    "EXCLUDED",
    "MAINTENANCE_ALLOWANCE_H",
    "UNAVAILABLE",
    "WindAvailability",
    "availability",
    "category_totals",
    "remap",
]

AVAILABLE, UNAVAILABLE, EXCLUDED = "available", "unavailable", "excluded"

CATEGORIES: dict[str, str] = {
    "full_performance": "Full performance",
    "partial_performance": "Partial performance (technical derating)",
    "technical_standby": "Technical standby",
    "environmental": "Out of environmental specification",
    "requested": "Requested shutdown or curtailment (grid operator or owner)",
    "electrical": "Out of electrical specification (grid)",
    "scheduled_maintenance": "Scheduled maintenance",
    "planned_corrective": "Planned corrective action",
    "forced_outage": "Forced outage",
    "no_info": "Information unavailable",
}
"""Information categories used here, with their descriptions."""

CONTRACTS: dict[str, dict[str, str]] = {
    "iec_operational": {
        "full_performance": AVAILABLE, "partial_performance": AVAILABLE, "technical_standby": AVAILABLE,
        "environmental": AVAILABLE, "requested": AVAILABLE, "electrical": AVAILABLE,
        "scheduled_maintenance": UNAVAILABLE, "planned_corrective": UNAVAILABLE, "forced_outage": UNAVAILABLE,
        "no_info": EXCLUDED,
    },
    "oem_warranty": {
        "full_performance": AVAILABLE, "partial_performance": AVAILABLE, "technical_standby": AVAILABLE,
        "environmental": EXCLUDED, "requested": EXCLUDED, "electrical": EXCLUDED,
        "scheduled_maintenance": UNAVAILABLE, "planned_corrective": UNAVAILABLE, "forced_outage": UNAVAILABLE,
        "no_info": AVAILABLE,
    },
}
"""Illustrative treatments: ``iec_operational`` reads the information model from the operator's side (every
operative state available, missing information excluded); ``oem_warranty`` is a typical manufacturer guarantee
(grid, environment and requested stops excused, missing information counted as available)."""

MAINTENANCE_ALLOWANCE_H: dict[str, float] = {"iec_operational": 0.0, "oem_warranty": 60.0}
"""Illustrative scheduled-maintenance allowances, hours per turbine and year."""


@dataclass(frozen=True)
class WindAvailability:
    """Availability of one group (a turbine-year, a farm-month, ...).

    Attributes
    ----------
    time, energy : float
        Time-based and energy-based availability.
    available_h, unavailable_h, excused_h : float
        Hours counted available, unavailable, and scheduled maintenance excused by the allowance.
    lost_unavailable_kwh, lost_total_kwh, potential_kwh : float
        Lost energy of unavailable categories (after the allowance), of all categories, and the potential
        (actual plus lost) of the categories not excluded.
    """

    time: float
    energy: float
    available_h: float
    unavailable_h: float
    excused_h: float
    lost_unavailable_kwh: float
    lost_total_kwh: float
    potential_kwh: float

    def as_dict(self) -> dict[str, float]:
        """Fields as a dictionary."""
        return asdict(self)


def category_totals(
    category: ArrayLike,
    interval_h: float,
    lost_kwh: ArrayLike | None = None,
    actual_kwh: ArrayLike | None = None,
    groups: ArrayLike | None = None,
    unknown: str = "forced_outage",
) -> pd.DataFrame:
    """Hours, lost and actual energy per (group, category).

    Parameters
    ----------
    category : array-like of str
        Category of every interval (any shape, e.g. time x turbine); missing values become ``unknown``.
    interval_h : float
        Interval length, hours.
    lost_kwh, actual_kwh : array-like, optional
        Lost energy (expected minus actual, during stops and derates) and actual energy per interval; negative
        actual energy (own consumption) counts as zero.
    groups : array-like, optional
        Group label of every interval (same shape), e.g. turbine-year; one group ``"all"`` by default.
    unknown : str, default "forced_outage"
        Category of intervals without one: the conservative choice counts them against the turbine.

    Returns
    -------
    pandas.DataFrame
        ``group``, ``category``, ``hours``, ``lost_kwh``, ``actual_kwh`` for every category of
        [`CATEGORIES`][res_asset_kpi.wind.availability.CATEGORIES] (zeros included).

    Examples
    --------
    >>> t = category_totals(["full_performance"] * 5 + ["forced_outage"], 1 / 6)
    >>> t.set_index("category").loc[["full_performance", "forced_outage"], "hours"].round(4).tolist()
    [0.8333, 0.1667]
    """
    cat = pd.Series(np.asarray(category, dtype=object).ravel()).fillna(unknown).astype(str)
    bad = sorted(set(cat) - set(CATEGORIES))
    if bad:
        raise ValueError(f"unknown categories {bad}; use {list(CATEGORIES)}")
    n = len(cat)
    lost = np.zeros(n) if lost_kwh is None else np.nan_to_num(np.asarray(lost_kwh, dtype=float).ravel())
    act = np.zeros(n) if actual_kwh is None else np.maximum(np.nan_to_num(np.asarray(actual_kwh, float).ravel()), 0)
    grp = np.full(n, "all", dtype=object) if groups is None else np.asarray(groups, dtype=object).ravel()
    if not (len(lost) == len(act) == len(grp) == n):
        raise ValueError("category, lost_kwh, actual_kwh and groups must have the same size")
    df = pd.DataFrame({"group": grp, "category": cat.to_numpy(), "hours": interval_h, "lost_kwh": lost,
                       "actual_kwh": act})
    full = pd.MultiIndex.from_product([pd.unique(grp), list(CATEGORIES)], names=["group", "category"])
    out = df.groupby(["group", "category"], sort=False)[["hours", "lost_kwh", "actual_kwh"]].sum()
    return out.reindex(full, fill_value=0.0).reset_index()


def availability(
    totals: pd.DataFrame,
    contract: str | Mapping[str, str] = "iec_operational",
    allowance_h: float = 0.0,
    n_turbine_years: float = 1.0,
) -> WindAvailability:
    """Time-based and energy-based availability from category totals of one group.

    Parameters
    ----------
    totals : pandas.DataFrame
        ``category``, ``hours`` and optionally ``lost_kwh`` and ``actual_kwh`` (rows of one group of
        [`category_totals`][res_asset_kpi.wind.availability.category_totals]).
    contract : str or mapping, default "iec_operational"
        Preset name from [`CONTRACTS`][res_asset_kpi.wind.availability.CONTRACTS] or a mapping category ->
        ``available`` / ``unavailable`` / ``excluded`` covering every category present.
    allowance_h : float, default 0.0
        Scheduled maintenance allowance, hours per turbine and year.
    n_turbine_years : float, default 1.0
        Turbine-years the totals cover (scales the allowance).

    Returns
    -------
    WindAvailability

    Examples
    --------
    A turbine-year with 8600 h in operation, 100 h of faults and 60 h of scheduled maintenance:

    >>> tot = pd.DataFrame({"category": ["full_performance", "forced_outage", "scheduled_maintenance"],
    ...                     "hours": [8600.0, 100.0, 60.0]})
    >>> round(availability(tot, "iec_operational").time, 4)
    0.9817
    >>> round(availability(tot, "oem_warranty", allowance_h=60.0).time, 4)
    0.9885
    """
    roles = CONTRACTS[contract] if isinstance(contract, str) else dict(contract)
    t = totals.groupby("category").sum(numeric_only=True)
    missing = sorted(set(t.index[t["hours"] > 0]) - set(roles))
    if missing:
        raise ValueError(f"the contract does not say how to treat {missing}")
    bad = sorted({r for r in roles.values()} - {AVAILABLE, UNAVAILABLE, EXCLUDED})
    if bad:
        raise ValueError(f"unknown treatments {bad}")
    role = t.index.map(lambda c: roles.get(c, EXCLUDED)).to_numpy()
    hours = t["hours"]
    lost = t["lost_kwh"] if "lost_kwh" in t else pd.Series(0.0, index=t.index)
    act = t["actual_kwh"] if "actual_kwh" in t else pd.Series(0.0, index=t.index)
    unavail_h = float(hours[role == UNAVAILABLE].sum())
    avail_h = float(hours[role == AVAILABLE].sum())
    sm = float(hours.get("scheduled_maintenance", 0.0)) if roles.get("scheduled_maintenance") == UNAVAILABLE else 0.0
    excused = min(sm, allowance_h * n_turbine_years)
    unavail_h -= excused
    lost_un = float(lost[role == UNAVAILABLE].sum())
    if sm > 0 and excused > 0:
        lost_un -= float(lost.get("scheduled_maintenance", 0.0)) * excused / sm
    keep = role != EXCLUDED
    potential = float(act[keep].sum() + lost[keep].sum())
    return WindAvailability(
        time=avail_h / (avail_h + unavail_h) if avail_h + unavail_h > 0 else float("nan"),
        energy=1.0 - lost_un / potential if potential > 0 else float("nan"),
        available_h=avail_h, unavailable_h=unavail_h, excused_h=excused, lost_unavailable_kwh=lost_un,
        lost_total_kwh=float(lost.sum()), potential_kwh=potential,
    )


def remap(category: ArrayLike, mapping: Mapping[str, str]) -> np.ndarray:
    """Re-label categories, e.g. to quantify what a misclassification does to contractual availability.

    Examples
    --------
    >>> remap(["requested", "forced_outage"], {"requested": "forced_outage"}).tolist()
    ['forced_outage', 'forced_outage']
    """
    c = np.asarray(category, dtype=object)
    out = c.copy()
    for src, dst in mapping.items():
        if dst not in CATEGORIES:
            raise ValueError(f"unknown category {dst!r}")
        out[c == src] = dst
    return out
