"""Equivalent full cycles, throughput and rainflow counting of the state of charge.

* **Equivalent full cycles** (EFC) on the basis a warranty names: discharged energy at the meter or at the DC
  terminals, or half the metered throughput ``(charge + discharge) / 2``, divided by the reference energy.
* **Rainflow counting** of the state of charge with the three-point method of ASTM E1049-85, the residue counted
  as half cycles, and an optional hysteresis filter against measurement noise. ``sum(depth x count)`` is the
  rainflow EFC.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "DEPTH_EDGES",
    "depth_histogram",
    "equivalent_full_cycles",
    "rainflow",
    "rainflow_efc",
    "reversals",
    "throughput",
]

ThroughputBasis = Literal["discharge_meter", "discharge_dc", "half_throughput_meter"]
DEPTH_EDGES = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001)
"""Default bin edges of the depth of discharge histogram (fractions of the reference energy)."""


def throughput(site: pd.DataFrame, interval_h: float, basis: ThroughputBasis = "discharge_meter") -> pd.Series:
    """Energy per interval that a warranty counts as throughput.

    Parameters
    ----------
    site : pandas.DataFrame
        ``p_meter`` (positive = discharge) and, for ``discharge_dc``, ``p_dc``.
    interval_h : float
        Interval length, hours.
    basis : {"discharge_meter", "discharge_dc", "half_throughput_meter"}, default "discharge_meter"

    Examples
    --------
    >>> site = pd.DataFrame({"p_meter": [-40.0, 30.0]})
    >>> throughput(site, 1.0, "half_throughput_meter").tolist()
    [20.0, 15.0]
    """
    if basis == "discharge_meter":
        return site["p_meter"].clip(lower=0) * interval_h
    if basis == "discharge_dc":
        if "p_dc" not in site:
            raise ValueError("throughput basis discharge_dc needs the signal p_dc")
        return site["p_dc"].clip(lower=0) * interval_h
    if basis == "half_throughput_meter":
        return site["p_meter"].abs() * interval_h / 2.0
    raise ValueError(f"unknown throughput basis {basis!r}")


def equivalent_full_cycles(energy: ArrayLike, reference_energy: float) -> float:
    """Equivalent full cycles: summed throughput energy / reference energy.

    Examples
    --------
    >>> equivalent_full_cycles([60.0, 90.0, 50.0], 100.0)
    2.0
    """
    return float(np.nansum(np.asarray(energy, dtype=float)) / reference_energy)


def reversals(x: ArrayLike, min_range: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Positions and values of the turning points of a signal.

    The first and last points are kept; NaNs and plateaus are dropped. With ``min_range > 0`` a turning point
    only counts once the signal has moved back by more than that (hysteresis).

    Examples
    --------
    >>> pos, val = reversals([0.5, 0.6, 0.9, 0.9, 0.2, 0.4, 0.3])
    >>> pos.tolist(), val.tolist()
    ([0, 2, 4, 5, 6], [0.5, 0.9, 0.2, 0.4, 0.3])
    """
    xa = np.asarray(x, dtype=float)
    pos = np.flatnonzero(~np.isnan(xa))
    v = xa[pos]
    if len(v) == 0:
        return pos, v
    keep = np.r_[True, np.diff(v) != 0]
    pos, v = pos[keep], v[keep]
    if len(v) < 3:
        return pos, v
    if min_range <= 0:
        d = np.sign(np.diff(v))
        idx = np.r_[0, np.flatnonzero(d[1:] != d[:-1]) + 1, len(v) - 1]
        return pos[idx], v[idx]
    out = [0]
    direction, cand, hi, lo = 0, 0, 0, 0
    for i in range(1, len(v)):
        if direction == 0:
            hi = i if v[i] > v[hi] else hi
            lo = i if v[i] < v[lo] else lo
            if v[hi] - v[0] > min_range:
                direction, cand = 1, hi
            elif v[0] - v[lo] > min_range:
                direction, cand = -1, lo
        elif direction > 0:
            if v[i] >= v[cand]:
                cand = i
            elif v[cand] - v[i] > min_range:
                out.append(cand)
                direction, cand = -1, i
        else:
            if v[i] <= v[cand]:
                cand = i
            elif v[i] - v[cand] > min_range:
                out.append(cand)
                direction, cand = 1, i
    if direction != 0:
        out.append(cand)
    if out[-1] != len(v) - 1:
        out.append(len(v) - 1)
    ix = np.asarray(out)
    return pos[ix], v[ix]


def rainflow(x: ArrayLike, min_range: float = 0.0) -> pd.DataFrame:
    """Rainflow cycles of a signal (three-point method, residue as half cycles).

    Parameters
    ----------
    x : array-like
        Signal in time order, e.g. the state of charge as a fraction.
    min_range : float, default 0.0
        Hysteresis of [`reversals`][res_asset_kpi.bess.cycles.reversals].

    Returns
    -------
    pandas.DataFrame
        One row per cycle: ``depth`` (range), ``mean``, ``count`` (1 or 0.5), ``start`` and ``end`` (positions).

    Examples
    --------
    The ASTM E1049-85 example (loads -2, 1, -3, 5, -1, 3, -4, 4, -2) gives two full cycles of range 4 and
    half cycles of ranges 3, 4, 8, 9, 8 and 6:

    >>> c = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    >>> c.groupby("depth")["count"].sum().to_dict()
    {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}
    """
    pos, v = reversals(x, min_range)
    stack_v: list[float] = []
    stack_p: list[int] = []
    rows: list[tuple[float, float, float, int, int]] = []
    for p, val in zip(pos, v, strict=True):
        stack_v.append(float(val))
        stack_p.append(int(p))
        while len(stack_v) >= 3:
            x_rng = abs(stack_v[-1] - stack_v[-2])
            y_rng = abs(stack_v[-2] - stack_v[-3])
            if x_rng < y_rng:
                break
            if len(stack_v) == 3:
                rows.append((y_rng, (stack_v[0] + stack_v[1]) / 2, 0.5, stack_p[0], stack_p[1]))
                del stack_v[0], stack_p[0]
            else:
                rows.append((y_rng, (stack_v[-2] + stack_v[-3]) / 2, 1.0, stack_p[-3], stack_p[-2]))
                del stack_v[-3:-1], stack_p[-3:-1]
    for i in range(len(stack_v) - 1):
        rows.append((abs(stack_v[i + 1] - stack_v[i]), (stack_v[i + 1] + stack_v[i]) / 2, 0.5, stack_p[i],
                     stack_p[i + 1]))
    return pd.DataFrame(rows, columns=["depth", "mean", "count", "start", "end"])


def rainflow_efc(cycles: pd.DataFrame) -> float:
    """Equivalent full cycles from rainflow cycles: ``sum(depth x count)``.

    Examples
    --------
    >>> rainflow_efc(rainflow([0.1, 0.9, 0.1, 0.9, 0.1]))
    1.6
    """
    return float((cycles["depth"] * cycles["count"]).sum())


def depth_histogram(cycles: pd.DataFrame, years: float = 1.0, edges: ArrayLike = DEPTH_EDGES) -> pd.DataFrame:
    """Cycles and their EFC per depth bin, per year.

    Examples
    --------
    >>> h = depth_histogram(rainflow([0.1, 0.9, 0.1, 0.9, 0.1]))
    >>> float(h.loc["80%-100%", "cycles"])
    2.0
    """
    e = np.asarray(edges, dtype=float)
    cut = pd.cut(cycles["depth"], e, right=False, include_lowest=True)
    h = pd.DataFrame({"cycles": cycles["count"], "efc": cycles["count"] * cycles["depth"]}) \
        .groupby(cut, observed=False).sum() / max(years, 1e-9)
    h.index = [f"{a:.0%}-{min(b, 1):.0%}" for a, b in zip(e[:-1], e[1:], strict=True)]
    return h
