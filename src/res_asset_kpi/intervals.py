"""Operations on half-open time intervals ``[start, end)``: union, difference, overlap and first-up attribution.

Event logs (stops, alarms, maintenance) overlap: a turbine stopped by a fault may also show a grid event and a
maintenance mode. Counting each event's own duration would count the same hour several times. The functions
here work on the time line instead. Times are passed as anything `pandas.to_datetime` accepts and are
handled internally as UTC nanoseconds.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = ["NS_PER_HOUR", "first_up", "overlap_hours", "subtract", "to_ns", "union"]

NS_PER_HOUR = 3.6e12


def to_ns(times: ArrayLike | pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """UTC datetimes (naive ones read as UTC) to int64 nanoseconds.

    Examples
    --------
    >>> to_ns(["1970-01-01 01:00"]).tolist()
    [3600000000000]
    """
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return np.asarray(idx.as_unit("ns").asi8, dtype=np.int64)


def _ts(ns: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ns, dtype="int64"), utc=True))


def union(starts: ArrayLike, ends: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Union of intervals as sorted, disjoint ``(starts, ends)`` arrays (int64 nanoseconds in and out).

    Examples
    --------
    >>> s, e = union([0, 5, 20], [10, 12, 30])
    >>> s.tolist(), e.tolist()
    ([0, 20], [12, 30])
    """
    s0, e0 = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    if len(s0) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    o = np.argsort(s0, kind="stable")
    s, e = s0[o], e0[o]
    run_end = np.maximum.accumulate(e)
    new = np.r_[True, s[1:] > run_end[:-1]]
    last = np.r_[np.flatnonzero(new)[1:] - 1, len(s) - 1]
    return s[new], run_end[last]


def subtract(a_starts: ArrayLike, a_ends: ArrayLike, b_starts: ArrayLike, b_ends: ArrayLike
             ) -> tuple[np.ndarray, np.ndarray]:
    """Intervals A minus the union of intervals B (int64 nanoseconds in and out).

    Examples
    --------
    >>> s, e = subtract([0], [100], [10, 50], [20, 60])
    >>> s.tolist(), e.tolist()
    ([0, 20, 60], [10, 50, 100])
    """
    b_s, b_e = union(b_starts, b_ends)
    out_s: list[int] = []
    out_e: list[int] = []
    for s, e in zip(np.asarray(a_starts, dtype=np.int64), np.asarray(a_ends, dtype=np.int64), strict=True):
        cur = int(s)
        k0 = int(np.searchsorted(b_e, s, side="right"))
        for k in range(k0, len(b_s)):
            if b_s[k] >= e:
                break
            if b_s[k] > cur:
                out_s.append(cur)
                out_e.append(int(b_s[k]))
            cur = max(cur, int(b_e[k]))
        if cur < e:
            out_s.append(cur)
            out_e.append(int(e))
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


def overlap_hours(starts: ArrayLike, ends: ArrayLike, window_start: object, window_end: object) -> np.ndarray:
    """Hours of each interval that fall inside ``[window_start, window_end)``.

    Examples
    --------
    >>> overlap_hours(["2024-01-01 22:00"], ["2024-01-02 03:00"], "2024-01-02", "2024-01-03").tolist()
    [3.0]
    """
    s, e = to_ns(starts), to_ns(ends)
    a, b = to_ns([window_start])[0], to_ns([window_end])[0]
    return np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None) / NS_PER_HOUR


def first_up(starts: ArrayLike, ends: ArrayLike) -> pd.DataFrame:
    """Split the union of overlapping intervals into segments owned by the earliest-starting active interval.

    Each moment covered by at least one interval belongs to the interval that started first among those still
    active (ties: the one listed first). An interval entirely inside an earlier one owns nothing; summing the
    segment hours per owner then counts every hour once.

    Parameters
    ----------
    starts, ends : array-like
        Interval bounds (datetimes).

    Returns
    -------
    pandas.DataFrame
        Columns ``start``, ``end`` (UTC), ``owner`` (position of the owning interval in the input) and
        ``hours``.

    Examples
    --------
    A fault from 10:00 to 14:00 and a grid event from 12:00 to 16:00:

    >>> seg = first_up(["2024-01-01 10:00", "2024-01-01 12:00"], ["2024-01-01 14:00", "2024-01-01 16:00"])
    >>> seg[["owner", "hours"]].to_dict("list")
    {'owner': [0, 1], 'hours': [4.0, 2.0]}
    """
    s, e = to_ns(starts), to_ns(ends)
    order = np.lexsort((np.arange(len(s)), s))
    ss, ee = s[order], e[order]
    bounds = np.unique(np.concatenate([ss, ee]))
    heap: list[int] = []
    rows: list[tuple[int, int, int]] = []
    j, n = 0, len(ss)
    for k in range(len(bounds) - 1):
        b = bounds[k]
        while j < n and ss[j] <= b:
            if ee[j] > ss[j]:
                heapq.heappush(heap, j)
            j += 1
        while heap and ee[heap[0]] <= b:
            heapq.heappop(heap)
        if not heap:
            continue
        owner = heap[0]
        if rows and rows[-1][2] == owner and rows[-1][1] == b:
            rows[-1] = (rows[-1][0], int(bounds[k + 1]), owner)
        else:
            rows.append((int(b), int(bounds[k + 1]), owner))
    a = np.array(rows, dtype=np.int64).reshape(-1, 3)
    out = pd.DataFrame({"start": _ts(a[:, 0]), "end": _ts(a[:, 1]), "owner": order[a[:, 2]]})
    out["hours"] = (a[:, 1] - a[:, 0]) / NS_PER_HOUR
    return out
