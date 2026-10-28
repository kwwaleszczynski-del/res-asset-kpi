"""Time stamps, time zones, regular grids and reporting periods.

Conventions used throughout the package:

* Every series is indexed by **UTC** time stamps that mark the **start** of the averaging interval.
* Reporting periods (days, months, years, contract years) are taken in **local time**, by default
  ``Europe/Warsaw``. A local day then has 23 or 25 hours on the days of the daylight saving time (DST) changes,
  March has 743 hours and October 745.
* Supported interval lengths are 5, 10, 15 and 30 minutes and 1 hour. A series can be put on a coarser grid
  (averaged or summed) or a finer one (spread) when one length divides the other.

Naive local time stamps are ambiguous once a year: in autumn the hour 02:00-03:00 occurs twice. Data loggers
export it twice, in time order, so within each device and signal the first occurrence is read as summer time
and the second as winter time. A repeated hour seen only once, a third occurrence and the skipped spring hour
cannot be placed; they become ``NaT`` and are reported in the log.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "SUPPORTED_STEPS",
    "check_step",
    "contract_years",
    "infer_step",
    "interval_hours",
    "period_bounds",
    "period_labels",
    "regularise",
    "to_utc",
]

log = logging.getLogger(__name__)

SUPPORTED_STEPS: tuple[pd.Timedelta, ...] = tuple(
    pd.Timedelta(s) for s in ("5min", "10min", "15min", "30min", "1h")
)
"""Supported interval lengths: 5, 10, 15 and 30 minutes and 1 hour."""
Label = Literal["start", "end", "center"]
Aggregation = Literal["mean", "sum", "instant", "circular"]
Freq = Literal["D", "M", "Q", "Y"]
_LABEL_SHIFT = {"start": 0.0, "end": 1.0, "center": 0.5}
_OFFSET = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def check_step(step: str | pd.Timedelta) -> pd.Timedelta:
    """Validate an interval length.

    Parameters
    ----------
    step : str or pandas.Timedelta
        For example ``"15min"`` or ``"1h"``.

    Returns
    -------
    pandas.Timedelta

    Raises
    ------
    ValueError
        If the length is not one of [`SUPPORTED_STEPS`][res_asset_kpi.timeseries.SUPPORTED_STEPS].

    Examples
    --------
    >>> check_step("15min")
    Timedelta('0 days 00:15:00')
    """
    td = pd.Timedelta(step)
    if td not in SUPPORTED_STEPS:
        raise ValueError(f"interval {step!r} not supported; use one of 5min, 10min, 15min, 30min, 1h")
    return td


def infer_step(timestamps: pd.DatetimeIndex | pd.Series | Sequence[pd.Timestamp]) -> pd.Timedelta:
    """Most common positive spacing of the time stamps, rounded to the minute (median of the differences).

    Examples
    --------
    >>> idx = pd.date_range("2024-01-01", periods=5, freq="10min", tz="UTC")
    >>> infer_step(idx)
    Timedelta('0 days 00:10:00')
    """
    t = pd.Series(pd.DatetimeIndex(timestamps)).sort_values()
    d = t.diff().dropna()
    d = d[d > pd.Timedelta(0)]
    if d.empty:
        raise ValueError("cannot infer the interval from fewer than two distinct time stamps")
    return pd.Timedelta(d.median()).round("1min")


def interval_hours(index: pd.DatetimeIndex) -> float:
    """Length of one interval of a regular index, in hours.

    Examples
    --------
    >>> interval_hours(pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"))
    0.25
    """
    if index.freq is not None:
        return float(pd.Timedelta(index.freq) / pd.Timedelta("1h"))
    return float(infer_step(index) / pd.Timedelta("1h"))


def _parse(text: pd.Series, fmt: str | None) -> pd.Series:
    if fmt is not None:
        return pd.to_datetime(text, format=fmt)
    try:
        return pd.to_datetime(text, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(text, format="mixed")


def to_utc(
    timestamps: pd.Series | pd.DatetimeIndex | Sequence[str],
    tz: str = "Europe/Warsaw",
    groups: pd.Series | pd.DataFrame | None = None,
    fmt: str | None = None,
) -> pd.Series:
    """Parse time stamps and convert them to UTC.

    Parameters
    ----------
    timestamps : pandas.Series, pandas.DatetimeIndex or sequence of str
        Strings or datetimes. Strings that all carry a UTC offset (``2024-10-27T02:30:00+02:00``) and
        time-zone-aware datetimes are exact and ``tz`` is ignored.
    tz : str, default "Europe/Warsaw"
        IANA time zone of naive time stamps; ``"UTC"`` for naive UTC.
    groups : pandas.Series or pandas.DataFrame, optional
        Keys (e.g. device and signal) within which the repeated autumn hour is resolved by order of
        appearance. Without it the whole input is one group.
    fmt : str, optional
        ``strftime`` pattern of the strings, e.g. ``"%d.%m.%Y %H:%M"``; ISO 8601 by default.

    Returns
    -------
    pandas.Series
        ``datetime64[ns, UTC]`` with the index of the input; time stamps that cannot be placed are ``NaT``.

    Raises
    ------
    ValueError
        If only some strings carry a UTC offset.

    Examples
    --------
    The autumn change in Poland: 02:30 local time occurs twice.

    >>> t = to_utc(["2024-10-27 01:30", "2024-10-27 02:30", "2024-10-27 02:30", "2024-10-27 03:30"])
    >>> [x.strftime("%H:%M") for x in t]
    ['23:30', '00:30', '01:30', '02:30']
    """
    s = timestamps.copy() if isinstance(timestamps, pd.Series) else pd.Series(timestamps)
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_convert("UTC").dt.as_unit("ns")
        parsed = s
    else:
        text = s.astype(str).str.strip()
        has_offset = text.str.contains(_OFFSET)
        if has_offset.all():
            return pd.to_datetime(text, utc=True, format=fmt or "ISO8601").dt.as_unit("ns")
        if has_offset.any():
            raise ValueError("some time stamps carry a UTC offset and some do not; cannot read them together")
        parsed = _parse(text, fmt)
    parsed = parsed.dt.as_unit("ns")
    if str(tz).upper() == "UTC":
        return parsed.dt.tz_localize("UTC")
    keys = pd.DataFrame({"__t": parsed.to_numpy()})
    if groups is not None:
        g = groups.to_frame() if isinstance(groups, pd.Series) else groups
        keys = pd.concat([keys, g.reset_index(drop=True).set_axis(keys.index)], axis=1)
    by = list(keys.columns)
    occurrence = keys.groupby(by, sort=False, dropna=False).cumcount().to_numpy()
    size = keys.groupby(by, sort=False, dropna=False)["__t"].transform("size").to_numpy()
    n = len(parsed)
    as_dst = parsed.dt.tz_localize(tz, ambiguous=np.ones(n, dtype=bool), nonexistent="NaT")
    as_std = parsed.dt.tz_localize(tz, ambiguous=np.zeros(n, dtype=bool), nonexistent="NaT")
    ambiguous = (as_dst != as_std).to_numpy() & as_dst.notna().to_numpy()
    local = as_dst.where(~ambiguous | (occurrence == 0), as_std)
    unplaceable = ambiguous & ((size < 2) | (occurrence >= 2))
    local = local.where(~unplaceable)
    n_gap = int((as_dst.isna() & parsed.notna()).sum())
    n_single = int((ambiguous & (size < 2)).sum())
    n_extra = int((ambiguous & (size >= 2) & (occurrence >= 2)).sum())
    if n_gap or n_single or n_extra:
        log.warning(
            "time stamps not placed in %s: %d in the skipped spring hour, %d repeated autumn hour seen once, "
            "%d seen more than twice",
            tz,
            n_gap,
            n_single,
            n_extra,
        )
    out = local.dt.tz_convert("UTC")
    out.index = s.index
    return out


def _aggregate(s: pd.Series, how: Aggregation, keys: object) -> pd.Series:
    """Aggregate a series by group keys (index level or resampler) with the rule of the signal."""
    if how == "circular":
        rad = np.deg2rad(s)
        sin = np.sin(rad).groupby(keys).mean()
        cos = np.cos(rad).groupby(keys).mean()
        return pd.Series(np.rad2deg(np.arctan2(sin, cos)) % 360.0, index=sin.index)
    g = s.groupby(keys)
    return g.sum(min_count=1) if how == "sum" else g.mean()


def regularise(
    values: pd.Series,
    step: str | pd.Timedelta,
    *,
    source_step: str | pd.Timedelta | None = None,
    label: Label = "start",
    how: Aggregation = "mean",
) -> pd.Series:
    """Put one signal on a regular UTC grid whose time stamps mark the start of each interval.

    Parameters
    ----------
    values : pandas.Series
        Values indexed by time-zone-aware time stamps (any order; duplicates are averaged or summed).
    step : str or pandas.Timedelta
        Interval length of the output grid (see [`SUPPORTED_STEPS`][res_asset_kpi.timeseries.SUPPORTED_STEPS]).
    source_step : str or pandas.Timedelta, optional
        Interval length of the input; inferred from the time stamps by default.
    label : {"start", "end", "center"}, default "start"
        Which instant of its averaging interval an input time stamp marks.
    how : {"mean", "sum", "instant", "circular"}, default "mean"
        ``mean`` for interval averages (power, irradiance, temperature), ``sum`` for amounts per interval
        (energy, precipitation; spread evenly when the grid is finer), ``instant`` for states read at an
        instant (state of charge, stored energy; not shifted by ``label``, interpolated linearly onto a finer
        grid), ``circular`` for directions in degrees.

    Returns
    -------
    pandas.Series
        Values on the complete grid from the first to the last interval; missing intervals are NaN.

    Raises
    ------
    ValueError
        If the index is naive, a step is not supported, or neither step divides the other.

    Examples
    --------
    Three 5-minute means labelled by the end of their interval become one 15-minute mean:

    >>> idx = pd.to_datetime(["2024-06-01 10:05", "2024-06-01 10:10", "2024-06-01 10:15"]).tz_localize("UTC")
    >>> regularise(pd.Series([1.0, 2.0, 6.0], index=idx), "15min", label="end")
    2024-06-01 10:00:00+00:00    3.0
    Freq: 15min, dtype: float64
    """
    target = check_step(step)
    idx = pd.DatetimeIndex(values.index)
    if idx.tz is None:
        raise ValueError("the index must be time-zone aware; convert it with to_utc()")
    idx = idx.tz_convert("UTC").as_unit("ns")
    src = check_step(source_step) if source_step is not None else check_step(infer_step(idx))
    if (target % src) and (src % target):
        raise ValueError(f"cannot put {src} intervals on a {target} grid: neither divides the other")
    if how != "instant":
        idx = idx - _LABEL_SHIFT[label] * src
    s = pd.Series(np.asarray(values, dtype=float), index=idx.round(src))
    s = _aggregate(s, how, s.index.rename(None)).sort_index()
    s.index = pd.DatetimeIndex(s.index, tz="UTC")
    s = s.dropna() if how == "instant" else s
    if s.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if src < target:
        if how == "instant":
            grid = pd.date_range(s.index.min().ceil(target), s.index.max(), freq=target)
            s = s.reindex(grid)
        else:
            s = _aggregate(s, how, s.index.floor(target))
    elif src > target:
        n = int(round(src / target))
        if how == "instant":  # a state is known at its instants only, not until the end of an interval
            fine = pd.date_range(s.index.min(), s.index.max(), freq=target)
            s = s.reindex(s.index.union(fine)).interpolate(method="time", limit_area="inside").reindex(fine)
        else:
            fine = pd.date_range(s.index.min(), s.index.max() + src - target, freq=target)
            s = s.reindex(fine).ffill(limit=n - 1)
            if how == "sum":
                s = s / n
    grid = pd.date_range(s.index.min().floor(target), s.index.max(), freq=target)
    return s.reindex(grid)


def period_labels(index: pd.DatetimeIndex, freq: Freq = "M", tz: str = "Europe/Warsaw") -> pd.Index:
    """Label of the local-time period each time stamp falls in.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Time-zone-aware time stamps (interval starts).
    freq : {"D", "M", "Q", "Y"}, default "M"
        Local day (``2024-03-31``), month (``2024-03``), quarter (``2024-Q1``) or year (``2024``).
    tz : str, default "Europe/Warsaw"
        Time zone in which the periods are counted.

    Returns
    -------
    pandas.Index
        One string label per time stamp.

    Examples
    --------
    23:30 UTC on 31 March is already 1 April in Warsaw:

    >>> idx = pd.DatetimeIndex(["2024-03-31 21:00", "2024-03-31 23:30"], tz="UTC")
    >>> list(period_labels(idx, "M"))
    ['2024-03', '2024-04']
    """
    if index.tz is None:
        raise ValueError("the index must be time-zone aware")
    loc = index.tz_convert(tz)
    y, m, d = loc.year.to_numpy(), loc.month.to_numpy(), loc.day.to_numpy()
    if freq == "D":
        code = y * 10000 + m * 100 + d
        fmt = "{0:04d}-{1:02d}-{2:02d}"
    elif freq == "M":
        code = y * 100 + m
        fmt = "{0:04d}-{1:02d}"
    elif freq == "Q":
        code = y * 10 + (m - 1) // 3 + 1
        fmt = "{0:04d}-Q{3}"
    elif freq == "Y":
        code = y
        fmt = "{0:04d}"
    else:
        raise ValueError(f"unknown period {freq!r}; use D, M, Q or Y")
    uniq, inv = np.unique(code, return_inverse=True)
    names = []
    for c in uniq:
        c = int(c)
        if freq == "D":
            parts = (c // 10000, c // 100 % 100, c % 100, 0)
        elif freq == "M":
            parts = (c // 100, c % 100, 0, 0)
        elif freq == "Q":
            parts = (c // 10, 0, 0, c % 10)
        else:
            parts = (c, 0, 0, 0)
        names.append(fmt.format(*parts))
    return pd.Index(np.asarray(names, dtype=object)[inv])


def period_bounds(
    start: pd.Timestamp | str, end: pd.Timestamp | str, freq: Freq = "M", tz: str = "Europe/Warsaw"
) -> pd.DataFrame:
    """Local-time periods covering ``[start, end)`` with their UTC bounds and length in hours.

    Parameters
    ----------
    start, end : pandas.Timestamp or str
        Time-zone-aware bounds (strings with an offset, or naive strings read as UTC).
    freq : {"D", "M", "Q", "Y"}, default "M"
    tz : str, default "Europe/Warsaw"

    Returns
    -------
    pandas.DataFrame
        Columns ``period``, ``start`` and ``end`` (UTC) and ``hours``.

    Examples
    --------
    >>> p = period_bounds("2024-03-01 00:00+01:00", "2024-04-01 00:00+02:00")
    >>> p[["period", "hours"]]
        period  hours
    0  2024-03  743.0
    """
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    t1 = t1.tz_localize("UTC") if t1.tzinfo is None else t1
    l0, l1 = t0.tz_convert(tz).tz_localize(None), t1.tz_convert(tz).tz_localize(None)
    offset = {"D": "D", "M": "MS", "Q": "QS", "Y": "YS"}[freq]
    first = {"D": l0.normalize(), "M": l0.to_period("M").to_timestamp(), "Q": l0.to_period("Q").to_timestamp(),
             "Y": l0.to_period("Y").to_timestamp()}[freq]
    starts = pd.date_range(first, l1 - pd.Timedelta("1ns"), freq=offset)
    starts = starts.append(pd.DatetimeIndex([starts[-1] + pd.tseries.frequencies.to_offset(offset)]))
    utc = starts.tz_localize(tz, ambiguous=np.zeros(len(starts), dtype=bool), nonexistent="shift_forward")
    utc = utc.tz_convert("UTC")
    labels = period_labels(utc[:-1], freq, tz)
    out = pd.DataFrame({"period": list(labels), "start": utc[:-1], "end": utc[1:]})
    out["hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600.0
    return out


def contract_years(index: pd.DatetimeIndex, start: pd.Timestamp | str, tz: str = "Europe/Warsaw") -> np.ndarray:
    """1-based contract year of each time stamp; year 1 starts at local midnight of ``start``.

    Years run from anniversary to anniversary of the start date (calendar years, 365 or 366 days). Time stamps
    before the start get 0.

    Examples
    --------
    >>> idx = pd.DatetimeIndex(["2024-06-30 21:59", "2024-06-30 22:00", "2026-01-01 00:00"], tz="UTC")
    >>> contract_years(idx, "2023-07-01").tolist()
    [1, 2, 3]
    """
    loc = index.tz_convert(tz)
    s = pd.Timestamp(start)
    s = s.tz_localize(tz) if s.tz is None else s.tz_convert(tz)
    years = np.zeros(len(loc), dtype=int)
    if len(loc) == 0:
        return years
    k = 0
    while True:
        lo = s + pd.DateOffset(years=k)
        if lo > loc.max():
            break
        hi = s + pd.DateOffset(years=k + 1)
        years[(loc >= lo) & (loc < hi)] = k + 1
        k += 1
    return years
