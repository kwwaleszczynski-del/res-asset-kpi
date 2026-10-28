"""Data quality flags: missing, stuck and out-of-range values, as bit flags per interval.

Flags are integers combined with ``|``, so one interval can carry several. Checks never change the data;
[`apply_flags`][res_asset_kpi.quality.apply_flags] sets the rejected values to NaN when the analyst decides so, and
[`coverage`][res_asset_kpi.quality.coverage] reports the share of usable intervals per period.

Default plausible ranges ([`RANGES`][res_asset_kpi.quality.RANGES]) are wide physical limits meant to catch unit and
scaling errors, not to judge a sensor; tighten them per site.

Examples
--------
>>> s = pd.Series([20.0, 21.0, 21.0, 21.0, 21.0, None, 400.0])
>>> f = quality_flags(s, "t_amb", stuck_run=4)
>>> [str(Flag(int(x))) for x in f]   # doctest: +NORMALIZE_WHITESPACE
['Flag.OK', 'Flag.STUCK', 'Flag.STUCK', 'Flag.STUCK', 'Flag.STUCK', 'Flag.MISSING', 'Flag.OUT_OF_RANGE']
"""

from __future__ import annotations

import enum
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "RANGES",
    "REJECT",
    "Flag",
    "apply_flags",
    "coverage",
    "flag_missing",
    "flag_out_of_range",
    "flag_stuck",
    "quality_flags",
    "summary",
]


class Flag(enum.IntFlag):
    """Data quality flags of one interval (combine with ``|``)."""

    OK = 0
    MISSING = 1
    """No value recorded."""
    STUCK = 2
    """The same value repeated over a run of intervals (frozen sensor or logger)."""
    OUT_OF_RANGE = 4
    """Outside the plausible range of the signal."""
    FILLED = 8
    """Value filled or corrected by the analyst (set by the user, never by the checks here)."""

    def __str__(self) -> str:
        return f"Flag.{self.name}" if self.name else repr(self)


REJECT = Flag.MISSING | Flag.STUCK | Flag.OUT_OF_RANGE
"""Flags that make a value unusable by default."""

RANGES: dict[str, tuple[float, float]] = {
    "poa": (-20.0, 1600.0),
    "ghi": (-20.0, 1500.0),
    "t_amb": (-50.0, 60.0),
    "t_mod": (-50.0, 100.0),
    "t_cell": (-50.0, 110.0),
    "wind_speed": (0.0, 60.0),
    "wind_dir": (0.0, 360.0),
    "nacelle_dir": (0.0, 360.0),
    "pressure": (50_000.0, 110_000.0),
    "setpoint": (0.0, 1.0),
    "soc": (0.0, 1.0),
    "pitch": (-10.0, 120.0),
}
"""Default plausible range per canonical signal, in canonical units (inclusive)."""


def _values(x: pd.Series | np.ndarray | Iterable[float]) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float), x.index
    return np.asarray(x, dtype=float), None


def _out(flags: np.ndarray, index: pd.Index | None) -> pd.Series:
    return pd.Series(flags.astype(np.uint8), index=index, name="flag")


def flag_missing(values: pd.Series | np.ndarray | Iterable[float]) -> pd.Series:
    """[`Flag.MISSING`][res_asset_kpi.quality.Flag.MISSING] where the value is NaN.

    Examples
    --------
    >>> flag_missing([1.0, float("nan")]).tolist()
    [0, 1]
    """
    v, idx = _values(values)
    return _out(np.where(np.isnan(v), Flag.MISSING, Flag.OK), idx)


def flag_stuck(
    values: pd.Series | np.ndarray | Iterable[float], min_run: int, ignore: Iterable[float] = ()
) -> pd.Series:
    """[`Flag.STUCK`][res_asset_kpi.quality.Flag.STUCK] on every value of a run of at least ``min_run`` equal values.

    Parameters
    ----------
    values : array-like
        One signal in time order.
    min_run : int
        Shortest run that counts as stuck (e.g. six hours of intervals for an ambient temperature, three
        10-minute records for an anemometer).
    ignore : iterable of float, optional
        Values whose runs are normal (e.g. ``0`` for irradiance at night or power of a stopped turbine).

    Examples
    --------
    >>> flag_stuck([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 6.0], min_run=3, ignore=[0.0]).tolist()
    [0, 0, 0, 2, 2, 2, 0]
    """
    if min_run < 2:
        raise ValueError("min_run must be at least 2")
    v, idx = _values(values)
    if len(v) == 0:
        return _out(np.zeros(0), idx)
    change = np.r_[True, v[1:] != v[:-1]]
    run_id = np.cumsum(change)
    run_len = np.bincount(run_id)[run_id]
    stuck = (run_len >= min_run) & np.isfinite(v)
    for x in ignore:
        stuck &= v != x
    return _out(np.where(stuck, Flag.STUCK, Flag.OK), idx)


def flag_out_of_range(
    values: pd.Series | np.ndarray | Iterable[float], lower: float | None = None, upper: float | None = None
) -> pd.Series:
    """[`Flag.OUT_OF_RANGE`][res_asset_kpi.quality.Flag.OUT_OF_RANGE] below ``lower`` or above ``upper``.

    Bounds are inclusive; ``None`` leaves that side open.

    Examples
    --------
    >>> flag_out_of_range([-1.0, 0.5, 2.0], 0.0, 1.0).tolist()
    [4, 0, 4]
    """
    v, idx = _values(values)
    bad = np.zeros(len(v), dtype=bool)
    with np.errstate(invalid="ignore"):
        if lower is not None:
            bad |= v < lower
        if upper is not None:
            bad |= v > upper
    return _out(np.where(bad, Flag.OUT_OF_RANGE, Flag.OK), idx)


def quality_flags(
    values: pd.Series | np.ndarray | Iterable[float],
    signal: str | None = None,
    *,
    stuck_run: int | None = None,
    stuck_ignore: Iterable[float] = (),
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
    """All checks of one signal combined.

    Parameters
    ----------
    values : array-like
        One signal in time order, canonical units.
    signal : str, optional
        Canonical signal name; selects the default range from [`RANGES`][res_asset_kpi.quality.RANGES].
    stuck_run : int, optional
        Run length for [`flag_stuck`][res_asset_kpi.quality.flag_stuck]; no stuck check without it.
    stuck_ignore : iterable of float, optional
        Passed to [`flag_stuck`][res_asset_kpi.quality.flag_stuck].
    lower, upper : float, optional
        Override the default range.

    Returns
    -------
    pandas.Series
        ``uint8`` flags.
    """
    lo, hi = RANGES.get(signal or "", (None, None))
    flags = flag_missing(values).to_numpy() | flag_out_of_range(
        values, lower if lower is not None else lo, upper if upper is not None else hi
    ).to_numpy()
    if stuck_run is not None:
        flags |= flag_stuck(values, stuck_run, stuck_ignore).to_numpy()
    _, idx = _values(values)
    return _out(flags, idx)


def apply_flags(values: pd.Series, flags: pd.Series | np.ndarray, reject: Flag = REJECT) -> pd.Series:
    """Values with every interval carrying a rejected flag set to NaN.

    Examples
    --------
    >>> apply_flags(pd.Series([1.0, 2.0]), np.array([0, Flag.STUCK])).tolist()
    [1.0, nan]
    """
    bad = (np.asarray(flags, dtype=np.int64) & int(reject)) != 0
    return values.mask(bad)


def coverage(
    flags: pd.Series | pd.DataFrame, periods: pd.Index | np.ndarray | None = None, reject: Flag = REJECT
) -> pd.Series | pd.DataFrame:
    """Share of intervals without a rejected flag, overall or per period label.

    Parameters
    ----------
    flags : pandas.Series or pandas.DataFrame
        Flags of one or several signals.
    periods : array-like, optional
        One label per interval, e.g. from [`period_labels`][res_asset_kpi.timeseries.period_labels].
    reject : Flag, default [`REJECT`][res_asset_kpi.quality.REJECT]

    Examples
    --------
    >>> coverage(pd.Series([0, 1, 0, 0]), periods=np.array(["a", "a", "b", "b"])).tolist()
    [0.5, 1.0]
    """
    ok = (flags.astype(np.int64) & int(reject)) == 0
    if periods is None:
        return ok.mean()
    return ok.groupby(np.asarray(periods)).mean()


def summary(flags: pd.Series) -> pd.Series:
    """Number of intervals carrying each flag.

    Examples
    --------
    >>> summary(pd.Series([0, 1, 3, 2])).to_dict()
    {'MISSING': 2, 'STUCK': 2, 'OUT_OF_RANGE': 0, 'FILLED': 0}
    """
    f = flags.to_numpy(dtype=np.int64)
    return pd.Series({str(fl.name): int(((f & int(fl)) != 0).sum()) for fl in Flag if fl is not Flag.OK})
