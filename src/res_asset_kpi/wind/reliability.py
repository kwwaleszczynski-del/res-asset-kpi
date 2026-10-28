"""Reliability indicators from failure records: MTBF and MTTR with confidence bounds, Crow-AMSAA and Laplace.

Definitions:

* **Operating time** - observation time minus all stop time (a failure cannot start while the unit is stopped),
  see [`operating_time`][res_asset_kpi.wind.reliability.operating_time].
* **MTBF** - operating hours / failures, with the two-sided chi-square interval of a time-truncated test:

      MTBF_lo = 2 T / chi2(1 - a/2; 2n + 2)        MTBF_hi = 2 T / chi2(a/2; 2n)

* **MTTR** - mean time to *restore*: from the stop to the restart, so it includes waiting, travel and logistics,
  not only hands-on repair time.
* **Crow-AMSAA** - the power-law non-homogeneous Poisson process with intensity
  ``u(t) = lambda beta t^(beta - 1)`` in unit age ``t``, fitted by maximum likelihood over all units and all
  their observation windows ``(S_j, T_j]`` (late starts and gaps handled exactly):

      ln L = n ln(lambda) + n ln(beta) + (beta - 1) sum ln t_i - lambda sum (T_j^beta - S_j^beta)

  ``beta < 1``: failures thin out (reliability grows), ``beta > 1``: they become more frequent. The verdict
  comes from the likelihood-ratio test of ``beta = 1``; the bounds of ``beta`` from the profile likelihood.
* **Laplace test** - model-free trend test: under a constant rate every failure time is uniform in its window, ``U = sum
  (t_i - (S_i + L_i / 2)) / sqrt(sum L_i^2 / 12)`` is standard normal; ``U > 0`` means failures cluster late.
  [`laplace_observed`][res_asset_kpi.wind.reliability.laplace_observed] applies it on each unit's observed-time clock
  (its windows joined end to end), so that it does not depend on the choice of the age origin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import optimize, stats

from res_asset_kpi.intervals import NS_PER_HOUR, subtract, to_ns, union

__all__ = [
    "HOURS_PER_YEAR",
    "CrowAMSAA",
    "crow_amsaa",
    "laplace_observed",
    "laplace_test",
    "mtbf_interval",
    "mtbf_mttr",
    "operating_time",
]

HOURS_PER_YEAR = 8766.0
"""Hours in an average year (365.25 days)."""


def operating_time(windows: pd.DataFrame, stops: pd.DataFrame, unit_col: str = "turbine_id") -> pd.DataFrame:
    """Operating intervals per unit: observation windows minus all stops.

    Parameters
    ----------
    windows : pandas.DataFrame
        Observation windows: ``unit_col``, ``start``, ``end`` (periods with valid logging).
    stops : pandas.DataFrame
        Every stop (failures and others): ``unit_col``, ``start``, ``end``.
    unit_col : str, default "turbine_id"

    Returns
    -------
    pandas.DataFrame
        ``unit_col``, ``start``, ``end``, ``hours`` and ``window`` (position of the observation window within
        the unit).

    Examples
    --------
    >>> w = pd.DataFrame({"turbine_id": ["T1"], "start": ["2024-01-01"], "end": ["2024-01-11"]})
    >>> s = pd.DataFrame({"turbine_id": ["T1"], "start": ["2024-01-02"], "end": ["2024-01-03"]})
    >>> float(operating_time(w, s)["hours"].sum())
    216.0
    """
    parts = []
    for uid, w in windows.groupby(unit_col, sort=True):
        e = stops[stops[unit_col] == uid]
        ws = np.sort(to_ns(w["start"]))
        s, t = subtract(*union(to_ns(w["start"]), to_ns(w["end"])), to_ns(e["start"]), to_ns(e["end"]))
        parts.append(pd.DataFrame({unit_col: uid, "a": s, "b": t, "window": np.searchsorted(ws, s, "right") - 1}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[unit_col, "a", "b", "window"])
    out["start"] = pd.to_datetime(out.pop("a").to_numpy(dtype="int64"), utc=True)
    out["end"] = pd.to_datetime(out.pop("b").to_numpy(dtype="int64"), utc=True)
    out["hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600.0
    return out[[unit_col, "start", "end", "hours", "window"]]


def mtbf_interval(hours: float, failures: int, conf: float = 0.9) -> tuple[float, float]:
    """Two-sided chi-square confidence interval of the MTBF for a time-truncated observation.

    Parameters
    ----------
    hours : float
        Operating hours.
    failures : int
        Number of failures.
    conf : float, default 0.9
        Confidence level.

    Returns
    -------
    tuple of float
        Lower and upper bound, hours (upper is infinite without failures).

    Examples
    --------
    >>> lo, hi = mtbf_interval(1000.0, 4)
    >>> round(lo, 1), round(hi, 1)
    (109.2, 731.9)
    """
    a = 1.0 - conf
    lo = 2.0 * hours / stats.chi2.ppf(1.0 - a / 2.0, 2 * failures + 2)
    hi = 2.0 * hours / stats.chi2.ppf(a / 2.0, 2 * failures) if failures > 0 else float("inf")
    return float(lo), float(hi)


def mtbf_mttr(
    failures: pd.DataFrame,
    operating_h: pd.Series,
    by: str | Sequence[str] | None = "turbine_id",
    unit_col: str = "turbine_id",
    conf: float = 0.9,
) -> pd.DataFrame:
    """Failures, MTBF with its interval, failure rate and restore-time statistics per group.

    Parameters
    ----------
    failures : pandas.DataFrame
        One row per failure with ``duration_h`` (restore time) and the grouping columns.
    operating_h : pandas.Series
        Operating hours per unit (index: unit id), e.g. ``operating_time(...).groupby(unit_col)["hours"].sum()``.
    by : str, sequence of str or None, default "turbine_id"
        Grouping. When it contains ``unit_col``, each group uses its unit's operating hours and every unit gets
        a row (also without failures); otherwise (e.g. by component, or ``None`` for the fleet) the operating
        hours of all units are used, since every unit carries every component.
    unit_col : str, default "turbine_id"
    conf : float, default 0.9
        Confidence level of the MTBF interval.

    Returns
    -------
    pandas.DataFrame
        ``failures``, ``operating_h``, ``mtbf_h``, ``mtbf_lo_h``, ``mtbf_hi_h``,
        ``failures_per_unit_year``, ``mttr_h``, ``ttr_median_h``, ``ttr_p90_h``, ``downtime_h``, sorted by
        downtime.

    Examples
    --------
    >>> f = pd.DataFrame({"turbine_id": ["T1", "T1", "T2"], "duration_h": [2.0, 6.0, 1.0]})
    >>> up = pd.Series({"T1": 4000.0, "T2": 4380.0})
    >>> mtbf_mttr(f, up)[["turbine_id", "failures", "mtbf_h", "mttr_h"]]
      turbine_id  failures  mtbf_h  mttr_h
    0         T1         2  2000.0     4.0
    1         T2         1  4380.0     1.0
    """
    keys = [] if by is None else [by] if isinstance(by, str) else list(by)
    per_unit = unit_col in keys
    if per_unit and keys == [unit_col]:
        groups = pd.DataFrame({unit_col: operating_h.index})
    elif keys:
        groups = failures[keys].drop_duplicates()
    else:
        groups = pd.DataFrame(index=[0])
    rows = []
    for _, k in groups.iterrows():
        m = np.ones(len(failures), dtype=bool)
        for c in keys:
            m &= failures[c].to_numpy() == k[c]
        d = failures.loc[m, "duration_h"].to_numpy(dtype=float)
        hours = float(operating_h.get(k[unit_col], 0.0)) if per_unit else float(operating_h.sum())
        n = len(d)
        lo, hi = mtbf_interval(hours, n, conf)
        rows.append({**{c: k[c] for c in keys}, "failures": n, "operating_h": hours,
                     "mtbf_h": hours / n if n else np.nan, "mtbf_lo_h": lo, "mtbf_hi_h": hi,
                     "failures_per_unit_year": n / hours * HOURS_PER_YEAR if hours else np.nan,
                     "mttr_h": d.mean() if n else np.nan, "ttr_median_h": np.median(d) if n else np.nan,
                     "ttr_p90_h": np.percentile(d, 90) if n else np.nan, "downtime_h": d.sum()})
    out = pd.DataFrame(rows)
    return out.sort_values("downtime_h", ascending=False, kind="stable").reset_index(drop=True) if len(out) else out


def _root(f: Callable[[float], float], a: float, b: float) -> float:
    """Root of ``f`` in [a, b]; NaN without a sign change or with non-finite values."""
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
        return float("nan")
    return float(optimize.brentq(f, a, b))


@dataclass(frozen=True)
class CrowAMSAA:
    """Result of a Crow-AMSAA (power-law NHPP) fit.

    Attributes
    ----------
    beta, beta_lo, beta_hi : float
        Shape and its profile-likelihood bounds.
    lam : float
        Scale, per hour**beta.
    failures, units : int
        Number of failures and of units.
    hours : float
        Observed unit-hours.
    p_trend : float
        p-value of the likelihood-ratio test of ``beta = 1``.
    laplace_u, p_laplace : float
        Laplace statistic and its two-sided p-value on the observed-time clock.
    """

    beta: float
    lam: float
    beta_lo: float
    beta_hi: float
    failures: int
    units: int
    hours: float
    p_trend: float
    laplace_u: float
    p_laplace: float

    @property
    def verdict(self) -> str:
        """``improving``, ``deteriorating`` or ``no significant trend`` at 5 % (likelihood-ratio test)."""
        if self.failures < 5:
            return "too few failures"
        if not np.isfinite(self.p_trend):
            return "no power-law fit"
        if self.p_trend >= 0.05:
            return "no significant trend"
        return "deteriorating" if self.beta > 1 else "improving"

    @property
    def laplace_agrees(self) -> bool:
        """The Laplace test is significant at 5 % in the direction of the verdict."""
        if self.verdict not in ("improving", "deteriorating") or not np.isfinite(self.p_laplace):
            return False
        return bool(self.p_laplace < 0.05 and (self.laplace_u > 0) == (self.beta > 1))

    def intensity(self, age_h: ArrayLike) -> np.ndarray:
        """Failures per hour per unit at unit age ``age_h``."""
        return self.lam * self.beta * np.asarray(age_h, dtype=float) ** (self.beta - 1.0)

    def as_dict(self) -> dict[str, object]:
        """Fields and derived values as a flat dictionary."""
        return {**asdict(self), "unit_years": self.hours / HOURS_PER_YEAR, "verdict": self.verdict,
                "laplace_agrees": self.laplace_agrees}


def _nhpp_windows(failures: pd.DataFrame, windows: pd.DataFrame, origin: Mapping[object, object] | None,
                  unit_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Failure ages and window bounds (S, T) in hours of unit age."""
    origin = origin or {}
    ages: list[float] = []
    ws: list[float] = []
    we: list[float] = []
    for uid, w in windows.sort_values("start").groupby(unit_col, sort=True):
        t0 = to_ns([origin.get(uid, w["start"].min())])[0]
        f = to_ns(failures.loc[failures[unit_col] == uid, "start"])
        for a, b in zip(to_ns(w["start"]), to_ns(w["end"]), strict=True):
            sel = f[(f >= a) & (f < b)]
            ws.append(max(a - t0, 0) / NS_PER_HOUR)
            we.append((b - t0) / NS_PER_HOUR)
            ages += list((sel - t0) / NS_PER_HOUR)
    return np.array(ages, dtype=float), np.array(ws, dtype=float), np.array(we, dtype=float)


def _loglik(beta: float, t: np.ndarray, s: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    n = len(t)
    lam = n / float((e**beta - s**beta).sum())
    return float(n * np.log(lam) + n * np.log(beta) + (beta - 1.0) * np.log(t).sum() - n), lam


def crow_amsaa(
    failures: pd.DataFrame,
    windows: pd.DataFrame,
    origin: Mapping[object, object] | None = None,
    unit_col: str = "turbine_id",
    conf: float = 0.9,
) -> CrowAMSAA:
    """Fit the power-law NHPP to the failures of several units over their observation windows.

    Parameters
    ----------
    failures : pandas.DataFrame
        ``unit_col`` and ``start`` of every failure.
    windows : pandas.DataFrame
        Observation windows: ``unit_col``, ``start``, ``end``.
    origin : mapping, optional
        Age origin per unit (e.g. commissioning date); default: the start of the unit's first window.
    unit_col : str, default "turbine_id"
    conf : float, default 0.9
        Confidence level of the bounds of ``beta``.

    Returns
    -------
    CrowAMSAA

    Examples
    --------
    Failures that thin out over 4000 hours of one unit:

    >>> start = pd.Timestamp("2024-01-01", tz="UTC")
    >>> hours = [10, 30, 70, 150, 300, 600, 1100, 1900, 3200]
    >>> f = pd.DataFrame({"turbine_id": "T1", "start": [start + pd.Timedelta(hours=h) for h in hours]})
    >>> w = pd.DataFrame({"turbine_id": ["T1"], "start": [start], "end": [start + pd.Timedelta(hours=4000)]})
    >>> ca = crow_amsaa(f, w)
    >>> round(ca.beta, 3), ca.verdict
    (0.361, 'improving')
    """
    t, s, e = _nhpp_windows(failures, windows, origin, unit_col)
    n = len(t)
    hours = float((e - s).sum())
    n_units = int(windows[unit_col].nunique())
    nan = float("nan")
    if n < 3:  # neither a fit nor a trend test means much with fewer failures
        return CrowAMSAA(nan, nan, nan, nan, n, n_units, hours, nan, nan, nan)
    u, p_lap = laplace_observed(failures, windows, unit_col)
    scale = e.max()  # beta does not depend on the time unit; this keeps T**beta finite
    t, s, e = np.maximum(t, 1e-6) / scale, s / scale, e / scale
    lt = np.log(t)
    ls = np.where(s > 0, np.log(np.maximum(s, 1e-300)), 0.0)
    le = np.log(e)

    def score(beta: float) -> float:
        eb, sb = e**beta, np.where(s > 0, s**beta, 0.0)
        return float(n / beta + lt.sum() - n * (eb * le - sb * ls).sum() / (eb - sb).sum())

    beta = _root(score, 1e-4, 20.0)
    if not np.isfinite(beta):
        return CrowAMSAA(nan, nan, nan, nan, n, n_units, hours, nan, u, p_lap)
    lmax, lam = _loglik(beta, t, s, e)
    cut = float(stats.chi2.ppf(conf, 1)) / 2.0

    def drop(b: float) -> float:
        return lmax - _loglik(b, t, s, e)[0] - cut

    lo = _root(drop, 1e-4, beta)
    hi = _root(drop, beta, 20.0)
    p_lr = float(stats.chi2.sf(2.0 * (lmax - _loglik(1.0, t, s, e)[0]), 1))
    return CrowAMSAA(float(beta), float(lam / scale**beta), lo, hi, n, n_units, hours, p_lr, u, p_lap)


def laplace_test(t: ArrayLike, s: ArrayLike, e: ArrayLike) -> tuple[float, float]:
    """Laplace trend statistic for event times ``t``, each observed in its own window ``(s, e)``.

    Returns
    -------
    tuple of float
        ``U`` (positive: events cluster late) and the two-sided p-value; NaN without events.

    Examples
    --------
    >>> u, p = laplace_test([90.0, 95.0, 99.0], [0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
    >>> round(u, 3), round(p, 4)
    (2.68, 0.0074)
    """
    ta, sa, ea = (np.asarray(x, dtype=float) for x in (t, s, e))
    if len(ta) == 0:
        return float("nan"), float("nan")
    length = ea - sa
    u = float((ta - (sa + length / 2.0)).sum() / np.sqrt((length**2).sum() / 12.0))
    return u, float(2.0 * stats.norm.sf(abs(u)))


def laplace_observed(failures: pd.DataFrame, windows: pd.DataFrame, unit_col: str = "turbine_id"
                     ) -> tuple[float, float]:
    """Laplace test on each unit's observed-time clock (its observation windows joined end to end).

    Gaps then shorten the clock instead of splitting it, so the test looks at the long-term trend and not at
    what happens around each gap.
    """
    taus: list[float] = []
    spans: list[float] = []
    for uid, w in windows.sort_values("start").groupby(unit_col, sort=True):
        a, b = to_ns(w["start"]), to_ns(w["end"])
        cum = np.r_[0, np.cumsum(b - a)]
        f = to_ns(failures.loc[failures[unit_col] == uid, "start"])
        k = np.searchsorted(a, f, side="right") - 1
        ok = (k >= 0) & (f < b[np.clip(k, 0, None)])
        taus += list((cum[k[ok]] + f[ok] - a[k[ok]]) / NS_PER_HOUR)
        spans += [cum[-1] / NS_PER_HOUR] * int(ok.sum())
    return laplace_test(np.array(taus), np.zeros(len(taus)), np.array(spans))
