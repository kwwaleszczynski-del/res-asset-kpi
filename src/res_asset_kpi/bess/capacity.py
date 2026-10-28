"""Capacity test evaluation: discharged energy, temperature correction, measurement uncertainty and verdict.

A capacity test (in the spirit of IEC 62933-2-1 and common LTSA practice; the procedure itself is set by the
contract) charges the battery to full, rests, and discharges at the test power until the EMS stops at its lower
limit. The discharged energy is integrated at the meter (or the DC terminals), optionally with the auxiliary
energy added back, and corrected to the reference temperature with a linear coefficient:

```text
E_corr = E_meas / (1 + alpha (T_test - T_ref))
```

Uncertainty, following the GUM with relative standard uncertainties added in quadrature and expanded with a
coverage factor ``k``:

```text
u_meter = class / sqrt(3),   u_CT = ct / sqrt(3),   u_VT = vt / sqrt(3)       (rectangular error limits)
u_temp  = sqrt((alpha u_T)^2 + (u_alpha |T_test - T_ref|)^2)                  (temperature correction)
U       = k sqrt(u_meter^2 + u_CT^2 + u_VT^2 + u_temp^2) E_corr
```

Decision rules: ``simple`` passes when ``E_corr >= E_guaranteed``; ``guard_band`` passes only when
``E_corr - U >= E_guaranteed``, fails when ``E_corr + U < E_guaranteed`` and is ``inconclusive`` in between.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["CapacityTest", "UncertaintyBudget", "discharge_segment", "evaluate_capacity_test", "relative_uncertainty"]


@dataclass(frozen=True)
class UncertaintyBudget:
    """Relative standard uncertainties of one capacity test."""

    meter: float
    ct: float
    vt: float
    temperature: float

    @property
    def combined(self) -> float:
        """Combined relative standard uncertainty (root sum of squares)."""
        return float(np.sqrt(self.meter**2 + self.ct**2 + self.vt**2 + self.temperature**2))


def relative_uncertainty(
    *,
    meter_class_pct: float = 0.2,
    ct_pct: float = 0.2,
    vt_pct: float = 0.2,
    temp_c: float = float("nan"),
    t_ref_c: float = 25.0,
    temp_coeff_per_k: float = 0.003,
    temp_coeff_u_per_k: float = 0.001,
    temp_u_k: float = 1.0,
) -> UncertaintyBudget:
    """Relative standard uncertainties of a capacity test (see the module description).

    Parameters
    ----------
    meter_class_pct, ct_pct, vt_pct : float, default 0.2
        Error limits of the energy meter and the current and voltage transformers, percent.
    temp_c : float
        Mean cell temperature during the test, degC; NaN means no temperature correction.
    t_ref_c : float, default 25.0
        Reference temperature.
    temp_coeff_per_k, temp_coeff_u_per_k : float, default 0.003 and 0.001
        Temperature coefficient of the dischargeable energy (1/K) and its standard uncertainty.
    temp_u_k : float, default 1.0
        Standard uncertainty of the test temperature, K.

    Examples
    --------
    >>> u = relative_uncertainty(temp_c=30.0)
    >>> round(u.meter, 6), round(u.temperature, 6), round(u.combined, 6)
    (0.001155, 0.005831, 0.006164)
    """
    temp = float(np.hypot(temp_coeff_per_k * temp_u_k, temp_coeff_u_per_k * abs(temp_c - t_ref_c))) \
        if np.isfinite(temp_c) else 0.0
    root3 = float(np.sqrt(3.0))
    return UncertaintyBudget(meter=meter_class_pct / 100 / root3, ct=ct_pct / 100 / root3,
                             vt=vt_pct / 100 / root3, temperature=temp)


@dataclass(frozen=True)
class CapacityTest:
    """Result of one capacity test.

    Attributes
    ----------
    e_measured, e_corrected, e_guaranteed : float
        Measured, temperature-corrected and guaranteed energy.
    temp_factor : float
        ``1 + alpha (T_test - T_ref)``.
    u_rel_combined : float
        Combined relative standard uncertainty.
    expanded_uncertainty : float
        ``U`` in energy units.
    margin : float
        ``E_corr - E_guaranteed``.
    verdict : str
        ``pass``, ``fail`` or ``inconclusive``.
    """

    e_measured: float
    temp_factor: float
    e_corrected: float
    u_rel_combined: float
    expanded_uncertainty: float
    e_guaranteed: float
    margin: float
    verdict: str

    def as_dict(self) -> dict[str, object]:
        """Fields as a dictionary."""
        return asdict(self)


def evaluate_capacity_test(
    e_measured: float,
    e_guaranteed: float,
    *,
    temp_c: float = float("nan"),
    t_ref_c: float = 25.0,
    temp_coeff_per_k: float = 0.003,
    temp_coeff_u_per_k: float = 0.001,
    temp_u_k: float = 1.0,
    meter_class_pct: float = 0.2,
    ct_pct: float = 0.2,
    vt_pct: float = 0.2,
    coverage_k: float = 2.0,
    decision_rule: Literal["simple", "guard_band"] = "guard_band",
) -> CapacityTest:
    """Temperature correction, expanded uncertainty and verdict of one capacity test.

    Parameters
    ----------
    e_measured : float
        Discharged energy of the test (e.g. from [`discharge_segment`][res_asset_kpi.bess.capacity.discharge_segment]).
    e_guaranteed : float
        Guaranteed energy at the time of the test, same unit.
    temp_c : float, optional
        Mean cell temperature during the test; without it no correction is applied.
    coverage_k : float, default 2.0
        Coverage factor of the expanded uncertainty.
    decision_rule : {"simple", "guard_band"}, default "guard_band"
    t_ref_c, temp_coeff_per_k, temp_coeff_u_per_k, temp_u_k, meter_class_pct, ct_pct, vt_pct
        See [`relative_uncertainty`][res_asset_kpi.bess.capacity.relative_uncertainty].

    Returns
    -------
    CapacityTest

    Examples
    --------
    >>> r = evaluate_capacity_test(91.0, 90.0, temp_c=30.0)
    >>> round(r.e_corrected, 3), round(r.expanded_uncertainty, 3), r.verdict
    (89.655, 1.105, 'inconclusive')
    >>> evaluate_capacity_test(91.0, 90.0, temp_c=30.0, decision_rule="simple").verdict
    'fail'
    """
    factor = 1.0 + temp_coeff_per_k * (temp_c - t_ref_c) if np.isfinite(temp_c) else 1.0
    e_corr = e_measured / factor
    u = relative_uncertainty(meter_class_pct=meter_class_pct, ct_pct=ct_pct, vt_pct=vt_pct, temp_c=temp_c,
                             t_ref_c=t_ref_c, temp_coeff_per_k=temp_coeff_per_k,
                             temp_coeff_u_per_k=temp_coeff_u_per_k, temp_u_k=temp_u_k)
    big_u = coverage_k * u.combined * e_corr
    if decision_rule == "guard_band":
        verdict = "pass" if e_corr - big_u >= e_guaranteed else ("fail" if e_corr + big_u < e_guaranteed
                                                                 else "inconclusive")
    elif decision_rule == "simple":
        verdict = "pass" if e_corr >= e_guaranteed else "fail"
    else:
        raise ValueError("decision_rule must be 'simple' or 'guard_band'")
    return CapacityTest(float(e_measured), float(factor), float(e_corr), u.combined, float(big_u),
                        float(e_guaranteed), float(e_corr - e_guaranteed), verdict)


def _steady_runs(p: np.ndarray, power: float, tol: float) -> list[tuple[int, int]]:
    """(first, last) positions of runs at the test power (within the tolerance), longest first."""
    m = np.abs(p - power) <= tol * power
    d = np.diff(np.r_[0, m.astype(int), 0])
    runs = [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True)]
    return sorted(runs, key=lambda r: r[0] - r[1])


def discharge_segment(
    p_discharge: ArrayLike,
    interval_h: float,
    test_power: float,
    steady_tolerance: float = 0.05,
    idle_share: float = 0.02,
) -> tuple[int, int, float]:
    """Find the test discharge in a power series and integrate it.

    The longest run at the test power (within ``steady_tolerance``) is widened to the whole contiguous discharge
    around it (power above ``idle_share`` of the test power), so that the ramps count too.

    Parameters
    ----------
    p_discharge : array-like
        Power per interval, positive = discharge (at the meter, or at the DC terminals).
    interval_h : float
        Interval length, hours.
    test_power : float
        Test power, same unit.
    steady_tolerance : float, default 0.05
        Relative band around the test power.
    idle_share : float, default 0.02
        Power below this share of the test power ends the discharge.

    Returns
    -------
    tuple
        First and last position of the discharge and its energy.

    Raises
    ------
    ValueError
        If the series never reaches the test power.

    Examples
    --------
    >>> discharge_segment([0.0, 20.0, 50.0, 50.0, 49.0, 10.0, 0.0], 0.25, 50.0)
    (1, 5, 44.75)
    """
    p = np.nan_to_num(np.asarray(p_discharge, dtype=float))
    runs = _steady_runs(p, test_power, steady_tolerance)
    if not runs:
        raise ValueError("no discharge at the test power")
    lo, hi = runs[0]
    eps = idle_share * test_power
    while lo > 0 and p[lo - 1] > eps:
        lo -= 1
    while hi < len(p) - 1 and p[hi + 1] > eps:
        hi += 1
    return int(lo), int(hi), float(p[lo:hi + 1].clip(min=0).sum() * interval_h)
