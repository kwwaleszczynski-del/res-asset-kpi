"""Energy bridge (waterfall) from a budget (e.g. P50) to the measured production, closing exactly.

**Sequential method** (loss accounting). The budget energy ``E_b`` is first re-stated for what was outside the
operator's control, with ratios applied in order (e.g. irradiance, temperature, degradation):

    bar_k = E_b r_1 ... r_(k-1) (r_k - 1)            E_w = E_b prod r_k

Then each loss ``k`` (soiling, snow, clipping, availability, curtailment, data gaps, ...) gets the bar *budgeted
loss minus measured loss*, in kWh. The measured loss fractions ``l_k`` are applied in order to the potential of
the actual plant ``X_a = E_act / prod (1 - l_k)``, the budgeted fractions ``b_k`` to the potential the budget
implies for that weather ``X_b = E_w / prod (1 - b_k)``:

    L_k = X_a prod_(j<k) (1 - l_j) l_k        B_k = X_b prod_(j<k) (1 - b_j) b_k        bar_k = B_k - L_k

The **residual** ``E_act - E_b - sum bars`` is what the measurements do not explain (model error, unmodelled
losses, sensor level). By construction the bars add up to ``E_act - E_b`` exactly.

**Shapley method.** The same month written as a product ``E_act = E_b prod f_k`` of the weather ratios, the loss
ratios ``(1 - l_k) / (1 - b_k)`` and the residual factor ``X_a / X_b``; each bar is the Shapley value of the
product game, which does not depend on the order of the factors. Its bars also sum to ``E_act - E_b``, but no
single bar equals a measured loss.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from math import factorial

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = ["energy_bridge", "shapley_product"]

log = logging.getLogger(__name__)


def shapley_product(base: float, factors: ArrayLike) -> np.ndarray:
    """Shapley values of the game ``v(S) = base * prod_(k in S) f_k``.

    They add up to ``base * (prod f - 1)``.

    Parameters
    ----------
    base : float
        Value of the empty coalition (the budget energy).
    factors : array-like
        Multiplicative factors.

    Returns
    -------
    numpy.ndarray
        One value per factor.

    Examples
    --------
    >>> shapley_product(100.0, [1.1, 0.9]).round(6).tolist()
    [9.5, -10.5]
    """
    f = np.asarray(factors, dtype=float)
    n = len(f)
    w = np.array([factorial(s) * factorial(n - 1 - s) / factorial(n) for s in range(n)])
    out = np.zeros(n)
    for k in range(n):
        e = np.zeros(n)  # elementary symmetric polynomials of the other factors
        e[0] = 1.0
        for x in np.delete(f, k):
            e[1:] = e[1:] + x * e[:-1]
        out[k] = base * (f[k] - 1.0) * float(np.dot(w, e))
    return out


def _column(x: ArrayLike, n: int, name: str) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.shape == (1,) and n > 1:
        a = np.repeat(a, n)
    if a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have one value per period")
    return a


def energy_bridge(
    e_budget: ArrayLike,
    e_actual: ArrayLike,
    weather: Mapping[str, ArrayLike],
    losses_actual: Mapping[str, ArrayLike],
    losses_budget: Mapping[str, ArrayLike],
    *,
    method: str = "sequential",
    min_keep: float = 0.05,
    index: Sequence[object] | pd.Index | None = None,
) -> pd.DataFrame:
    """Bars of the energy bridge from budget to actual, one row per period.

    Parameters
    ----------
    e_budget, e_actual : float or array-like
        Budget and measured energy per period (any energy unit, the same for both).
    weather : mapping of str to float or array-like
        Ordered ratios actual / budget of the conditions outside the operator's control, e.g.
        ``{"irradiance": r_irr, "temperature": r_T, "degradation": r_D}``.
    losses_actual : mapping of str to float or array-like
        Ordered measured loss fractions ``l_k`` (0..1), e.g. ``{"soiling": 0.021, "availability": 0.004}``.
    losses_budget : mapping of str to float or array-like
        Budgeted loss fractions ``b_k`` with the same keys (0 where the budget assumes none).
    method : {"sequential", "shapley"}, default "sequential"
        Allocation method (see the module description).
    min_keep : float, default 0.05
        Smallest ``prod (1 - l_k)`` for which the actual potential is computed as ``E_act / prod (1 - l_k)``.
        In a period with less production left (e.g. a month-long outage) the measured losses are applied to
        the budget's potential instead, and a warning is logged.
    index : sequence, optional
        Period labels of the rows.

    Returns
    -------
    pandas.DataFrame
        One column per weather step and loss, then ``residual``; also ``e_budget``, ``e_actual``,
        ``e_weather`` (the budget re-stated for the weather), ``potential_actual``, ``potential_budget`` and
        ``closure`` (``e_actual - e_budget - sum of bars``, zero up to rounding).

    Examples
    --------
    >>> b = energy_bridge(1000.0, 905.0, {"irradiance": 0.95},
    ...                   {"soiling": 0.03, "availability": 0.01}, {"soiling": 0.02, "availability": 0.0})
    >>> b[["irradiance", "soiling", "availability", "residual"]].round(2)
       irradiance  soiling  availability  residual
    0       -50.0    -8.88         -9.14    -26.97
    >>> abs(float(b["closure"].iloc[0])) < 1e-9
    True
    """
    if set(losses_actual) != set(losses_budget):
        raise ValueError("losses_actual and losses_budget need the same keys")
    if method not in ("sequential", "shapley"):
        raise ValueError("method must be 'sequential' or 'shapley'")
    eb0 = np.atleast_1d(np.asarray(e_budget, dtype=float))
    n = max(len(eb0), len(np.atleast_1d(np.asarray(e_actual, dtype=float))))
    eb = _column(e_budget, n, "e_budget")
    ea = _column(e_actual, n, "e_actual")
    r = {k: _column(v, n, f"weather[{k!r}]") for k, v in weather.items()}
    la = {k: _column(v, n, f"losses_actual[{k!r}]") for k, v in losses_actual.items()}
    lb = {k: _column(losses_budget[k], n, f"losses_budget[{k!r}]") for k in losses_actual}
    for name, d in (("losses_actual", la), ("losses_budget", lb)):
        for k, v in d.items():
            if ((v < 0) | (v >= 1)).any():
                raise ValueError(f"{name}[{k!r}] must be in [0, 1)")
    out: dict[str, np.ndarray] = {}
    acc = eb.copy()
    for k, v in r.items():
        out[k] = acc * (v - 1.0)
        acc = acc * v
    e_w = acc
    keep_a = np.prod([1.0 - v for v in la.values()], axis=0) if la else np.ones(n)
    keep_b = np.prod([1.0 - v for v in lb.values()], axis=0) if lb else np.ones(n)
    pot_b = e_w / keep_b
    low = keep_a <= min_keep
    if low.any():
        log.warning("%d period(s) with almost no production left: measured losses taken on the budget's "
                    "potential", int(low.sum()))
    pot_a = np.where(low, pot_b, ea / np.where(low, 1.0, keep_a))
    acc_a, acc_b = pot_a.copy(), pot_b.copy()
    for k in la:
        loss_a = acc_a * la[k]
        loss_b = acc_b * lb[k]
        out[k] = loss_b - loss_a
        acc_a = acc_a - loss_a
        acc_b = acc_b - loss_b
    bars = list(r) + list(la)
    out["residual"] = ea - eb - np.sum([out[k] for k in bars], axis=0)
    if method == "shapley":
        loss_ratios = [(1.0 - la[k]) / (1.0 - lb[k]) for k in la]
        # residual factor = what the other factors leave unexplained; equals X_a / X_b outside the fallback
        explained = e_w * (np.prod(loss_ratios, axis=0) if loss_ratios else np.ones(n))
        residual_factor = np.divide(ea, explained, out=np.ones(n), where=explained > 0)
        factors = [r[k] for k in r] + loss_ratios + [residual_factor]
        sh = np.array([shapley_product(eb[i], [f[i] for f in factors]) for i in range(n)])
        for j, k in enumerate([*bars, "residual"]):
            out[k] = sh[:, j]
    frame = pd.DataFrame(out, index=index)
    frame["e_budget"] = eb
    frame["e_actual"] = ea
    frame["e_weather"] = e_w
    frame["potential_actual"] = pot_a
    frame["potential_budget"] = pot_b
    frame["closure"] = ea - eb - frame[[*bars, "residual"]].sum(axis=1).to_numpy()
    return frame
