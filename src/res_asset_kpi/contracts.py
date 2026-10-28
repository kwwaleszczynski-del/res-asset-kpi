"""Contract terms in YAML: guaranteed values, exclusions, allowances and liquidated damages (LD).

The KPI functions of [`res_asset_kpi.pv`][res_asset_kpi.pv], [`res_asset_kpi.wind`][res_asset_kpi.wind] and
[`res_asset_kpi.bess`][res_asset_kpi.bess] measure; this module compares the measurements with what a contract
guarantees and prices the shortfall. Contract terms are kept in YAML so that they can be reviewed next to the agreement
they come from.

!!! warning

    The example files shipped in ``res_asset_kpi/data`` ([`example_path`][res_asset_kpi.contracts.example_path]) are
    **illustrative**. They are not taken from any real agreement. Thresholds, exclusions, damages formulas and caps
    differ from one contract to the next and must be checked against the actual contract, which always prevails. Nothing
    here is legal advice.

Damages methods (``guarantees.<name>.damages.method``), with shortfall ``s = max(0, g_eff - measured)``:

| method | amount | typical use |
|---|---|---|
| ``energy`` | ``s x E_basis [MWh] x price_per_mwh`` | PV or wind availability, PR |
| ``per_point`` | ``rate_per_pp x 100 s x multiplier`` | BESS availability, RTE |
| ``rte_energy_cost`` | ``E_out x (1 / RTE - 1 / RTE_g) x price_per_mwh`` when ``RTE < RTE_g`` | BESS RTE |
| ``per_mwh`` | ``rate_per_mwh x max(0, E_g - E_corr)`` on a failed test | BESS capacity |

Each guarantee's damages can be capped per period (``cap``);
[`apply_annual_cap`][res_asset_kpi.contracts.apply_annual_cap] caps the total of a year. The guaranteed value can step
down with the contract year (``annual_degradation``, relative) and be reduced by a measurement ``tolerance``: ``g_eff =
g (1 - d)^(year - 1) - tolerance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike

__all__ = [
    "Bonus",
    "Contract",
    "ContractError",
    "Damages",
    "Guarantee",
    "apply_annual_cap",
    "assess",
    "damages_amount",
    "effective_guarantee",
    "example_path",
    "guaranteed_retention",
    "load_contract",
]

Method = Literal["energy", "per_point", "rte_energy_cost", "per_mwh"]
_METHODS = ("energy", "per_point", "rte_energy_cost", "per_mwh")
_PERIODS = ("month", "quarter", "year", "contract_year")


class ContractError(ValueError):
    """A contract file does not follow the expected structure."""


@dataclass(frozen=True)
class Damages:
    """Liquidated damages clause of one guarantee."""

    method: Method = "energy"
    price_per_mwh: float = 0.0
    rate_per_pp: float = 0.0
    multiplier: float = 1.0
    rate_per_mwh: float = 0.0
    cap: float | None = None


@dataclass(frozen=True)
class Bonus:
    """Bonus above a threshold: ``(measured - threshold) x E_basis [MWh] x price_per_mwh x share``."""

    threshold: float
    share: float = 0.0
    price_per_mwh: float = 0.0


@dataclass(frozen=True)
class Guarantee:
    """One guaranteed KPI.

    Attributes
    ----------
    guaranteed : float
        Guaranteed value in contract year 1 (a fraction for ratios, an energy for capacity).
    basis : str
        Which measurement is compared (e.g. ``time`` or ``energy`` availability, ``pr_wc``, ``incl_aux``).
    annual_degradation, tolerance : float
        See [`effective_guarantee`][res_asset_kpi.contracts.effective_guarantee].
    min_data_share : float
        Below this share of valid data the test of a period is inconclusive.
    damages : Damages
    bonus : Bonus or None
    parameters : dict
        Asset-specific test settings (thresholds, windows, retention curves), passed through unchanged.
    """

    guaranteed: float
    basis: str = ""
    annual_degradation: float = 0.0
    tolerance: float = 0.0
    min_data_share: float = 0.0
    damages: Damages = field(default_factory=Damages)
    bonus: Bonus | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Contract:
    """Contract terms loaded from YAML (see the example files)."""

    name: str
    asset: Literal["pv", "wind", "bess"]
    currency: str = "EUR"
    period: str = "year"
    contract_start: str | None = None
    excluded_events: tuple[str, ...] = ()
    event_treatments: dict[str, str] = field(default_factory=dict)
    planned_allowance_h: float = 0.0
    annual_cap: float | None = None
    guarantees: dict[str, Guarantee] = field(default_factory=dict)


def _build(cls: type[Any], raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise ContractError(f"{where}: expected a mapping")
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ContractError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**raw)


def load_contract(source: str | Path | dict[str, Any]) -> Contract:
    """Read and check contract terms from a YAML file or a dictionary.

    Parameters
    ----------
    source : str, pathlib.Path or dict
        Path of the YAML file, or its parsed content.

    Returns
    -------
    Contract

    Raises
    ------
    ContractError
        Unknown keys, unknown damages methods or periods, or a missing guaranteed value.

    Examples
    --------
    >>> c = load_contract(example_path("pv_om_contract"))
    >>> c.asset, c.guarantees["availability"].guaranteed
    ('pv', 0.98)
    """
    raw = dict(source) if isinstance(source, dict) else yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ContractError("a contract file must hold a mapping")
    guarantees = {}
    for name, g in (raw.pop("guarantees", None) or {}).items():
        if not isinstance(g, dict) or "guaranteed" not in g:
            raise ContractError(f"guarantee {name!r}: 'guaranteed' is required")
        g = dict(g)
        dmg = _build(Damages, g.pop("damages", None) or {}, f"guarantee {name!r} damages")
        if dmg.method not in _METHODS:
            raise ContractError(f"guarantee {name!r}: unknown damages method {dmg.method!r}; use {_METHODS}")
        bonus = _build(Bonus, g.pop("bonus"), f"guarantee {name!r} bonus") if g.get("bonus") else None
        g.pop("bonus", None)
        guarantees[str(name)] = _build(Guarantee, {**g, "damages": dmg, "bonus": bonus}, f"guarantee {name!r}")
    pm = raw.pop("planned_maintenance", None) or {}
    if set(pm) - {"allowance_h"}:
        raise ContractError("planned_maintenance: only 'allowance_h' is known")
    raw["planned_allowance_h"] = float(pm.get("allowance_h", 0.0))
    raw["excluded_events"] = tuple(raw.get("excluded_events") or ())
    raw["event_treatments"] = dict(raw.get("event_treatments") or {})
    if raw.get("contract_start") is not None:
        raw["contract_start"] = str(raw["contract_start"])
    c = _build(Contract, {**raw, "guarantees": guarantees}, "contract")
    if c.asset not in ("pv", "wind", "bess"):
        raise ContractError("asset must be pv, wind or bess")
    if c.period not in _PERIODS:
        raise ContractError(f"period must be one of {_PERIODS}")
    if c.period == "contract_year" and c.contract_start is None:
        raise ContractError("period contract_year needs contract_start")
    return c


def example_path(name: str) -> Path:
    """Path of an illustrative contract or mapping shipped with the package.

    Parameters
    ----------
    name : str
        ``pv_om_contract``, ``wind_availability_warranty``, ``bess_ltsa``, ``mapping_wide`` or ``mapping_long``.

    Examples
    --------
    >>> example_path("bess_ltsa").name
    'bess_ltsa.example.yaml'
    """
    path = Path(str(resources.files("res_asset_kpi") / "data" / f"{name}.example.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"no example named {name!r}")
    return path


def effective_guarantee(guarantee: Guarantee, contract_year: int | ArrayLike = 1) -> np.ndarray:
    """Guaranteed value in a contract year: ``g (1 - d)^(year - 1) - tolerance``.

    Examples
    --------
    >>> g = Guarantee(guaranteed=0.78, annual_degradation=0.005, tolerance=0.02)
    >>> effective_guarantee(g, [1, 3]).round(5).tolist()
    [0.76, 0.75222]
    """
    y = np.maximum(np.asarray(contract_year, dtype=float) - 1.0, 0.0)
    return guarantee.guaranteed * (1.0 - guarantee.annual_degradation) ** y - guarantee.tolerance


def guaranteed_retention(
    years: ArrayLike, retention: ArrayLike, elapsed_years: ArrayLike, efc_total: ArrayLike | None = None,
    efc_per_year: float | None = None,
) -> np.ndarray:
    """Guaranteed state of health from a warranty retention table, linear between points.

    With ``efc_per_year``, the table is read at the equivalent age ``max(years, EFC / efc_per_year)``, so that
    cycling faster than the warranty allows ages the guarantee faster.

    Examples
    --------
    >>> guaranteed_retention([0, 1, 2], [1.0, 0.97, 0.95], [1.5], efc_total=[730], efc_per_year=365).tolist()
    [0.95]
    """
    age = np.asarray(elapsed_years, dtype=float)
    if efc_total is not None and efc_per_year:
        age = np.maximum(age, np.asarray(efc_total, dtype=float) / efc_per_year)
    return np.asarray(np.interp(age, np.asarray(years, float), np.asarray(retention, float)), dtype=float)


def damages_amount(
    damages: Damages,
    measured: ArrayLike,
    guaranteed: ArrayLike,
    *,
    basis_energy_mwh: ArrayLike | None = None,
    e_out_mwh: ArrayLike | None = None,
) -> np.ndarray:
    """Damages per period before any annual cap (see the table in the module description).

    Parameters
    ----------
    damages : Damages
    measured, guaranteed : array-like
        Measured and (effective) guaranteed values per period.
    basis_energy_mwh : array-like, optional
        Energy the shortfall is priced on (``energy`` method), e.g. expected energy of the counted intervals for
        availability or the reference energy for PR.
    e_out_mwh : array-like, optional
        Discharged energy of each window (``rte_energy_cost`` method).

    Examples
    --------
    >>> d = Damages(method="energy", price_per_mwh=60.0, cap=20000.0)
    >>> damages_amount(d, [0.975, 0.99], 0.98, basis_energy_mwh=[30000.0, 30000.0]).round(2).tolist()
    [9000.0, 0.0]
    """
    m, g = np.asarray(measured, dtype=float), np.asarray(guaranteed, dtype=float)
    short = np.where(np.isfinite(m), np.maximum(0.0, g - m), np.nan)
    if damages.method == "energy":
        if basis_energy_mwh is None:
            raise ValueError("the energy method needs basis_energy_mwh")
        amount = short * np.asarray(basis_energy_mwh, dtype=float) * damages.price_per_mwh
    elif damages.method == "per_point":
        amount = damages.rate_per_pp * short * 100.0 * damages.multiplier
    elif damages.method == "rte_energy_cost":
        if e_out_mwh is None:
            raise ValueError("the rte_energy_cost method needs e_out_mwh")
        with np.errstate(divide="ignore", invalid="ignore"):
            extra = np.asarray(e_out_mwh, dtype=float) * (1.0 / m - 1.0 / g)
        amount = np.where(m < g, extra, 0.0) * damages.price_per_mwh
        amount = np.where(np.isfinite(m), amount, np.nan)
    else:  # per_mwh: shortfall is in energy already
        amount = damages.rate_per_mwh * short
    if damages.cap is not None:
        amount = np.minimum(amount, damages.cap)
    return np.asarray(amount, dtype=float)


def assess(
    guarantee: Guarantee,
    measured: pd.Series,
    *,
    contract_year: ArrayLike | int = 1,
    basis_energy_mwh: ArrayLike | None = None,
    e_out_mwh: ArrayLike | None = None,
    data_share: ArrayLike | None = None,
) -> pd.DataFrame:
    """Compare measured values per period with a guarantee and price the shortfall.

    Parameters
    ----------
    guarantee : Guarantee
    measured : pandas.Series
        Measured value per period (index: period label).
    contract_year : int or array-like, default 1
        Contract year of each period, for the step-down of the guaranteed value.
    basis_energy_mwh, e_out_mwh : array-like, optional
        See [`damages_amount`][res_asset_kpi.contracts.damages_amount]; also prices the bonus.
    data_share : array-like, optional
        Share of valid data per period; below ``guarantee.min_data_share`` the result is inconclusive (no
        verdict, no damages).

    Returns
    -------
    pandas.DataFrame
        ``measured``, ``guaranteed_effective``, ``shortfall``, ``passed`` (True, False or None),
        ``damages`` and ``bonus`` per period.

    Examples
    --------
    >>> c = load_contract(example_path("pv_om_contract"))
    >>> a = pd.Series({"2023": 0.975, "2024": 0.997})
    >>> r = assess(c.guarantees["availability"], a, basis_energy_mwh=[30000.0, 31000.0])
    >>> r[["passed", "damages", "bonus"]].round(1)
         passed  damages  bonus
    2023  False   9000.0    0.0
    2024   True      0.0  1860.0
    """
    m = measured.to_numpy(dtype=float)
    g = np.broadcast_to(effective_guarantee(guarantee, contract_year), m.shape).astype(float)
    enough = np.ones(len(m), dtype=bool) if data_share is None else \
        np.asarray(data_share, dtype=float) >= guarantee.min_data_share
    dmg = damages_amount(guarantee.damages, m, g, basis_energy_mwh=basis_energy_mwh, e_out_mwh=e_out_mwh)
    bonus = np.zeros(len(m))
    b = guarantee.bonus
    if b is not None and basis_energy_mwh is not None:
        over = np.where(np.isfinite(m), np.maximum(0.0, m - b.threshold), 0.0)
        bonus = over * np.asarray(basis_energy_mwh, dtype=float) * b.price_per_mwh * b.share
    passed = [None if not ok or not np.isfinite(x) else bool(x >= gg) for x, gg, ok in zip(m, g, enough, strict=True)]
    return pd.DataFrame({
        "measured": m,
        "guaranteed_effective": g,
        "shortfall": np.where(np.isfinite(m), np.maximum(0.0, g - m), np.nan),
        "passed": pd.Series(passed, dtype=object).to_numpy(),
        "damages": np.where(enough, dmg, np.nan),
        "bonus": np.where(enough, bonus, 0.0),
    }, index=measured.index)


def apply_annual_cap(damages: pd.Series, years: ArrayLike, cap: float | None) -> pd.Series:
    """Scale the damages of each year down so that their sum does not exceed ``cap``.

    Examples
    --------
    >>> d = pd.Series([300.0, 500.0, 100.0])
    >>> apply_annual_cap(d, [2024, 2024, 2025], 400.0).tolist()
    [150.0, 250.0, 100.0]
    """
    if cap is None:
        return damages.copy()
    y = np.asarray(years)
    totals = damages.groupby(y).transform("sum").to_numpy(dtype=float)
    scale = np.where(totals > cap, cap / np.where(totals > 0, totals, 1.0), 1.0)
    return damages * scale
