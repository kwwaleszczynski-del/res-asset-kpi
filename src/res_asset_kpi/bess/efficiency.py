"""Round-trip efficiency (RTE) of a battery at the meter, corrected for the change of stored energy.

Per interval the metered power ``p_meter`` (positive = export, discharge) is split into energy out and energy in.
Where the auxiliaries (HVAC, transformers, control) sit behind the revenue meter (``aux_metering =
"behind_meter"``), the meter sees the net flow and the flow of the power conversion system is
``p_meter + p_aux``; where they have their own supply (``"separate"``), their energy is added to the input for
the RTE that includes them.

Over a window that does not end at the state of charge it started with, the RTE is corrected for the change of
stored energy ``dE`` with a one-way efficiency ``eta`` between the meter and the store:

```text
dE >= 0:  RTE = (E_out + eta dE) / E_in          what is left in store is worth eta dE at the meter
dE <  0:  RTE = E_out / (E_in + |dE| / eta)      what was taken from store would need |dE| / eta to refill
```

``eta`` is given or found as ``sqrt(RTE)`` of the whole record by fixed-point iteration. With ``dE = 0`` both
lines reduce to ``E_out / E_in``. The stored energy is read at the instants that bound each window (the start of
the window and the start of the next one).

Any consistent units work: MW and MWh, or kW and kWh.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from res_asset_kpi import timeseries

__all__ = ["MIN_EFC", "auto_eta", "corrected_rte", "interval_flows", "stored_energy", "window_keys", "window_rte"]

AuxMetering = Literal["behind_meter", "separate", "none"]
WindowFreq = Literal["D", "M", "Q", "Y", "CY"]
MIN_EFC: dict[str, float] = {"D": 0.3, "M": 5.0, "Q": 15.0, "Y": 50.0, "CY": 50.0}
"""Default minimum equivalent full cycles for a window's RTE to be assessed."""


def corrected_rte(e_out: ArrayLike, e_in: ArrayLike, d_stored: ArrayLike, eta: float) -> np.ndarray:
    """RTE corrected for the change of stored energy (see the module description).

    Parameters
    ----------
    e_out, e_in : array-like
        Energy discharged and charged over each window.
    d_stored : array-like
        Change of stored energy over each window (end minus start); NaN gives NaN.
    eta : float
        One-way efficiency between the meter and the store.

    Examples
    --------
    A day that charged 100 and discharged 80 but ended 5 higher than it started:

    >>> corrected_rte([80.0], [100.0], [5.0], 0.95).round(4).tolist()
    [0.8475]
    """
    eo, ei, de = (np.asarray(x, dtype=float) for x in (e_out, e_in, d_stored))
    d = np.nan_to_num(de)
    gain = d > 0
    num = eo + np.where(gain, eta * d, 0.0)
    den = ei + np.where(gain, 0.0, -d / eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return np.where(np.isnan(de), np.nan, r)


def auto_eta(e_out: float, e_in: float, d_stored: float, iterations: int = 50) -> float:
    """One-way efficiency ``sqrt(RTE)`` consistent with the corrected RTE of the whole record.

    Examples
    --------
    >>> round(auto_eta(81.0, 100.0, 0.0), 4)
    0.9
    """
    eta = 0.95
    for _ in range(iterations):
        r = float(corrected_rte([e_out], [e_in], [d_stored], eta)[0])
        if not np.isfinite(r) or r <= 0:
            return 0.95
        eta = min(float(np.sqrt(r)), 1.0)
    return eta


def stored_energy(site: pd.DataFrame, soc_energy: float | None = None) -> pd.Series | None:
    """Stored energy at every instant: ``e_stored`` if present, else ``soc`` times the energy SoC = 1 stands for."""
    if "e_stored" in site:
        return site["e_stored"].astype(float)
    if "soc" in site and soc_energy:
        return site["soc"].astype(float) * soc_energy
    return None


def interval_flows(site: pd.DataFrame, aux_metering: AuxMetering = "behind_meter",
                   soc_energy: float | None = None) -> pd.DataFrame:
    """Energy per interval: meter out and in, PCS out and in, auxiliaries, DC, and the stored energy.

    Parameters
    ----------
    site : pandas.DataFrame
        Regular UTC index (interval starts) and canonical columns: ``p_meter`` (required), ``p_aux``,
        ``p_dc``, ``e_stored`` or ``soc``.
    aux_metering : {"behind_meter", "separate", "none"}, default "behind_meter"
        Where the auxiliary consumption is metered (see the module description).
    soc_energy : float, optional
        Energy that SoC = 1 stands for, to turn ``soc`` into stored energy.

    Returns
    -------
    pandas.DataFrame
        ``e_out``, ``e_in``, ``e_aux``, ``e_out_pcs``, ``e_in_pcs``, (``e_dc_dis``, ``e_dc_ch``) and
        ``stored`` (NaN columns where the signal is missing).
    """
    dt = timeseries.interval_hours(pd.DatetimeIndex(site.index))
    p = site["p_meter"].astype(float)
    out = pd.DataFrame(index=site.index)
    out["e_out"] = p.clip(lower=0) * dt
    out["e_in"] = (-p).clip(lower=0) * dt
    aux = site["p_aux"].astype(float) if "p_aux" in site else None
    out["e_aux"] = aux * dt if aux is not None else np.nan
    pcs = p + aux if aux is not None and aux_metering == "behind_meter" else p if aux_metering == "separate" else None
    out["e_out_pcs"] = pcs.clip(lower=0) * dt if pcs is not None else np.nan
    out["e_in_pcs"] = (-pcs).clip(lower=0) * dt if pcs is not None else np.nan
    if "p_dc" in site:
        out["e_dc_dis"] = site["p_dc"].clip(lower=0) * dt
        out["e_dc_ch"] = (-site["p_dc"]).clip(lower=0) * dt
    e = stored_energy(site, soc_energy)
    out["stored"] = e if e is not None else np.nan
    return out


def window_keys(index: pd.DatetimeIndex, freq: WindowFreq = "M", tz: str = "Europe/Warsaw",
                contract_start: pd.Timestamp | str | None = None) -> pd.Series:
    """Window label of every interval: local day, month, quarter, year, or contract year (``CY``)."""
    if freq == "CY":
        if contract_start is None:
            raise ValueError("contract years need contract_start")
        keys = np.asarray(timeseries.contract_years(index, contract_start, tz)).astype(str)
    else:
        keys = np.asarray(timeseries.period_labels(index, freq, tz))
    return pd.Series(keys, index=index)


def _state_at(e: pd.Series, pos: np.ndarray, tolerance: int = 2) -> np.ndarray:
    """Stored energy at row positions (instants); gaps up to ``tolerance`` samples are interpolated."""
    filled = e.interpolate(limit=tolerance, limit_area="inside").to_numpy(float)
    vals = np.full(len(pos), np.nan)
    ok = (pos >= 0) & (pos < len(filled))
    vals[ok] = filled[pos[ok]]
    return vals


def _window_sums(flows: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Sums of the flows and the change of stored energy over each window.

    If no state is known after the last interval of the record, that interval is left out of the last window.
    """
    n = len(flows)
    pos = pd.Series(np.arange(n), index=flows.index)
    g = pos.groupby(keys.to_numpy(), sort=True)
    first, last = g.min(), g.max()
    end = (last + 1).to_numpy()
    stored = flows["stored"]
    e_end = _state_at(stored, end)
    shorten = np.isnan(e_end) & (end == n)
    end = np.where(shorten, end - 1, end)
    e_end = np.where(shorten, _state_at(stored, end), e_end)
    e_start = _state_at(stored, first.to_numpy())
    cum = flows.drop(columns="stored").fillna(0.0).cumsum()
    cum = pd.concat([pd.DataFrame(0.0, index=[0], columns=cum.columns), cum.reset_index(drop=True)],
                    ignore_index=True)
    sums = pd.DataFrame(cum.to_numpy()[end] - cum.to_numpy()[first.to_numpy()], index=first.index,
                        columns=cum.columns)
    sums["coverage"] = flows["e_out"].notna().astype(float).groupby(keys.to_numpy(), sort=True).mean().to_numpy()
    step_h = (flows.index[1] - flows.index[0]) / pd.Timedelta("1h") if n > 1 else 0.0
    sums["hours"] = (end - first.to_numpy()) * step_h
    sums["stored_start"] = e_start
    sums["stored_end"] = e_end
    sums["d_stored"] = e_end - e_start
    for c in ("e_aux", "e_out_pcs", "e_in_pcs", "e_dc_dis", "e_dc_ch"):
        if c in flows and flows[c].isna().all():
            sums[c] = np.nan
    sums.index.name = "window"
    return sums


def window_rte(
    site: pd.DataFrame,
    freq: WindowFreq = "M",
    *,
    reference_energy: float,
    aux_metering: AuxMetering = "behind_meter",
    soc_energy: float | None = None,
    tz: str = "Europe/Warsaw",
    contract_start: pd.Timestamp | str | None = None,
    eta: float | Literal["auto"] = "auto",
    min_efc: float | None = None,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Energy in and out and RTE per window, with and without the auxiliaries.

    Parameters
    ----------
    site : pandas.DataFrame
        Regular UTC index and canonical columns ``p_meter`` and ``e_stored`` or ``soc`` (needed for the SoC
        correction); ``p_aux`` for the split with and without auxiliaries.
    freq : {"D", "M", "Q", "Y", "CY"}, default "M"
        Window: local day, month, quarter, year, or contract year from ``contract_start``.
    reference_energy : float
        Energy for the equivalent full cycles (e.g. usable energy), same unit as the energies.
    aux_metering : {"behind_meter", "separate", "none"}, default "behind_meter"
        Where the auxiliary consumption is metered (see the module description).
    soc_energy : float, optional
        Energy that SoC = 1 stands for (needed when the site has ``soc`` and no ``e_stored``).
    tz : str, default "Europe/Warsaw"
        Time zone of the windows (a local day has 23 or 25 hours at the DST changes).
    contract_start : pandas.Timestamp or str, optional
        Start of contract year 1 for ``freq="CY"``.
    eta : float or "auto", default "auto"
        One-way efficiency for the correction; ``auto`` = ``sqrt(RTE)`` of the whole record (excluding the
        auxiliaries when possible).
    min_efc : float, optional
        Minimum equivalent full cycles for ``valid`` (default [`MIN_EFC`][res_asset_kpi.bess.efficiency.MIN_EFC]
        of the window).
    min_coverage : float, default 0.95
        Minimum share of intervals with a meter reading for ``valid``.

    Returns
    -------
    pandas.DataFrame
        Per window: summed flows, ``coverage``, ``hours``, ``stored_start``, ``stored_end``, ``d_stored``,
        ``rte_raw`` (uncorrected ``E_out / E_in``), ``rte`` (including auxiliaries), ``rte_excl_aux``,
        ``efc`` (PCS input energy / reference energy), ``eta`` and ``valid``.

    Examples
    --------
    One day of one cycle, ending 2 MWh higher than it started:

    >>> idx = pd.date_range("2024-06-01 00:00", periods=6, freq="4h", tz="Europe/Warsaw").tz_convert("UTC")
    >>> site = pd.DataFrame({"p_meter": [-25.0, 0.0, 19.5, 0.0, 0.0, 0.0],
    ...                      "e_stored": [10.0, 105.0, 105.0, 22.0, 12.0, 12.0]}, index=idx)
    >>> w = window_rte(site, "D", reference_energy=100.0, aux_metering="none", eta=0.95, min_efc=0.5)
    >>> w[["e_in", "e_out", "d_stored", "rte_raw", "rte"]].round(4)
                e_in  e_out  d_stored  rte_raw   rte
    window
    2024-06-01 100.0   78.0       2.0     0.78  0.799
    """
    flows = interval_flows(site, aux_metering, soc_energy)
    keys = window_keys(pd.DatetimeIndex(site.index), freq, tz, contract_start)
    s = _window_sums(flows, keys)
    have_pcs = bool(s["e_in_pcs"].notna().any())
    if eta == "auto":
        total = _window_sums(flows, pd.Series("all", index=flows.index)).iloc[0]
        cols = ("e_out_pcs", "e_in_pcs") if have_pcs else ("e_out", "e_in")
        eta = auto_eta(total[cols[0]], total[cols[1]], total["d_stored"])
    eta_v = float(eta)
    e_in_incl = s["e_in"] + s["e_aux"] if aux_metering == "separate" and s["e_aux"].notna().any() else s["e_in"]
    s["rte_raw"] = s["e_out"] / s["e_in"].where(s["e_in"] > 0)
    s["rte"] = corrected_rte(s["e_out"], e_in_incl, s["d_stored"], eta_v)
    s["rte_excl_aux"] = corrected_rte(s["e_out_pcs"], s["e_in_pcs"], s["d_stored"], eta_v) if have_pcs else np.nan
    basis = s["e_in_pcs"] if have_pcs else s["e_in"]
    s["efc"] = basis / reference_energy
    s["eta"] = eta_v
    need = MIN_EFC.get(freq, 0.0) if min_efc is None else min_efc
    s["valid"] = (s["efc"] >= need) & (s["coverage"] >= min_coverage) & s["d_stored"].notna()
    return s
