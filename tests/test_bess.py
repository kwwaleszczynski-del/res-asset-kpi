"""Battery storage: RTE, LTSA availability, cycles and rainflow, capacity tests; hand-worked examples and properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from res_asset_kpi import bess


def _site(p: list[float], e: list[float] | None = None, start: str = "2024-06-01", freq: str = "1h",
          tz: str = "Europe/Warsaw", **cols: list[float]) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(p), freq=freq, tz=tz).tz_convert("UTC")
    data: dict[str, list[float]] = {"p_meter": p, **cols}
    if e is not None:
        data["e_stored"] = e
    return pd.DataFrame(data, index=idx)


def test_corrected_rte_by_hand() -> None:
    r = bess.corrected_rte([80.0, 80.0, 80.0, 80.0], [100.0, 100.0, 100.0, 0.0], [0.0, 5.0, -5.0, 0.0], 0.9)
    assert r[0] == pytest.approx(0.8)
    assert r[1] == pytest.approx((80 + 0.9 * 5) / 100)
    assert r[2] == pytest.approx(80 / (100 + 5 / 0.9))
    assert np.isnan(r[3])
    assert np.isnan(bess.corrected_rte([1.0], [1.0], [np.nan], 0.9)[0])
    assert bess.auto_eta(0.0, 0.0, 0.0) == 0.95


def test_auto_eta_is_consistent() -> None:
    eta = bess.auto_eta(85.0, 100.0, -3.0)
    assert eta**2 == pytest.approx(float(bess.corrected_rte([85.0], [100.0], [-3.0], eta)[0]), rel=1e-6)


def test_window_rte_daily_with_dst_and_aux() -> None:
    # 2024-10-27 has 25 hours in Warsaw; charge 4 h at 25 MW, discharge 4 h at 20 MW at the meter, aux 0.5 MW
    p = [0.0] * 25 + [0.0] * 24
    p[2:6] = [-25.0] * 4
    p[12:16] = [20.0] * 4
    aux = [0.5] * 49
    e = [20.0] * 50
    e[6:13] = [110.0] * 7
    e[13:16] = [90.0, 70.0, 50.0]
    site = _site(p, e[:49], start="2024-10-27", aux=aux)
    site = site.rename(columns={"aux": "p_aux"})
    w = bess.window_rte(site, "D", reference_energy=100.0, aux_metering="behind_meter", eta=0.95, min_efc=0.5)
    day = w.loc["2024-10-27"]
    assert day["hours"] == 25.0
    assert day["e_in"] == pytest.approx(100.0) and day["e_out"] == pytest.approx(80.0)
    assert day["d_stored"] == pytest.approx(0.0)
    assert day["rte"] == pytest.approx(0.8)
    # PCS flow = meter + aux: charge 24.5 x 4 h = 98 MWh; discharge 20.5 x 4 h = 82 MWh, idle aux counted as output
    assert day["e_in_pcs"] == pytest.approx(98.0)
    assert day["e_out_pcs"] == pytest.approx(82.0 + 0.5 * 17)
    assert day["efc"] == pytest.approx(0.98) and bool(day["valid"])
    assert w.loc["2024-10-28", "hours"] == 23.0  # the last interval has no state after it


def test_window_rte_separate_aux_soc_auto_eta_and_contract_years() -> None:
    p = [-50.0, -50.0, 0.0, 45.0, 45.0, 0.0]
    soc = [0.1, 0.6, 1.1, 1.1, 0.6, 0.1]
    site = _site(p, None, soc=soc, p_aux=[1.0] * 6, tz="UTC")
    w = bess.window_rte(site, "Y", reference_energy=100.0, aux_metering="separate", soc_energy=100.0,
                        tz="UTC", min_efc=0.0)
    row = w.iloc[0]
    assert row["e_aux"] == pytest.approx(5.0)
    assert row["rte"] == pytest.approx(90.0 / (100.0 + 5.0))  # separate supply: aux energy added to the input
    assert row["rte_excl_aux"] == pytest.approx(0.9)
    assert row["eta"] == pytest.approx(np.sqrt(0.9))
    cy = bess.window_rte(site, "CY", reference_energy=100.0, aux_metering="none", soc_energy=100.0, tz="UTC",
                         contract_start="2024-06-01")
    assert list(cy.index) == ["1"] and np.isnan(cy["rte_excl_aux"].iloc[0])
    with pytest.raises(ValueError, match="contract_start"):
        bess.window_rte(site, "CY", reference_energy=100.0)
    no_state = bess.window_rte(_site(p, None, tz="UTC"), "M", reference_energy=100.0, tz="UTC")
    assert not no_state["valid"].any()


def test_interval_flows_with_dc() -> None:
    site = _site([10.0, -10.0], [50.0, 40.0], tz="UTC", p_dc=[11.0, -9.0])
    f = bess.interval_flows(site, "none")
    assert f["e_dc_dis"].tolist() == [11.0, 0.0] and f["e_dc_ch"].tolist() == [0.0, 9.0]
    assert f["e_out_pcs"].isna().all()
    assert bess.stored_energy(site.drop(columns="e_stored")) is None


def test_measured_availability_formulas() -> None:
    site = pd.DataFrame({"p_avail_dis": [50.0, 47.0, 30.0], "p_avail_ch": [60.0, 50.0, 50.0],
                         "e_avail": [100.0, 50.0, 100.0]})
    assert bess.measured_availability(site, power=50.0, charge_power=50.0).tolist() == [1.0, 0.0, 0.0]
    assert bess.measured_availability(site, power=50.0, threshold=0.9).tolist() == [1.0, 1.0, 0.0]
    cap = bess.measured_availability(site, power=50.0, formula="capacity", combine="mean")
    assert cap.tolist() == pytest.approx([1.0, 0.97, 0.8])
    en = bess.measured_availability(site, power=50.0, formula="energy", energy=100.0, directions=["discharge"])
    assert en.tolist() == pytest.approx([1.0, 0.47, 0.6])
    with pytest.raises(ValueError, match="e_avail"):
        bess.measured_availability(site.drop(columns="e_avail"), power=50.0, formula="energy", energy=1.0)
    with pytest.raises(ValueError, match="needs the signal"):
        bess.measured_availability(site.drop(columns="p_avail_ch"), power=50.0)
    with pytest.raises(ValueError, match="unknown direction"):
        bess.measured_availability(site, power=50.0, directions=["sideways"])
    with pytest.raises(ValueError, match="unknown formula"):
        bess.measured_availability(site, power=50.0, formula="mood")  # type: ignore[arg-type]


def test_ltsa_availability_by_hand() -> None:
    # 20 hours of a 1-year period with an allowance of 876 h/year -> 2 h pro-rated to 20 h
    idx = pd.date_range("2025-03-01", periods=20, freq="1h", tz="UTC")
    a = pd.Series(1.0, index=idx)
    a.iloc[0:3] = 0.0     # planned maintenance 3 h: 2 h in the allowance, 1 h counted
    a.iloc[5] = np.nan    # missing data
    a.iloc[8:10] = 0.0    # force majeure: excluded
    a.iloc[12] = 0.5      # partial
    ev = pd.DataFrame({"start": [idx[0], idx[8]], "end": [idx[3], idx[10]],
                       "category": ["planned_maintenance", "force_majeure"]})
    tr = {"planned_maintenance": "allowance", "force_majeure": "excluded"}
    r = bess.ltsa_availability(a, events=ev, treatments=tr, planned_allowance_h=876.0, guaranteed=0.97).iloc[0]
    assert r["hours_planned_in_allowance"] == 2.0 and r["hours_planned_counted"] == 1.0
    assert r["hours_excluded"] == 2.0 + 2.0 + 1.0
    assert r["availability"] == pytest.approx((15 - 1 - 0.5) / 15)
    assert r["guaranteed"] == 0.97
    r2 = bess.ltsa_availability(a, events=ev, treatments=tr, planned_allowance_h=876.0, missing_data="unavailable")
    assert r2.iloc[0]["availability"] == pytest.approx((16 - 2 - 0.5) / 16)
    r3 = bess.ltsa_availability(a, missing_data="available", periods=["a"] * 10 + ["b"] * 10)
    assert r3.loc["a", "availability"] == pytest.approx(5 / 10)
    with pytest.raises(ValueError, match="not in the treatments"):
        bess.ltsa_availability(a, events=ev, treatments={"force_majeure": "excluded"})
    with pytest.raises(ValueError, match="unknown treatments"):
        bess.interval_treatment(idx, None, {"x": "ignored"})
    bad = ev.assign(end=ev["start"])
    with pytest.raises(ValueError, match="end must be after start"):
        bess.ltsa_availability(a, events=bad, treatments=tr)
    with pytest.raises(ValueError, match="periods"):
        bess.ltsa_availability(a, periods=["a"])


def test_overlapping_events_priority() -> None:
    idx = pd.date_range("2025-03-01", periods=4, freq="1h", tz="UTC")
    ev = pd.DataFrame({"start": [idx[0], idx[1]], "end": [idx[3], idx[2]],
                       "category": ["forced_outage", "grid_operator_instruction"]})
    t = bess.interval_treatment(idx, ev, {"forced_outage": "counted", "grid_operator_instruction": "deemed_available"})
    assert t["treatment"].tolist() == ["counted", "deemed_available", "counted", "counted"]


@settings(max_examples=60, deadline=None)
@given(st.lists(st.one_of(st.floats(0.0, 1.0), st.just(float("nan"))), min_size=1, max_size=48),
       st.sampled_from(["excluded", "available", "unavailable"]), st.floats(0.0, 5000.0))
def test_ltsa_availability_is_a_share(values: list[float], missing: str, allowance: float) -> None:
    idx = pd.date_range("2025-01-01", periods=len(values), freq="1h", tz="UTC")
    a = pd.Series(values, index=idx)
    ev = pd.DataFrame({"start": [idx[0]], "end": [idx[-1] + pd.Timedelta("1h")], "category": ["planned"]})
    r = bess.ltsa_availability(a, events=ev, treatments={"planned": "allowance"}, planned_allowance_h=allowance,
                               missing_data=missing)  # type: ignore[arg-type]
    v = r["availability"].iloc[0]
    assert np.isnan(v) or 0.0 <= v <= 1.0


def test_throughput_and_efc() -> None:
    site = pd.DataFrame({"p_meter": [-50.0, 40.0], "p_dc": [-48.0, 42.0]})
    assert bess.throughput(site, 0.5).tolist() == [0.0, 20.0]
    assert bess.throughput(site, 0.5, "discharge_dc").tolist() == [0.0, 21.0]
    assert bess.equivalent_full_cycles(bess.throughput(site, 0.5, "half_throughput_meter"), 10.0) == 2.25
    with pytest.raises(ValueError, match="p_dc"):
        bess.throughput(site.drop(columns="p_dc"), 1.0, "discharge_dc")
    with pytest.raises(ValueError, match="unknown throughput"):
        bess.throughput(site, 1.0, "both")  # type: ignore[arg-type]


def test_reversals_hysteresis_and_edges() -> None:
    noisy = [0.2, 0.21, 0.205, 0.8, 0.79, 0.795, 0.3, 0.31, 0.9]
    pos, val = bess.reversals(noisy, min_range=0.05)
    assert val.tolist() == [0.2, 0.8, 0.3, 0.9]
    pos0, _ = bess.reversals(noisy)
    assert len(pos0) > 4
    assert bess.reversals([np.nan, np.nan])[0].size == 0
    assert bess.reversals([0.5, 0.6])[1].tolist() == [0.5, 0.6]
    down = bess.reversals([0.9, 0.85, 0.5, 0.52, 0.1], min_range=0.05)[1]
    assert down.tolist() == [0.9, 0.1]


def test_rainflow_astm_example_and_efc() -> None:
    c = bess.rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    full = c[c["count"] == 1.0]
    assert sorted(full["depth"].tolist()) == [4.0]
    assert c["count"].sum() == pytest.approx(4.0)
    daily = np.tile([0.1, 0.9], 10)
    cyc = bess.rainflow(daily)
    assert bess.rainflow_efc(cyc) == pytest.approx(0.8 * 9.5)
    h = bess.depth_histogram(cyc, years=0.5)
    assert h.loc["80%-100%", "cycles"] == pytest.approx(19.0)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.floats(0.0, 1.0), min_size=0, max_size=80), st.sampled_from([0.0, 0.02, 0.1]))
def test_rainflow_preserves_path_length(x: list[float], hysteresis: float) -> None:
    pos, v = bess.reversals(x, hysteresis)
    c = bess.rainflow(x, hysteresis)
    assert (2.0 * c["depth"] * c["count"]).sum() == pytest.approx(np.abs(np.diff(v)).sum(), abs=1e-9)
    assert (c["depth"] >= 0).all() and set(c["count"]) <= {0.5, 1.0}


def test_capacity_test_by_hand() -> None:
    r = bess.evaluate_capacity_test(102.0, 95.0, temp_c=20.0, decision_rule="guard_band")
    factor = 1 + 0.003 * (20 - 25)
    assert r.temp_factor == pytest.approx(factor)
    assert r.e_corrected == pytest.approx(102.0 / factor)
    u = np.sqrt(3 * (0.002 / np.sqrt(3)) ** 2 + 0.003**2 + 0.005**2)
    assert r.u_rel_combined == pytest.approx(u)
    assert r.expanded_uncertainty == pytest.approx(2 * u * 102.0 / factor)
    assert r.verdict == "pass" and r.margin == pytest.approx(102.0 / factor - 95.0)
    assert bess.evaluate_capacity_test(80.0, 95.0).verdict == "fail"
    assert bess.evaluate_capacity_test(80.0, 95.0).temp_factor == 1.0
    assert bess.relative_uncertainty().temperature == 0.0
    assert r.as_dict()["verdict"] == "pass"
    with pytest.raises(ValueError, match="decision_rule"):
        bess.evaluate_capacity_test(1.0, 1.0, decision_rule="coin")  # type: ignore[arg-type]


def test_discharge_segment_errors() -> None:
    with pytest.raises(ValueError, match="test power"):
        bess.discharge_segment([0.0, 10.0, 0.0], 1.0, 50.0)
    lo, hi, e = bess.discharge_segment([30.0, 50.0, 50.0], 1.0, 50.0)
    assert (lo, hi, e) == (0, 2, 130.0)
