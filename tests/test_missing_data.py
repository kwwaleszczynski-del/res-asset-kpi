"""Gaps and edge cases: missing intervals, missing irradiance or expected power, partial events, units, mappings.

Each test reproduces a case where version 0.1.0 returned a number that looked valid but was biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from res_asset_kpi import bess, pv, schema, timeseries, units, wind


def _battery_day() -> pd.DataFrame:
    """One hourly day: PCS charges 25 MW for 4 h and discharges 20 MW for 4 h; 1 MW auxiliaries behind the meter."""
    idx = pd.date_range("2024-06-01", periods=24, freq="1h", tz="UTC")
    pcs = np.zeros(24)
    pcs[2:6] = -25.0
    pcs[12:16] = 20.0
    e = np.full(24, 20.0)
    e[6:13] = 110.0
    e[13:16] = [87.8, 65.6, 43.3]
    e[16:] = 21.1
    return pd.DataFrame({"p_meter": pcs - 1.0, "p_aux": np.ones(24), "e_stored": e}, index=idx)


RTE_KW = {"reference_energy": 100.0, "aux_metering": "behind_meter", "eta": 0.9, "tz": "UTC", "min_efc": 0.5}


def test_window_rte_with_a_missing_meter_value_is_not_valid() -> None:
    site = _battery_day()
    full = bess.window_rte(site, "D", **RTE_KW).iloc[0]
    assert full["coverage"] == 1.0 and bool(full["valid"]) and np.isfinite(full["rte"])
    gap = site.copy()
    gap.iloc[13, 0] = np.nan
    w = bess.window_rte(gap, "D", **RTE_KW).iloc[0]
    assert w["coverage"] == pytest.approx(23 / 24)
    assert np.isnan(w["rte"]) and np.isnan(w["rte_raw"]) and np.isnan(w["rte_excl_aux"]) and not bool(w["valid"])
    accepted = bess.window_rte(gap, "D", **{**RTE_KW, "min_coverage": 0.9}).iloc[0]
    assert bool(accepted["valid"]) and accepted["rte"] < full["rte"]  # the bias is now the caller's choice


def test_window_rte_counts_rows_missing_from_the_index() -> None:
    site = _battery_day().drop(_battery_day().index[[13, 14]])
    w = bess.window_rte(site, "D", **RTE_KW).iloc[0]
    assert w["coverage"] == pytest.approx(22 / 24) and w["hours"] == 23.0  # no state after the last row
    assert np.isnan(w["rte"]) and not bool(w["valid"])
    with pytest.raises(ValueError, match="min_coverage"):
        bess.window_rte(site, "D", **{**RTE_KW, "min_coverage": 0.0})


def test_auto_eta_uses_complete_windows_only() -> None:
    two_days = pd.concat([_battery_day(), _battery_day().shift(24, freq="h")])
    sums = bess.window_rte(two_days, "D", **RTE_KW)  # flows and dE do not depend on eta
    auto = bess.window_rte(two_days, "D", **{**RTE_KW, "eta": "auto"})
    whole = bess.auto_eta(sums["e_out_pcs"].sum(), sums["e_in_pcs"].sum(), sums["d_stored"].sum())
    assert auto["eta"].iloc[0] == pytest.approx(whole)  # complete record: same as the whole-record value
    gap = two_days.copy()
    gap.iloc[36, 0] = np.nan  # the second day is incomplete
    w = bess.window_rte(gap, "D", **{**RTE_KW, "eta": "auto"})
    day1 = sums.iloc[0]
    assert w["eta"].iloc[0] == pytest.approx(bess.auto_eta(day1["e_out_pcs"], day1["e_in_pcs"], day1["d_stored"]))


def test_pv_availability_reports_missing_irradiance() -> None:
    running = pd.DataFrame({"A": [True] * 4, "B": [True, False, False, True]})
    poa = [400.0, np.nan, np.nan, 400.0]
    a = pv.availability(running, 1.0, poa_wm2=poa).iloc[0]
    assert a["poa_missing_h"] == pytest.approx(2.0)  # the stop no longer passes for night unnoticed
    d = pv.availability(running, 1.0, poa_wm2=poa, missing_poa="daylight").iloc[0]
    assert d["availability_time"] == pytest.approx(0.75) and d["stopped_h"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing_poa"):
        pv.availability(running, 1.0, poa_wm2=poa, missing_poa="night")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one value per interval"):
        pv.availability(running, 1.0, poa_wm2=[400.0])


def test_pv_energy_availability_needs_expected_power() -> None:
    running = pd.DataFrame({"A": [True] * 4, "B": [True, False, False, True]})
    exp = np.full((4, 2), 50.0)
    exp[1:3, 1] = np.nan
    a = pv.availability(running, 1.0, poa_wm2=[400.0] * 4, expected_kw=exp).iloc[0]
    assert a["availability_time"] == pytest.approx(0.75)
    assert np.isnan(a["availability_energy"]) and np.isnan(a["lost_kwh"])
    assert a["expected_missing_h"] == pytest.approx(1.0)
    night = exp.copy()
    night[1:3, 1] = 50.0
    night[0, 0] = np.nan  # missing expected power outside counted intervals does not matter
    b = pv.availability(running, 1.0, poa_wm2=[0.0, 400.0, 400.0, 400.0], expected_kw=night).iloc[0]
    assert b["expected_missing_h"] == 0.0 and b["availability_energy"] == pytest.approx(1 - 100.0 / 300.0)


def test_regularise_needs_complete_source_intervals() -> None:
    idx = pd.date_range("2024-06-01 10:00", periods=4, freq="15min", tz="UTC")
    one = pd.Series([10.0, np.nan, np.nan, np.nan], index=idx)
    assert np.isnan(timeseries.regularise(one, "1h", how="sum").iloc[0])
    assert np.isnan(timeseries.regularise(one, "1h").iloc[0])
    assert timeseries.regularise(one, "1h", min_share=0.25).tolist() == [10.0]
    assert np.isnan(timeseries.regularise(one, "1h", how="sum", min_share=0.25).iloc[0])  # sums stay strict
    absent = pd.Series([10.0], index=idx[:1])
    assert np.isnan(timeseries.regularise(absent, "1h", how="sum", source_step="15min").iloc[0])
    three = pd.Series([1.0, 2.0, np.nan, 3.0], index=idx)
    assert timeseries.regularise(three, "1h", min_share=0.75).tolist() == [2.0]
    with pytest.raises(ValueError, match="min_share"):
        timeseries.regularise(one, "1h", min_share=1.5)
    long = pd.DataFrame({"timestamp": idx, "device_id": "M", "signal": "e_export", "value": [1.0, 2.0, np.nan, 3.0]})
    assert np.isnan(schema.to_wide(long, "1h")[("M", "e_export")].iloc[0])


def test_wind_allowance_scales_with_the_turbine_years_covered() -> None:
    fleet = pd.DataFrame({"category": ["full_performance", "scheduled_maintenance", "forced_outage"],
                          "hours": [6 * 8760 - 360 - 100, 360.0, 100.0]})
    a = wind.availability(fleet, "oem_warranty", allowance_h=60.0)
    assert a.excused_h == pytest.approx(360.0) and a.time == pytest.approx(1 - 100 / (6 * 8760 - 360))
    one = wind.availability(fleet, "oem_warranty", allowance_h=60.0, n_turbine_years=1.0)
    assert one.excused_h == pytest.approx(60.0)
    month = pd.DataFrame({"category": ["full_performance", "scheduled_maintenance"], "hours": [714.0, 30.0]})
    m = wind.availability(month, "oem_warranty", allowance_h=60.0)
    assert m.excused_h == pytest.approx(60.0 * 744 / 8760)


def test_discharge_segment_refuses_a_gap_in_the_discharge() -> None:
    with pytest.raises(ValueError, match="missing power"):
        bess.discharge_segment([0, 50, 50, np.nan, 50, 50, 50, 0], 1.0, 50.0)
    assert bess.discharge_segment([np.nan, 0, 50, 50, 0], 1.0, 50.0) == (2, 3, 100.0)  # gap outside: fine


def test_ltsa_events_count_for_the_share_of_the_interval_they_cover() -> None:
    idx = pd.date_range("2025-03-01", periods=10, freq="1h", tz="UTC")
    m = pd.Series(1.0, index=idx)
    m.iloc[2:5] = 0.0  # outage 02:00-05:00
    ev = pd.DataFrame({"start": [idx[2] + pd.Timedelta("5min")], "end": [idx[5]], "category": ["grid"]})
    r = bess.ltsa_availability(m, events=ev, treatments={"grid": "deemed_available"}).iloc[0]
    assert r["availability"] == pytest.approx((7 + 55 / 60 + 2) / 10)
    assert r["hours_deemed_available"] == pytest.approx(2 + 55 / 60)
    inside = pd.DataFrame({"start": [idx[3] + pd.Timedelta("10min")], "end": [idx[3] + pd.Timedelta("40min")],
                           "category": ["fm"]})
    r2 = bess.ltsa_availability(m, events=inside, treatments={"fm": "excluded"}).iloc[0]
    assert r2["hours_excluded"] == pytest.approx(0.5)  # an event inside one interval no longer vanishes
    both = pd.DataFrame({"start": [idx[2], idx[2] + pd.Timedelta("30min")],
                         "end": [idx[3], idx[4]], "category": ["grid", "fm"]})
    r3 = bess.ltsa_availability(m, events=both, treatments={"grid": "deemed_available", "fm": "excluded"}).iloc[0]
    assert r3["hours_excluded"] == pytest.approx(1.5) and r3["hours_deemed_available"] == pytest.approx(0.5)


def test_ltsa_partial_allowance_and_missing_data() -> None:
    idx = pd.date_range("2025-03-01", periods=8, freq="1h", tz="UTC")
    m = pd.Series([0.0, 0.0, 0.0, np.nan, 1.0, 1.0, 1.0, 1.0], index=idx)
    ev = pd.DataFrame({"start": [idx[0] + pd.Timedelta("30min")], "end": [idx[2] + pd.Timedelta("30min")],
                       "category": ["pm"]})
    r = bess.ltsa_availability(m, events=ev, treatments={"pm": "allowance"}, planned_allowance_h=8760.0).iloc[0]
    assert r["hours_planned_in_allowance"] == pytest.approx(2.0) and r["hours_planned_counted"] == 0.0
    assert r["hours_missing"] == pytest.approx(1.0)
    assert r["availability"] == pytest.approx(4 / 5)  # counted: 0.5 + 0.5 + 0 of hours 0-2, then four available
    tight = bess.ltsa_availability(m, events=ev, treatments={"pm": "allowance"}, planned_allowance_h=1095.0,
                                   missing_data="unavailable").iloc[0]
    assert tight["hours_planned_in_allowance"] == pytest.approx(0.5)
    assert tight["hours_planned_counted"] == pytest.approx(1.5) and tight["hours_missing"] == pytest.approx(1.0)


def test_mapping_sign_and_ambiguous_milli_units() -> None:
    raw = pd.DataFrame({"timestamp": ["2024-01-01 00:00"], "device_id": ["M1"], "signal": ["P_IMP"],
                        "value": [5.0]})
    long = schema.read(raw, {"timezone": "UTC", "signals": {"P_IMP": {"signal": "p_meter", "unit": "MW", "sign": -1}}})
    assert long["value"].iloc[0] == -5000.0
    with pytest.raises(schema.SchemaError, match="sign"):
        schema.load_mapping({"signals": {"P": {"signal": "p_meter", "sign": 2}}})
    with pytest.raises(ValueError, match="ambiguous"):
        units.convert(1.0, "mW", "W")
    assert units.convert(1.0, "mw", "kW") == 1000.0  # lower-case mw is the usual spelling of MW


def test_bridge_edge_periods(caplog: pytest.LogCaptureFixture) -> None:
    b = pv.energy_bridge([0.0, 100.0], [50.0, 90.0], {"irr": [1.0, 1.0]}, {"a": [0.1, 0.1]}, {"a": [0.0, 0.0]},
                         method="shapley")
    assert b.iloc[0][["irr", "a", "residual"]].isna().all() and "without budget energy" in caplog.text
    assert abs(b["closure"].iloc[1]) < 1e-9
    outage = pv.energy_bridge(100.0, 0.0, {}, {"availability": 1.0}, {"availability": 0.01}, method="shapley")
    assert outage[["availability", "residual"]].sum(axis=1).iloc[0] == pytest.approx(-100.0)


def test_capacity_test_docstring_example_counts() -> None:
    c = bess.rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2])
    assert c[c["count"] == 1.0]["depth"].tolist() == [4.0]
    assert sorted(c[c["count"] == 0.5]["depth"].tolist()) == [3.0, 4.0, 6.0, 8.0, 8.0, 9.0]
