"""Wind: availability by category, power curves, reliability; hand-worked examples and properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from res_asset_kpi import wind

CATS = list(wind.CATEGORIES)


def test_category_totals_and_availability_by_hand() -> None:
    cat = np.array([["full_performance", "forced_outage"], ["environmental", None], ["requested", "full_performance"]])
    lost = np.array([[0.0, 100.0], [0.0, 50.0], [80.0, 0.0]])
    act = np.array([[300.0, -2.0], [0.0, 0.0], [0.0, 250.0]])
    groups = np.array([["T1", "T2"]] * 3)
    tot = wind.category_totals(cat, 1.0, lost, act, groups)
    t2 = tot[tot["group"] == "T2"].set_index("category")
    assert t2.loc["forced_outage", "hours"] == 2.0  # the missing category counts as a forced outage
    assert t2.loc["forced_outage", "actual_kwh"] == 0.0  # own consumption counts as zero
    assert len(tot) == 2 * len(CATS)
    t1 = tot[tot["group"] == "T1"]
    a = wind.availability(t1, "iec_operational")
    assert a.time == 1.0 and a.energy == 1.0
    a2 = wind.availability(tot[tot["group"] == "T2"], "iec_operational")
    assert a2.time == pytest.approx(1 / 3)
    assert a2.energy == pytest.approx(1 - 150.0 / 400.0)
    warranty = wind.availability(t1, "oem_warranty")
    assert warranty.time == 1.0 and warranty.potential_kwh == pytest.approx(300.0)
    assert set(warranty.as_dict()) >= {"time", "energy", "excused_h"}


def test_maintenance_allowance_excuses_hours_and_energy() -> None:
    tot = pd.DataFrame({"category": ["full_performance", "scheduled_maintenance", "forced_outage"],
                        "hours": [1700.0, 100.0, 50.0], "lost_kwh": [0.0, 200.0, 100.0],
                        "actual_kwh": [5000.0, 0.0, 0.0]})
    a = wind.availability(tot, "oem_warranty", allowance_h=30.0, n_turbine_years=2.0)
    assert a.excused_h == 60.0
    assert a.time == pytest.approx(1700.0 / (1700.0 + 90.0))
    assert a.lost_unavailable_kwh == pytest.approx(300.0 - 200.0 * 0.6)
    assert a.energy == pytest.approx(1 - 180.0 / 5300.0)


def test_availability_errors_and_remap() -> None:
    with pytest.raises(ValueError, match="unknown categories"):
        wind.category_totals(["spinning"], 1.0)
    with pytest.raises(ValueError, match="same size"):
        wind.category_totals(["forced_outage"], 1.0, lost_kwh=[1.0, 2.0])
    tot = pd.DataFrame({"category": ["forced_outage"], "hours": [1.0]})
    with pytest.raises(ValueError, match="does not say"):
        wind.availability(tot, {"full_performance": "available"})
    with pytest.raises(ValueError, match="unknown treatments"):
        wind.availability(tot, {"forced_outage": "forgiven"})
    with pytest.raises(ValueError, match="unknown category"):
        wind.remap(["requested"], {"requested": "grid"})
    empty = pd.DataFrame({"category": ["no_info"], "hours": [10.0]})
    assert np.isnan(wind.availability(empty, "iec_operational").time)


def test_misclassification_lowers_availability() -> None:
    cat = np.array(["full_performance"] * 90 + ["requested"] * 5 + ["forced_outage"] * 5)
    base = wind.availability(wind.category_totals(cat, 1.0), "oem_warranty")
    worse = wind.availability(wind.category_totals(wind.remap(cat, {"requested": "forced_outage"}), 1.0),
                              "oem_warranty")
    assert base.time == pytest.approx(90 / 95) and worse.time == pytest.approx(90 / 100)


@settings(max_examples=80, deadline=None)
@given(st.lists(st.sampled_from(CATS), min_size=1, max_size=200), st.floats(0, 200), st.sampled_from(
    ["iec_operational", "oem_warranty"]))
def test_wind_availability_is_a_share(cats: list[str], allowance: float, contract: str) -> None:
    rng = np.random.default_rng(len(cats))
    lost = rng.uniform(0, 500, len(cats))
    act = rng.uniform(-10, 2000, len(cats))
    a = wind.availability(wind.category_totals(cats, 1 / 6, lost, act), contract, allowance_h=allowance)
    for v in (a.time, a.energy):
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_air_density_and_normalisation_by_hand() -> None:
    assert wind.standard_pressure(0.0) == pytest.approx(101325.0)
    rho = wind.air_density(0.0, elevation_m=0.0)
    assert rho == pytest.approx(101325.0 / (287.05 * 273.15))
    assert wind.normalise_wind_speed(8.0, wind.RHO_0) == pytest.approx(8.0)
    assert wind.normalise_wind_speed(8.0, 1.225 * 1.331) == pytest.approx(8.0 * 1.1, rel=1e-6)
    with pytest.raises(ValueError, match="pressure_pa or elevation_m"):
        wind.air_density(10.0)


def test_bin_power_curve_by_hand() -> None:
    ws = [7.74, 7.76, 8.0, 8.24, 8.26, 8.3, 8.4, np.nan]
    p = [900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1350.0, 1400.0, 5.0]
    c = wind.bin_power_curve(ws, p)
    assert c.index.tolist() == [8.0, 8.5]  # 7.74 rounds into the 7.5 bin, which has one record
    row = c.loc[8.0]
    assert row["n"] == 3 and row["power_kw"] == pytest.approx(1100.0)
    assert row["u_a_kw"] == pytest.approx(np.std([1000.0, 1100.0, 1200.0], ddof=1) / np.sqrt(3))
    assert c.loc[8.5, "ws"] == pytest.approx(np.mean([8.26, 8.3, 8.4]))
    assert wind.bin_power_curve(ws, p, min_count=1).index.tolist() == [7.5, 8.0, 8.5]


def test_database_complete_needs_bins_and_hours() -> None:
    ws = np.repeat(np.arange(2.0, 14.01, 0.5), 60)
    p = np.clip((ws - 3.0) * 300.0, 0.0, 2000.0)
    assert wind.database_complete(wind.bin_power_curve(ws, p), 2000.0)
    short = wind.bin_power_curve(ws[::4], p[::4])
    assert not wind.database_complete(short, 2000.0)  # 375 records = 62.5 h < 180 h
    assert not wind.database_complete(wind.bin_power_curve(ws, p * 0.5), 2000.0)  # never reaches 85 %


def test_aep_rayleigh_by_hand() -> None:
    # a step curve: 0 below 5 m/s, 1000 kW from 5 m/s to cut-out 25 m/s; AEP = 8760 (F(25) - F(5)) MWh
    aep = wind.aep_rayleigh([4.999, 5.0, 25.0], [0.0, 1000.0, 1000.0], 7.0, step=0.001)
    f = lambda v: 1 - np.exp(-np.pi / 4 * (v / 7.0) ** 2)  # noqa: E731
    assert aep == pytest.approx(8760.0 * (f(25.0) - f(5.0)), rel=1e-3)
    assert wind.aep_rayleigh([0.0, 25.0], [1000.0, 1000.0], 7.0, availability=0.97) == pytest.approx(
        0.97 * wind.aep_rayleigh([0.0, 25.0], [1000.0, 1000.0], 7.0))
    assert np.isnan(wind.curve_power([3.0, 4.0], [0.0, 1.0], [np.nan])[0])


# --------------------------------------------------------------------------------------------- reliability


def _ts(h: float) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h)


def test_mtbf_interval_matches_chi_square() -> None:
    lo, hi = wind.mtbf_interval(5000.0, 10, conf=0.8)
    assert lo == pytest.approx(10000.0 / stats.chi2.ppf(0.9, 22))
    assert hi == pytest.approx(10000.0 / stats.chi2.ppf(0.1, 20))
    assert wind.mtbf_interval(100.0, 0)[1] == np.inf


def test_operating_time_and_mtbf_mttr_by_hand() -> None:
    windows = pd.DataFrame({"turbine_id": ["T1", "T1", "T2"], "start": [_ts(0), _ts(600), _ts(0)],
                            "end": [_ts(500), _ts(1000), _ts(1000)]})
    stops = pd.DataFrame({"turbine_id": ["T1", "T1", "T2", "T2"], "start": [_ts(100), _ts(450), _ts(10), _ts(15)],
                          "end": [_ts(110), _ts(650), _ts(20), _ts(30)]})
    up = wind.operating_time(windows, stops)
    hours = up.groupby("turbine_id")["hours"].sum()
    assert hours.to_dict() == {"T1": 500 - 10 - 50 + 400 - 50, "T2": 1000 - 20}
    assert up.loc[up["turbine_id"] == "T1", "window"].tolist() == [0, 0, 1]
    fail = pd.DataFrame({"turbine_id": ["T1", "T1", "T2"], "start": [_ts(100), _ts(450), _ts(10)],
                         "duration_h": [10.0, 200.0, 20.0], "component": ["pitch", "gearbox", "pitch"]})
    k = wind.mtbf_mttr(fail, hours).set_index("turbine_id")
    assert k.loc["T1", "mtbf_h"] == pytest.approx(790.0 / 2)
    assert k.loc["T1", "mttr_h"] == pytest.approx(105.0)
    assert k.loc["T2", "failures_per_unit_year"] == pytest.approx(1 / 980 * 8766)
    comp = wind.mtbf_mttr(fail, hours, by="component").set_index("component")
    assert comp.loc["pitch", "mtbf_h"] == pytest.approx(1770.0 / 2)
    fleet = wind.mtbf_mttr(fail, hours, by=None)
    assert fleet["failures"].iloc[0] == 3 and fleet["downtime_h"].iloc[0] == 230.0
    both = wind.mtbf_mttr(fail, hours, by=["turbine_id", "component"])
    assert len(both) == 3
    none = wind.mtbf_mttr(fail.iloc[:0], pd.Series({"T9": 100.0}))
    assert none["failures"].iloc[0] == 0 and np.isnan(none["mtbf_h"].iloc[0])
    assert wind.operating_time(windows.iloc[:0], stops).empty


def test_crow_amsaa_closed_form_single_window() -> None:
    hours = [20.0, 70.0, 160.0, 300.0, 520.0, 800.0, 1200.0, 1700.0]
    f = pd.DataFrame({"turbine_id": "T1", "start": [_ts(h) for h in hours]})
    w = pd.DataFrame({"turbine_id": ["T1"], "start": [_ts(0)], "end": [_ts(2000)]})
    ca = wind.crow_amsaa(f, w)
    beta = len(hours) / np.sum(np.log(2000.0 / np.array(hours)))
    assert ca.beta == pytest.approx(beta, rel=1e-8)
    assert ca.lam == pytest.approx(len(hours) / 2000.0**beta, rel=1e-6)
    assert ca.beta_lo < ca.beta < ca.beta_hi
    assert ca.hours == pytest.approx(2000.0) and ca.units == 1
    assert ca.intensity([100.0])[0] == pytest.approx(ca.lam * ca.beta * 100.0 ** (ca.beta - 1))
    d = ca.as_dict()
    assert d["verdict"] == ca.verdict and d["unit_years"] == pytest.approx(2000 / 8766)


def test_crow_amsaa_recovers_trend_over_several_units() -> None:
    rng = np.random.default_rng(7)
    beta_true, lam = 1.6, 1e-4
    rows, wins = [], []
    for u in range(8):
        s, t_end = 200.0 * u, 4000.0
        # power-law NHPP: N(t) = lam t^beta; invert cumulative exponential arrivals
        cum = lam * s**beta_true + np.cumsum(rng.exponential(1.0, 200))
        ages = (cum / lam) ** (1 / beta_true)
        ages = ages[ages < t_end]
        origin = _ts(-s)
        rows += [{"turbine_id": f"T{u}", "start": origin + pd.Timedelta(hours=a)} for a in ages]
        wins.append({"turbine_id": f"T{u}", "start": _ts(0), "end": origin + pd.Timedelta(hours=t_end)})
    origin_map = {f"T{u}": _ts(-200.0 * u) for u in range(8)}
    ca = wind.crow_amsaa(pd.DataFrame(rows), pd.DataFrame(wins), origin=origin_map, conf=0.99)
    assert abs(ca.beta - beta_true) < 0.3  # about 450 failures: standard error of beta near 0.1
    assert ca.beta_lo < beta_true < ca.beta_hi
    assert ca.verdict == "deteriorating" and ca.laplace_agrees


def test_crow_amsaa_degenerate_cases() -> None:
    w = pd.DataFrame({"turbine_id": ["T1"], "start": [_ts(0)], "end": [_ts(100)]})
    few = wind.crow_amsaa(pd.DataFrame({"turbine_id": ["T1"], "start": [_ts(5)]}), w)
    assert np.isnan(few.beta) and few.verdict == "too few failures" and not few.laplace_agrees
    flat = wind.crow_amsaa(pd.DataFrame({"turbine_id": "T1", "start": [_ts(h) for h in (10, 30, 50, 70, 90)]}), w)
    assert flat.verdict == "no significant trend"


def test_laplace_by_hand() -> None:
    u, p = wind.laplace_test([25.0, 50.0, 75.0], [0.0, 0.0, 0.0], [100.0, 100.0, 100.0])
    assert u == pytest.approx(0.0) and p == pytest.approx(1.0)
    assert np.isnan(wind.laplace_test([], [], [])[0])
    # observed clock: two windows of 50 h joined; a failure at 10 h into the second one is at 60 h
    f = pd.DataFrame({"turbine_id": ["T1"], "start": [_ts(110)]})
    w = pd.DataFrame({"turbine_id": ["T1", "T1"], "start": [_ts(0), _ts(100)], "end": [_ts(50), _ts(150)]})
    u2, _ = wind.laplace_observed(f, w)
    assert u2 == pytest.approx((60.0 - 50.0) / np.sqrt(100.0**2 / 12))
