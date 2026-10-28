"""PV: performance ratios, availability and the energy bridge; hand-worked examples and properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from res_asset_kpi import pv


def test_pr_family_by_hand() -> None:
    # 100 kWp, two 15-minute intervals: 600 W/m2 at 40 degC and 900 W/m2 at 55 degC, gamma -0.4 %/K
    p = [48.0, 66.0]
    g = [600.0, 900.0]
    tc = [40.0, 55.0]
    t = pv.pr_terms(p, g, 100.0, 0.25, t_cell_c=tc, gamma_per_k=-0.004, t_typ_c=45.0)
    e = (48.0 + 66.0) * 0.25  # 28.5 kWh
    ref = 100.0 * (0.6 + 0.9) * 0.25  # 37.5 kWh
    ref25 = 100.0 * (0.6 * (1 - 0.004 * 15) + 0.9 * (1 - 0.004 * 30)) * 0.25
    refwc = 100.0 * (0.6 * (1 + 0.004 * 5) + 0.9 * (1 - 0.004 * 10)) * 0.25
    assert t.energy_kwh == pytest.approx(e)
    assert t.insolation_kwh_m2 == pytest.approx(0.375)
    assert t.pr == pytest.approx(e / ref)
    assert t.pr_temperature_corrected == pytest.approx(e / ref25)
    assert t.pr_weather_corrected == pytest.approx(e / refwc)
    assert t.as_dict()["intervals"] == 2.0


def test_pr_missing_values_and_mask() -> None:
    t = pv.pr_terms([50.0, np.nan, 40.0, 30.0], [500.0, 800.0, 400.0, 20.0], 100.0, 1.0,
                    mask=np.array([True, True, True, False]))
    assert t.intervals == 2 and t.pr == pytest.approx(90.0 / 90.0)
    assert np.isnan(t.pr_temperature_corrected) and np.isnan(t.pr_weather_corrected)
    assert np.isnan(pv.performance_ratio([0.0], [0.0], 100.0, 1.0))
    no_typ = pv.pr_terms([50.0], [500.0], 100.0, 1.0, t_cell_c=[30.0], gamma_per_k=-0.004)
    assert np.isfinite(no_typ.pr_temperature_corrected) and np.isnan(no_typ.pr_weather_corrected)
    with pytest.raises(ValueError, match="same length"):
        pv.pr_terms([1.0], [1.0, 2.0], 1.0, 1.0)
    with pytest.raises(ValueError, match="t_cell_c"):
        pv.pr_terms([1.0], [1.0], 1.0, 1.0, t_cell_c=[1.0, 2.0], gamma_per_k=-0.004)


def test_typical_temperature_makes_annual_pr_wc_equal_pr() -> None:
    rng = np.random.default_rng(1)
    g = rng.uniform(50, 1000, 500)
    tc = 10 + g / 30 + rng.normal(0, 2, 500)
    p = 0.0009 * g * 100 * (1 - 0.004 * (tc - 25))
    t_typ = pv.typical_cell_temperature(g, tc)
    t = pv.pr_terms(p, g, 100.0, 1.0, t_cell_c=tc, gamma_per_k=-0.004, t_typ_c=t_typ)
    assert t.pr_weather_corrected == pytest.approx(t.pr, rel=1e-12)
    assert t.pr_temperature_corrected == pytest.approx(0.9, rel=1e-12)
    assert pv.typical_cell_temperature([100.0, np.nan], [20.0, 30.0], mask=[True, True]) == 20.0


def test_pr_by_period() -> None:
    frame = pd.DataFrame({"p_ac": [40.0, 80.0, 90.0], "poa": [500.0, 1000.0, 1000.0], "t_cell": [25.0] * 3})
    out = pv.pr_by_period(frame, ["a", "a", "b"], 100.0, 1.0, gamma_per_k=-0.004, t_typ_c=25.0)
    assert out.loc["a", "pr"] == pytest.approx(0.8)
    assert out.loc["b", "pr_25"] == pytest.approx(0.9)
    assert out.loc["b", "pr_wc"] == pytest.approx(0.9)


def test_cell_temperature_models() -> None:
    assert pv.cell_temperature_from_module([30.0], [1000.0], 2.0).tolist() == [32.0]
    tc = pv.cell_temperature_pvsyst([800.0], [20.0], [2.0], u_c=25.0, u_v=1.2)
    assert tc[0] == pytest.approx(20.0 + 0.9 * 800.0 * 0.9 / (25.0 + 2.4))


def test_availability_by_hand_with_exclusions_and_allowance() -> None:
    # 3 inverters of 100, 100 and 200 kWp; 6 daylight hours, 1 night hour
    running = pd.DataFrame({
        "A": [True, True, False, False, True, True, True],
        "B": [True, True, True, True, False, np.nan, True],
        "C": [True, False, True, True, True, True, False],
    })
    poa = [300.0] * 6 + [0.0]
    excluded = np.zeros((7, 3), dtype=bool)
    excluded[1, 2] = True  # C down in hour 1 during a grid outage
    planned = pd.Series([False, False, True, True, False, False, False])  # A planned for 2 h, plant-wide flag
    a = pv.availability(running, 1.0, capacity=[100, 100, 200], poa_wm2=poa, excluded=excluded,
                        planned=planned, planned_allowance_h=0.5,
                        expected_kw=np.full((7, 3), [80.0, 80.0, 160.0]))
    row = a.loc["all"]
    w = np.array([0.25, 0.25, 0.5])
    # planned: weighted daylight hours 1.0 per hour for known units; hour 2 within allowance? cum 1.0 > 0.5 -> no
    counted = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=float)
    up = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=float)
    assert row["availability_time"] == pytest.approx((up @ w).sum() / (counted @ w).sum())
    lim = np.array([80.0, 80.0, 160.0])
    lost = ((counted - up) * lim).sum()
    assert row["lost_kwh"] == pytest.approx(lost)
    assert row["availability_energy"] == pytest.approx(1 - lost / (counted * lim).sum())
    assert row["planned_h"] == pytest.approx(2.0) and row["planned_counted_h"] == pytest.approx(2.0)
    assert row["excluded_share"] == pytest.approx(0.5 / 5.75)


def test_availability_allowance_and_periods() -> None:
    running = pd.DataFrame({"A": [False, False, True, False]})
    planned = [True, True, False, True]
    a = pv.availability(running, 1.0, planned=planned, planned_allowance_h=1.0, periods=["m1", "m1", "m1", "m2"])
    assert a.loc["m1", "availability_time"] == pytest.approx(0.5)  # first planned hour excused, second counted
    assert a.loc["m2", "availability_time"] == pytest.approx(0.0) or np.isnan(a.loc["m2", "availability_time"])
    assert np.isnan(a.loc["m1", "availability_energy"])


def test_availability_errors() -> None:
    running = pd.DataFrame({"A": [True], "B": [True]})
    with pytest.raises(ValueError, match="capacity"):
        pv.availability(running, 1.0, capacity=[1.0])
    with pytest.raises(ValueError, match="one value per interval"):
        pv.availability(running, 1.0, excluded=[True, False])
    with pytest.raises(ValueError, match="shape"):
        pv.availability(running, 1.0, excluded=np.ones((1, 3), dtype=bool))
    with pytest.raises(ValueError, match="expected_kw"):
        pv.availability(running, 1.0, expected_kw=np.ones((2, 2)))
    with pytest.raises(ValueError, match="periods"):
        pv.availability(running, 1.0, periods=["a", "b"])


@settings(max_examples=60, deadline=None)
@given(
    arrays(np.float64, (24, 3), elements=st.sampled_from([0.0, 1.0, np.nan])),
    arrays(np.bool_, (24, 3)),
    arrays(np.float64, (24, 3), elements=st.floats(0.0, 500.0)),
    st.floats(0.0, 10.0),
)
def test_availability_is_a_share(state: np.ndarray, excl: np.ndarray, lim: np.ndarray, allowance: float) -> None:
    a = pv.availability(pd.DataFrame(state), 0.5, capacity=[1, 2, 3], excluded=excl, planned=~excl,
                        planned_allowance_h=allowance, expected_kw=lim).iloc[0]
    for col in ("availability_time", "availability_energy"):
        assert np.isnan(a[col]) or 0.0 <= a[col] <= 1.0
    assert 0.0 <= a["stopped_h"] <= a["counted_h"] + 1e-12


def test_bridge_sequential_by_hand() -> None:
    b = pv.energy_bridge(
        e_budget=[1000.0], e_actual=[920.0],
        weather={"irradiance": [1.05], "temperature": [0.99]},
        losses_actual={"soiling": [0.03], "availability": [0.02]},
        losses_budget={"soiling": [0.02], "availability": [0.01]},
        index=["2024-07"],
    ).loc["2024-07"]
    assert b["irradiance"] == pytest.approx(50.0)
    assert b["temperature"] == pytest.approx(1050.0 * -0.01)
    e_w = 1000 * 1.05 * 0.99
    x_a = 920.0 / (0.97 * 0.98)
    x_b = e_w / (0.98 * 0.99)
    assert b["soiling"] == pytest.approx(x_b * 0.02 - x_a * 0.03)
    assert b["availability"] == pytest.approx(x_b * 0.98 * 0.01 - x_a * 0.97 * 0.02)
    assert b["residual"] == pytest.approx(x_a - x_b)  # losses equal budget -> residual is the potential gap
    assert abs(b["closure"]) < 1e-9


def test_bridge_shapley_and_fallback(caplog: pytest.LogCaptureFixture) -> None:
    kw = {"weather": {"irradiance": 0.9}, "losses_actual": {"availability": 0.97},
          "losses_budget": {"availability": 0.01}}
    b = pv.energy_bridge(1000.0, 25.0, **kw)
    assert "almost no production" in caplog.text
    assert b["potential_actual"].iloc[0] == pytest.approx(b["potential_budget"].iloc[0])
    sh = pv.energy_bridge(1000.0, 25.0, method="shapley", **kw)
    bars = ["irradiance", "availability", "residual"]
    assert sh[bars].sum(axis=1).iloc[0] == pytest.approx(25.0 - 1000.0)
    with pytest.raises(ValueError, match="same keys"):
        pv.energy_bridge(1.0, 1.0, {}, {"a": 0.1}, {"b": 0.1})
    with pytest.raises(ValueError, match="method"):
        pv.energy_bridge(1.0, 1.0, {}, {}, {}, method="lmdi")
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        pv.energy_bridge(1.0, 1.0, {}, {"a": 1.0}, {"a": 0.0})
    with pytest.raises(ValueError, match="one value per period"):
        pv.energy_bridge([1.0, 2.0], [1.0, 2.0], {"x": [1.0, 1.0, 1.0]}, {}, {})


def test_shapley_two_factors_by_hand() -> None:
    # v({}) = 100, v({1}) = 110, v({2}) = 90, v({1,2}) = 99: phi_1 = (10 + 9) / 2, phi_2 = (-10 - 11) / 2
    assert pv.shapley_product(100.0, [1.1, 0.9]).tolist() == pytest.approx([9.5, -10.5])


fraction = st.floats(0.0, 0.6)


@settings(max_examples=150, deadline=None)
@given(
    st.floats(1.0, 1e7), st.floats(0.0, 1e7),
    st.lists(st.floats(0.5, 1.5), min_size=0, max_size=4),
    st.lists(st.tuples(fraction, fraction), min_size=0, max_size=6),
    st.sampled_from(["sequential", "shapley"]),
)
def test_bridge_closes_exactly(e_b: float, e_a: float, ratios: list[float],
                               losses: list[tuple[float, float]], method: str) -> None:
    weather = {f"w{i}": r for i, r in enumerate(ratios)}
    la = {f"l{i}": a for i, (a, _) in enumerate(losses)}
    lb = {f"l{i}": b for i, (_, b) in enumerate(losses)}
    b = pv.energy_bridge(e_b, e_a, weather, la, lb, method=method)
    bars = [*weather, *la, "residual"]
    total = float(b[bars].sum(axis=1).iloc[0])
    assert total == pytest.approx(e_a - e_b, rel=1e-9, abs=1e-6 * max(e_a, e_b))
