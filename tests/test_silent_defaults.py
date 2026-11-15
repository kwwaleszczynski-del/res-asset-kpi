"""No silent constants: an efficiency, a temperature factor or a category that nobody measured is reported, not assumed.

Each test reproduces an input on which version 0.1.1 returned a plausible number built on a constant (0.95, 1.0, 0 kWh)
without saying so.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from res_asset_kpi import bess, wind

KW = {"reference_energy": 100.0, "aux_metering": "none", "tz": "UTC", "min_efc": 0.5}


def _days(n: int) -> pd.DataFrame:
    """n hourly days; the PCS charges 25 MW and discharges 20 MW for 4 h each: RTE 0.8, back to the start state."""
    idx = pd.date_range("2024-06-01", periods=24 * n, freq="1h", tz="UTC")
    p = np.zeros(24 * n)
    e = np.full(24 * n, 20.0)
    for d in range(n):
        o = 24 * d
        p[o + 2:o + 6] = -25.0
        p[o + 12:o + 16] = 20.0
        e[o + 6:o + 13] = 20.0 + 100.0 * 0.894
        e[o + 13:o + 16] = [20.0 + 89.4 - 22.4 * k for k in (1, 2, 3)]
    return pd.DataFrame({"p_meter": p, "e_stored": e}, index=idx)


def _masked_ends() -> pd.DataFrame:
    """Three days whose stored energy is missing at the start and at the end of the record."""
    site = _days(3)
    site.iloc[:3, 1] = np.nan
    site.iloc[-3:, 1] = np.nan
    return site


def test_auto_eta_is_nan_when_the_energies_do_not_determine_it(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert np.isnan(bess.auto_eta(240.0, 300.0, float("nan")))  # the sum over a record with a masked end
    assert "not determined" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert np.isnan(bess.auto_eta(0.0, 0.0, 0.0))  # no complete window at all
        assert np.isnan(bess.auto_eta(0.0, 100.0, 0.0))  # charge without discharge
    assert caplog.text.count("not determined") == 2
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert bess.auto_eta(81.0, 100.0, 0.0) == pytest.approx(0.9)
    assert caplog.text == ""


def test_auto_eta_says_when_it_limits_the_efficiency(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert bess.auto_eta(130.0, 100.0, 0.0) == 1.0  # more energy out than in
    assert caplog.text.count("exceeds 1") == 1


def test_window_rte_without_a_complete_window_reports_no_efficiency(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        w = bess.window_rte(_masked_ends(), "M", **KW, eta="auto")
    assert np.isnan(w["eta"].iloc[0]) and np.isnan(w["rte"].iloc[0]) and not bool(w["valid"].iloc[0])
    assert "no complete window" in caplog.text
    no_state = bess.window_rte(_days(2).drop(columns="e_stored"), "D", **KW, eta="auto")
    assert no_state["eta"].isna().all() and no_state["rte"].isna().all()


def test_window_rte_takes_the_efficiency_from_the_complete_windows() -> None:
    w = bess.window_rte(_masked_ends(), "D", **KW, eta="auto")
    assert w["valid"].tolist() == [False, True, False]
    assert w["eta"].iloc[0] == pytest.approx(np.sqrt(0.8), rel=2e-3) and w["eta"].nunique() == 1
    assert w["rte"].iloc[1] == pytest.approx(0.8, rel=2e-3)


def test_efficiency_of_one_stage_is_not_taken_as_given_by_the_next() -> None:
    """The chain daily -> monthly (as in a pipeline that passes the daily efficiency on) must not turn NaN into 0.95."""
    site = _masked_ends()
    daily = bess.window_rte(site, "D", **KW, eta="auto")
    eta = float(daily["eta"].iloc[0])
    monthly = bess.window_rte(site.iloc[3:-3], "M", **KW, eta=eta, min_coverage=0.9)
    assert np.isfinite(eta) and monthly["eta"].iloc[0] == eta
    stuck = bess.window_rte(_masked_ends().iloc[:23], "D", **KW, eta="auto")  # no complete day at all
    none = float(stuck["eta"].iloc[0])
    assert np.isnan(none)
    chained = bess.window_rte(site, "M", **KW, eta=none)
    assert np.isnan(chained["eta"].iloc[0]) and np.isnan(chained["rte"].iloc[0])


def test_a_window_that_ends_where_it_started_does_not_need_the_efficiency() -> None:
    assert bess.corrected_rte([80.0], [100.0], [0.0], float("nan"))[0] == pytest.approx(0.8)
    assert np.isnan(bess.corrected_rte([80.0], [100.0], [5.0], float("nan"))[0])
    assert np.isnan(bess.corrected_rte([80.0], [100.0], [-5.0], float("nan"))[0])
    balanced = bess.window_rte(_days(1), "D", **KW, eta=float("nan"))
    assert balanced["rte"].iloc[0] == pytest.approx(0.8, rel=2e-3)  # d_stored is 0: no correction needed


def test_capacity_test_without_a_temperature_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        r = bess.evaluate_capacity_test(91.0, 90.0)
    assert r.temp_corrected is False and r.temp_factor == 1.0 and r.as_dict()["temp_corrected"] is False
    assert "without a test temperature" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        t = bess.evaluate_capacity_test(91.0, 90.0, temp_c=35.0)
    assert t.temp_corrected is True and caplog.text == ""
    # why it matters: the missing temperature term makes the guard-band verdict more confident
    assert r.verdict == "pass" and t.verdict != "pass" and r.u_rel_combined < t.u_rel_combined


def test_category_totals_report_what_they_assume(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        t = wind.category_totals(["forced_outage", None, "forced_outage", "full_performance"], 1.0,
                                 lost_kwh=[100.0, 100.0, np.nan, np.nan], actual_kwh=[0.0, 0.0, 0.0, 50.0])
    assert "1 interval(s) without a category counted as 'forced_outage'" in caplog.text
    assert "1 interval(s) outside full performance have no lost energy" in caplog.text  # the NaN at full performance
    row = t.set_index("category").loc["forced_outage"]  # is expected and does not count
    assert row["hours"] == 3.0 and row["lost_kwh"] == 200.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        wind.category_totals(["forced_outage", "full_performance"], 1.0, lost_kwh=[10.0, np.nan],
                             actual_kwh=[0.0, 5.0])
    assert caplog.text == ""
