"""Units, time handling (DST, grids, periods), schema and mappings, quality flags, interval operations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from res_asset_kpi import intervals, quality, schema, timeseries, units
from res_asset_kpi.contracts import example_path

# ----------------------------------------------------------------------------------------------------- units


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (2500.0, "W", "kW", 2.5),
        (1.0, "MWh", "kWh", 1000.0),
        (3.6, "MJ", "kWh", 1.0),
        (212.0, "degF", "degC", 100.0),
        (0.0, "degC", "K", 273.15),
        (36.0, "km/h", "m/s", 10.0),
        (10.0, "mph", "m/s", 4.4704),
        (45.0, "%", "fraction", 0.45),
        (1013.25, "hPa", "Pa", 101325.0),
        (1.0, "kW/m2", "W/m²", 1000.0),
        (2.0, "cm", "mm", 20.0),
    ],
)
def test_convert(value: float, src: str, dst: str, expected: float) -> None:
    assert units.convert(value, src, dst) == pytest.approx(expected)


def test_convert_rejects_other_dimension_and_unknown_unit() -> None:
    with pytest.raises(ValueError, match="cannot convert"):
        units.convert(1.0, "kW", "kWh")
    with pytest.raises(ValueError, match="unknown unit"):
        units.normalise_unit("furlong")


def test_convert_keeps_pandas_types() -> None:
    s = pd.Series([1000.0, np.nan], index=["a", "b"])
    out = units.convert(s, "W", "kW")
    assert isinstance(out, pd.Series)
    assert out["a"] == 1.0 and np.isnan(out["b"])
    assert units.canonical_unit("GWh") == "kwh"
    assert units.normalise_unit("degK") == "k"


@given(st.floats(-1e6, 1e6), st.sampled_from(["degC", "degF", "K"]), st.sampled_from(["degC", "degF", "K"]))
def test_convert_round_trip(x: float, a: str, b: str) -> None:
    assert units.convert(units.convert(x, a, b), b, a) == pytest.approx(x, rel=1e-9, abs=1e-6)


# ------------------------------------------------------------------------------------------------ timeseries


def test_to_utc_resolves_autumn_hour_per_group() -> None:
    ts = ["2024-10-27 02:00", "2024-10-27 02:00", "2024-10-27 02:00", "2024-10-27 02:00"]
    groups = pd.Series(["A", "A", "B", "B"])
    out = timeseries.to_utc(ts, "Europe/Warsaw", groups)
    assert [t.strftime("%H:%M") for t in out] == ["00:00", "01:00", "00:00", "01:00"]


def test_to_utc_drops_unplaceable_times(caplog: pytest.LogCaptureFixture) -> None:
    out = timeseries.to_utc(["2024-03-31 02:30", "2024-10-27 02:30", "2024-10-27 03:30"])
    assert out.isna().tolist() == [True, True, False]
    assert "skipped spring hour" in caplog.text
    triple = timeseries.to_utc(["2024-10-27 02:15"] * 3)
    assert triple.isna().tolist() == [False, False, True]


def test_to_utc_offsets_utc_and_errors() -> None:
    exact = timeseries.to_utc(["2024-10-27T02:30:00+01:00", "2024-10-27T02:30:00+02:00"])
    assert exact.iloc[0] - exact.iloc[1] == pd.Timedelta("1h")
    naive = timeseries.to_utc(pd.Series(["01.07.2024 12:00"]), "UTC", fmt="%d.%m.%Y %H:%M")
    assert naive.iloc[0] == pd.Timestamp("2024-07-01 12:00", tz="UTC")
    aware = timeseries.to_utc(pd.Series(pd.date_range("2024-01-01", periods=2, freq="h", tz="Europe/Warsaw")))
    assert str(aware.dt.tz) == "UTC"
    mixed = timeseries.to_utc(pd.Series(["2024-01-01T00:00:00", "2024-01-01 00:15"]), "UTC")
    assert mixed.notna().all()
    with pytest.raises(ValueError, match="some time stamps"):
        timeseries.to_utc(["2024-01-01T00:00+01:00", "2024-01-01 00:15"])


def test_step_helpers() -> None:
    assert timeseries.check_step("1h") == pd.Timedelta("60min")
    with pytest.raises(ValueError, match="not supported"):
        timeseries.check_step("7min")
    with pytest.raises(ValueError, match="fewer than two"):
        timeseries.infer_step(pd.DatetimeIndex(["2024-01-01"]))
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"], tz="UTC")
    assert timeseries.interval_hours(idx) == pytest.approx(1 / 6)


def test_regularise_down_up_and_kinds() -> None:
    idx = pd.date_range("2024-06-01 00:00", periods=4, freq="15min", tz="UTC")
    mean = timeseries.regularise(pd.Series([1.0, 2.0, 3.0, 6.0], index=idx), "1h")
    assert mean.tolist() == [3.0]
    energy = timeseries.regularise(pd.Series([1.0, 2.0, np.nan, 6.0], index=idx), "1h", how="sum")
    assert energy.tolist() == [9.0]
    hourly = pd.Series([4.0, 8.0], index=pd.date_range("2024-06-01", periods=2, freq="1h", tz="UTC"))
    spread = timeseries.regularise(hourly, "15min", how="sum")
    assert spread.tolist() == [1.0] * 4 + [2.0] * 4
    held = timeseries.regularise(hourly, "15min")
    assert held.tolist() == [4.0] * 4 + [8.0] * 4
    soc = timeseries.regularise(hourly, "15min", how="instant")
    assert soc.tolist() == [4.0, 5.0, 6.0, 7.0, 8.0]
    soc_down = timeseries.regularise(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx), "30min", how="instant")
    assert soc_down.tolist() == [1.0, 3.0]
    wind = timeseries.regularise(pd.Series([350.0, 10.0, 20.0, 340.0], index=idx), "1h", how="circular")
    assert wind.iloc[0] == pytest.approx(0.0, abs=1e-9) or wind.iloc[0] == pytest.approx(360.0)


def test_regularise_labels_duplicates_and_errors() -> None:
    idx = pd.DatetimeIndex(["2024-06-01 00:10", "2024-06-01 00:20", "2024-06-01 00:20"], tz="UTC")
    s = timeseries.regularise(pd.Series([1.0, 2.0, 4.0], index=idx), "10min", label="end")
    assert s.index[0] == pd.Timestamp("2024-06-01 00:00", tz="UTC")
    assert s.tolist() == [1.0, 3.0]
    c = timeseries.regularise(pd.Series([1.0, 2.0], index=pd.DatetimeIndex(
        ["2024-06-01 00:05", "2024-06-01 00:15"], tz="UTC")), "10min", label="center")
    assert c.index[0] == pd.Timestamp("2024-06-01 00:00", tz="UTC")
    gap = timeseries.regularise(pd.Series([1.0, 2.0], index=pd.DatetimeIndex(
        ["2024-06-01 00:00", "2024-06-01 00:30"], tz="UTC")), "10min", source_step="10min")
    assert np.isnan(gap.iloc[1]) and len(gap) == 4
    with pytest.raises(ValueError, match="time-zone aware"):
        timeseries.regularise(pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="h")), "1h")
    with pytest.raises(ValueError, match="neither divides"):
        timeseries.regularise(pd.Series([1.0, 2.0], index=pd.date_range(
            "2024-01-01", periods=2, freq="10min", tz="UTC")), "15min")
    empty = timeseries.regularise(pd.Series([np.nan, np.nan], index=pd.date_range(
        "2024-01-01", periods=2, freq="1h", tz="UTC")), "1h", how="instant")
    assert empty.empty


def test_period_labels_and_bounds_across_dst() -> None:
    idx = pd.DatetimeIndex(["2024-02-15 12:00", "2024-12-31 23:30"], tz="UTC")
    assert list(timeseries.period_labels(idx, "D")) == ["2024-02-15", "2025-01-01"]
    assert list(timeseries.period_labels(idx, "Q")) == ["2024-Q1", "2025-Q1"]
    assert list(timeseries.period_labels(idx, "Y")) == ["2024", "2025"]
    with pytest.raises(ValueError, match="unknown period"):
        timeseries.period_labels(idx, "W")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time-zone aware"):
        timeseries.period_labels(pd.DatetimeIndex(["2024-01-01"]), "M")
    months = timeseries.period_bounds("2024-01-01 00:00+01:00", "2025-01-01 00:00+01:00")
    assert dict(zip(months["period"], months["hours"], strict=True))["2024-03"] == 743.0
    assert dict(zip(months["period"], months["hours"], strict=True))["2024-10"] == 745.0
    assert months["hours"].sum() == 8784.0
    days = timeseries.period_bounds("2024-10-26 22:00", "2024-10-28 23:00", "D")
    assert days["hours"].tolist() == [25.0, 24.0]
    year = timeseries.period_bounds(pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-04-01", tz="UTC"), "Y")
    assert year["period"].tolist() == ["2024"]
    q = timeseries.period_bounds("2024-01-01 00:00+01:00", "2024-07-01 00:00+02:00", "Q")
    assert q["period"].tolist() == ["2024-Q1", "2024-Q2"]


def test_contract_years() -> None:
    idx = pd.DatetimeIndex(["2022-12-31 22:00", "2023-12-31 22:59", "2023-12-31 23:00"], tz="UTC")
    assert timeseries.contract_years(idx, pd.Timestamp("2023-01-01", tz="Europe/Warsaw")).tolist() == [0, 1, 2]
    assert timeseries.contract_years(pd.DatetimeIndex([], tz="UTC"), "2023-01-01").tolist() == []


# ---------------------------------------------------------------------------------------------------- schema


def test_example_mappings_load() -> None:
    wide = schema.load_mapping(example_path("mapping_wide"))
    assert wide.format == "wide" and wide.label == "end" and wide.columns["INV01 Pac [W]"].device == "INV01"
    long = schema.load_mapping(example_path("mapping_long"))
    assert long.signals["SOC_PCT"].unit == "%" and long.devices == {"BMS-01": "BESS"}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"colour": "red"}, "unknown keys"),
        ({"format": "tall"}, "format"),
        ({"label": "middle"}, "label"),
        ({"step": "7min"}, "not supported"),
        ({"format": "wide"}, "needs 'columns'"),
        ({"signals": {"x": {"signal": "temperature"}}}, "not a canonical"),
        ({"signals": {"x": {"signal": "p_ac", "unit": "kWh"}}}, "does not measure"),
        ({"signals": {"x": {"signal": "p_ac", "colour": 1}}}, "unknown keys"),
    ],
)
def test_mapping_errors(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        schema.load_mapping(raw)


def test_read_wide_with_units_and_dst() -> None:
    raw = pd.DataFrame({
        "Time": ["2024-10-27 02:15", "2024-10-27 02:15", "2024-10-27 03:30"],
        "Pac [W]": [1000.0, 2000.0, "n/a"],
        "T [F]": [32.0, 50.0, 68.0],
    })
    m = {"format": "wide", "timestamp_column": "Time",
         "columns": {"Pac [W]": {"device": "INV1", "signal": "p_ac", "unit": "W"},
                     "T [F]": {"signal": "t_amb", "unit": "degF"}}}
    long = schema.read(raw, m)
    p = long[long["signal"] == "p_ac"]
    assert p["timestamp"].dt.strftime("%H:%M").tolist() == ["00:15", "01:15", "02:30"]
    assert p["value"].iloc[0] == 1.0 and np.isnan(p["value"].iloc[2])
    assert long.loc[long["signal"] == "t_amb", "value"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert set(long.loc[long["signal"] == "t_amb", "device_id"]) == {"PLANT"}


def test_read_long_mapping_and_errors() -> None:
    raw = pd.DataFrame({"ts": ["2024-01-01 00:00", "2024-01-01 00:00"], "device_id": ["BMS-01", "M1"],
                        "signal": ["SOC_PCT", "P_POI_kW"], "value": [50.0, -10.0]})
    m = {"timestamp_column": "ts", "timezone": "UTC",
         "signals": {"SOC_PCT": {"signal": "soc", "unit": "%"}, "P_POI_kW": {"signal": "p_meter", "unit": "kW"}},
         "devices": {"BMS-01": "BESS"}}
    long = schema.read(raw, m)
    assert long.set_index("signal")["value"].to_dict() == {"soc": 0.5, "p_meter": -10.0}
    assert long.set_index("signal")["device_id"]["soc"] == "BESS"
    chain = pd.DataFrame({"timestamp": ["2024-01-01"], "device_id": ["X"], "signal": ["t_amb"], "value": [50.0]})
    out = schema.read(chain, {"timezone": "UTC", "signals": {"t_amb": {"signal": "t_amb", "unit": "degF"}}})
    assert out["value"].iloc[0] == pytest.approx(10.0)
    with pytest.raises(schema.SchemaError, match="time stamp column"):
        schema.read(raw, {"timestamp_column": "time"})
    with pytest.raises(schema.SchemaError, match="missing"):
        schema.read(raw.drop(columns="value"), {"timestamp_column": "ts"})
    with pytest.raises(schema.SchemaError, match="not canonical"):
        schema.read(raw.rename(columns={"ts": "timestamp"}))
    with pytest.raises(schema.SchemaError, match="not in the table"):
        schema.read(pd.DataFrame({"timestamp": ["2024-01-01"]}),
                    {"format": "wide", "columns": {"P": {"signal": "p_ac"}}})
    with pytest.raises(schema.SchemaError, match="missing"):
        schema.validate_long(pd.DataFrame({"timestamp": []}))


def test_to_wide_mixed_intervals_and_round_trip() -> None:
    t5 = pd.date_range("2024-07-01 10:00", periods=6, freq="5min", tz="UTC")
    t15 = pd.date_range("2024-07-01 10:00", periods=2, freq="15min", tz="UTC")
    long = pd.concat([
        pd.DataFrame({"timestamp": t5, "device_id": "INV1", "signal": "p_ac", "value": np.arange(6.0)}),
        pd.DataFrame({"timestamp": t15, "device_id": "PLANT", "signal": "e_export", "value": [3.0, 6.0]}),
        pd.DataFrame({"timestamp": t15[:1], "device_id": "PLANT", "signal": "poa", "value": [500.0]}),
    ], ignore_index=True)
    wide = schema.to_wide(long)
    assert wide.attrs["step"] == pd.Timedelta("15min")
    assert wide[("INV1", "p_ac")].tolist() == [1.0, 4.0]
    assert wide[("PLANT", "e_export")].tolist() == [3.0, 6.0]
    back = schema.to_long(wide)
    assert len(back) == 5 and set(back.columns) == set(schema.LONG_COLUMNS)
    fine = schema.to_wide(long, "5min")
    assert fine[("PLANT", "e_export")].iloc[:3].tolist() == [1.0, 1.0, 1.0]
    plain = pd.DataFrame({("A", "p_ac"): [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    plain.columns = list(plain.columns)
    assert schema.to_long(plain)["device_id"].tolist() == ["A"]
    with pytest.raises(schema.SchemaError, match="cannot infer"):
        schema.to_wide(long.iloc[-1:])
    with pytest.raises(schema.SchemaError, match="no values"):
        schema.to_wide(long.iloc[:0], "15min")


def test_read_then_to_wide_uses_mapping_step_and_label() -> None:
    raw = pd.DataFrame({"Timestamp": ["2024-07-01 12:05", "2024-07-01 12:10", "2024-07-01 12:15"],
                        "INV01 Pac [W]": [3000.0, 6000.0, 9000.0]})
    m = {"format": "wide", "timestamp_column": "Timestamp", "label": "end", "step": "15min",
         "columns": {"INV01 Pac [W]": {"device": "INV01", "signal": "p_ac", "unit": "W"}}}
    wide = schema.to_wide(schema.read(raw, m))
    assert wide.index[0] == pd.Timestamp("2024-07-01 10:00", tz="UTC")
    assert wide[("INV01", "p_ac")].tolist() == [6.0]


# --------------------------------------------------------------------------------------------------- quality


def test_quality_flags_and_helpers() -> None:
    s = pd.Series([1.0, 1.0, 1.0, np.nan, -50.0, 2.0])
    f = quality.quality_flags(s, "poa", stuck_run=3)
    assert f.tolist() == [2, 2, 2, 1, 4, 0]
    assert quality.quality_flags(s, "poa", lower=-100.0).tolist()[4] == 0
    assert quality.apply_flags(s, f).isna().sum() == 5
    assert quality.apply_flags(s, f, quality.Flag.MISSING).isna().sum() == 1
    assert quality.coverage(f) == pytest.approx(1 / 6)
    frame = pd.DataFrame({"a": f, "b": np.zeros(6, dtype=int)})
    assert quality.coverage(frame).tolist() == pytest.approx([1 / 6, 1.0])
    assert quality.summary(f | int(quality.Flag.FILLED))["FILLED"] == 6
    assert str(quality.Flag.STUCK) == "Flag.STUCK"
    assert "STUCK" in str(quality.Flag.STUCK | quality.Flag.MISSING)
    assert quality.flag_stuck([], 3).empty
    with pytest.raises(ValueError, match="min_run"):
        quality.flag_stuck([1.0], 1)
    assert quality.flag_out_of_range(np.array([np.nan, 5.0]), upper=1.0).tolist() == [0, 4]


@given(st.lists(st.sampled_from([0.0, 1.0, np.nan]), max_size=60), st.integers(2, 6))
def test_stuck_runs_are_long_enough(values: list[float], n: int) -> None:
    v = np.asarray(values)
    flagged = quality.flag_stuck(v, n).to_numpy() != 0
    for i in np.flatnonzero(flagged):
        j, k = i, i
        while j > 0 and v[j - 1] == v[i]:
            j -= 1
        while k < len(v) - 1 and v[k + 1] == v[i]:
            k += 1
        assert k - j + 1 >= n


# ------------------------------------------------------------------------------------------------- intervals


def test_first_up_and_overlap() -> None:
    seg = intervals.first_up(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:30", "2024-01-01 05:00"],
        ["2024-01-01 03:00", "2024-01-01 02:00", "2024-01-01 04:00", "2024-01-01 06:00"],
    )
    assert seg.groupby("owner")["hours"].sum().to_dict() == {0: 3.0, 2: 1.0, 3: 1.0}
    assert intervals.first_up([], []).empty


@given(st.lists(st.tuples(st.integers(0, 1000), st.integers(0, 200)), max_size=25))
def test_union_and_first_up_count_every_moment_once(spans: list[tuple[int, int]]) -> None:
    s = np.array([a for a, _ in spans], dtype=np.int64) * 10**9
    e = s + np.array([d for _, d in spans], dtype=np.int64) * 10**9
    us, ue = intervals.union(s, e)
    assert (np.diff(us) > 0).all() and (us[1:] > ue[:-1]).all()
    covered = set()
    for a, d in spans:
        covered.update(range(a, a + d))
    assert int((ue - us).sum() // 10**9) == len(covered)
    seg = intervals.first_up(pd.to_datetime(s, utc=True), pd.to_datetime(e, utc=True))
    assert seg["hours"].sum() * 3600 == pytest.approx(len(covered))
    rs, re_ = intervals.subtract([0], [2000 * 10**9], s, e)
    assert int((re_ - rs).sum() // 10**9) == 2000 - len(covered)
