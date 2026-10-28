"""Input data model: a long table of measurements, a wide table with a YAML column mapping, canonical signals.

**Long format** (the native one), one row per time stamp, device and signal:

| column | meaning |
|---|---|
| timestamp | start (or end, or centre: see ``label``) of the averaging interval |
| device_id | any string: an inverter, a turbine, a meter, ``PLANT`` for site signals |
| signal | a canonical signal name ([`SIGNALS`][res_asset_kpi.schema.SIGNALS]) or a name mapped to one |
| value | number in the canonical unit of the signal, or in the unit the mapping states |

**Wide format**: one row per time stamp and one column per device and signal, e.g. a logger export. A mapping
names the time stamp column and says, for every column used, which device and signal it holds and in which
unit. Mappings are plain dictionaries or YAML files:

    format: wide
    timestamp_column: Timestamp
    timezone: Europe/Warsaw     # naive local time with DST; ISO strings with offsets are read exactly
    label: end                  # the logger stamps the end of each average
    step: 15min                 # output grid
    columns:
      "Irradiance POA [W/m2]": {signal: poa, unit: W/m2}
      "INV01 Pac [W]":         {device: INV01, signal: p_ac, unit: W}

For a long table the mapping renames signals and devices instead:

    format: long
    timezone: Etc/GMT+8
    signals:
      ambient_temp_f: {signal: t_amb, unit: degF}
      P_POI_kW:       {signal: p_meter, unit: kW, sign: -1}
    devices: {MET: PLANT}

``read`` turns either form into a tidy long table with UTC time stamps and canonical units; ``to_wide`` puts it
on a regular grid with one column per ``(device, signal)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from res_asset_kpi import timeseries, units

__all__ = [
    "LONG_COLUMNS",
    "SIGNALS",
    "ColumnSpec",
    "Mapping",
    "SchemaError",
    "SignalInfo",
    "SignalSpec",
    "load_mapping",
    "read",
    "to_long",
    "to_wide",
    "validate_long",
]

LONG_COLUMNS = ("timestamp", "device_id", "signal", "value")
"""Columns of the long format, in order."""


class SchemaError(ValueError):
    """Input data or a mapping does not follow the data model."""


@dataclass(frozen=True)
class SignalInfo:
    """Canonical unit, aggregation rule and meaning of a signal."""

    unit: str
    aggregation: timeseries.Aggregation
    description: str


SIGNALS: dict[str, SignalInfo] = {
    # meteorology
    "poa": SignalInfo("W/m2", "mean", "plane-of-array irradiance"),
    "ghi": SignalInfo("W/m2", "mean", "global horizontal irradiance"),
    "t_amb": SignalInfo("degC", "mean", "ambient air temperature"),
    "t_mod": SignalInfo("degC", "mean", "back-of-module temperature"),
    "t_cell": SignalInfo("degC", "mean", "PV cell temperature"),
    "wind_speed": SignalInfo("m/s", "mean", "wind speed (nacelle anemometer or met mast)"),
    "wind_dir": SignalInfo("deg", "circular", "wind direction"),
    "pressure": SignalInfo("Pa", "mean", "air pressure"),
    "precip": SignalInfo("mm", "sum", "precipitation per interval"),
    # electrical, PV and wind
    "p_ac": SignalInfo("kW", "mean", "AC active power, positive = export"),
    "p_dc": SignalInfo("kW", "mean", "DC power, positive = discharge or generation"),
    "e_export": SignalInfo("kWh", "sum", "exported energy per interval"),
    "e_import": SignalInfo("kWh", "sum", "imported energy per interval"),
    "setpoint": SignalInfo("fraction", "mean", "active power limit as a share of the rating"),
    "rotor_speed": SignalInfo("rpm", "mean", "rotor speed"),
    "pitch": SignalInfo("deg", "mean", "blade pitch angle"),
    "nacelle_dir": SignalInfo("deg", "circular", "nacelle position (yaw)"),
    # battery storage
    "p_meter": SignalInfo("kW", "mean", "active power at the revenue meter, positive = export"),
    "p_aux": SignalInfo("kW", "mean", "auxiliary consumption, positive = consumption"),
    "soc": SignalInfo("fraction", "instant", "state of charge"),
    "e_stored": SignalInfo("kWh", "instant", "stored energy"),
    "p_avail_dis": SignalInfo("kW", "mean", "available discharge power"),
    "p_avail_ch": SignalInfo("kW", "mean", "available charge power (magnitude)"),
    "e_avail": SignalInfo("kWh", "mean", "available (dischargeable) energy"),
    "t_cell_avg": SignalInfo("degC", "mean", "mean battery cell temperature"),
    "t_cell_max": SignalInfo("degC", "mean", "highest battery cell temperature"),
}
"""Canonical signals: unit, aggregation rule (how intervals are combined) and meaning."""


@dataclass(frozen=True)
class ColumnSpec:
    """What one column of a wide table holds."""

    signal: str
    device: str | None = None
    unit: str | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class SignalSpec:
    """How a signal name of a long table maps to a canonical signal."""

    signal: str
    unit: str | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class Mapping:
    """Column and signal mapping of an input table (see the module description).

    Attributes
    ----------
    format : {"long", "wide"}
    timestamp_column : str
        Name of the time stamp column.
    timezone : str
        Time zone of naive time stamps.
    label : {"start", "end", "center"}
        Which instant of its interval a time stamp marks.
    step : str or None
        Output interval length; ``None`` takes the coarsest interval found.
    plant_device : str
        Device id given to wide columns without a ``device``.
    columns : dict of str to ColumnSpec
        Wide format: the columns used.
    signals : dict of str to SignalSpec
        Long format: renamed signals (others must already be canonical).
    devices : dict of str to str
        Long format: renamed devices.
    """

    format: Literal["long", "wide"] = "long"
    timestamp_column: str = "timestamp"
    timezone: str = "Europe/Warsaw"
    label: timeseries.Label = "start"
    step: str | None = None
    plant_device: str = "PLANT"
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    signals: dict[str, SignalSpec] = field(default_factory=dict)
    devices: dict[str, str] = field(default_factory=dict)


_MAPPING_KEYS = {"format", "timestamp_column", "timezone", "label", "step", "plant_device", "columns", "signals",
                 "devices"}


def _spec(cls: type[Any], name: str, raw: Any) -> Any:
    raw = {"signal": raw} if isinstance(raw, str) else dict(raw or {})
    allowed = set(cls.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise SchemaError(f"{name}: unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")
    spec = cls(**raw)
    if spec.signal not in SIGNALS:
        raise SchemaError(f"{name}: {spec.signal!r} is not a canonical signal; known: {sorted(SIGNALS)}")
    if spec.unit is not None and units.dimension(spec.unit) != units.dimension(SIGNALS[spec.signal].unit):
        raise SchemaError(f"{name}: unit {spec.unit!r} does not measure {spec.signal} ({SIGNALS[spec.signal].unit})")
    return spec


def load_mapping(source: str | Path | dict[str, Any]) -> Mapping:
    """Read a mapping from a YAML file or a dictionary and check it.

    Parameters
    ----------
    source : str, pathlib.Path or dict
        Path of a YAML file, or the parsed content.

    Returns
    -------
    Mapping

    Raises
    ------
    SchemaError
        Unknown keys, unknown signals, or units of the wrong dimension.

    Examples
    --------
    >>> m = load_mapping({"format": "wide", "timestamp_column": "Time",
    ...                   "columns": {"Pac [W]": {"device": "INV1", "signal": "p_ac", "unit": "W"}}})
    >>> m.columns["Pac [W]"].unit
    'W'
    """
    raw = source if isinstance(source, dict) else yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - _MAPPING_KEYS
    if unknown:
        raise SchemaError(f"mapping: unknown keys {sorted(unknown)}; allowed {sorted(_MAPPING_KEYS)}")
    fmt = raw.get("format", "long")
    if fmt not in ("long", "wide"):
        raise SchemaError("mapping: format must be 'long' or 'wide'")
    label = raw.get("label", "start")
    if label not in ("start", "end", "center"):
        raise SchemaError("mapping: label must be 'start', 'end' or 'center'")
    if raw.get("step") is not None:
        timeseries.check_step(raw["step"])
    columns = {str(k): _spec(ColumnSpec, f"column {k!r}", v) for k, v in (raw.get("columns") or {}).items()}
    signals = {str(k): _spec(SignalSpec, f"signal {k!r}", v) for k, v in (raw.get("signals") or {}).items()}
    if fmt == "wide" and not columns:
        raise SchemaError("mapping: a wide table needs 'columns'")
    return Mapping(
        format=fmt,
        timestamp_column=raw.get("timestamp_column", "timestamp"),
        timezone=raw.get("timezone", "Europe/Warsaw"),
        label=label,
        step=raw.get("step"),
        plant_device=raw.get("plant_device", "PLANT"),
        columns=columns,
        signals=signals,
        devices={str(k): str(v) for k, v in (raw.get("devices") or {}).items()},
    )


def validate_long(df: pd.DataFrame) -> pd.DataFrame:
    """Check a long table and return a clean copy.

    The copy has exactly the columns ``timestamp``, ``device_id``, ``signal`` (strings) and ``value`` (float;
    non-numeric entries become NaN).

    Raises
    ------
    SchemaError
        Missing columns, or signals that are not canonical.

    Examples
    --------
    >>> raw = pd.DataFrame({"timestamp": ["2024-01-01 00:00"], "device_id": ["INV1"], "signal": ["p_ac"],
    ...                     "value": ["12.5"]})
    >>> float(validate_long(raw)["value"].iloc[0])
    12.5
    """
    missing = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"long format needs columns {list(LONG_COLUMNS)}; missing {missing}")
    out = df[list(LONG_COLUMNS)].copy()
    out["device_id"] = out["device_id"].astype(str)
    out["signal"] = out["signal"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype(float)
    unknown = sorted(set(out["signal"]) - set(SIGNALS))
    if unknown:
        raise SchemaError(f"signals {unknown} are not canonical; map them (see SIGNALS) or rename them")
    return out


def _to_canonical(values: pd.Series, signal: str, unit: str | None, scale: float) -> pd.Series:
    v = values * scale
    return units.convert(v, unit, SIGNALS[signal].unit) if unit is not None else v


def read(raw: pd.DataFrame, mapping: Mapping | dict[str, Any] | str | Path | None = None) -> pd.DataFrame:
    """Turn a long or wide input table into a tidy long table with UTC time stamps and canonical units.

    Parameters
    ----------
    raw : pandas.DataFrame
        Input table.
    mapping : Mapping, dict, str or pathlib.Path, optional
        The mapping (see [`load_mapping`][res_asset_kpi.schema.load_mapping]). Without one the table must already
        be a long table with canonical signals and naive ``Europe/Warsaw`` or offset-carrying time stamps.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` (UTC, as given by ``label``), ``device_id``, ``signal``, ``value``. Rows whose
        time stamp cannot be placed (DST) are dropped with a warning in the log.

    Examples
    --------
    >>> wide = pd.DataFrame({"Time": ["2024-07-01 12:00", "2024-07-01 12:15"], "Pac [W]": [1000.0, 2000.0]})
    >>> m = {"format": "wide", "timestamp_column": "Time",
    ...      "columns": {"Pac [W]": {"device": "INV1", "signal": "p_ac", "unit": "W"}}}
    >>> read(wide, m)
                      timestamp device_id signal  value
    0 2024-07-01 10:00:00+00:00      INV1   p_ac    1.0
    1 2024-07-01 10:15:00+00:00      INV1   p_ac    2.0
    """
    m = mapping if isinstance(mapping, Mapping) else load_mapping(mapping) if mapping is not None else Mapping()
    if m.timestamp_column not in raw.columns:
        raise SchemaError(f"time stamp column {m.timestamp_column!r} not found")
    if m.format == "wide":
        frames = []
        for col, spec in m.columns.items():
            if col not in raw.columns:
                raise SchemaError(f"mapped column {col!r} not in the table")
            values = pd.to_numeric(raw[col], errors="coerce").astype(float)
            frames.append(pd.DataFrame({
                "timestamp": raw[m.timestamp_column].to_numpy(),
                "device_id": spec.device or m.plant_device,
                "signal": spec.signal,
                "value": _to_canonical(values, spec.signal, spec.unit, spec.scale).to_numpy(),
            }))
        long = pd.concat(frames, ignore_index=True)
    else:
        long = raw.rename(columns={m.timestamp_column: "timestamp"})
        missing = [c for c in LONG_COLUMNS if c not in long.columns]
        if missing:
            raise SchemaError(f"long format needs columns {list(LONG_COLUMNS)}; missing {missing}")
        long = long[list(LONG_COLUMNS)].copy().reset_index(drop=True)
        long["device_id"] = long["device_id"].astype(str).replace(m.devices)
        long["signal"] = long["signal"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="coerce").astype(float)
        original = long["signal"].copy()  # renames must not chain (a -> b, then b -> c)
        for name, sspec in m.signals.items():
            sel = original == name
            if sel.any():
                long.loc[sel, "value"] = _to_canonical(long.loc[sel, "value"], sspec.signal, sspec.unit, sspec.scale)
                long.loc[sel, "signal"] = sspec.signal
    long = validate_long(long)
    long["timestamp"] = timeseries.to_utc(long["timestamp"], m.timezone, long[["device_id", "signal"]])
    long = long.dropna(subset=["timestamp"]).reset_index(drop=True)
    long.attrs["label"] = m.label
    if m.step is not None:
        long.attrs["step"] = m.step
    return long


def to_wide(
    long: pd.DataFrame,
    step: str | pd.Timedelta | None = None,
    label: timeseries.Label | None = None,
) -> pd.DataFrame:
    """Put a tidy long table on one regular UTC grid, one column per ``(device, signal)``.

    Each device and signal is regularised on its own (see [`regularise`][res_asset_kpi.timeseries.regularise]) with the
    aggregation rule of the signal, then all are aligned on the union of their spans.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of [`read`][res_asset_kpi.schema.read] (or any long table with UTC time stamps and canonical signals).
    step : str or pandas.Timedelta, optional
        Output interval; default: ``long.attrs["step"]`` or the coarsest interval found.
    label : {"start", "end", "center"}, optional
        Default: ``long.attrs["label"]`` or ``"start"``.

    Returns
    -------
    pandas.DataFrame
        Index: interval starts (UTC); columns: ``MultiIndex`` (device, signal). ``attrs["step"]`` holds the
        interval length.

    Examples
    --------
    >>> long = pd.DataFrame({"timestamp": pd.date_range("2024-07-01 10:00", periods=4, freq="5min", tz="UTC"),
    ...                      "device_id": "INV1", "signal": "p_ac", "value": [1.0, 2.0, 3.0, 4.0]})
    >>> wide = to_wide(long, "15min")
    >>> wide[("INV1", "p_ac")].tolist()
    [2.0, 4.0]
    >>> wide.index[0]
    Timestamp('2024-07-01 10:00:00+0000', tz='UTC')
    """
    df = long.dropna(subset=["timestamp"])
    lab = label or long.attrs.get("label", "start")
    steps = {
        k: timeseries.check_step(timeseries.infer_step(g)) if g.nunique() > 1 else None
        for k, g in df.groupby(["device_id", "signal"], sort=True)["timestamp"]
    }
    known = [s for s in steps.values() if s is not None]
    want = step if step is not None else long.attrs.get("step")
    target = timeseries.check_step(want) if want is not None else max(known) if known else None
    if target is None:
        raise SchemaError("cannot infer the interval: give step")
    series = {}
    for (dev, sig), g in df.groupby(["device_id", "signal"], sort=True):
        s = pd.Series(g["value"].to_numpy(float), index=pd.DatetimeIndex(g["timestamp"]))
        series[(dev, sig)] = timeseries.regularise(
            s, target, source_step=steps[(dev, sig)] or target, label=lab, how=SIGNALS[sig].aggregation
        )
    starts = [s.index.min() for s in series.values() if len(s)]
    ends = [s.index.max() for s in series.values() if len(s)]
    if not starts:
        raise SchemaError("no values to put on a grid")
    grid = pd.date_range(min(starts), max(ends), freq=target, name="interval_start")
    wide = pd.DataFrame({k: s.reindex(grid) for k, s in series.items()}, index=grid)
    wide.columns = pd.MultiIndex.from_tuples(list(series), names=["device", "signal"])
    wide.attrs["step"] = target
    return wide


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Inverse of [`to_wide`][res_asset_kpi.schema.to_wide]: back to the long format (missing values dropped).

    Examples
    --------
    >>> idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    >>> wide = pd.DataFrame({("INV1", "p_ac"): [1.0, float("nan")]}, index=idx)
    >>> to_long(wide)
                      timestamp device_id signal  value
    0 2024-01-01 00:00:00+00:00      INV1   p_ac    1.0
    """
    cols = wide.columns if isinstance(wide.columns, pd.MultiIndex) else pd.MultiIndex.from_tuples(wide.columns)
    frames = [
        pd.DataFrame({"timestamp": wide.index, "device_id": dev, "signal": sig,
                      "value": wide[(dev, sig)].to_numpy(float)})
        for dev, sig in cols
    ]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(LONG_COLUMNS))
    return out.dropna(subset=["value"]).reset_index(drop=True)
