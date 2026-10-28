# Data, time and units

## Long and wide tables

The native input is a **long table**, one row per time stamp, device and signal:

| timestamp | device_id | signal | value |
|---|---|---|---|
| 2024-10-27 02:15 | INV01 | p_ac | 142.6 |
| 2024-10-27 02:15 | PLANT | poa | 0.0 |

Signal names are canonical (`p_ac`, `poa`, `t_amb`, `soc`, ... see `schema.SIGNALS` in the
[API reference](../api/core.md)), each with a canonical unit and an aggregation rule. A **wide table** (one column
per signal, as most loggers export) is read with a mapping that says, for every column, which device and signal
it holds and in which unit:

```yaml
format: wide
timestamp_column: Timestamp
timezone: Europe/Warsaw     # naive local time with DST; ISO strings with offsets are read exactly
label: end                  # the logger stamps the end of each average
step: 15min                 # output grid
columns:
  "Irradiance POA [W/m2]": {signal: poa, unit: W/m2}
  "INV01 Pac [W]":         {device: INV01, signal: p_ac, unit: W}
  "PPC limit [%]":         {signal: setpoint, unit: "%"}
```

A long table whose names or units differ is mapped the same way, with `signals` (raw name -> canonical signal
and unit) and `devices` (renames). Both example mappings ship with the package
(`contracts.example_path("mapping_wide")`, `contracts.example_path("mapping_long")`).

```python
from res_asset_kpi import schema

long = schema.read(raw, "mapping.yaml")     # tidy long table: UTC time stamps, canonical units
wide = schema.to_wide(long)                 # regular grid, columns (device, signal)
```

## Time

- Every series is indexed by **UTC time stamps marking the start** of the averaging interval. A logger that stamps
  the end (`label: end`) or the centre (`label: center`) is shifted back on reading.
- **Naive local time stamps** are converted with the mapping's time zone (default `Europe/Warsaw`). In autumn the
  hour 02:00-03:00 occurs twice. Within each device and signal, the first occurrence is read as summer time (CEST)
  and the second as winter time (CET), which is what an export in time order contains. A repeated hour seen only
  once, a third occurrence and the skipped spring hour cannot be placed: they become missing and the count is
  logged.
- **Reporting periods** are local: `timeseries.period_labels` gives local days, months, quarters or years, and
  `timeseries.period_bounds` their UTC bounds and lengths (March 743 hours, October 745, a DST day 23 or 25).
  Contract years run from anniversary to anniversary of a start date (`timeseries.contract_years`).

```python
>>> from res_asset_kpi import timeseries
>>> t = timeseries.to_utc(["2024-10-27 01:30", "2024-10-27 02:30", "2024-10-27 02:30", "2024-10-27 03:30"])
>>> [x.strftime("%H:%M") for x in t]          # UTC
['23:30', '00:30', '01:30', '02:30']
```

## Intervals

Signals may come at 5, 10, 15 or 30 minutes or 1 hour, and differently per device. `to_wide` puts each on the
output grid by its aggregation rule:

| rule | signals | coarser grid | finer grid |
|---|---|---|---|
| `mean` | power, irradiance, temperature, speed | mean of the intervals | value repeated |
| `sum` | energy, precipitation | sum | spread evenly |
| `instant` | state of charge, stored energy | value at the grid instant | linear interpolation |
| `circular` | wind direction, nacelle position | circular mean | value repeated |

Steps that do not divide each other (10 and 15 minutes) are refused rather than approximated. Downsampled means
use the intervals present; check the [coverage](quality.md) before trusting a period.

## Units

`units.convert` converts between units of one dimension and refuses to mix dimensions; the mapping converts every
column to the canonical unit of its signal once. Function parameters carry their unit in the name
(`p_stc_kw`, `poa_wm2`, `interval_h`, `temp_c`). Battery functions accept any consistent pair (MW and MWh, or kW
and kWh).

| dimension | canonical | accepted |
|---|---|---|
| power | kW | W, MW, GW |
| energy | kWh | Wh, MWh, GWh, J, MJ, GJ |
| irradiance | W/m2 | kW/m2 |
| temperature | degC | degF, K |
| speed | m/s | km/h, mph, kn |
| fraction | fraction | %, pu |
| pressure | Pa | hPa, kPa, mbar, bar |
