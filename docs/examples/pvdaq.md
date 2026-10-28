# PVDAQ system 34

[`examples/pvdaq_system34.py`](https://github.com/kwwaleszczynski-del/res-asset-kpi/blob/main/examples/pvdaq_system34.py)
runs the PV part of the library on NREL PVDAQ system 34: a 146.64 kWp rooftop array in Las Vegas with one 135 kW
inverter (PV Data Acquisition database, OEDI, CC BY 4.0), 2016-2018 at 15 minutes. The repository holds no data;
the script reads the long table that the companion project
[pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge) builds from the public files.

```bash
python examples/pvdaq_system34.py --data <folder with long_34.parquet> --out results/
```

## Reading the data

The file is a long table with the met station's own signal names and units, in local standard time
(`Etc/GMT+8`, no DST) and stamped at the end of each interval. The mapping converts both:

```python
MAPPING = {
    "format": "long", "timezone": "Etc/GMT+8", "label": "end", "step": "15min",
    "signals": {
        "poa_irradiance": {"signal": "poa", "unit": "W/m2"},
        "ambient_temp_f": {"signal": "t_amb", "unit": "degF"},
        "module_temp_f": {"signal": "t_mod", "unit": "degF"},
        "wind_speed_mph": {"signal": "wind_speed", "unit": "mph"},
    },
    "devices": {"MET": "PLANT"},
}
wide = schema.to_wide(schema.read(raw, MAPPING))
```

## Data quality

Six hours of identical values count as stuck (zeros of irradiance and wind at night are ignored):

| signal | missing | stuck | usable share |
|---|---:|---:|---:|
| poa | 4 715 | 0 | 0.955 |
| t_amb | 9 948 | 31 511 | 0.606 |
| t_mod | 4 400 | 0 | 0.958 |
| wind_speed | 4 400 | 1 138 | 0.947 |
| p_ac | 2 205 | 0 | 0.979 |

The ambient temperature sensor sits frozen for weeks at a time, as pv-budget-bridge also found. The flagged
values are set to missing, so the cell temperature (PVsyst heat balance, \(U_c = 21\), \(U_v = 1.35\)) and with it
the temperature-corrected ratios use only intervals with a real reading.

## Results

Daylight intervals (POA above 50 W/m²); \(T_\mathrm{typ} = 56.1\) °C is the irradiance-weighted cell temperature of
the three years, standing in for a typical year. The inverter counts as running above 2 % of its rating; the
expected power for the energy basis comes from a temperature-corrected model scaled to the plant's median ratio of
measured to modelled power.

| year | energy (MWh) | H_POA (kWh/m²) | PR | PR_25 | PR_wc | A_time | A_energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2016 | 175.0 | 2039 | 0.585 | 0.677 | 0.583 | 0.948 | 0.946 |
| 2017 | 178.3 | 1988 | 0.612 | 0.712 | 0.612 | 0.994 | 0.999 |
| 2018 | 124.3 | 1405 | 0.603 | 0.698 | 0.601 | 0.994 | 0.999 |

Over the year \(\mathrm{PR}_\mathrm{wc} \approx \mathrm{PR}\), as it should be, while \(\mathrm{PR}_{25}\) is ten
points higher: in the desert sun the cells run around 56 °C, far from 25 °C. Month by month the weather-corrected
ratio removes the summer dip of the plain one:

| month | PR | PR_25 | PR_wc | A_time | A_energy | lost (kWh) |
|---|---:|---:|---:|---:|---:|---:|
| 2016-05 | 0.632 | 0.730 | 0.628 | 0.994 | 0.999 | 15 |
| 2016-06 | 0.119 | 0.150 | 0.127 | 0.195 | 0.202 | 10 147 |
| 2016-07 | 0.595 | 0.764 | 0.648 | 0.990 | 0.998 | 42 |
| 2016-08 | 0.605 | 0.753 | 0.641 | 0.989 | 0.999 | 24 |

June 2016 is the inverter outage that pv-budget-bridge reports as the main availability loss of the year; the
energy basis prices it at about 10 MWh. 2018 holds fewer months of data.

The absolute PR is low because the met station's irradiance sensor reads 14-22 % above both a clear-sky model and
ERA5 (pv-budget-bridge, which rescales the sensor before its bridge, reports PR 0.63-0.67). The library does not
rescale anything on its own: that is an analyst's decision, made explicitly and recorded.
