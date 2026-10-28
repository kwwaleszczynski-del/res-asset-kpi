# Kelmarsh wind farm

[`examples/kelmarsh.py`](https://github.com/kwwaleszczynski-del/res-asset-kpi/blob/main/examples/kelmarsh.py)
runs the wind part of the library on the Kelmarsh wind farm (UK): six Senvion MM92 turbines of 2050 kW, whose
10-minute SCADA data and status logs Cubico Sustainable Investments published on Zenodo (record 16807551,
CC BY 4.0). The repository holds no data; the script reads the yearly Parquet files that the companion project
[wind-power-curve-kelmarsh](https://github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh) builds from the
record.

```bash
python examples/kelmarsh.py --data <folder holding processed/> --out results/
```

It is an example of the API on real data, not a study: the operator's IEC categories are taken as logged, and a
failure is any stop the log files as a forced outage (overlapping ones merged). The companion project
[wind-om-reliability](https://github.com/kwwaleszczynski-del/wind-om-reliability) builds failure episodes with far
more care (restart attempts, fleet-wide events, code mappings) and therefore counts fewer failures; the
[validation](../validation.md) runs the library on its episodes and reproduces its numbers exactly.

## Availability by category

Stops overlap in the log (a fault during a maintenance visit, a grid event during a fault). The stops of each
turbine are split with `intervals.first_up`, which gives every moment to the stop that began first, so that no
hour is counted twice; the rest of the year is full performance.

| year | `iec_operational` | `oem_warranty` |
|---|---:|---:|
| 2016 (from 15 April) | 0.9921 | 0.9960 |
| 2017 | 0.9923 | 0.9964 |
| 2018 | 0.9940 | 0.9982 |
| 2019 | 0.9953 | 0.9976 |
| 2020 | 0.9916 | 0.9947 |
| 2021 | 0.9678 | 0.9724 |

The warranty view is higher every year because it excuses grid, environmental and requested stops and 60 hours
of scheduled maintenance per turbine-year.

## Power curves

Records without a logged stop and with positive power, density-normalised (site elevation 150 m), binned at
0.5 m/s:

| turbine | 2017: P(8 m/s) kW | 2017: AEP at 6.5 m/s (MWh) | 2021: P(8 m/s) kW | 2021: AEP at 6.5 m/s (MWh) |
|---|---:|---:|---:|---:|
| T1 | 979.8 | 6125 | 994.9 | 6167 |
| T2 | 1040.4 | 6369 | 1067.3 | 6466 |
| T3 | 1025.5 | 6314 | 1023.9 | 6299 |
| T4 | 1030.3 | 6316 | 1003.7 | 6200 |
| T5 | 1061.0 | 6428 | 1057.4 | 6390 |
| T6 | 1032.3 | 6294 | 1037.7 | 6283 |

All twelve data sets are complete in the sense of the standard. The script rebuilds the wind speed as the mean
of the two nacelle anemometers: the SCADA "wind speed" changes definition in April 2018 and would otherwise fake a
power curve improvement of several percent, a finding of wind-power-curve-kelmarsh. With that fixed, the curves of
2017 and 2021 agree within 3 %.

## Reliability

Operating time is the period since commercial operation minus every stop.

| turbine | failures | operating h | MTBF h (90 % interval) | MTTR h | median TTR h |
|---|---:|---:|---:|---:|---:|
| T1 | 113 | 48 788 | 432 (369-508) | 2.46 | 0.25 |
| T2 | 106 | 49 033 | 463 (394-547) | 0.85 | 0.20 |
| T3 | 79 | 48 796 | 618 (512-751) | 1.73 | 0.18 |
| T4 | 96 | 49 000 | 510 (431-609) | 0.97 | 0.18 |
| T5 | 79 | 48 475 | 614 (509-746) | 4.81 | 0.26 |
| T6 | 120 | 48 262 | 402 (346-471) | 3.06 | 0.18 |

Most forced outages are cleared in minutes (automatic resets); a few long ones carry the mean. Crow-AMSAA over
the fleet gives \(\beta = 0.883\) (90 % interval 0.825-0.944) from 593 failures in 34.3 turbine-years; the
likelihood-ratio test rejects \(\beta = 1\) (p = 0.002), so failures thin out with age. The Laplace test on the
observed-time clock does not confirm it (U = -0.62, p = 0.53): the trend is mild, and the two tests answer
slightly different questions (a power law against any monotone change of the rate). With failure episodes built
as in wind-om-reliability, both tests agree (\(\beta = 0.82\)).
