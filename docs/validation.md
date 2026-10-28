# Validation

The library's code was extracted from six open analysis projects by the same author (MIT licence). To make sure
nothing changed on the way, `validation/parity.py` runs each source project's own functions and the library on the
**same inputs** and compares the results number by number. Inputs are public data (NREL PVDAQ, the Kelmarsh,
Penmanshiel and Hill of Towie wind farms) or the source projects' synthetic plants.

| Source project | What the library took from it |
|---|---|
| [pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge) | PR family, contract availability and PR tests, energy bridge, input schema and DST handling |
| [pv-outage-classifier](https://github.com/kwwaleszczynski-del/pv-outage-classifier) | treatment of event classes (unavailable, excluded, available) in PV availability, long-format conventions |
| [wind-downtime-availability](https://github.com/kwwaleszczynski-del/wind-downtime-availability) | IEC 61400-26 categories, contract presets, time and energy availability |
| [wind-power-curve-kelmarsh](https://github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh) | air density, normalisation, method of bins, AEP |
| [wind-om-reliability](https://github.com/kwwaleszczynski-del/wind-om-reliability) | operating time, MTBF and MTTR, Crow-AMSAA, Laplace test, interval operations |
| [bess-ops-warranty](https://github.com/kwwaleszczynski-del/bess-ops-warranty) | SoC-corrected RTE, LTSA availability, rainflow and cycles, capacity tests, damages |

## Results

41 checks over 989 293 values; the largest relative difference is \(1.3 \cdot 10^{-13}\), in the Shapley bars
of the energy bridge, where the library computes the residual factor in a different but equivalent order.
Everything else is identical to the last bit.

| Source project | Check | Values compared | Largest relative difference |
|---|---|---:|---:|
| pv-budget-bridge | pl_demo: monthly PR, PR_25, PR_wc | 144 | 0 (identical) |
| pv-budget-bridge | pl_demo: bridge bars, sequential (kWh) | 480 | 0 (identical) |
| pv-budget-bridge | pl_demo: bridge bars, Shapley (kWh) | 480 | 1.8e-14 |
| pv-budget-bridge | pl_demo: contract availability, time and energy | 8 | 0 (identical) |
| pv-budget-bridge | pl_demo: availability damages and bonus | 8 | 0 (identical) |
| pv-budget-bridge | pl_demo: PR test, measured PR_wc and effective guarantee | 8 | 0 (identical) |
| pv-budget-bridge | pl_demo: PR test damages | 4 | 0 (identical) |
| pv-budget-bridge | pvdaq_34: monthly PR, PR_25, PR_wc | 111 | 0 (identical) |
| pv-budget-bridge | pvdaq_34: bridge bars, sequential (kWh) | 360 | 0 (identical) |
| pv-budget-bridge | pvdaq_34: bridge bars, Shapley (kWh) | 360 | 1.3e-13 |
| pv-budget-bridge | pvdaq_34: contract availability, time and energy | 6 | 0 (identical) |
| pv-budget-bridge | pvdaq_34: availability damages and bonus | 7 | 1.7e-16 |
| pv-budget-bridge | pvdaq_34: PR test, measured PR_wc and effective guarantee | 7 | 0 (identical) |
| pv-budget-bridge | pvdaq_34: PR test damages | 3 | 0 (identical) |
| wind-downtime-availability | time, energy availability and lost energy from category totals | 1,974 | 5.5e-16 |
| wind-downtime-availability | time availability vs the published turbine-year table | 658 | 1.2e-16 |
| bess-ops-warranty | RTE windows by local day (energy, dE, RTE with and without aux, EFC, valid) | 5,840 | 0 (identical) |
| bess-ops-warranty | RTE windows by month (energy, dE, RTE with and without aux, EFC, valid) | 192 | 0 (identical) |
| bess-ops-warranty | RTE windows by contract year (energy, dE, RTE with and without aux, EFC, valid) | 16 | 0 (identical) |
| bess-ops-warranty | one-way efficiency auto | 1 | 0 (identical) |
| bess-ops-warranty | measured availability per interval | 70,080 | 0 (identical) |
| bess-ops-warranty | LTSA availability per month | 168 | 0 (identical) |
| bess-ops-warranty | LTSA availability per year | 14 | 0 (identical) |
| bess-ops-warranty | rainflow cycles, hysteresis 0.0 | 63,955 | 0 (identical) |
| bess-ops-warranty | rainflow EFC and depth histogram, hysteresis 0.0 | 15 | 0 (identical) |
| bess-ops-warranty | rainflow cycles, hysteresis 0.01 | 8,795 | 0 (identical) |
| bess-ops-warranty | rainflow EFC and depth histogram, hysteresis 0.01 | 15 | 0 (identical) |
| bess-ops-warranty | throughput per interval, discharge_meter | 70,080 | 0 (identical) |
| bess-ops-warranty | throughput per interval, half_throughput_meter | 70,080 | 0 (identical) |
| bess-ops-warranty | throughput per interval, discharge_dc | 70,080 | 0 (identical) |
| bess-ops-warranty | capacity tests (4): energy, correction, uncertainty, margin | 24 | 0 (identical) |
| bess-ops-warranty | guaranteed retention at equivalent age | 4 | 0 (identical) |
| wind-power-curve-kelmarsh | air density, Kelmarsh 2020 (all records) | 311,918 | 0 (identical) |
| wind-power-curve-kelmarsh | normalised wind speed | 311,918 | 0 (identical) |
| wind-power-curve-kelmarsh | bins (ws, power, std, n, u_A) per turbine | 1,165 | 0 (identical) |
| wind-power-curve-kelmarsh | AEP for Rayleigh 5, 6.5, 8 m/s per turbine (MWh) | 18 | 0 (identical) |
| wind-om-reliability | operating hours per turbine | 6 | 0 (identical) |
| wind-om-reliability | MTBF, CI, MTTR, TTR by turbine_id | 54 | 0 (identical) |
| wind-om-reliability | MTBF, CI, MTTR, TTR by component | 117 | 0 (identical) |
| wind-om-reliability | Crow-AMSAA beta, bounds, lambda, LR p, Laplace (fleet, turbines, components) | 119 | 0 (identical) |
| wind-om-reliability | fleet Crow-AMSAA beta vs the published table | 1 | 0 (identical) |

pv-outage-classifier is not compared number by number: its availability works on the event objects of its own
detector, while the library takes the same information as masks of running, excluded and planned intervals.

## What the extraction found

- **Shapley bridge in near-total outages.** When the measured losses leave almost no production (less than 5 % of
  the potential), pv-budget-bridge falls back to its model's potential, and its Shapley bars then no longer add up
  to the difference between budget and actual. A property test (Hypothesis) found the case in the library's
  first version; the library now takes the residual factor as whatever the other factors leave unexplained, so
  both methods close exactly in every month. The sequential bars, which pv-budget-bridge reports by default, were
  never affected.
- **Trend tests with few failures.** wind-om-reliability returns no Laplace statistic for groups with fewer than
  three failures; the library first did, then was aligned with it (neither a fit nor a trend test means much
  there).

## Running it

Each check runs in the Python environment of its source project, which has that project's dependencies, and
reads that project's data folder:

```bash
python validation/parity.py pv          --source ../pv-budget-bridge --data ../pv-budget-bridge/data
python validation/parity.py wind-avail  --source ../wind-downtime-availability
python validation/parity.py power-curve --source ../wind-power-curve-kelmarsh
python validation/parity.py reliability --source ../wind-om-reliability --data ../wind-power-curve-kelmarsh/data
python validation/parity.py bess        --source ../bess-ops-warranty
```

The script exits with an error when any relative difference exceeds \(10^{-9}\) or a missing value appears on
one side only.
