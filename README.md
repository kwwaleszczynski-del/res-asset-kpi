# res-asset-kpi

[![tests](https://github.com/kwwaleszczynski-del/res-asset-kpi/actions/workflows/ci.yml/badge.svg)](https://github.com/kwwaleszczynski-del/res-asset-kpi/actions/workflows/ci.yml)
[![docs](https://github.com/kwwaleszczynski-del/res-asset-kpi/actions/workflows/docs.yml/badge.svg)](https://kwwaleszczynski-del.github.io/res-asset-kpi/)
[![PyPI](https://img.shields.io/pypi/v/res-asset-kpi.svg)](https://pypi.org/project/res-asset-kpi/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23742958.svg)](https://doi.org/10.5281/zenodo.23742958)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Key performance indicators for operating **PV plants, wind turbines and battery storage**, and the contract logic
around them: guaranteed values, exclusions and liquidated damages.

Asset managers, O&M contractors and lenders' engineers compute the same indicators for every portfolio: the
performance ratio, availability as the contract defines it, the bridge from budget to actual production, power
curves, failure rates, round-trip efficiency, cycles. This library writes them once, with explicit units, explicit
time handling (UTC grids, local reporting periods, the DST hours) and a test against a hand-worked example for
every formula. Its code comes from six open analysis projects and reproduces their results exactly
([validation](https://kwwaleszczynski-del.github.io/res-asset-kpi/validation/)).

> [!WARNING]
> **Not a certified implementation.** The IEC standards referred to (IEC 61724-1, IEC 61400-12-1, IEC 61400-26,
> IEC 62933-2-1) are not freely available. The methods follow public descriptions and the cited literature, and no
> text of the standards is reproduced. The contract examples are illustrative and not legal advice.

**Documentation:** https://kwwaleszczynski-del.github.io/res-asset-kpi/

## What it covers

| Asset | Indicators |
|---|---|
| all | long and wide input with a YAML mapping, explicit units, UTC grids for 5/10/15/30/60-minute data, the repeated and the skipped DST hour, local periods and contract years, data quality flags |
| PV | performance ratio, PR corrected to 25 °C, weather-corrected PR; capacity-weighted time and energy availability with exclusions and a maintenance allowance; budget-to-actual energy bridge that closes exactly |
| Wind | time and energy availability by IEC 61400-26 information category; power curve by the method of bins, AEP; MTBF and MTTR with confidence bounds, Crow-AMSAA, Laplace test |
| Battery | round-trip efficiency corrected for the state of charge, with and without auxiliaries; LTSA availability; equivalent full cycles, rainflow; capacity tests with GUM uncertainty |
| Contracts | guarantees, step-down, tolerance, damages, bonus and caps from YAML |

## Install

```bash
pip install res-asset-kpi
```

Python 3.11+; depends on numpy, pandas, scipy and PyYAML.

## Examples

Performance ratio, plain and corrected to 25 °C cell temperature:

```python
from res_asset_kpi import pv

t = pv.pr_terms(p_ac_kw=[48.0, 66.0], poa_wm2=[600.0, 900.0], p_stc_kw=100.0, interval_h=0.25,
                t_cell_c=[40.0, 55.0], gamma_per_k=-0.004)
t.pr, t.pr_temperature_corrected            # (0.76, 0.8407...)
```

```math
\mathrm{PR} = \frac{\sum_i E_i}{\sum_i P_\mathrm{STC}\, G_i / G_\mathrm{STC}\, \Delta t}
\qquad
\mathrm{PR}_{25} = \frac{\sum_i E_i}{\sum_i P_\mathrm{STC}\, G_i / G_\mathrm{STC}\,\left[1 + \gamma\,(T_{c,i} - 25)\right] \Delta t}
```

A logger export in local time, with the autumn hour logged twice, onto a regular UTC grid:

```python
import pandas as pd
from res_asset_kpi import schema

long = schema.read(pd.read_csv("export.csv"), "mapping.yaml")   # units converted, DST resolved
wide = schema.to_wide(long)                                      # index: interval start, UTC
```

Wind availability as an illustrative turbine warranty counts it, and the cost of calling grid stops faults:

```python
from res_asset_kpi import wind

totals = wind.category_totals(categories, interval_h=1 / 6, lost_kwh=lost, actual_kwh=actual)
wind.availability(totals, "oem_warranty", allowance_h=60).time
```

Round-trip efficiency of a battery per month, corrected for the change of stored energy $`\Delta E`$ with the
one-way efficiency $`\eta`$:

```math
\mathrm{RTE} = \frac{E_\mathrm{out} + \eta\,\Delta E}{E_\mathrm{in}} \quad (\Delta E \ge 0),
\qquad
\mathrm{RTE} = \frac{E_\mathrm{out}}{E_\mathrm{in} + |\Delta E| / \eta} \quad (\Delta E < 0)
```

```python
from res_asset_kpi import bess

monthly = bess.window_rte(site, "M", reference_energy=100.0, aux_metering="behind_meter")
monthly[["rte", "rte_excl_aux", "efc", "valid"]]
```

Two worked examples run on public data: NREL PVDAQ system 34
([`examples/pvdaq_system34.py`](examples/pvdaq_system34.py)) and the Kelmarsh wind farm
([`examples/kelmarsh.py`](examples/kelmarsh.py)). The repository holds no data; the scripts read the files that
the companion projects download.

## Quality

- Typed API (mypy strict), numpydoc docstrings whose examples run as doctests.
- Hand-worked tests for every formula; property tests with Hypothesis (the energy bridge closes, availabilities
  stay in $`[0, 1]`$, rainflow preserves the path length, interval operations count every moment once).
- Coverage above 90 %, CI on Python 3.11-3.13 and on the oldest supported dependencies.
- [`validation/parity.py`](validation/parity.py) reproduces the source projects' results: 41 checks over about
  one million values, largest relative difference $`1.3 \cdot 10^{-13}`$.

## Source projects

| Project | Contributed |
|---|---|
| [pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge) | PR family, PV contract tests, energy bridge, input schema |
| [pv-outage-classifier](https://github.com/kwwaleszczynski-del/pv-outage-classifier) | event treatments in PV availability |
| [wind-downtime-availability](https://github.com/kwwaleszczynski-del/wind-downtime-availability) | IEC 61400-26 categories and availability |
| [wind-power-curve-kelmarsh](https://github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh) | power curve binning, AEP |
| [wind-om-reliability](https://github.com/kwwaleszczynski-del/wind-om-reliability) | MTBF, MTTR, Crow-AMSAA, Laplace |
| [bess-ops-warranty](https://github.com/kwwaleszczynski-del/bess-ops-warranty) | RTE, LTSA availability, cycles, capacity tests, damages |

## Used by

These projects install the library from PyPI and compute their indicators with it. The switch left every figure in
their READMEs unchanged.

| Project | Uses |
|---|---|
| [pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge) | `pv.pr_terms`, `pv.availability`, `pv.shapley_product`, `contracts.assess` |
| [wind-downtime-availability](https://github.com/kwwaleszczynski-del/wind-downtime-availability) | `wind.category_totals`, `wind.availability` |
| [wind-power-curve-kelmarsh](https://github.com/kwwaleszczynski-del/wind-power-curve-kelmarsh) | `wind.air_density`, `wind.bin_power_curve`, `wind.aep_rayleigh` |
| [wind-om-reliability](https://github.com/kwwaleszczynski-del/wind-om-reliability) | `wind.operating_time`, `wind.mtbf_mttr`, `wind.crow_amsaa`, `intervals` |
| [bess-ops-warranty](https://github.com/kwwaleszczynski-del/bess-ops-warranty) | `bess.interval_flows`, `bess.corrected_rte`, `bess.ltsa_availability`, `bess.rainflow`, `bess.evaluate_capacity_test`, `contracts.damages_amount` |

## Citing

Please cite the software through its Zenodo DOI, [10.5281/zenodo.23742958](https://doi.org/10.5281/zenodo.23742958)
(all versions; each release also has its own DOI, e.g. 0.1.0:
[10.5281/zenodo.23742959](https://doi.org/10.5281/zenodo.23742959)). See also [`CITATION.cff`](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
