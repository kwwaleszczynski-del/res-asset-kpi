# res-asset-kpi

Key performance indicators for operating **PV plants, wind turbines and battery storage**, with the contract
logic around them: guaranteed values, exclusions and liquidated damages.

The library collects calculations that asset managers, O&M contractors and lenders' engineers repeat for every
portfolio, written once with explicit units, explicit time handling and tests that check each formula against a
hand-worked example. Its code comes from six open analysis projects (see [Validation](validation.md)); this
package gives them one tested, typed API and reproduces their numbers exactly.

!!! warning "Not a certified implementation"

    The IEC standards referred to here (IEC 61724-1, IEC 61400-12-1, IEC 61400-26, IEC 62933-2-1) are not freely
    available. The methods are implemented from public descriptions and the literature cited on each page, and
    no text of the standards is reproduced. Results are not certified and must not replace a measurement under
    the standard where a contract requires one. Contract examples are illustrative, not legal advice.

## What it covers

| Asset | Indicators | Module |
|---|---|---|
| all | long and wide input with a YAML mapping, explicit units, UTC grids, DST, 5/10/15/30/60-minute intervals, data quality flags | [`schema`](api/core.md), [`timeseries`](api/core.md), [`units`](api/core.md), [`quality`](api/core.md) |
| PV | performance ratio, PR corrected to 25 °C, weather-corrected PR; time and energy availability with exclusions; budget-to-actual energy bridge that closes exactly | [`pv`](api/pv.md) |
| Wind | time and energy availability by IEC 61400-26 information category; power curves by the method of bins, AEP; MTBF, MTTR, Crow-AMSAA, Laplace test | [`wind`](api/wind.md) |
| Battery | round-trip efficiency corrected for the state of charge, with and without auxiliaries; LTSA availability; equivalent full cycles, rainflow; capacity tests with uncertainty | [`bess`](api/bess.md) |
| Contracts | guarantees, step-down, tolerance, damages and caps from YAML | [`contracts`](api/contracts.md) |

## Install

```bash
pip install res-asset-kpi
```

Releases are on [PyPI](https://pypi.org/project/res-asset-kpi/) and archived on Zenodo:
[10.5281/zenodo.23742958](https://doi.org/10.5281/zenodo.23742958).

Python 3.11 or newer; the dependencies are numpy, pandas, scipy and PyYAML.

## A first example

Performance ratio of a 100 kWp array over two quarter-hours, plain and corrected to 25 °C cell temperature:

```python
from res_asset_kpi import pv

terms = pv.pr_terms(
    p_ac_kw=[48.0, 66.0],          # mean AC power per interval
    poa_wm2=[600.0, 900.0],        # plane-of-array irradiance
    p_stc_kw=100.0,                # DC rating
    interval_h=0.25,
    t_cell_c=[40.0, 55.0],         # cell temperature
    gamma_per_k=-0.004,            # power temperature coefficient
)
terms.pr, terms.pr_temperature_corrected      # (0.76, 0.8407...)
```

From a logger export to a regular grid in UTC, with the repeated autumn hour placed correctly:

```python
import pandas as pd
from res_asset_kpi import schema

raw = pd.read_csv("export.csv")                       # one column per signal, local time stamps
long = schema.read(raw, "mapping.yaml")               # see res_asset_kpi/data/mapping_wide.example.yaml
wide = schema.to_wide(long)                           # columns (device, signal), index = interval start, UTC
```

The [guide](guide/data.md) explains the data model, the [methods](methods/pv.md) pages give every formula, and the
two [examples](examples/pvdaq.md) run the library on public data.

## Design

- **Explicit units.** Parameter names carry the unit (`p_stc_kw`, `poa_wm2`, `interval_h`); input tables are
  converted to canonical units once, by the mapping.
- **Explicit time.** Every series is indexed by UTC interval starts; reporting periods are taken in local time
  (`Europe/Warsaw` by default), so a day has 23 or 25 hours at the DST changes and months have their true length.
- **No hidden data handling.** Quality checks flag and never change data; contract treatments (excluded, deemed
  available, allowance) are parameters, not assumptions.
- **Tested formulas.** Every formula has a hand-worked test; properties such as the closure of the energy bridge,
  availability in \([0, 1]\) and the path length of rainflow counting are checked with Hypothesis.
