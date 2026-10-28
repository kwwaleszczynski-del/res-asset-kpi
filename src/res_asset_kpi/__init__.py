"""Key performance indicators for operating PV, wind and battery assets.

Subpackages and modules:

* [`res_asset_kpi.pv`][res_asset_kpi.pv] - performance ratio (plain, temperature-corrected, weather-corrected),
  availability, budget-to-actual energy bridge;
* [`res_asset_kpi.wind`][res_asset_kpi.wind] - availability by IEC 61400-26 categories, power curves by the method of
  bins, MTBF, MTTR, Crow-AMSAA;
* [`res_asset_kpi.bess`][res_asset_kpi.bess] - round-trip efficiency corrected for the state of charge, LTSA
  availability, equivalent cycles and rainflow, capacity tests with uncertainty;
* [`res_asset_kpi.contracts`][res_asset_kpi.contracts] - guarantees, exclusions and liquidated damages from YAML;
* [`res_asset_kpi.schema`][res_asset_kpi.schema], [`res_asset_kpi.timeseries`][res_asset_kpi.timeseries],
  [`res_asset_kpi.units`][res_asset_kpi.units], [`res_asset_kpi.quality`][res_asset_kpi.quality],
  [`res_asset_kpi.intervals`][res_asset_kpi.intervals] - input data, time, units, quality flags.

The formulas follow public descriptions of IEC 61724-1, IEC 61400-12-1, IEC 61400-26 and IEC 62933-2-1 and of the
cited literature; this is not a certified implementation of any standard.
"""

from res_asset_kpi import bess, contracts, intervals, pv, quality, schema, timeseries, units, wind
from res_asset_kpi._version import __version__

__all__ = [
    "__version__",
    "bess",
    "contracts",
    "intervals",
    "pv",
    "quality",
    "schema",
    "timeseries",
    "units",
    "wind",
]
