# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.1.1] - unreleased

Fixes from an independent review. Missing data no longer turn into silent zeros; on complete data every result is
unchanged (the regression check against the original projects still reproduces them exactly).

### Fixed

- `bess.window_rte`: the record is put on its full regular grid, so rows missing from the index count as missing;
  `coverage` is the share of the window's intervals with a meter reading. A window with a gap gets NaN ratios and
  `valid = False`, since its stored-energy change spans the gap while its metered energy does not (one lost
  discharge hour of a day lowered the RTE from 0.65 to 0.49 and still reported it valid). `min_coverage` now
  defaults to 1.0; the automatic one-way efficiency uses the complete windows only.
- `pv.availability`: missing irradiance no longer reads as night. Such intervals are left out and reported in the
  new column `poa_missing_h`, or counted as daylight with `missing_poa="daylight"`. A counted interval without
  expected power makes `availability_energy`, `expected_kwh` and `lost_kwh` NaN (new column
  `expected_missing_h`) instead of counting a stop as zero loss.
- `timeseries.regularise` and `schema.to_wide`: on a coarser grid a sum needs every source interval and a mean the
  share `min_share` (default 1.0) of them; a quarter-hour of energy no longer passes for a full hour.
- `wind.availability`: the maintenance allowance is scaled by the turbine-years the totals cover, by default their
  hours / 8760 (it was one turbine-year whatever the totals covered: a fleet-year got one turbine's allowance, a
  month a full year's). `n_turbine_years` still overrides it.
- `bess.discharge_segment`: a missing value next to the test discharge raises instead of cutting the test energy
  short.
- `bess.ltsa_availability`: events count for the share of each interval they cover instead of whole intervals by
  their start (an event inside one interval used to vanish).
- `pv.energy_bridge`: a measured loss of 1.0 (a month without production) is accepted; the Shapley bars of a
  period without budget energy are NaN with a warning instead of an unclosed bridge.
- Mappings accept `sign: -1`, as documented; `mW` and `mWh` are refused as ambiguous instead of being read as MW.
- Documentation: the rainflow example counts one full cycle of range 4, not two; night consumption in the PR and
  the charge-side cycles of `window_rte` are stated; the parity check is described as a regression check against
  the source projects' commits before they imported the library, and refuses a source tree that imports it.

### Changed

- Defaults that can change results on incomplete data: `window_rte(min_coverage)` 0.95 -> 1.0;
  `wind.availability(n_turbine_years)` 1.0 -> derived from the hours of the totals; `regularise` and `to_wide`
  need complete source intervals (`min_share=1.0`). Pass the old values explicitly to keep the old behaviour.
- Tests: the energy-bridge property test recomputes every sequential bar instead of only the closure, which the
  residual guarantees by construction; new tests reproduce each case above and fail on 0.1.0.

## [0.1.0] - 2026-10-28

First release: the core extracted from six open analysis projects, with one typed API and tests.

### Added

- Input data model: long table (`timestamp, device_id, signal, value`) and wide table with a YAML mapping,
  canonical signals with units and aggregation rules, unit conversion, regular UTC grids for 5, 10, 15 and
  30-minute and hourly data, the repeated and the skipped DST hour, local reporting periods and contract years.
- Data quality flags: missing, stuck and out-of-range values; coverage per period.
- Interval operations on event logs: union, difference, overlap, first-up attribution.
- PV: performance ratio, PR corrected to 25 °C, weather-corrected PR (Dierauf et al. 2013), two cell temperature
  models; capacity-weighted time and energy availability with exclusions and a planned-maintenance allowance;
  budget-to-actual energy bridge (sequential and Shapley), closing exactly.
- Wind: time and energy availability by IEC 61400-26 information category with contract presets and a
  maintenance allowance; air density, wind speed normalisation, power curve by the method of bins with category A
  uncertainty, data set completeness, AEP for a Rayleigh distribution; operating time, MTBF with chi-square
  bounds, MTTR, Crow-AMSAA with profile-likelihood bounds and likelihood-ratio test, Laplace test.
- Battery storage: round-trip efficiency corrected for the change of stored energy, with and without
  auxiliaries, per local day, month, quarter, year or contract year; LTSA availability (time, capacity, energy
  formulas; deemed available, excluded, allowance and counted events); throughput, equivalent full cycles,
  rainflow counting with hysteresis; capacity test evaluation with temperature correction, GUM uncertainty and
  guard-band decision rule.
- Contracts: guarantees, step-down, tolerance, exclusions, damages methods, bonus and caps from YAML, with
  illustrative example files for PV, wind and battery contracts.
- Validation script reproducing the source projects' results (41 checks, largest relative difference 1.3e-13),
  examples on NREL PVDAQ system 34 and the Kelmarsh wind farm, MkDocs documentation, CI on Python 3.11-3.13,
  release workflow for PyPI Trusted Publishing (disabled until enabled), citation metadata.
