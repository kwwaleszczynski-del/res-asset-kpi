# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

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
