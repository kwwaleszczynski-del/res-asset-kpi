# Contracts

The KPI functions measure; [`contracts`](../api/contracts.md) compares the measurements with what a contract
guarantees and prices the shortfall. The terms live in YAML, so that they can be reviewed next to the agreement
they come from.

!!! warning "Illustrative examples, not legal advice"

    The three example files (`pv_om_contract`, `wind_availability_warranty`, `bess_ltsa`) are not taken from any
    real agreement. Thresholds, exclusions, damages formulas and caps differ from one contract to the next and
    must be checked against the actual contract, which always prevails.

## Structure

```yaml
name: Illustrative PV O&M contract
asset: pv                      # pv | wind | bess
currency: EUR
period: year                   # month | quarter | year | contract_year (needs contract_start)
excluded_events: [force_majeure, grid_operator_instruction, grid_outage]
planned_maintenance:
  allowance_h: 24
guarantees:
  availability:
    basis: time
    guaranteed: 0.98
    damages: {method: energy, price_per_mwh: 60.0, cap: 20000.0}
    bonus: {threshold: 0.995, share: 0.5, price_per_mwh: 60.0}
  performance_ratio:
    basis: pr_wc
    guaranteed: 0.78
    annual_degradation: 0.005
    tolerance: 0.02
    min_data_share: 0.8
    damages: {method: energy, price_per_mwh: 60.0, cap: 50000.0}
```

Battery and wind files add `event_treatments` (category -> treatment) and asset-specific `parameters` per
guarantee (thresholds, windows, the retention curve), which the loader passes through unchanged. Unknown keys are
errors, so a typo does not silently change a clause.

## Guaranteed value

The guaranteed value may step down each contract year and be reduced by a measurement tolerance:

$$
g_\mathrm{eff}(y) = g\,(1 - d)^{\,y - 1} - \mathrm{tolerance}
$$

A period with less valid data than `min_data_share` gets no verdict and no damages.

## Damages

With the shortfall \(s = \max(0, g_\mathrm{eff} - m)\) for a measured value \(m\):

| method | amount | typical use |
|---|---|---|
| `energy` | \(s \cdot E_\mathrm{basis}\,[\mathrm{MWh}] \cdot \mathrm{price}\) | PV and wind availability, PR |
| `per_point` | \(\mathrm{rate} \cdot 100\, s \cdot \mathrm{multiplier}\) | battery availability, RTE |
| `rte_energy_cost` | \(E_\mathrm{out} \left(\frac{1}{\mathrm{RTE}} - \frac{1}{\mathrm{RTE}_g}\right) \mathrm{price}\) when \(\mathrm{RTE} < \mathrm{RTE}_g\) | battery RTE |
| `per_mwh` | \(\mathrm{rate} \cdot \max(0, E_g - E_\mathrm{corr})\) on a failed test | battery capacity |

\(E_\mathrm{basis}\) is the energy the shortfall is priced on: the expected energy of the counted intervals for
PV availability, the reference energy of the PR intervals for a PR guarantee, the potential energy for wind. A
bonus above a threshold pays \((m - \mathrm{threshold}) \cdot E_\mathrm{basis} \cdot \mathrm{price} \cdot
\mathrm{share}\). Each guarantee's damages may be capped per period; `apply_annual_cap` scales all damages of a
year down to an annual cap.

```python
from res_asset_kpi import contracts

c = contracts.load_contract(contracts.example_path("pv_om_contract"))
result = contracts.assess(c.guarantees["availability"], availability_per_year,
                          basis_energy_mwh=expected_kwh_per_year / 1000)
result[["passed", "damages", "bonus"]]
```
