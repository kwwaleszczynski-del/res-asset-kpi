# PV methods

Implemented from public descriptions of IEC 61724-1 and IEC TS 63019 and from Dierauf et al. (2013); not a
certified implementation of the standards.

## Performance ratio

With \(E_i\) the AC energy of interval \(i\), \(P_\mathrm{STC}\) the DC rating, \(G_i\) the plane-of-array
irradiance, \(G_\mathrm{STC} = 1000\ \mathrm{W/m^2}\), \(\gamma\) the power temperature coefficient (1/K, negative)
and \(T_{c,i}\) the cell temperature, summed over the intervals used:

$$
\mathrm{PR} = \frac{\sum_i E_i}{\sum_i P_\mathrm{STC}\, \dfrac{G_i}{G_\mathrm{STC}}\, \Delta t}
\qquad
\mathrm{PR}_{25} = \frac{\sum_i E_i}{\sum_i P_\mathrm{STC}\, \dfrac{G_i}{G_\mathrm{STC}}
\left[1 + \gamma\,(T_{c,i} - 25)\right] \Delta t}
$$

The **weather-corrected PR** (Dierauf et al., 2013, NREL/TP-5200-57991) corrects to \(T_\mathrm{typ}\), the
irradiance-weighted mean cell temperature of a typical year, instead of 25 °C:

$$
\mathrm{PR}_\mathrm{wc} = \frac{\sum_i E_i}{\sum_i P_\mathrm{STC}\, \dfrac{G_i}{G_\mathrm{STC}}
\left[1 + \gamma\,(T_{c,i} - T_\mathrm{typ})\right] \Delta t}
\qquad
T_\mathrm{typ} = \frac{\sum_j G_j\, T_{c,j}}{\sum_j G_j}
$$

Over the typical year itself \(\mathrm{PR}_\mathrm{wc} = \mathrm{PR}\) exactly when the plant follows the
temperature model; month by month it removes the seasonal swing that \(\mathrm{PR}\) shows and that
\(\mathrm{PR}_{25}\) overcorrects. Intervals with a missing power or irradiance are left out of numerator and
denominator; when cell temperatures are given, intervals without one are left out of all three ratios.

Cell temperature models: from a back-of-module sensor \(T_c = T_m + G/1000 \cdot \Delta T\) (King et al., 2004,
SAND2004-3535) and the PVsyst heat balance \(T_c = T_a + \alpha G (1 - \eta) / (U_c + U_v v)\).

```python
t = pv.pr_terms(p_ac_kw, poa_wm2, p_stc_kw=146.6, interval_h=0.25, t_cell_c=t_cell, gamma_per_k=-0.0045,
                t_typ_c=pv.typical_cell_temperature(poa_tmy, t_cell_tmy), mask=poa_wm2 > 50)
t.pr, t.pr_temperature_corrected, t.pr_weather_corrected
```

## Availability

Units \(u\) (inverters) carry capacity weights \(w_u\), \(\sum_u w_u = 1\). An interval counts when it is daylight
(irradiance above a threshold, 50 W/m² by default), the unit's state is known, and it is neither excluded nor
planned maintenance within the allowance:

$$
A_\mathrm{time} = \frac{\sum_{i,u} w_u\, [\text{running}_{i,u} \wedge \text{counted}_{i,u}]}
{\sum_{i,u} w_u\, [\text{counted}_{i,u}]}
\qquad
A_\mathrm{energy} = 1 - \frac{\sum_{i,u} E^\mathrm{exp}_{i,u}\, [\text{stopped}_{i,u} \wedge \text{counted}_{i,u}]}
{\sum_{i,u} E^\mathrm{exp}_{i,u}\, [\text{counted}_{i,u}]}
$$

- **Excluded** periods (grid outage, grid-operator instruction, force majeure: whatever the contract places outside
  the operator's responsibility) leave numerator and denominator.
- **Planned maintenance** is excluded until its capacity-weighted daylight hours in the period exceed the
  allowance; afterwards it counts as downtime.
- The **energy basis** weighs each stopped interval by the energy the unit was expected to produce (after soiling,
  snow and clipping), so an outage on a sunny noon costs more than one at dawn.

## Energy bridge

The bridge explains the difference between the budget energy \(E_b\) (e.g. P50) and the measured energy
\(E_\mathrm{act}\) of a month. Its bars add up to \(E_\mathrm{act} - E_b\) exactly.

**Weather steps.** Ratios outside the operator's control (irradiance, temperature, degradation) re-state the budget
in order:

$$
\text{bar}_k = E_b \prod_{j<k} r_j \,(r_k - 1)
\qquad
E_w = E_b \prod_k r_k
$$

**Loss steps.** For each loss \(k\) (soiling, snow, clipping, availability, curtailment, data gaps, ...) the bar is
the budgeted loss minus the measured loss, in energy. Measured fractions \(\ell_k\) apply in order to the actual
plant's potential, budgeted fractions \(b_k\) to the potential the budget implies for that weather:

$$
X_a = \frac{E_\mathrm{act}}{\prod_k (1 - \ell_k)}
\qquad
X_b = \frac{E_w}{\prod_k (1 - b_k)}
$$

$$
L_k = X_a \prod_{j<k} (1 - \ell_j)\, \ell_k
\qquad
B_k = X_b \prod_{j<k} (1 - b_j)\, b_k
\qquad
\text{bar}_k = B_k - L_k
$$

**Residual.** \(E_\mathrm{act} - E_b - \sum \text{bars}\). Summing the bars shows that it always equals
\(X_a - X_b\): the gap between the potential of the actual plant and the potential the budget implies for the
weather that occurred, which the measurements do not explain (model error, unmodelled losses, sensor level).

**Shapley variant.** The month is written as a product
\(E_\mathrm{act} = E_b \prod_k f_k\) of the weather ratios, the loss ratios \((1 - \ell_k)/(1 - b_k)\) and the
residual factor \(X_a / X_b\). Each bar is the Shapley value of the game \(v(S) = E_b \prod_{k \in S} f_k\):

$$
\phi_k = \sum_{S \subseteq N \setminus \{k\}} \frac{|S|!\,(n - |S| - 1)!}{n!}\, E_b\, (f_k - 1) \prod_{j \in S} f_j
$$

It does not depend on the order of the factors, and the bars still close exactly, but no single bar equals a
measured loss. The sequential method is the default because each loss bar is then a quantity one can check.

```python
bars = pv.energy_bridge(
    e_budget=table["E_budget"], e_actual=table["E_actual"],
    weather={"irradiance": table["r_irr"], "temperature": table["r_temp"], "degradation": table["r_deg"]},
    losses_actual={"soiling": table["soiling"], "availability": table["unavailability"]},
    losses_budget={"soiling": 0.02, "availability": 0.01},
    index=table.index,
)
```

## References

- Dierauf, T., Growitz, A., Kurtz, S., Becerra Cruz, J. L., Riley, E., Hansen, C. (2013). *Weather-Corrected
  Performance Ratio.* NREL/TP-5200-57991.
- King, D. L., Boyson, W. E., Kratochvil, J. A. (2004). *Photovoltaic Array Performance Model.* SAND2004-3535.
- Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games II*, 307-317.
