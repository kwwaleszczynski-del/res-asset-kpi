# Battery storage methods

The test procedures and guarantee definitions of battery contracts vary; the formulas below follow common LTSA
and warranty practice and the spirit of IEC 62933-2-1, not a certified reading of the standard.

## Round-trip efficiency

The metered power \(p\) (positive = export) splits into energy out \(E_\mathrm{out} = \sum \max(p, 0)\,\Delta t\)
and energy in \(E_\mathrm{in} = \sum \max(-p, 0)\,\Delta t\). Where the auxiliaries (HVAC, transformers, control)
sit behind the revenue meter, the flow of the power conversion system is \(p + p_\mathrm{aux}\), which gives the
RTE without auxiliaries; where they have their own supply, their energy is added to \(E_\mathrm{in}\) for the RTE
with auxiliaries.

A window (day, month, contract year) seldom ends at the state of charge it started with. With
\(\Delta E\) the change of stored energy between the instants that bound the window and \(\eta\) the one-way
efficiency between the meter and the store:

$$
\mathrm{RTE} =
\begin{cases}
\dfrac{E_\mathrm{out} + \eta\,\Delta E}{E_\mathrm{in}} & \Delta E \ge 0 \\[2ex]
\dfrac{E_\mathrm{out}}{E_\mathrm{in} + |\Delta E| / \eta} & \Delta E < 0
\end{cases}
$$

Energy left in store is worth \(\eta \Delta E\) at the meter; energy taken from store would have needed
\(|\Delta E| / \eta\) to put back. \(\eta\) is given, or found as \(\sqrt{\mathrm{RTE}}\) of the whole record by
fixed-point iteration. A window is assessed (`valid`) when it has enough equivalent full cycles (0.3 a day,
5 a month by default) and data, because on days with little throughput the auxiliaries and the SoC resolution
dominate the ratio.

## LTSA availability

Per interval the measured availability follows from the available power the EMS reports in each direction
\(d\) against the contracted power:

$$
a_i^\mathrm{time} = \left[\frac{P_{\mathrm{avail},d}}{P_{\mathrm{contract},d}} \ge \theta \;\; \forall d\right]
\qquad
a_i^\mathrm{cap} = \min_d \min\!\left(1, \frac{P_{\mathrm{avail},d}}{P_{\mathrm{contract},d}}\right)
\qquad
a_i^\mathrm{energy} = a_i^\mathrm{cap} \min\!\left(1, \frac{E_\mathrm{avail}}{E_\mathrm{contract}}\right)
$$

Events from the operator's log decide how an interval counts: **deemed available** (\(a_i = 1\): a grid-operator
instruction, a grid outage, an owner request), **excluded** (force majeure, the capacity test), **allowance**
(planned maintenance excluded until the yearly allowance, pro-rated to the period, is used up) or **counted**.
When events overlap, excluded wins over deemed available, which wins over allowance. Then

$$
A = \frac{\sum_{i \notin \mathrm{excl}} a_i\, \Delta t}{\sum_{i \notin \mathrm{excl}} \Delta t}.
$$

Intervals without data and without an event are excluded, or counted as available or unavailable, as the
contract says.

## Cycles

**Equivalent full cycles** divide the throughput a warranty names (discharge at the meter or at the DC terminals,
or half of charge plus discharge) by the reference energy.

**Rainflow counting** of the state of charge uses the three-point method of ASTM E1049-85: reversals of the
signal enter a stack; whenever the latest range is at least as large as the one before it, that earlier range is
counted as a full cycle (or a half cycle when it contains the first point) and removed; what is left at the end
counts as half cycles. An optional hysteresis ignores reversals smaller than a threshold (measurement noise).
\(\sum \text{depth} \times \text{count}\) is the rainflow EFC; twice that sum equals the total path length of the
reversals, which the tests check for random signals.

## Capacity test

The discharged energy of the test is integrated over the whole contiguous discharge around the run at the test
power, corrected to the reference temperature with a linear coefficient \(\alpha\):

$$
E_\mathrm{corr} = \frac{E_\mathrm{meas}}{1 + \alpha\,(T_\mathrm{test} - T_\mathrm{ref})}
$$

Its uncertainty follows the GUM, relative standard uncertainties combined in quadrature, rectangular limits for
the meter class and the current and voltage transformers:

$$
u_c^2 = \left(\frac{c_\mathrm{meter}}{\sqrt 3}\right)^2 + \left(\frac{c_\mathrm{CT}}{\sqrt 3}\right)^2
+ \left(\frac{c_\mathrm{VT}}{\sqrt 3}\right)^2 + (\alpha\, u_T)^2 + \left(u_\alpha\, |T_\mathrm{test} - T_\mathrm{ref}|\right)^2,
\qquad U = k\, u_c\, E_\mathrm{corr}
$$

Decision rules: **simple** passes when \(E_\mathrm{corr} \ge E_g\); **guard band** passes when
\(E_\mathrm{corr} - U \ge E_g\), fails when \(E_\mathrm{corr} + U < E_g\) and is inconclusive in between. The
guaranteed energy \(E_g\) can follow a warranty retention curve read at the equivalent age
\(\max(t, \mathrm{EFC} / \mathrm{EFC}_\mathrm{year})\) (`contracts.guaranteed_retention`).

## References

- ASTM E1049-85 (reapproved 2017). *Standard Practices for Cycle Counting in Fatigue Analysis.*
- JCGM 100:2008. *Evaluation of measurement data: Guide to the expression of uncertainty in measurement (GUM).*
- DNV (2021). *Energy storage capacity warranties: beyond the fine print.*
