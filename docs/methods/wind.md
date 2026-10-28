# Wind methods

Implemented from public descriptions of IEC 61400-26-1/-2 and IEC 61400-12-1 and from the reliability
literature cited below; not a certified implementation of the standards.

## Availability by information category

Every interval of every turbine carries one information category. The list follows the top levels of the
standard's information model (its tree is finer: "out of environmental specification" alone covers low wind, high
wind, icing and temperature limits):

| category | meaning |
|---|---|
| `full_performance` | generating without limits |
| `partial_performance` | generating with a technical derating |
| `technical_standby` | ready, waiting (e.g. low wind, cable untwisting) |
| `environmental` | out of environmental specification |
| `requested` | requested shutdown or curtailment (grid operator or owner) |
| `electrical` | out of electrical specification (grid) |
| `scheduled_maintenance` | scheduled maintenance |
| `planned_corrective` | planned corrective action |
| `forced_outage` | forced outage |
| `no_info` | information unavailable |

A **contract** says for each category whether its time is available, unavailable or excluded:

$$
A_\mathrm{time} = \frac{T_\mathrm{available}}{T_\mathrm{available} + T_\mathrm{unavailable}}
\qquad
A_\mathrm{energy} = 1 - \frac{E_\mathrm{lost,\,unavailable}}{E_\mathrm{actual} + E_\mathrm{lost}}
$$

with the energy sums taken over the categories that are not excluded. A **maintenance allowance** (hours per
turbine and year) moves scheduled maintenance from unavailable to excluded until it is used up; its lost energy is
excused in the same proportion. Two illustrative presets ship with the library: `iec_operational` (every operative
state available, missing information excluded) and `oem_warranty` (grid, environment and requested stops
excused, scheduled maintenance allowance 60 h, missing information counted as available). `wind.remap` re-labels
categories, which shows what a misclassification costs: counting grid stops as turbine faults lowers a warranty
availability by as much as the stops last.

## Power curve

**Air density** of dry air from temperature and pressure (the ISO standard atmosphere at the site elevation when
the pressure is not measured):

$$
\rho = \frac{p}{R_d\, T}, \qquad R_d = 287.05\ \mathrm{J\,kg^{-1}\,K^{-1}},
\qquad p(z) = 101325\,(1 - 2.25577 \cdot 10^{-5} z)^{5.25588}
$$

**Normalisation** of the wind speed to the reference density \(\rho_0 = 1.225\ \mathrm{kg/m^3}\) for
pitch-regulated turbines:

$$
V_n = V \left(\frac{\rho}{\rho_0}\right)^{1/3}
$$

**Method of bins.** Bins of 0.5 m/s centred on multiples of 0.5 m/s; in each bin the mean normalised wind speed
and mean power, and the statistical (category A) uncertainty of the bin mean \(s_i / \sqrt{N_i}\). Bins with fewer
than three 10-minute records are dropped. A data set is **complete** when every bin from 1 m/s below cut-in to
1.5 times the wind speed at 85 % of rated power is present and the records cover at least 180 hours.

**Annual energy production** for a Rayleigh distribution of mean \(\bar V\), the curve held at its last value up
to cut-out:

$$
\mathrm{AEP} = N_h \sum_i \left[F(V_i) - F(V_{i-1})\right] \frac{P_{i-1} + P_i}{2},
\qquad F(V) = 1 - \exp\!\left(-\frac{\pi}{4}\left(\frac{V}{\bar V}\right)^2\right)
$$

with \(N_h = 8760\) hours. The nacelle anemometer of a SCADA data set is not the met mast of the standard, so the
absolute curve carries the anemometer's bias; comparisons between turbines and years remain valid when the
anemometer and its transfer function do not change (see the [Kelmarsh example](../examples/kelmarsh.md)).

## Reliability

**Operating time** is the observation time minus every stop: a failure cannot start while the turbine stands.

**MTBF** \(= T / n\) for \(T\) operating hours and \(n\) failures, with the two-sided chi-square interval of a
time-truncated observation at confidence \(1 - a\):

$$
\mathrm{MTBF}_\mathrm{lo} = \frac{2T}{\chi^2_{1-a/2,\;2n+2}}
\qquad
\mathrm{MTBF}_\mathrm{hi} = \frac{2T}{\chi^2_{a/2,\;2n}}
$$

**MTTR** is the mean time to *restore*, from the stop to the restart: it includes waiting, travel and logistics,
not only hands-on repair time. Its median and 90th percentile are reported too, since restore times are heavily
skewed.

**Crow-AMSAA** (power-law non-homogeneous Poisson process, Crow 1974). The failure intensity at unit age \(t\) is
\(u(t) = \lambda \beta t^{\beta - 1}\). Over units observed in windows \((S_j, T_j]\) (late starts and gaps are
handled exactly) with failure ages \(t_i\), the log-likelihood is

$$
\ln L = n \ln\lambda + n \ln\beta + (\beta - 1) \sum_i \ln t_i - \lambda \sum_j \left(T_j^\beta - S_j^\beta\right).
$$

For fixed \(\beta\) the maximum in \(\lambda\) is \(n / \sum_j (T_j^\beta - S_j^\beta)\); \(\hat\beta\) solves the
profile score equation. Bounds on \(\beta\) come from the profile likelihood, the verdict from the
likelihood-ratio test of \(\beta = 1\): \(\beta < 1\) failures thin out (reliability grows), \(\beta > 1\) they
become more frequent. For one window starting at age 0 the estimate reduces to the textbook
\(\hat\beta = n / \sum_i \ln(T / t_i)\).

**Laplace test** (model-free): under a constant rate each failure time is uniform in its window, and

$$
U = \frac{\sum_i \left(t_i - S_i - L_i / 2\right)}{\sqrt{\sum_i L_i^2 / 12}}
$$

is standard normal (\(L_i\) the window length); \(U > 0\) means failures cluster late. It is applied on each
unit's observed-time clock (its windows joined end to end), so that it does not depend on the age origin.

## References

- Crow, L. H. (1974). Reliability analysis for complex, repairable systems. In *Reliability and Biometry*,
  SIAM, 379-410.
- Rausand, M., Høyland, A. (2004). *System Reliability Theory*, 2nd ed. Wiley (chi-square bounds, Laplace test).
- Tavner, P. (2012). *Offshore Wind Turbines: Reliability, Availability and Maintenance*. IET.
