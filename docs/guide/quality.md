# Quality flags

Checks never change the data. Each returns one integer flag per interval; flags combine with `|`, so one
interval can carry several.

| flag | value | meaning |
|---|---:|---|
| `MISSING` | 1 | no value recorded |
| `STUCK` | 2 | the same value repeated over a run of intervals (frozen sensor or logger) |
| `OUT_OF_RANGE` | 4 | outside the plausible range of the signal |
| `FILLED` | 8 | set by the analyst for values filled or corrected; never set by the checks |

```python
from res_asset_kpi import quality

flags = quality.quality_flags(frame["t_amb"], "t_amb", stuck_run=24)   # six hours of 15-minute values
quality.summary(flags)                     # intervals per flag
quality.coverage(flags, periods)           # share of usable intervals per period
t_amb = quality.apply_flags(frame["t_amb"], flags)   # rejected values set to NaN, by the analyst's decision
```

**Stuck values.** A run of at least `stuck_run` exactly equal values is flagged. Values whose runs are normal
(`0` for irradiance at night or for the power of a stopped turbine) are passed as `stuck_ignore`. Typical run
lengths are six hours for temperatures and three 10-minute records for an anemometer above 1 m/s.

**Ranges.** The defaults (`quality.RANGES`) are wide physical limits meant to catch unit and scaling errors, for
example a temperature in °F read as °C or a state of charge in percent read as a fraction. They do not judge a
sensor; tighten them per site.

In the [PVDAQ example](../examples/pvdaq.md) the ambient temperature sensor of system 34 stays frozen for
31 511 quarter-hours (about 30 % of the three years); the flags remove those intervals from the temperature-corrected
ratios instead of letting a constant 0 °F distort them.
