# Power Analysis

**Utilities to estimate sample sizes and power for harmonisation experiments.**

::: DiagnoseHarmonisation.PowerAnalysis
    options:
      filters:
        - "!^_"
      members_order: source

## Example

Programmatic usage:

```python
from DiagnoseHarmonisation.PowerAnalysis import estimate_power
estimate_power(effect_size=0.5, n=50, alpha=0.05)
```
