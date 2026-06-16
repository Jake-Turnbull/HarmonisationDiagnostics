# HarmonisationFunctions

**A set of harmonisation functions that are frequently used in the literature. We will continue to expand these with methods we develop.**

::: DiagnoseHarmonisation.HarmonisationFunctions
    options:
      filters:
        - "!^_"
      members_order: source

## Examples

- Local priors: directional bias weighting

```python
from DiagnoseHarmonisation.HarmonisationFunctions import combat_modular

# data: (n_features, n_samples) array
# batch: length n_samples vector of batch labels
# coords: optional feature coordinates used for spatial weighting (n_features, 2)
out = combat_modular(
    data,
    batch,
    mean_model='ols',
    prior_mode='local',
    prior_weight_methods=['correlation_similarity', 'directional_bias'],
    prior_weight_opts={
        'method_weights': {'correlation_similarity': 0.4, 'directional_bias': 0.6},
        'min_effective': 5,
        'directional': {'dir_sigma': 0.5, 'dir_power': 1.5, 'use_global_sign': True}
    },
    return_priors=True,
)
priors = out['priors']
local = priors['local_priors']
weights = local['weights']  # shape (n_batch, n_features, n_features)
```

- GAM mean model with spline fallback

```python
out = combat_modular(data, batch, mod=covariates_df, mean_model='gam', gam_opts={'n_splines':8})
```

