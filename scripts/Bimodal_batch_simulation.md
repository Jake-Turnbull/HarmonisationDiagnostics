# Two-Batch ComBat Stress-Test Simulation

`scripts/Bimodal_batch_simulation.py` generates balanced two-batch data and compares standard ComBat with SW-ComBat. The observed value follows

$$
y_{ij} = X_i\beta_j + \gamma_{b_i,j} + \sqrt{\delta_{b_i,j}}\epsilon_{ij},
$$

where the batch-free target is $X_i\beta_j + \epsilon_{ij}$. Here `delta` is the variance ratio, so its square root scales residuals. Additive effects are centered across batches; log-delta effects are reciprocal across batches, giving a geometric mean delta of one. `n_samples` means total N. For example, N=65 is divided into batches of 32 and 33.

## Scenario Families

The families are selectable independently and are not crossed into a Cartesian product by default.

- `alignment`: preserves the original `simple`, `opposing`, `correlated_independent`, and `correlated_shared` scenarios. The correlated cases contrast feature correlation with whether batch effects share that same grouping.
- `effect_shapes`: adds `unimodal`, `bimodal`, `multimodal`, `overlapping`, and `heavy_tailed` additive and log-delta effect populations. Their oracle matrix is built from the true injected additive and log-delta feature effects.

The effect-shape regimes use distinct group locations with scenario-specific spread; `overlapping` uses wider spread, and `heavy_tailed` adds Student-t variation. The oracle similarity is derived from the true injected additive and log-delta effects.
The three inferred similarities and oracle are evaluated through the local-prior implementation in `HarmonisationFunctions.py`.

## Methods and Metrics

The benchmark includes `Raw`, global `ComBat`, and four local-prior SW runs that all call `combat_modular(prior_mode="local")`, which delegates to `HarmonisationFunctions._construct_local_priors()`:

- `SW-mean-difference` uses the helper's `magnitude_similarity` method (rank similarity of per-feature standardized means).
- `SW-correlation` uses the helper's `correlation_similarity` method.
- `SW-variance-difference` uses `variance_similarity`, which affects the scale (`a`/`b`) priors; as implemented in `_construct_local_priors()`, its gamma mean weights remain uniform when no mean-similarity method is requested.
- `SW-oracle` uses the supported `custom` method with an RBF similarity computed from the true injected additive gamma and log-delta feature profiles. The helper applies this matrix to mean-effect pooling; scale pooling continues to use the helper's within-batch variance similarity.

No monkey-patched pooling switches or alternative SW implementations are benchmarked. The helper's low-sample warnings are retained. `raw_metrics.csv` records RMSE and MAE to clean data, parameter errors, and prior diagnostics. Pairwise SW-versus-ComBat comparisons use the same scenario, N, feature count, seed, and experiment.

## Figures

Figure outputs are written separately for `alignment` and `effect_shapes`, so original and added regimes do not share panels. Each family has distinct figure files:

- `<family>_figure1_feature_correlations_and_effects`: feature correlations and true batchwise additive/log-delta effects.
- `<family>_figure2_similarity_matrices_low_n` and `<family>_figure2_similarity_matrices_high_n`: matrices returned by `_construct_local_priors()` for the mean-difference, correlation, and oracle methods, plus scale weights for variance-difference, at low and high sample sizes.
- `<family>_figure3_rmse_by_sample_size`: RMSE to clean data by total N, with batch counts and separate feature-size panels.
- `<family>_figure4_paired_rmse_improvement`: paired RMSE improvement versus global ComBat; the zero line marks where SW performs worse.

Additional outputs include `run_config.json`, `raw_metrics.csv`, paired CSVs, per-regime representative truth `.npz` files, the true-effect oracle CSV, and actual method-specific mean/scale prior-weight CSVs at low and high N.

## Running

The default run covers both families, with total sample sizes 20, 40, 65, 100, and 200; three seeds; and 50 and 100 features. This is a full benchmark. For a quick run:

```bash
python scripts/Bimodal_batch_simulation_2.py \
  --output scripts/sw_similarity_quick_check \
  --sample-sizes 20 40 100 \
  --seeds 1 2 \
  --features 50 \
  --scenario-family effect_shapes
```

Choose `--scenario-family alignment`, `--scenario-family effect_shapes`, or `--scenario-family all`. Optionally select scenario names with `--scenarios bimodal heavy_tailed`; names must belong to the chosen family. This simulator is fixed at two batches, and total N must be at least two.

The generator and plotting tests can be run with:

```bash
pytest -q tests/test_sw_similarity_benchmark.py
```

## Interpretation

These are controlled simulations, not clinical validation. Oracle similarity uses the known injected additive and log-delta effects and is an optimistic reference. Seeds are simulation replicates; intervals are descriptive normal-approximation intervals, not formal hypothesis tests. Small total N implies fewer samples per batch and greater reliance on estimated priors.
