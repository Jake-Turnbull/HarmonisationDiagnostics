# Cross-Sectional Comparison Report

This page documents the multi-method comparison workflow used to evaluate raw and harmonised datasets side by side.

The comparison report expects a dictionary of equally shaped data matrices that share the same batch labels and optional covariates. It runs the same diagnostic suite on each method, then combines the outputs into a scorecard and a short recommendation summary.

## Behaviour notes

- Inputs are normalized from pandas-native types:
  - `datasets` entries can be `pandas.DataFrame` or array-like.
  - `batch` can be a `Series`, one-column `DataFrame`, or array-like.
  - multi-column batch `DataFrame` inputs raise a clear validation error.
- If covariates are not provided, the report continues and skips only LMM diagnostics. The report summary records this as `skipped_missing_covariates`.
- Feature label policy in comparison feature-wise plots:
  - `<= 20` features: diagonal labels (`45°`).
  - `> 20` features: tick labels hidden while keeping plotted values.
- In `save_data=True` mode, per-method exports include:
  - UMAP embeddings (when computed).
  - PCA scores, explained variance, and PC-correlation tables.
  - Frobenius covariance matrices (raw and normalized).
  - Per-batch scree and cumulative variance tables.

## Public entry point

::: DiagnoseHarmonisation.DiagnosticReport.CrossSectionalComparisonReport
    options:
      members_order: source
      show_source: false

## Supporting helpers

The report uses a small set of helper utilities to validate inputs, run the diagnostics, and aggregate the results:

::: DiagnoseHarmonisation.DiagnosticReport.validate_comparison_datasets
    options:
      members_order: source
      show_source: false

::: DiagnoseHarmonisation.DiagnosticReport._run_single_method_diagnostics
    options:
      members_order: source
      show_source: false

::: DiagnoseHarmonisation.DiagnosticReport.summarise_method_performance
    options:
      members_order: source
      show_source: false

::: DiagnoseHarmonisation.DiagnosticReport.generate_comparison_advice
    options:
      members_order: source
      show_source: false

::: DiagnoseHarmonisation.DiagnosticReport._save_comparison_results
    options:
      members_order: source
      show_source: false