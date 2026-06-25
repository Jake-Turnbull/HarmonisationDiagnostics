# Cross-Sectional Comparison Report

This page documents the multi-method comparison workflow used to evaluate raw and harmonised datasets side by side.

The comparison report expects a dictionary of equally shaped data matrices that share the same batch labels and optional covariates. It runs the same diagnostic suite on each method, then combines the outputs into a scorecard and a short recommendation summary.

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