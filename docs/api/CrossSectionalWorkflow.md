# Cross-sectional Workflow

**High-level workflow orchestration for cross-sectional diagnostic runs.**

::: DiagnoseHarmonisation.cross_sectional_workflow
    options:
      filters:
        - "!^_"
      members_order: source

## Typical usage

Invoke from the CLI or call programmatically to run the standard diagnostic sequence producing plots and a report.

## Input handling

- Subject ID alignment is enforced before diagnostics.
- Batch can be auto-detected, explicitly selected, or intentionally omitted.
- Covariate selection may be empty. In that case the workflow continues and passes `covariates=None` to the report layer.
- When covariates are empty, the report warns that LMM diagnostics are skipped while all non-LMM diagnostics still run.
