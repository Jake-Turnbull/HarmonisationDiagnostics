# Command-line Interface (CLI)

**Entry points for running the diagnostic pipeline from the command line.**

::: DiagnoseHarmonisation.cli
    options:
      filters:
        - "!^_"
      members_order: source

## Usage

Run the main CLI from the package:

```bash
python -m DiagnoseHarmonisation.cli --help
```

This module exposes the high-level entrypoints used by the project-level `DHarm` console script and includes helpers for parsing arguments and launching the cross-sectional workflow.

## Common flags

- `--data, -d`: Path to data file (subjects x features).
- `--covariates, -c`: Path to covariates file (one row per subject).
- `--batch-col`: 1-based index of covariates column containing batch labels (optional).
- `--outdir`: Output directory for reports.
- `-v/--verbose`: Verbose logging.
