# Longitudinal CLI

**Command-line entrypoints and helpers for longitudinal workflows.**

::: DiagnoseHarmonisation.longitudinal_cli
    options:
      filters:
        - "!^_"
      members_order: source

## Usage

```bash
python -m DiagnoseHarmonisation.longitudinal_cli --help
```

Alternatively:

```bash
harmdiag-longitudinal --help
```

This module contains argument parsing and runner functions tailored to longitudinal and test-retest analyses.

## Common flags

- `--data, -d`: Path to longitudinal data file.
- `--subject-id-col`: Column name for subject IDs (required).
- `--timepoint-col`: Column name for visit/timepoint (required).
- `--batch-col`: Column name for batch/site/scanner (required).
- `--feature-cols` / `--features-file`: Select features by comma-list or file.
- `--inspect`: Print columns and help choose indices.

