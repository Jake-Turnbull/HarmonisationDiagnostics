# Using DiagnoseHarmonisation from the Command Line for Cross-sectional data

While the ideal way we would recommend using this package is within a python script, we also offer two command-line entry points for the cross-sectional report:

- `DHarm gui` launches a desktop GUI with file pickers, dropdowns, and checkboxes.
- `DHarm run` keeps the scripted terminal workflow for users who want to pass paths and options directly.

After installing the package and ensuring it is on your Python path, users can choose either mode depending on their workflow.

## Desktop GUI

Run:

  DHarm gui

This opens a Tkinter-based desktop window where you can:

- choose the data file and covariates file
- choose the output directory
- select the subject ID columns
- select the batch column or explicitly opt into no batch column
- choose which covariates to include
- set the report name
- toggle whether aligned data should be saved
- toggle timestamped report names

The GUI keeps advanced report settings on sensible defaults for the first version and shows status messages while the report is running.

If no covariates are selected, the run continues and emits a warning that LMM diagnostics are skipped. All other diagnostics still run and the report is generated.

## Scripted CLI Options

The options for `DHarm run` are shown below:

  "DHarm",  description="Harmonisation Diagnostics CLI for scripted runs and the desktop cross-sectional GUI."
  

    "run", help="Run the diagnostics pipeline from data and covariates CSVs"
    "--data", "-d", required=True, help="Path to data CSV (subjects x IDPs). First row must be feature names."
    "--covariates", "-c", required=True, help="Path to covariates CSV (first column subject ID)."
    "--batch-col", type=int, default=None, help="1-based column number in covariates CSV where batch is located. If omitted, tries to auto-detect by header."
    "--data-id-col", default=None, help="Data subject ID column name (defaults to first column)."
    "--cov-id-col", default=None, help="Covariates subject ID column name (defaults to first column)."
    "--outdir", default=None, help="Directory to write summary / report files."
    "-v", "--verbose", action="store_true", help="Verbose output."
    "--report-name", default=None, help="Optional name for the report (used in filenames)."
    "--save-data", default = True, help="Whether to save the aligned data and covariates used for the report (for debugging)."
    "--save-data-name", default=None, help="Optional name for the saved data files (used in filenames)."

### Additional CLI behaviour notes

- Data and batch inputs are normalized from pandas-friendly formats where relevant in the workflow layer.
- Feature-heavy plots follow a readability policy:
  - `<= 20` features: diagonal labels.
  - `> 20` features: labels hidden, values still plotted.

# Using DiagnoseHarmonisation from the Command Line for Longitudinal data

While the ideal way we would recommend using this package is within a python script, we also offer command-line entry point for the longitudinal report:

## Scripted CLI Options

The options for `DHarm-longitudinal run` are shown below:

```
DHarm-longitudinal run --help

usage: DHarm-longitudinal run [-h] --data DATA --subject-id-col SUBJECT_ID_COL --timepoint-col
                                 TIMEPOINT_COL --batch-col BATCH_COL
                                 (--feature-cols FEATURE_COLS | --features-file FEATURES_FILE)
                                 [--covariates COVARIATES] [--cov-subject-id-col COV_SUBJECT_ID_COL]
                                 [--cov-timepoint-col COV_TIMEPOINT_COL]
                                 [--covariate-cols COVARIATE_COLS | --covariates-file COVARIATES_FILE]
                                 [--covariate-names COVARIATE_NAMES] [--outdir OUTDIR]
                                 [--report-name REPORT_NAME] [--save-data] [--save-data-name SAVE_DATA_NAME]
                                 [-v]

options:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  Path to longitudinal data CSV/XLS/XLSX.
  --subject-id-col SUBJECT_ID_COL
                        Subject ID column name in the data file.
  --timepoint-col TIMEPOINT_COL
                        Timepoint/visit column name in the data file.
  --batch-col BATCH_COL
                        Batch/site/scanner column name in the data file.
  --feature-cols FEATURE_COLS
                        Comma-separated feature/IDP column names.
  --features-file FEATURES_FILE
                        Text file containing one feature/IDP column name per line.
  --covariates COVARIATES, -c COVARIATES
                        Optional path to separate longitudinal covariates CSV/XLS/XLSX. If omitted, covariates
                        are read from --data.
  --cov-subject-id-col COV_SUBJECT_ID_COL
                        Subject ID column name in the separate covariates file.
  --cov-timepoint-col COV_TIMEPOINT_COL
                        Timepoint/visit column name in the separate covariates file.
  --covariate-cols COVARIATE_COLS
                        Comma-separated covariate column names.
  --covariates-file COVARIATES_FILE
                        Text file containing one covariate column name per line.
  --covariate-names COVARIATE_NAMES
                        Optional comma-separated covariate display names. Must match the order and number of
                        selected covariates.
  --outdir OUTDIR       Directory to write report files if LongitudinalReport supports save_dir.
  --report-name REPORT_NAME
                        Optional name for the report.
  --save-data           Attempt to save aligned input data. Warning: current LongitudinalReport may fail with
                        covariates because its save_data block expects array-like covariates, while modelling
                        expects dict covariates.
  --save-data-name SAVE_DATA_NAME
                        Optional name for saved input data files if supported.
  -v, --verbose         Verbose output.
```

## Notes

We offer some support for different spreedsheet types (e.g xlsx) as well as some support for missing values. However, it is worth noting that if this missingness is relatively high, the pipeline will fail to run (specifically when trying to fit linear mixed effect mdoels). This is true for both data and covariates.

As such we recommend that users use their own imputation approaches or ommit features with large portions of missingnes (>10%). The imputation we do is batch specific, so if batches are small it becomes more unreliable.
