# Using DiagnoseHarmonisation from the Command Line

While the ideal way we would recommend using this package is within a python script, we also offer two command-line entry points for the cross-sectional report:

- `harmdiag gui` launches a desktop GUI with file pickers, dropdowns, and checkboxes.
- `harmdiag run` keeps the scripted terminal workflow for users who want to pass paths and options directly.

After installing the package and ensuring it is on your Python path, users can choose either mode depending on their workflow.

## Desktop GUI

Run:

    harmdiag gui

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

## Scripted CLI Options

The options for `harmdiag run` are shown below:

    "harmdiag",  description="Harmonisation Diagnostics CLI for scripted runs and the desktop cross-sectional GUI."
  

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

## Notes

We offer some support for different spreedsheet types (e.g xlsx) as well as some support for missing values. However, it is worth noting that if this missingness is relatively high, the pipeline will fail to run (specifically when trying to fit linear mixed effect mdoels). This is true for both data and covariates.

As such we recommend that users use their own imputation approaches or ommit features with large portions of missingnes (>10%). The imputation we do is batch specific, so if batches are small it becomes more unreliable.
