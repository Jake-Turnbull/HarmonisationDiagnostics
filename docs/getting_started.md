# Getting started

Here we will provide a brief example of how to use DiagnoseHarmonisation within a standard workflow, giving an example of how one would use the python version (which has full functionality) and how one would use the command-line tools on spreadsheet files.

By far the easiest way to run this code is by using a python script and loading your data in as arrays.

## 1. Install from Github

In terminal, run either:

    pip install DiagnoseHarmonisation

    or for the development version:

    pip install git+https://github.com/Jake-Turnbull/HarmonisationDiagnostics.git

Or alternatively clone locally:
    git clone https://github.com/Jake-Turnbull/HarmonisationDiagnostics.git
    cd HarmonisationDiagnostics
    pip install -e .

## 1a. Launch the desktop GUI

If you would prefer not to type file paths and column names manually, you can launch the cross-sectional desktop GUI:

    DHarm gui

This opens a window for selecting the data file, covariates file, output directory, batch column, subject ID columns, and covariates to include in the report.

## 2. Data requirements

The minimum arguments required to run DiagnoseHarmonize are:

    data: NumPy array (samples x features)
    batch: Vector (array or list) of batch labels (Samples x 1)
    Covariates: NumPy array (Samples x covariates)

For additional arguments, please check the DiagnosticReports docs.
Note, while covariates aren't inherently required, in order to get an informative result they are recommended. In the case that no covariates are used, the CrossSectionalReport will throw an error. This will be fixed in a later patch but for now please include an intercept (vector of ones) as a placeholder.

## 3. Generate a Cross-Sectional Diagnostic Report

There are two main cross-sectional workflows: a single-dataset report and a multi-method comparison report.

The single-dataset report provides the standard diagnostic summary for one harmonised dataset.

Using the full report:

    from DiagnoseHarmonisation import DiagnosticReport
        report = DiagnosticReport.CrossSectionalReport(
            data=X,
            batch=batch,
            covariates=covars)

This will produce a detailed HTML file containing a full analysis of batch and covariate effects.

To compare several harmonised outputs side by side, pass a dictionary of method names and datasets:

```python
from DiagnoseHarmonisation import DiagnosticReport

report = DiagnosticReport.CrossSectionalComparisonReport(
    datasets={
        "Raw": X_raw,
        "ComBat": X_combat,
        "CovBat": X_covbat,
    },
    batch=batch,
    covariates=covars,
    covariate_names=["age", "sex"],
)
```

This comparison report checks that every dataset shares the same shape and then generates:

1. A dataset summary block with sample counts, batch counts, missingness, and covariate names.
2. Per-method diagnostic summaries.
3. A category-based scorecard covering additive, multiplicative, linear-modelling, distributional, and PCA diagnostics.
4. Side-by-side comparison plots.

You can also generate the same cross-sectional report without writing Python by either:

    DHarm gui

or:

    DHarm run --data data.csv --covariates covariates.csv

## 4. Generate a Longitudinal Diagnostic Report

To generate evaluation report on longitudinal/test-retest data:

```
DiagnosticReport.LongitudinalReport(
    data=X,
    subject_ids=subject_var,
    batch=batch,
    timepoints=timepoint_var,
    features=idp_names,
    covariates=covars
)
```
This will produce a detailed HTML file containing a full analysis of batch and covariate effects.

You can also generate the same longitudinal data report without writing Python (in bash cell) by:

```
DHarm-longitudinal run \
  --data idps_with_batch_included.csv \
  --subject-id-col subjectid_column_name \
  --timepoint-col timepoint_column_name \
  --batch-col batch_column_name \
  --features-file idps_list.txt \
  --covariates-file covariates_list.txt

```

## 5. Applying harmonisation methods

Assuming you detect significant batch effects, you would then select a harmonisation method based on which you have observed. For example, if the batch effect is only additive (difference in means) you may simply revert to regression. If the effect is more complex however, you may choose a more advanced method such as CovBat:

    from DiagnoseHarmonisation import HarmonisationFunctions

    X_new = HarmonisationFunctions.combat(X, 
    batch, 
    mod,
    covbat_mode=True
    )

## 5. Checking harmonisation efficacy

Now that you have your harmonised data, you can simply rerun the tool on the new data to see which metrics show improvement and whether or not batch effects persist in any of them. It is worth saying here that you may not require them to be completely removed depending on your experimental goal. For example, depending on your analysis, a simple mean correction may suffice.

The new comparison report is the best way to evaluate several candidate harmonisation methods against the same baseline because it keeps the diagnostics, the scorecard, and the visual summaries aligned.
