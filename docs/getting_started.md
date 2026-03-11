# Getting started 

Here we will provide a brief example of how to use DiagnoseHarmonization within a standard workflow, giving an example of how one would use the python version (which has full functionality) and how one would use the terminal instance of the tool (on csv files).

By far the easiest way to run this code is by using a python script and loading your data in as arrays.

## 1. Install from Github:

In terminal, run:

    pip install git+https://github.com/Jake-Turnbull/HarmonizationDiagnostics.git

Or alternatively clone locally:
    git clone https://github.com/Jake-Turnbull/HarmonizationDiagnostics.git
    cd HarmonizationDiagnostics
    pip install -e .

## 2. Data requirements:

The minimum arguments required to run DiagnoseHarmonize are:

    data: NumPy array (samples x features)
    batch: Vector (array or list) of batch labels (Samples x 1)
    Covariates: NumPy array (Samples x covariates)

For additional arguments, please check the DiagnosticReports docs.
Note, while covariates aren't inherently required, in order to get an informative result they are recommended. In the case that no covariates are used, the CrossSectionalReport will throw an error. This will be fixed in a later patch but for now please include an intercept (vector of ones) as a placeholder.

## 3. Generate a Cross-Sectional Diagnostic Report:

There are two main functions for the generation of a cross-sectional report, a full one with detailed analysis across multiple different metrics and advanced visualisations and a minimal version, which simply returns additive, multiplicative and a visual representation of overall distributional differences.

Using the full report:

    from DiagnoseHarmonization import DiagnosticReport
        report = DiagnosticReport.CrossSectionalReport(
            data=X,
            batch=batch,
            covariates=covars)

This will produce a detailed HTML file containing a full analysis of batch and covariate effects.

## 4. Applying harmonization methods:

Assuming you detect significant batch effects, you would then select a harmonisation method based on which you have observed. For example, if the batch effect is only additive (difference in means) you may simply revert to regression. If the effect is more complex however, you may choose a more advanced method such as CovBat:

    from DiagnoseHarmonization import HarmonizationFunctions

    X_new = HarmonizationFunctions.combat(X, 
    batch, 
    mod,
    covbat_mode=True
    )

## 5. Checking harmonization efficacy

Now that you have your harmonised data, you can simply rerun the tool on the new data to see which metrics show improvement and whether or not batch effects persist in any of them. It is worth saying here that you may not require them to be completely removed depending on your experimental goal. For example, depending on your analysis, a simple mean correction may suffice.
