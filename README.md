# DiagnoseHarmonise (DHARM)


[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19595960-blue)](https://doi.org/10.5281/zenodo.19595960)


DiagnoseHarmonize is an **in-development** library for the streamlined application and assessment of harmonisation algorithms at the summary-measure level. It also serves as a centralised location for popular, well-validated harmonisation methods from the literature. Full documentation is available here **[DiagnoseHarmonisation](https://jake-turnbull.github.io/HarmonisationDiagnostics/)**.

In an upcoming paper, we plan to demonstrate that systematic evaluation and reporting of different components of batch effects is not only beneficial for choosing an appropriate harmonisation strategy, but essential for evaluating how well harmonisation has worked.

## Installation and Usage

Install by downloading directly or by running: pip install git+https://github.com/Jake-Turnbull/HarmonisationDiagnostics.git in the terminal.

Load different components of the module by calling

```
from DiagnoseHarmonisation import ModuleName
```
The commands can then be ran using ModuleName.FunctionName()

The two main commands are those in the DiagnosticReport module:
```
DiagnosticReport.CrossSectionalReport()
DiagnosticReport.LongitudinalReport()
```

## Support and Contact

If you find any issues or bugs in the code, please raise an issue or contact one of the following:

- **Jake Turnbull**: [jacob.turnbull@ndcn.ox.ac.uk](mailto:jacob.turnbull@ndcn.ox.ac.uk)
- **Gaurav Bhalerao**: [gaurav.bhalerao@ndcn.ox.ac.uk](mailto:gaurav.bhalerao@ndcn.ox.ac.uk)

---

## Overview

This library is intended to support the streamlined analysis and application of harmonisation for MRI data. Consistent reporting of different components of batch differences should be carried out both pre- and post-harmonisation, both to confirm that harmonisation was needed and to verify that it was successful. 

While this tool was developed for MRI data, there is no inherent reason it cannot be used in other research scenarios.

The purpose of harmonisation is to remove technical variation driven by differences in data acquisition (e.g. across sites), while preserving meaningful biological signals of interest.

Harmonisation efficacy should therefore be assessed across two broad categories:

1. **Reduction or removal of batch effects**, i.e. unwanted technical differences between datasets.
2. **Preservation of biological signal**, ensuring that meaningful variability is retained.

This library provides a set of functions to assess the severity, nature, and distribution of batch effects across features in multi-batch data. These diagnostics are intended to provide guidance on the most appropriate harmonisation strategy to apply.

Harmonisation is goal-specific, so its integration into experimental design should be carefully considered. Diagnostic reports can serve as a practical method for informing experimental design decisions.

## DiagnosticReport.py

Main set of callable functions. Takes in data, batch and covariates to provide a statistical analysis of batch differences and covariate effects within the data, returning a structured report that assess each component of the data. 

The library currently offers two main implementations, one for cross sectional data and one for longitudinal data:

**CrossSectionalReport():**

Single callable function that takes a data set and batch, returning a full organised analysis as a single easily understood HTML file.

    Arguments:  
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        
    Optional arguments:
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results. Default = True
        save_data_name (str, optional): Filename for saved data. Default = Report name
        save_dir (str or os.PathLike, optional): Directory to save report and data. Default = pwd
        report_name (str, optional): Name of the report file. Default = CrossSectionalReport_timestamp
        SaveArtifacts (bool, optional): Whether to save plots as pngs. Default = False 
        rep (StatsReporter, optional): Existing report object to use. Default = Generate report object
        show (bool, optional): Whether to display plots interactively. Default = False (recommended to keep as false)
        
**LongitudinalReport():**
Requires an additional vector of subject IDs. Longitudinal harmonisation has the added goal of ensuring that between-subject variability is preserved or recovered after harmonisation.
This report assesses additive, multiplicative, and distributional components of batch effects under the assumption that batch effects affect all observations of a participant similarly across features.
It also evaluates consistency of subject ranking across sites (e.g. if subject A has larger ROI values than subject B at one site, this ordering should be preserved across sites).

    
    Arguments: 
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        subject_ids (list or np.ndarray): Subject IDs for each sample.

    Optional Arguments:
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.

## LoggingTool.py

Enhanced logging and HTML report generation for diagnostic reports.
Provides the StatsReporter class that allows logging text and plots, organizing them into sections, and writing a structured HTM report with a table of contents.
If individuals would like to use this library to create their own analysis scripts, we suggest using the logging tool as an easy way to organise and return results (see script for more detail)

## DiagnosticFunctions.py

Definitions for each of the functions called by the different reporting tools in DiagnosticReport.py are written here. These functions can be called independantly of the main diagnostic reports if the user would prefer to focus on a single test.

## PlotDiagnosticResults.py

Complementary plotting functions for the functions in DiagnosticFunctions.py. Some of these functions require the output from a corresponding diagnostic function in order to run so keep this in mind this if using them outside of the reports

## HarmonisationFunctions.py

While not the main purpose of this library, we do provide access to some well validated harmonisation methods for derived measures. These have all been tested and confirmed to be within machine precision to the other more widely used publically available versions.


## Simulator.py

Batch effect simulator that opens an interactive web-browser and allows the user to generate simulated datasets with varying numbers of unique batches,
severity of batch effects (additive and multiplicative) and different covariate effects.

The user can then visualise the feature-wise difference in batches using histograms and box-plots, generate a cross-sectional diagnostic report to view the effects in more detail and apply harmonisation (using ComBat). This allows the user to get a direct comparisson of the before/after of applying harmonisation by comparing the reports in a semi-realistic scenario.

To run the simulator, run **streamlit run simulator.py** in the terminal.
