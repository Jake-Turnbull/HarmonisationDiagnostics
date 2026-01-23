# DiagnoseHarmonize v0.9.1

DiagnoseHarmonize is an **in-development** library for the streamlined application and assessment of harmonization algorithms at the summary-measure level. It also serves as a centralised location for popular, well-validated harmonization methods from the literature.

In an upcoming paper, we plan to demonstrate that systematic evaluation and reporting of different components of batch effects is not only beneficial for choosing an appropriate harmonisation strategy, but essential for evaluating how well harmonisation has worked.

## Installation and Usage

Install by downloading directly or by running: git clone https://github.com/Jake-Turnbull/HarmonizationDiagnostics.git in the terminal.

Load different components of the module by calling

```
from DiagnoseHarmonization import ModuleName
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
Assumes that each row represents an independent sample. This report focuses on:
Statistical differences between batches
Potential confounding between batch and covariates
It should be used before harmonisation to assess:
Severity of mean differences
Scaling differences
Distributional differences
It also provides visualisation of batch effects and covariates using PCA clustering and correlation analyses.

    Arguments:
        data (np.ndarray): Data matrix (samples × features)
        batch (list or np.ndarray): Batch labels for each sample
        covariates (np.ndarray, optional): Covariate matrix (samples × covariates)
        covariate_names (list of str, optional): Names of covariates

    
**LongitudinalReport():**
Requires an additional vector of subject IDs. Longitudinal harmonisation has the added goal of ensuring that between-subject variability is preserved or recovered after harmonisation.
This report assesses additive, multiplicative, and distributional components of batch effects under the assumption that batch effects affect all observations of a participant similarly across features.
It also evaluates consistency of subject ranking across sites (e.g. if subject A has larger ROI values than subject B at one site, this ordering should be preserved across sites).

    
    Arguments:      
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        subject_ids (list or np.ndarray): Subject IDs for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.)


There are additionally optional arguments for each function to allow users to specify savepath, data etc
    Optional arguments for each function: 
        save_data (bool, optional): Whether to save input data and results, default False
        save_data_name (str, optional): Filename for saved data: Will generate one if not given
        save_dir (str or os.PathLike, optional): Directory to save report and data: Will save to working directory if not given
        report_name (str, optional): Name of the report file: Generate name based on date and time if not given
        SaveArtifacts (bool, optional): Whether to save plots: Default False
        rep (StatsReporter, optional): Existing report object to use: NOT RECOMMENDED TO CHANGE (the statsreporter functions will handle this in the function)
        show (bool, optional): Whether to display plots interactively: Default False

**Future implementations**
We also plan to add a third function, UnknownBatchReport(), which would be applied to datasets where there aren't distinct batches, or cases where there are many batches each with small sample sizes (e.g < 20). This script would take image quality metrics (IQMs) for combination with new methods for harmonisation that don't use a batch label.

## LoggingTool.py

Enhanced logging and HTML report generation for diagnostic reports.
Provides the StatsReporter class that allows logging text and plots, organizing them into sections, and writing a structured HTM report with a table of contents.
If individuals would like to use this library to create their own analysis scripts, we suggest using the logging tool as an easy way to organise and return results (see script for more detail)

**Key Functions:**

    - log_section(section_id, title): mark a new named section in the log 
    - log_plot(fig, caption, section=None): attach a plot to a section (defaults to last section)
    - write_report(...) builds a TOC with hyperlinks and places each section's plots immediately after its logs.

## DiagnosticFunctions.py

Definitions for each of the functions called by the different reporting tools in DiagnosticReport.py are written here.
Each function will show either the additive, multiplicative or distribution difference between batches. Additionally, covariate effects using Linear models (either mixed effects or if this cannot be fit, fixed effects through OLS)
    
**Functions include:**

    - Cohens_D: Calculate Cohen's d effect size between batches for each feature.
    - Mahalanobis_Distance: Calculate Mahalanobis distance between batches.
    - PC_Correlations: Perform PCA and correlate top PCs with batch and covariates.
    - fit_lmm_safe: Robustly fit a Linear Mixed Model with fallbacks and diagnostics.
    - Variance_Ratios: Calculate variance ratios between batches for each feature.
    - KS_Test: Performs two-sample Kolmogorov-Smirnov test between batches for each feature.

## PlotDiagnosticResults.py

Complementary plotting functions for the functions in DiagnosticFunctions.py
**Functions Include:**

    - Z_Score_Plot: Plot histogram and heatmap of Z-scored data by batch.
    - Cohens_D_plot: Plot Cohen's d effect sizes with histograms.
    - variance_ratio_plot: Plot variance ratios between batches.
    - PC_corr_plot: Generate PCA diagnostic plots including scatter plots and correlation heatmaps.
    - PC_clustering_plot: K-means clustering and silhouette analysis of PCA results by batch.
    - Ks_Plot: Plot KS statistic between batches.

## HarmonizationFunctions.py

Collection of widely used functions for applying harmonisation to tabular data:

    - ComBat: Use ComBat to harmonise a given dataset 
    - lme_harmonization: Use linear mixed modelling to remove an estimated batch effect from the data
    Future implementations:
    - Long_Combat
    - CovBat
    - ComBat-GAMs
    - IQM_harmonisation  

## Simulator.py

Batch effect simulator that opens an interactive web-browser and allows the user to generate simulated datasets with varying numbers of unique batches,
severity of batch effects (additive and multiplicative) and different covariate effects. 

The user can then visualise the feature-wise difference in batches using histograms and box-plots, generate a cross-sectional diagnostic report to view the effects in more detail and       apply harmonisation (using ComBat). This allows the user to get a direct comparisson of the before/after of applying harmonisation by comparing the reports in a semi-realistic scenario.

To run the simulator, run **streamlit run simulator.py** in the terminal
    
