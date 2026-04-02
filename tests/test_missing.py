# This is a test to see if the diagnostic report can handle missing data correctly 

import pytest
from DiagnoseHarmonisation import DiagnosticReport
import numpy as np
from pathlib import Path
import pandas as pd

save_dir = "/Users/jacob.turnbull/VS_code_projects/diagnostic_full_run/"

def test_missing_data(tmp_path = save_dir):
    """
    Run the full DiagnosticReport pipeline once and produce a single HTML report.
    - Writes report to a temporary directory (tmp_path / "diagnostic_full_run")
    - Asserts that a timestamped DiagnosticReport_*.html file was created
    - Prints the path to the report for manual viewing
    - Optionally opens the report automatically when OPEN_REPORT=1
    """
    # -------------------------
    # Prepare synthetic data
    # -------------------------
    np.random.seed(27)
    n_samples = 800
    n_features = 100

    data = np.random.randn(n_samples, n_features)
    covariate_cat = np.random.randint(0, 2, size=n_samples)    # categorical
    print(covariate_cat)
    # Mean center the categorical covariate, testing this as divide by zero errors in PCA correlations otherwise
    # Define age between 20 and 80 from normal distribution
    covariate_cont = 20 + 60 * np.random.rand(n_samples)   
    covariates = np.column_stack((covariate_cat, covariate_cont))

    covariate_cat = covariate_cat - np.mean(covariate_cat)           # mean center
    covariate_cont = covariate_cont- np.mean(covariate_cont)             # mean center

    batch = np.array(["Siemens"] * int(n_samples/4) + ["Philips"] * int(n_samples/4)  + ["GE"] * int(n_samples/4)  + ["Magnetom"] * int(n_samples/4) )
    # Construct mixed effects model to add some batch and covariate effects
    # Define age between 20 and 80 from normal distribution
    variable_names = ['Sex', 'Age']

    # Simulate more realistic batch effects
    for i in range(n_samples):
        for j in range(n_features):
            if batch[i] == "Siemens":
                # Draw from a normal distribution with a higher mean, normaly distribute positive shift along features
                data[i, j] += np.random.normal(loc=0.7, scale=1.0)
            elif batch[i] == "Philips":
                data[i, j] += np.random.normal(loc=0.25, scale=2.0)
            elif batch[i] == "GE":
                data[i, j] += np.random.normal(loc=-0.25, scale=2.5)
            elif batch[i] == "Magnetom":
                data[i, j] += np.random.normal(loc=-0.7, scale=0.6)
                
    # Simulate covariate effect of age and sex (when age increases, feature values decrease) 
    # (when sex = 0/1 (female/male), feature values decrease/increase to simulate volume differences)

    # Simulate a real covariate effect, e.g non-linearly decreasing feature value with age and normally distributed differences in sex
    for i in range(n_samples):
        for j in range(n_features):
            data[i, j] += -0.08 * (covariate_cont[i])  # Linear effect of age
            data[i, j] += 0.8 * covariate_cat[i]  # Effect of sex
    # -------------------------
    # Where to save the report
    # -------------------------
    Report_name="Test_no_missing"
    out_dir = tmp_path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # -------------------------
    # Run the DiagnosticReport
    # -------------------------
    timestamped_reports = False

    try:
        # call signature:
        # DiagnosticReport(data, batch, covariates=None, variable_names=None,
        #                  save_dir=None, SaveArtifacts=False, rep=None, show=False)
        from DiagnoseHarmonisation import DiagnosticReport
        DiagnosticReport.CrossSectionalReport(
            data=data, # Required: data matrix (samples x features)
                batch=batch, # Required: batch vector (samples,)
                    covariates=covariates, # Optional: covariate matrix (samples x covariates)
                        covariate_names=variable_names, # Optional: names of covariates
                            save_dir=str(out_dir), # Optional: directory to save report
                            save_data=True, # Whether to save data used in report, default False
                                report_name=Report_name, # Optional: base name of report file
                                SaveArtifacts=False, # Whether to save artifacts, default False
                                    rep=None, # Optional: report object
                                        show=False, # Whether to display the report, default False
                                        timestamped_reports=timestamped_reports # Whether to use timestamped report names
                                            
        )

    except Exception as e:
        # If the pipeline raises, fail the test but show exception
        pytest.fail(f"DiagnosticReport raised an exception: {e}")
    Report_name="Test_missing"
    out_dir.mkdir(parents=True, exist_ok=True)


    # Remove random data from 2 batches (i.e set to zero) to simulate missing data and test that the report can handle this without crashing
    # For each feature, randomly select 10% of samples from Siemens and 10% of samples from Philips to set to zero (simulate missing data)
    for j in range(n_features):
        # Get indices of Siemens and Philips samples
        siemens_indices = np.where(batch == "Siemens")[0]
        philips_indices = np.where(batch == "Philips")[0]
        # Randomly select 10% of indices from each batch
        siemens_missing = np.random.choice(siemens_indices, size=int(0.1 * len(siemens_indices)), replace=False)
        philips_missing = np.random.choice(philips_indices, size=int(0.1 * len(philips_indices)), replace=False)
        # Set selected indices to zero (simulate missing data)
        data[siemens_missing, j] = np.nan
        data[philips_missing, j] = np.nan
    
    try:

        DiagnosticReport.CrossSectionalReport(
            data=data, # Required: data matrix (samples x features)
                batch=batch, # Required: batch vector (samples,)
                    covariates=covariates, # Optional: covariate matrix (samples x covariates)
                        covariate_names=variable_names, # Optional: names of covariates
                            save_dir=str(out_dir), # Optional: directory to save report
                            save_data=True, # Whether to save data used in report, default False
                                report_name=Report_name+"_missing", # Optional: base name of report file
                                SaveArtifacts=False, # Whether to save artifacts, default False
                                    rep=None, # Optional: report object
                                        show=False, # Whether to display the report, default False
                                        timestamped_reports=timestamped_reports # Whether to use timestamped report names
                                            
        )
    except Exception as e:
        # If the pipeline raises, fail the test but show exception
        pytest.fail(f"DiagnosticReport raised an exception: {e}")
    Report_name="Test_missing"
    out_dir.mkdir(parents=True, exist_ok=True)
