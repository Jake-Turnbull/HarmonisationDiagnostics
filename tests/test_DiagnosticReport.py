import os
import numpy as np
from pathlib import Path
import pytest
import time
import webbrowser

# Adjust import to match your package layout
# from DiagnoseHarmonization.DiagnosticReport import DiagnosticReport
# If DiagnosticReport is defined in a module file DiagnosticReport.py inside DiagnoseHarmonization package:

save_dir = "/Users/jacob.turnbull/VS_code_projects/diagnostic_full_run/"

def test_full_pipeline_generates_report(tmp_path = save_dir):
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

    batch = np.array(["Siemens"] * int(n_samples/2) + ["Philips"] * int(n_samples/8)  + ["GE"] * int(n_samples/8)  + ["Magnetom"] * int(n_samples/4) )
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
    Report_name="Test_run"
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
        from DiagnoseHarmonization import DiagnosticReport
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

    # -------------------------
    # Find the generated report
    # -------------------------
    
    # Check for report with expected name pattern defined by variable Report_name

    if Report_name is None:
        Report_name = "DiagnosticReport"
        if timestamped_reports == True or timestamped_reports == None:
        # If timestamped, we need to match the pattern with wildcard
            Report_name = "DiagnosticReport"
            reports = sorted(out_dir.glob(f"{Report_name}_*.html"))
            assert len(reports) > 0, f"HTML with the right name file was not generated, expected pattern: looked for file: {Report_name}_*.html in {out_dir}"
    else:   
        reports = sorted(out_dir.glob(f"{Report_name}*.html"))
        print(reports)
        assert len(reports) > 0, f"HTML with the right name file was not generated, expected pattern: looked for file: {Report_name}*.html in {out_dir}"

    # pick the most recent report
    report_path = reports[-1]

    # Basic sanity checks
    assert report_path.exists() and report_path.stat().st_size > 100, "Report file is missing or unexpectedly small."

    # Print the full path so the tester can open it manually; -s flag required to see this in pytest output
    print("\n==== Diagnostic report generated ====")
    print(f"Report path: {report_path}")
    print("Open this file in your browser to view the report.")
    print("====================================\n")

    # Success
    assert True
    #%%
    covariate_cat = np.random.randint(0, 1, size=n_samples)    # categorical
    print( covariate_cat)


def test_min_script(tmp_path=save_dir):
    """
    Test that the minimal script runs without error and produces a report.
    This is a more basic test than test_full_pipeline_generates_report, and can be used to quickly check that the core functionality of DiagnosticReport is working.
    """
    # Minimal data and batch
    data = np.random.randn(100, 20)
    batch = np.array(["A"] * 50 + ["B"] * 50)
    # Run DiagnosticReport with minimal inputs
    try:
        from DiagnoseHarmonization import DiagnosticReport
        DiagnosticReport.CrossSectionalReportMin(
            data=data,
            batch=batch,
            save_dir=str(tmp_path),
            report_name="Minimal_Test",
            show=False,
            timestamped_reports=False,
            save_data=False
        )
    except Exception as e:
        pytest.fail(f"DiagnosticReport raised an exception in minimal script: {e}")

    # Check that report was generated
    report_path = Path(tmp_path) / "Minimal_Test.html"
    assert report_path.exists() and report_path.stat().st_size > 100, "Minimal script did not generate expected report."