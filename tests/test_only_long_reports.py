import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from DiagnoseHarmonisation import PlotDiagnosticResults
"""
Test functions for the longitudinal code changes in the DiagnosticReport, PlotDiagnosticResults, and related modules.

This reflects advised changes discussed with Gaurav in October/September 2026.
"""

def test_longitudinal_report_runs_without_covariates(test_results_dir, monkeypatch):
    from DiagnoseHarmonisation import PlotDiagnosticResults
    monkeypatch.setattr(PlotDiagnosticResults, "clustering_analysis_all", lambda *args, **kwargs: None)
    n_samples = 80
    n_features = 10
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n_samples, n_features))
    # Introduce subject random intercepts
    subject_ID = np.array(np.linspace(1, n_samples, n_samples, dtype=int))

    # Number of time points per subject
    n_timepoints = 4

    # Make data array with subject random intercepts
    # Data array is now 4 times as many rows:
    data = np.repeat(data, n_timepoints, axis=0)
    subject_array = np.repeat(subject_ID, n_timepoints)
    batch = np.array(["A"] * (n_samples // 2 * n_timepoints) + ["B"] * (n_samples // 2 * n_timepoints))
    timepoints = np.tile(np.arange(1, n_timepoints + 1), n_samples)
    
    features = [f"Feature_{i}" for i in range(n_features)]
    # Create covariates array 
    # Covaraties, age and sex, for each new timepoint, add random small age increase
    from DiagnoseHarmonisation import DiagnosticReport

    DiagnosticReport.LongitudinalReport(
        data,
        batch,
        subject_array,
        timepoints=timepoints,
        covariates=None,
        features=features,
        save_dir=str(test_results_dir / "long_no_covariates"),
        report_name="Long_without_Covariates.html",
        show=False,
        save_data=False,
        timestamped_reports=False,
    )

    report_path = test_results_dir / "long_no_covariates" / "Long_without_Covariates.html"
    assert report_path.exists() and report_path.stat().st_size > 100


# Test actual known effects:

def test_longitudinal_report_with_known_effects(test_results_dir, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(test_results_dir / "mplconfig_long_known_effects"))
    from DiagnoseHarmonisation import PlotDiagnosticResults
    monkeypatch.setattr(PlotDiagnosticResults, "clustering_analysis_all", lambda *args, **kwargs: None)
    n_samples = 100
    n_features = 50
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n_samples, n_features))
    # Introduce subject random intercepts
    subject_ID = np.array(np.linspace(1, n_samples, n_samples, dtype=int))

    # Number of time points per subject
    n_timepoints = 4

    # Make data array with subject random intercepts
    # Data array is now 4 times as many rows:
    data = np.repeat(data, n_timepoints, axis=0)
    subject_array = np.repeat(subject_ID, n_timepoints)
    batch = np.array(["A"] * (n_samples // 2 * n_timepoints) + ["B"] * (n_samples // 2 * n_timepoints))
    timepoints = np.tile(np.arange(1, n_timepoints + 1), n_samples)
    
    features = [f"Feature_{i}" for i in range(n_features)]
    # Create covariates array 
    # Covaraties, age and sex, for each new timepoint, add random small age increase

    age = np.repeat(rng.integers(20, 50, size=n_samples), n_timepoints) + np.tile(np.arange(n_timepoints), n_samples)
    sex = np.repeat(rng.choice([0, 1], size=n_samples), n_timepoints)
    covariates = np.column_stack((age, sex))
    covariate_names = ["age", "sex"]

    # create a small sex and slightly large age effect (fixed)
    data += 0.1 * age[:, np.newaxis] + 0.2 * sex[:, np.newaxis]
    # Now the data has known effects from age and sex

    # Add random subject-specific intercepts
    subject_intercepts = rng.normal(scale=0.5, size=n_samples)
    for n in range(1, n_samples + 1):   
        data[subject_array == n, :] += subject_intercepts[n - 1]

    # make covariates a dictionary for the report (why did you do this Gaurav???)
    covariates_dict = {name: covariates[:, i] for i, name in enumerate(covariate_names)}

    from DiagnoseHarmonisation import DiagnosticReport

    DiagnosticReport.LongitudinalReport(
        data,
        batch,
        subject_array,
        timepoints=timepoints,
        covariates=covariates_dict,
        features=features,
        covariate_names=covariate_names,
        save_dir=str(test_results_dir / "long_known_effects"),
        report_name="Long_with_Known_Effects.html",
        show=False,
        save_data=False,
        timestamped_reports=False,
    )

    report_path = test_results_dir / "long_known_effects" / "Long_with_Known_Effects.html"
    assert report_path.exists() and report_path.stat().st_size > 100
    report_html = report_path.read_text(encoding="utf-8")
    assert "Batch-covariate confounding diagnostics" in report_html
    assert "timepoint" in report_html


def test_longitudinal_report_warns_and_skips_rank_deficient_mahalanobis(
    test_results_dir, monkeypatch
):
    from DiagnoseHarmonisation import DiagnosticFunctionsLong, DiagnosticReport, PlotDiagnosticResults

    monkeypatch.setattr(PlotDiagnosticResults, "Long_Robust_Zscores", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotDiagnosticResults, "plot_SubjectOrder", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotDiagnosticResults, "plot_WithinSubjVar", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotDiagnosticResults, "plot_AddMultEffects", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotDiagnosticResults, "plot_MixedEffectsPart1", lambda *args, **kwargs: None)
    monkeypatch.setattr(PlotDiagnosticResults, "plot_MixedEffectsPart2", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "SubjectOrder_long",
        lambda **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "WithinSubjVar_long",
        lambda **kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "AdditiveEffect_long",
        lambda **kwargs: (pd.DataFrame({"Feature": ["f1", "f2"], "p-value": [0.2, 0.3]}), [], "full", "reduced"),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MixedEffects_long",
        lambda **kwargs: (pd.DataFrame({"IDP": ["f1", "f2"], "n_is_batchSig": [0, 0], "ICC": [0.4, 0.5]}), []),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MultiplicativeEffect_long",
        lambda **kwargs: (pd.DataFrame({"Feature": ["f1", "f2"], "p-value": [0.4, 0.5]}), []),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MultiVariateBatchDifference_long",
        lambda **kwargs: (
            {"pairwise_raw": {}, "centroid_raw": {}, "batches": ["A", "B"]},
            {"raw_diagnostics": {"covariance_rank": 1, "n_features": 2}},
        ),
    )
    monkeypatch.setattr(
        PlotDiagnosticResults,
        "plot_MultivariateBatchDifference",
        lambda **kwargs: pytest.fail("rank-deficient Mahalanobis results must not be plotted"),
    )

    n_subjects = 6
    subject_ids = np.repeat([f"s{i}" for i in range(n_subjects)], 2)
    timepoints = np.tile(["t1", "t2"], n_subjects)
    batch = np.repeat(["A", "A", "A", "B", "B", "B"], 2)
    data = np.column_stack((np.arange(12, dtype=float), np.arange(12, dtype=float)))
    report_path = test_results_dir / "long_rank_deficient" / "Long_Rank_Deficient.html"
    DiagnosticReport.LongitudinalReport(
        data=data,
        batch=batch,
        subject_ids=subject_ids,
        timepoints=timepoints,
        covariates=None,
        features=["f1", "f2"],
        save_dir=test_results_dir / "long_rank_deficient",
        report_name="Long_Rank_Deficient",
        save_data=False,
        timestamped_reports=False,
    )

    report_html = report_path.read_text(encoding="utf-8")
    assert "Longitudinal Mahalanobis distance skipped" in report_html
    assert "Between-subject variability" in report_html


def test_long_with_non_numeric_covariates(test_results_dir, monkeypatch):
    from DiagnoseHarmonisation import PlotDiagnosticResults
    monkeypatch.setattr(PlotDiagnosticResults, "plot_MixedEffectsPart1", lambda *args, **kwargs: None)
    monkeypatch.setenv("MPLCONFIGDIR", str(test_results_dir / "mplconfig_long_known_effects"))
    from DiagnoseHarmonisation import PlotDiagnosticResults
    monkeypatch.setattr(PlotDiagnosticResults, "clustering_analysis_all", lambda *args, **kwargs: None)
    n_samples = 100
    n_features = 50
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n_samples, n_features))
    # Introduce subject random intercepts
    subject_ID = np.array(np.linspace(1, n_samples, n_samples, dtype=int))

    # Number of time points per subject
    n_timepoints = 4

    # Make data array with subject random intercepts
    # Data array is now 4 times as many rows:
    data = np.repeat(data, n_timepoints, axis=0)
    subject_array = np.repeat(subject_ID, n_timepoints)
    batch = np.array(["A"] * (n_samples // 2 * n_timepoints) + ["B"] * (n_samples // 2 * n_timepoints))
    timepoints = np.tile(np.arange(1, n_timepoints + 1), n_samples)
    
    features = [f"Feature_{i}" for i in range(n_features)]
    # Create covariates array 
    # Covaraties, age and sex, for each new timepoint, add random small age increase

    age = np.repeat(rng.integers(20, 50, size=n_samples), n_timepoints) + np.tile(np.arange(n_timepoints), n_samples)
    sex = np.repeat(rng.choice(['M', 'F'], size=n_samples), n_timepoints)
    covariates = np.column_stack((age, sex))
    covariate_names = ["age", "sex"]

    # create a small sex and slightly large age effect (fixed)
    data += 0.1 * age[:, np.newaxis] + 0.2 * (sex[:, np.newaxis] == 'M')  # Convert sex to numeric effect
    # Now the data has known effects from age and sex

    # Add random subject-specific intercepts
    subject_intercepts = rng.normal(scale=0.5, size=n_samples)
    for n in range(1, n_samples + 1):   
        data[subject_array == n, :] += subject_intercepts[n - 1]

    # make covariates a dictionary for the report (why did you do this Gaurav???)
    covariates_dict = {name: covariates[:, i] for i, name in enumerate(covariate_names)}

    from DiagnoseHarmonisation import DiagnosticReport

    DiagnosticReport.LongitudinalReport(
        data,
        batch,
        subject_array,
        timepoints=timepoints,
        covariates=covariates_dict,
        features=features,
        covariate_names=covariate_names,
        save_dir=str(test_results_dir / "long_known_effects"),
        report_name="Long_with_Known_Effects.html",
        show=False,
        save_data=False,
        timestamped_reports=False,
    )

    report_path = test_results_dir / "long_known_effects" / "Long_with_Known_Effects.html"
    assert report_path.exists() and report_path.stat().st_size > 100
    report_html = report_path.read_text(encoding="utf-8")
    assert "Batch-covariate confounding diagnostics" in report_html
    assert "timepoint" in report_html