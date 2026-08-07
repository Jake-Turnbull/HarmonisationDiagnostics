from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import DiagnosticReport


def _build_longitudinal_demo_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_subjects = 8
    n_timepoints = 3
    n_features = 4

    subject_ids = np.repeat([f"s{i+1:02d}" for i in range(n_subjects)], n_timepoints)
    timepoints = np.tile(["t1", "t2", "t3"], n_subjects)

    # Keep batches balanced across rows while preserving repeated measures layout.
    batch = np.array(["A"] * (len(subject_ids) // 2) + ["B"] * (len(subject_ids) - len(subject_ids) // 2))

    raw = rng.normal(loc=0.0, scale=1.0, size=(len(subject_ids), n_features))
    raw[batch == "A", :] += 0.75

    harmonised = raw.copy()
    harmonised[batch == "A", :] -= 0.60

    covariates = {
        "age": np.linspace(30.0, 65.0, len(subject_ids)).tolist(),
        "sex": np.tile([0, 1], len(subject_ids) // 2).tolist(),
    }

    features = [f"feature_{i+1}" for i in range(n_features)]

    return {
        "datasets": {"Raw": raw, "ShiftCorrected": harmonised, "ShiftCorrected2": harmonised + 0.5},
        "batch": batch,
        "subject_ids": subject_ids,
        "timepoints": timepoints,
        "covariates": covariates,
        "features": features,
    }


def test_longitudinal_comparison_report_generates_html(test_results_dir):
    data = _build_longitudinal_demo_data()
    output_dir = test_results_dir / "longitudinal_comparison_report"

    report = DiagnosticReport.LongitudinalComparisonReport(
        datasets=data["datasets"],
        batch=data["batch"],
        subject_ids=data["subject_ids"],
        timepoints=data["timepoints"],
        covariates=data["covariates"],
        covariate_names=["age", "sex"],
        features=data["features"],
        save_data=False,
        save_dir=output_dir,
        report_name="Longitudinal_Comparison_Report",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
    )

    report_path = Path(report.report_path)
    assert report_path.exists() and report_path.stat().st_size > 100
    assert report_path.suffix == ".html"

    assert isinstance(report.comparison_results, dict)
    assert isinstance(report.comparison_scorecard, pd.DataFrame)
    assert isinstance(report.comparison_advice, dict)

    required_columns = {
        "subject_stability_score",
        "batch_removal_score",
        "biological_preservation_score",
        "overall_score",
        "overall_rank",
        "median_spearman_rho",
        "median_icc",
    }
    assert required_columns.issubset(set(report.comparison_scorecard.columns))

    assert report.comparison_advice.get("best_subject_stability") is not None
    assert report.comparison_advice.get("best_overall") is not None


def test_longitudinal_comparison_report_no_covariates_runs(test_results_dir):
    data = _build_longitudinal_demo_data(seed=123)
    output_dir = test_results_dir / "longitudinal_comparison_report_no_covariates"

    report = DiagnosticReport.LongitudinalComparisonReport(
        datasets=data["datasets"],
        batch=data["batch"],
        subject_ids=data["subject_ids"],
        timepoints=data["timepoints"],
        covariates=None,
        features=data["features"],
        save_data=False,
        save_dir=output_dir,
        report_name="Longitudinal_Comparison_No_Covariates",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
    )

    report_path = Path(report.report_path)
    assert report_path.exists() and report_path.suffix == ".html"
    assert "overall_score" in report.comparison_scorecard.columns


def test_longitudinal_comparison_report_rejects_shape_mismatch():
    data = _build_longitudinal_demo_data(seed=99)
    datasets = dict(data["datasets"])
    datasets["Broken"] = datasets["Raw"][:-1, :]

    with pytest.raises(ValueError, match="identical shape"):
        DiagnosticReport.LongitudinalComparisonReport(
            datasets=datasets,
            batch=data["batch"],
            subject_ids=data["subject_ids"],
            timepoints=data["timepoints"],
            covariates=data["covariates"],
            features=data["features"],
            save_data=False,
            timestamped_reports=False,
        )
