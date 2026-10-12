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


def test_longitudinal_comparison_skips_rank_deficient_mahalanobis(monkeypatch):
    from DiagnoseHarmonisation import DiagnosticFunctionsLong

    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "SubjectOrder_long",
        lambda **kwargs: pd.DataFrame({"SpearmanRho": [0.6], "pValue": [0.02]}),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "WithinSubjVar_long",
        lambda **kwargs: pd.DataFrame(
            {"subject": ["s1"], "n_obs": [2], "metric_type": ["RPD"], "f1": [3.0]}
        ),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "AdditiveEffect_long",
        lambda **kwargs: (pd.DataFrame({"p-value": [0.2]}), []),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MixedEffects_long",
        lambda **kwargs: (pd.DataFrame({"ICC": [0.7], "n_is_batchSig": [1]}), []),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MultiplicativeEffect_long",
        lambda **kwargs: (pd.DataFrame({"p-value": [0.3]}), []),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "MultiVariateBatchDifference_long",
        lambda **kwargs: (
            {"pairwise_raw": {}, "centroid_raw": {}, "batches": ["A", "B"]},
            {"raw_diagnostics": {"covariance_rank": 1, "n_features": 2}},
        ),
    )

    data = np.arange(24, dtype=float).reshape(12, 2)
    result = DiagnosticReport._run_single_longitudinal_method_diagnostics(
        method_name="RankDeficient",
        data=data,
        batch=np.array(["A"] * 6 + ["B"] * 6),
        subject_ids=np.repeat([f"s{i}" for i in range(6)], 2),
        timepoints=np.tile(["t1", "t2"], 6),
        covariates=None,
        feature_names=["f1", "f2"],
    )

    assert result.multivariate_batch_difference is None
    assert any("rank deficient (1 of 2)" in warning for warning in result.errors)
    assert result.subject_order is not None
    assert np.isnan(result.summary_metrics["mean_mahalanobis_distance"])


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
