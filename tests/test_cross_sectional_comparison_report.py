from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from DiagnoseHarmonisation import DiagnosticReport


def test_cross_sectional_comparison_report_generates_html(test_results_dir):
    rng = np.random.default_rng(42)
    n_samples = 24
    n_features = 6

    batch = np.array(["A"] * 12 + ["B"] * 12)
    covariates = np.column_stack(
        (
            np.linspace(25, 55, n_samples),
            np.tile([0, 1], n_samples // 2),
        )
    )
    feature_names = [f"feature_{index + 1}" for index in range(n_features)]
    covariate_names = ["age", "sex"]

    raw = rng.normal(size=(n_samples, n_features))
    raw[:12] += 0.8

    harmonised = raw.copy()
    harmonised[:12] -= 0.6

    output_dir = test_results_dir / "comparison_report"
    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw, "ShiftCorrected": harmonised},
        batch=batch,
        covariates=covariates,
        covariate_names=covariate_names,
        feature_names=feature_names,
        save_dir=output_dir,
        save_data=False,
        report_name="Comparison_Report",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    report_path = Path(report.report_path)
    assert report_path.exists() and report_path.stat().st_size > 100


def test_cross_sectional_comparison_report_accepts_pandas_inputs_and_saves_enriched_outputs(test_results_dir):
    rng = np.random.default_rng(7)
    n_samples = 20
    n_features = 8

    index = [f"sub_{i:02d}" for i in range(n_samples)]
    columns = [f"feat_{j+1}" for j in range(n_features)]
    raw = pd.DataFrame(rng.normal(size=(n_samples, n_features)), index=index, columns=columns)
    corrected = raw.copy()
    corrected.iloc[:10, :] = corrected.iloc[:10, :] - 0.5

    batch_df = pd.DataFrame({"batch": np.array(["A"] * 10 + ["B"] * 10)}, index=index)
    covariates_df = pd.DataFrame(
        {
            "age": np.linspace(30, 60, n_samples),
            "sex": np.tile([0, 1], n_samples // 2),
        },
        index=index,
    )

    output_dir = test_results_dir / "comparison_report_saved"
    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw, "Corrected": corrected},
        batch=batch_df,
        covariates=covariates_df,
        covariate_names=["age", "sex"],
        feature_names=None,
        save_dir=output_dir,
        save_data=True,
        save_data_name="compare",
        report_name="Comparison_Report_Saved",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    saved = report.comparison_saved_paths
    assert "Raw" in saved
    assert "Corrected" in saved

    required_keys = {
        "pca_explained_variance",
        "pca_correlations",
        "covariance_frobenius",
        "covariance_frobenius_normalized",
        "batch_scree",
    }
    assert required_keys.issubset(set(saved["Raw"].keys()))

    for key in required_keys:
        assert Path(saved["Raw"][key]).exists()


def test_cross_sectional_comparison_report_skips_lmm_when_covariates_missing(test_results_dir):
    rng = np.random.default_rng(11)
    n_samples = 16
    n_features = 5

    batch = np.array(["A"] * 8 + ["B"] * 8)
    raw = rng.normal(size=(n_samples, n_features))
    harmonised = raw.copy()
    harmonised[:8] -= 0.3

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw, "Harmonised": harmonised},
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_no_covariates",
        save_data=False,
        report_name="Comparison_No_Covariates",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    for method_result in report.comparison_results.values():
        assert isinstance(method_result.lmm_summary, dict)
        assert method_result.lmm_summary.get("status") == "skipped_missing_covariates"