from __future__ import annotations

from pathlib import Path

import numpy as np

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