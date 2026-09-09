from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DiagnoseHarmonisation import DiagnosticReport
from DiagnoseHarmonisation import DiagnosticFunctions
from DiagnoseHarmonisation import PlotComparisonResults


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
        UMAP_embedding=True,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
        probability_distribution=False,
    )

    report_path = Path(report.report_path)
    assert report_path.exists() and report_path.stat().st_size > 100
    assert "biological_effects_score" in report.comparison_scorecard.columns
    assert "weighted_covariate_pc_correlation_top3" in report.comparison_scorecard.columns
    assert report.comparison_advice.get("best_biological") is not None


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

def test_cross_sectional_comparison_report_unbalanced_batches(test_results_dir):
    rng = np.random.default_rng(17)
    n_samples = 3000
    n_features = 7
    batch = np.array(["A"] * 500 + ["B"] * 2500)
    assert len(batch) == n_samples
    raw = rng.normal(size=(n_samples, n_features))
    harmonised = raw.copy()
    harmonised[:500] -= 0.5

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw, "Harmonised": harmonised},
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_unbalanced_batches",
        save_data=False,
        report_name="Comparison_Unbalanced_Batches",
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

def test_cross_sectional_accepts_missing_data(test_results_dir):
    rng = np.random.default_rng(13)
    n_samples = 18
    n_features = 4

    batch = np.array(["A"] * 9 + ["B"] * 9)
    raw = rng.normal(size=(n_samples, n_features))
    harmonised = raw.copy()
    harmonised[:9] -= 0.4

    # Introduce missing values
    raw[2, 1] = np.nan
    harmonised[5, 3] = np.nan

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw, "Harmonised": harmonised},
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_missing_data",
        save_data=False,
        report_name="Comparison_Missing_Data",
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


def test_compare_zscore_histograms_use_per_batch_proportions():
    class _Result:
        pass

    batch = np.array(["A"] * 6 + ["B"] * 18)

    # Same within-batch shape but different sample counts.
    vals_a = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype=float)
    vals_b = np.tile(np.array([-1.0, 0.0, 1.0], dtype=float), 6)
    z = np.concatenate([vals_a, vals_b]).reshape(-1, 1)

    res = _Result()
    res.zscore_raw = z

    figs = PlotComparisonResults.plot_compare_zscore_distributions(
        {"Method": res},
        batch=batch,
        use_residual=False,
        probability_distribution=True,
    )
    assert len(figs) == 1

    _, fig = figs[0]
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Proportion within batch"

    n_bins = 59  # np.linspace(-5, 5, 60) -> 59 bars per batch
    heights = np.array([patch.get_height() for patch in ax.patches], dtype=float)
    assert heights.size == 2 * n_bins

    heights_a = heights[:n_bins]
    heights_b = heights[n_bins:]

    # Each batch histogram should sum to 1.0 (per-batch probabilities).
    assert np.isclose(np.sum(heights_a), 1.0)
    assert np.isclose(np.sum(heights_b), 1.0)

    # Same underlying distribution should give matching bar heights despite unequal counts.
    assert np.allclose(heights_a, heights_b)
    plt.close(fig)


def test_compare_zscore_histograms_can_use_frequencies():
    class _Result:
        pass

    batch = np.array(["A"] * 6 + ["B"] * 18)
    vals_a = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0], dtype=float)
    vals_b = np.tile(np.array([-1.0, 0.0, 1.0], dtype=float), 6)
    z = np.concatenate([vals_a, vals_b]).reshape(-1, 1)

    res = _Result()
    res.zscore_raw = z

    figs = PlotComparisonResults.plot_compare_zscore_distributions(
        {"Method": res},
        batch=batch,
        use_residual=False,
        probability_distribution=False,
    )
    assert len(figs) == 1

    _, fig = figs[0]
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Frequency"

    heights = np.array([patch.get_height() for patch in ax.patches], dtype=float)
    n_bins = 59
    assert heights.size == 2 * n_bins

    # In frequency mode, each batch histogram sums to its own sample count.
    assert np.isclose(np.sum(heights[:n_bins]), vals_a.size)
    assert np.isclose(np.sum(heights[n_bins:]), vals_b.size)
    plt.close(fig)


def test_compare_effect_plots_only_share_y_limits_when_requested():
    class _Result:
        def __init__(self, cohens_d, variance_ratios):
            self.cohens_d = cohens_d
            self.variance_ratios = variance_ratios

    results = {
        "Small": _Result(np.array([[0.1, 0.2]]), np.array([[1.1, 0.9]])),
        "Large": _Result(np.array([[3.0, 4.0]]), np.array([[4.0, 0.25]])),
    }

    for plotter in (
        PlotComparisonResults.plot_compare_cohens_d,
        PlotComparisonResults.plot_compare_variance_ratios,
    ):
        _, automatic_fig = plotter(results)[-1]
        automatic_limits = [ax.get_ylim() for ax in automatic_fig.axes[:2]]
        assert automatic_limits[0] != automatic_limits[1]
        plt.close(automatic_fig)

        _, shared_fig = plotter(results, shared_y=True)[-1]
        shared_limits = [ax.get_ylim() for ax in shared_fig.axes[:2]]
        assert shared_limits[0] == shared_limits[1]
        plt.close(shared_fig)


def test_comparison_report_uses_global_zscore_not_per_batch(test_results_dir):
    rng = np.random.default_rng(123)
    n_a, n_b = 8, 24
    n_features = 5

    batch = np.array(["A"] * n_a + ["B"] * n_b)
    raw = rng.normal(size=(n_a + n_b, n_features))
    raw[:n_a, :] += 3.0

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets={"Raw": raw},
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "cross_sectional_report_no_covariates",
        save_data=False,
        report_name="Comparison_Global_Zscore",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    observed = np.asarray(report.comparison_results["Raw"].zscore_raw, dtype=float)
    expected_global = DiagnosticFunctions.robust_z_score(raw)

    per_batch = np.empty_like(raw, dtype=float)
    per_batch[batch == "A", :] = DiagnosticFunctions.robust_z_score(raw[batch == "A", :])
    per_batch[batch == "B", :] = DiagnosticFunctions.robust_z_score(raw[batch == "B", :])

    assert np.allclose(observed, expected_global, equal_nan=True)
    assert not np.allclose(observed, per_batch, equal_nan=True)


def test_cross_sectional_works_with_many_datasets(test_results_dir):
    rng = np.random.default_rng(123)
    n_a, n_b = 8, 24
    n_features = 5

    batch = np.array(["A"] * n_a + ["B"] * n_b)
    raw = rng.normal(size=(n_a + n_b, n_features))
    raw[:n_a, :] += 3.0

    datasets = {"Raw": raw, "Raw_Copy": raw.copy()}

    # Add a batch effect
    harm1 = raw.copy()
    harm2 = raw.copy()
    harm3 = raw.copy()
    harm4 = raw.copy()
    # Add noisy bias to each at different levels
    harm1[batch == "B", :] += 2.0
    harm2[batch == "B", :] += 3.0
    harm3[batch == "B", :] += 4.0
    harm4[batch == "B", :] += 5.0

    datasets["Harm1"] = harm1
    datasets["Harm2"] = harm2
    datasets["Harm3"] = harm3
    datasets["Harm4"] = harm4
  

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets=datasets,
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "cross_sectional_report_many_datasets",
        save_data=False,
        report_name="Comparison_Many_Datasets",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    assert "Raw" in report.comparison_results
    assert "Raw_Copy" in report.comparison_results
    assert "Harm1" in report.comparison_results
    assert "Harm2" in report.comparison_results
    assert "Harm3" in report.comparison_results
    assert "Harm4" in report.comparison_results

def test_cross_sectional_many_datasets_many_batches(test_results_dir):
    rng = np.random.default_rng(123)
    n_a, n_b, n_c = 8, 24, 16
    n_features = 5

    batch = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c)
    raw = rng.normal(size=(n_a + n_b + n_c, n_features))
    raw[:n_a, :] += 3.0
    raw[n_a:n_a+n_b, :] += 2.0
    raw[n_a+n_b:, :] += 1.0

    # Create five different versions of the raw data with varying batch effects
    harm1 = raw.copy()
    harm2 = raw.copy()
    harm3 = raw.copy()
    harm4 = raw.copy()
    harm5 = raw.copy()

    harm1[batch == "B", :] += 2.0
    harm2[batch == "B", :] += 3.0
    harm3[batch == "B", :] += 4.0
    harm4[batch == "B", :] += 5.0
    harm5[batch == "B", :] += 6.0

    datasets = {
        "Raw": raw,
        "Harm1": harm1,
        "Harm2": harm2,
        "Harm3": harm3,
        "Harm4": harm4,
        "Harm5": harm5
    }

    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets=datasets,
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_report_many_datasets_many_batches",
        save_data=False,
        report_name="Comparison_Many_Datasets_Many_Batches",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    assert "Raw" in report.comparison_results
    assert "Harm1" in report.comparison_results
    assert "Harm2" in report.comparison_results
    assert "Harm3" in report.comparison_results
    assert "Harm4" in report.comparison_results
    assert "Harm5" in report.comparison_results


def test_font_size_in_report(test_results_dir):
    rng = np.random.default_rng(123)    
    n_a, n_b, n_c, n_d, n_e, n_f = 8, 24, 8, 16, 12, 10
    n_features = 5
    batch = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c + ["D"] * n_d + ["E"] * n_e + ["F"] * n_f)
    raw = rng.normal(size=(n_a + n_b + n_c + n_d + n_e + n_f, n_features))  
    raw[:n_a, :] += 3.0
    raw[n_a:n_a+n_b, :] += 2.0
    raw[n_a+n_b:n_a+n_b+n_c, :] += 1.5
    raw[n_a+n_b+n_c:n_a+n_b+n_c+n_d, :] += 1.0
    raw[n_a+n_b+n_c+n_d:n_a+n_b+n_c+n_d+n_e, :] += 0.5
    raw[n_a+n_b+n_c+n_d+n_e:, :] += 0.0

    # duplicate the raw data to create a comparison dataset
    raw_copy = raw.copy()
    datasets = {
        "Raw": raw,
        "Raw_Copy": raw_copy+0.5,
        "Raw_Copy_Offset": raw_copy+1.0,
        "Raw_Copy_Offset_2": raw_copy+1.5,
        "Raw_Copy_Offset_3": raw_copy+2.0,
        "Raw_Copy_Offset_4": raw_copy+2.5,
    }

    # Run the report 3 times with different fontsize arguments:
    report_default_fontsize = DiagnosticReport.CrossSectionalComparisonReport(
        datasets=datasets,
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_report_fontsize",
        save_data=False,
        report_name="Comparison_Default_Fontsize",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
    )

    report_small_fontsize = DiagnosticReport.CrossSectionalComparisonReport(
        datasets=datasets,
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_report_fontsize",
        save_data=False,
        report_name="Comparison_Small_Fontsize",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
        fontsize=7,
    )

    report_large_fontsize = DiagnosticReport.CrossSectionalComparisonReport(
        datasets=datasets,
        batch=batch,
        covariates=None,
        covariate_names=None,
        feature_names=[f"f{i+1}" for i in range(n_features)],
        save_dir=test_results_dir / "comparison_report_fontsize",
        save_data=False,
        report_name="Comparison_Large_Fontsize",
        SaveArtifacts=False,
        show=False,
        timestamped_reports=False,
        ratio_type="rest",
        UMAP_embedding=False,
        plot_covariate_embeddings=False,
        allow_many_covariate_embeddings=False,
        fontsize=16,
    )
    assert report_default_fontsize is not None
    assert report_small_fontsize is not None
    assert report_large_fontsize is not None