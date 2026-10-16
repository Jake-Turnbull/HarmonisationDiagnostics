from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import PlotComparisonResults, PlotDiagnosticResults
from DiagnoseHarmonisation.DiagnosticFunctionsLong import _add_fdr_pvalues
from DiagnoseHarmonisation.DiagnosticReport import (
    LongitudinalDiagnosticResult,
    _extract_longitudinal_summary_metrics,
)


def test_longitudinal_effect_pvalues_are_bh_fdr_adjusted():
    effects = pd.DataFrame({"Feature": ["f1", "f2", "f3"], "p-value": [0.01, 0.04, np.nan]})

    adjusted = _add_fdr_pvalues(effects)

    np.testing.assert_allclose(adjusted["p-value_fdr"][:2], [0.02, 0.04])
    assert np.isnan(adjusted["p-value_fdr"].iloc[2])


def test_longitudinal_summary_uses_fdr_pvalues_when_available():
    result = LongitudinalDiagnosticResult(
        method_name="method",
        data=np.empty((0, 0)),
        additive_effects=pd.DataFrame({"p-value": [0.01, 0.2], "p-value_fdr": [0.08, 0.2]}),
        multiplicative_effects=pd.DataFrame({"p-value": [0.01, 0.2], "p-value_fdr": [0.08, 0.2]}),
    )

    metrics = _extract_longitudinal_summary_metrics(result)

    assert metrics["prop_additive_significant"] == 0.0
    assert metrics["prop_multiplicative_significant"] == 0.0


def test_comparison_plot_grid_dimensions_are_configurable():
    method_result = SimpleNamespace(
        zscore_raw=np.arange(4, dtype=float).reshape(4, 1),
        within_subject_variability=pd.DataFrame({"x": [1.0, 2.0]}),
    )

    cross_sectional_figures = PlotComparisonResults.plot_compare_zscore_distributions(
        {"method": method_result},
        batch=["A", "A", "B", "B"],
        height_in_plotting_grid=3.0,
        width_in_plotting_grid=5.0,
    )
    longitudinal_figures = PlotComparisonResults.plot_compare_long_within_subject_variability(
        {"method": method_result},
        height_in_plotting_grid=3.0,
        width_in_plotting_grid=5.0,
    )

    assert cross_sectional_figures[0][1].get_size_inches() == pytest.approx([5.0, 3.0])
    assert longitudinal_figures[0][1].get_size_inches() == pytest.approx([5.0, 3.0])
    for _, figure in cross_sectional_figures + longitudinal_figures:
        figure.canvas.draw()
        axes = figure.axes[0]
        panel_aspect = axes.get_window_extent().width / axes.get_window_extent().height
        assert panel_aspect > 1.2

    for _, figure in cross_sectional_figures + longitudinal_figures:
        plt.close(figure)


def test_covariate_adjustment_removes_subject_intercepts_and_preserves_batch():
    subject_effects = np.repeat([-3.0, -1.0, 1.0, 3.0], 2)
    batch_effects = np.tile([0.0, 5.0], 4)
    covariates = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    frame = pd.DataFrame(
        {
            "feature": subject_effects + 10.0 + batch_effects + 2.0 * covariates,
            "batch": ["A", "B"] * 4,
            "subject": np.repeat(["s1", "s2", "s3", "s4"], 2),
        }
    )
    covariate_map = {"age": covariates}

    adjusted = PlotDiagnosticResults._adjust_for_covariates_preserve_batch(
        frame,
        idp_cols=["feature"],
        batch_col="batch",
        covariates=covariate_map,
        subject_id="subject",
    )

    batch_means = adjusted.groupby("batch")["feature"].mean()
    assert adjusted.groupby("batch")["feature"].std().max() < frame.groupby("batch")["feature"].std().max()
    assert batch_means["B"] - batch_means["A"] == pytest.approx(5.0)


def test_subject_order_uses_median_rho_heatmap_and_idp_medians():
    subject_order = pd.DataFrame(
        {
            "IDP": ["a", "b", "c", "a", "b", "c"],
            "TimeA": ["t1"] * 3 + ["t2"] * 3,
            "TimeB": ["t2"] * 3 + ["t3"] * 3,
            "SpearmanRho": [0.1, 0.5, 0.9, -0.2, 0.2, 0.8],
            "pValue": [0.01, 0.2, 0.03, 0.04, 0.2, 0.01],
        }
    )

    figure = PlotDiagnosticResults.plot_SubjectOrder(subject_order, times_order=["t1", "t2", "t3"])

    assert len(figure.axes) == 3
    heatmap = figure.axes[0].images[0].get_array()
    assert heatmap[0, 1] == 0.5
    assert heatmap[1, 2] == 0.2
    idp_medians = figure.axes[1].collections[0].get_offsets()
    np.testing.assert_allclose(idp_medians[:, 0], [-0.05, 0.35, 0.85])
    plt.close(figure)


def test_within_subject_large_data_shows_one_median_dot_per_idp():
    values = np.arange(11 * 2, dtype=float).reshape(11, 2)
    df = pd.DataFrame(values, columns=["f1", "f2"])
    df.insert(0, "subject", [f"s{i}" for i in range(11)])

    figure = PlotDiagnosticResults.plot_WithinSubjVar(df, idp_cols=["f1", "f2"])

    assert len(figure.axes) == 1
    offsets = figure.axes[0].collections[0].get_offsets()
    assert len(offsets) == 2
    np.testing.assert_allclose(offsets[:, 1], [10.0, 11.0])
    assert figure.axes[0].collections[0].get_facecolors()[0, :3] == pytest.approx(
        plt.matplotlib.colors.to_rgb("tab:blue")
    )
    plt.close(figure)


def test_within_subject_small_data_keeps_subject_specific_panels():
    df = pd.DataFrame(np.arange(100, dtype=float).reshape(10, 10), columns=[f"f{i}" for i in range(10)])
    df.insert(0, "subject", [f"s{i}" for i in range(10)])

    figure = PlotDiagnosticResults.plot_WithinSubjVar(df, idp_cols=[f"f{i}" for i in range(10)])

    assert len(figure.axes) == 3
    assert len(figure.axes[0].collections) == 10
    plt.close(figure)


def test_within_subject_summary_triggers_when_only_feature_count_exceeds_ten():
    idps = [f"f{i}" for i in range(11)]
    df = pd.DataFrame(np.arange(22, dtype=float).reshape(2, 11), columns=idps)
    df.insert(0, "subject", ["s1", "s2"])

    figure = PlotDiagnosticResults.plot_WithinSubjVar(df, idp_cols=idps)

    assert len(figure.axes) == 1
    assert len(figure.axes[0].collections[0].get_offsets()) == 11
    plt.close(figure)


def test_additive_multiplicative_pvalues_are_blue_dots_with_threshold():
    effects = pd.DataFrame({"Feature": ["f1", "f2"], "p-value": [0.01, 0.1]})

    figure = PlotDiagnosticResults.plot_AddMultEffects(effects)

    assert len(figure.axes) == 1
    ax = figure.axes[0]
    assert len(ax.collections) == 1
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], [2.0, 1.0])
    assert ax.collections[0].get_facecolors()[0, :3] == pytest.approx(
        plt.matplotlib.colors.to_rgb("tab:blue")
    )
    assert any(np.isclose(line.get_ydata()[0], -np.log10(0.05)) for line in ax.lines)
    plt.close(figure)


def test_effect_plot_uses_fdr_pvalues_and_labels_significance_threshold():
    effects = pd.DataFrame(
        {
            "Feature": ["f1", "f2"],
            "p-value": [0.01, 0.1],
            "p-value_fdr": [0.02, 0.2],
        }
    )

    figure = PlotDiagnosticResults.plot_AddMultEffects(effects)

    ax = figure.axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 1], -np.log10([0.02, 0.2]))
    assert any("BH-FDR-adjusted p-value = 0.05" in text.get_text() for text in ax.texts)
    assert "BH-FDR-adjusted p-value" in ax.get_ylabel()
    plt.close(figure)


def test_mixed_effect_counts_and_icc_use_blue_dots_and_reference_lines():
    df = pd.DataFrame(
        {"IDP": ["f1", "f2"], "n_is_batchSig": [1, 3], "ICC": [0.6, 0.85], "WCV": [2, 4]}
    )

    count_figure, count_axes = PlotDiagnosticResults.plot_MixedEffectsPart1(
        df, metrics=["n_is_batchSig"], display="single", metric="n_is_batchSig"
    )
    icc_figure, icc_axes = PlotDiagnosticResults.plot_MixedEffectsPart1(
        df, metrics=["ICC"], display="single", metric="ICC"
    )
    box_mode_figure, box_mode_axes = PlotDiagnosticResults.plot_MixedEffectsPart1(
        df, metrics=["ICC"], display="single", metric="ICC", plot_type="box"
    )

    assert len(count_axes[0].collections) == 1
    assert not count_axes[0].patches
    assert len(icc_axes[0].collections) == 1
    assert {round(line.get_ydata()[0], 2) for line in icc_axes[0].lines} >= {0.5, 0.75, 0.9}
    assert len(box_mode_axes[0].collections) == 1
    assert not box_mode_axes[0].patches
    for figure in (count_figure, icc_figure, box_mode_figure):
        plt.close(figure)


def test_large_biological_effect_plot_hides_idp_labels_and_confidence_intervals():
    idps = [f"f{i}" for i in range(11)]
    df = pd.DataFrame(
        {
            "IDP": idps,
            "age_est": np.arange(11, dtype=float),
            "age_pval": np.repeat(0.2, 11),
            "age_ciL": np.arange(11, dtype=float) - 0.5,
            "age_ciU": np.arange(11, dtype=float) + 0.5,
        }
    )

    figure, axes = PlotDiagnosticResults.plot_MixedEffectsPart2(df, fix_eff=("age",))

    assert len(axes[0].collections) == 1
    assert len(axes[0].lines) == 1
    plt.close(figure)


def test_comparison_longitudinal_plots_use_requested_summaries():
    results = {
        "method": SimpleNamespace(
            subject_order=pd.DataFrame(
                {
                    "TimeA": ["t1", "t1"],
                    "TimeB": ["t2", "t2"],
                    "IDP": ["a", "b"],
                    "SpearmanRho": [0.2, 0.8],
                    "pValue": [0.01, 0.02],
                }
            ),
            within_subject_variability=pd.DataFrame(
                {"subject": ["s1", "s2"], "f1": [1.0, 3.0], "f2": [5.0, 7.0]}
            ),
            additive_effects=pd.DataFrame(
                {"Feature": ["f1", "f2"], "p-value": [0.01, 0.1], "p-value_fdr": [0.02, 0.2]}
            ),
            multiplicative_effects=pd.DataFrame(
                {"Feature": ["f1", "f2"], "p-value": [0.02, 0.2], "p-value_fdr": [0.04, 0.2]}
            ),
            mixed_effects=pd.DataFrame(
                {
                    "IDP": ["f1", "f2"],
                    "n_is_batchSig": [1, 2],
                    "ICC": [0.6, 0.8],
                    "age_est": [0.1, 0.2],
                    "age_pval": [0.01, 0.2],
                    "age_ciL": [0.0, 0.1],
                    "age_ciU": [0.2, 0.3],
                }
            ),
        )
    }

    order_fig = PlotComparisonResults.plot_compare_long_subject_order(results)[0][1]
    variability_fig = PlotComparisonResults.plot_compare_long_within_subject_variability(results)[0][1]
    effect_figs = PlotComparisonResults.plot_compare_long_additive_multiplicative(results)
    mixed_figs = PlotComparisonResults.plot_compare_long_mixed_effects(results)
    biological_fig = PlotComparisonResults.plot_compare_long_biological_effects(results)[0][1]

    assert len(order_fig.axes) == 3
    np.testing.assert_allclose(order_fig.axes[1].collections[0].get_offsets()[:, 0], [0.2, 0.8])
    assert len(variability_fig.axes[0].collections) == 1
    assert len(effect_figs) == 2
    assert all(len(fig.axes[0].collections) == 1 for _, fig in effect_figs)
    np.testing.assert_allclose(
        effect_figs[0][1].axes[0].collections[0].get_offsets()[:, 1], -np.log10([0.02, 0.2])
    )
    np.testing.assert_allclose(
        effect_figs[1][1].axes[0].collections[0].get_offsets()[:, 1], -np.log10([0.04, 0.2])
    )
    assert len(mixed_figs) == 2
    assert len(mixed_figs[0][1].axes[0].collections) == 1
    assert {round(line.get_ydata()[0], 2) for line in mixed_figs[1][1].axes[0].lines} >= {0.5, 0.75, 0.9}
    assert len(biological_fig.axes[0].collections) == 4
    assert len(biological_fig.axes[0].collections[0].get_offsets()) == 2
    for figure in [order_fig, variability_fig, biological_fig, *(fig for _, fig in effect_figs + mixed_figs)]:
        plt.close(figure)


def test_small_biological_effect_plot_keeps_confidence_intervals_and_labels():
    df = pd.DataFrame(
        {
            "IDP": ["f1", "f2"],
            "age_est": [0.1, 0.2],
            "age_pval": [0.01, 0.2],
            "age_ciL": [0.0, 0.1],
            "age_ciU": [0.2, 0.3],
        }
    )

    figure, axes = PlotDiagnosticResults.plot_MixedEffectsPart2(df, fix_eff=("age",))

    assert len(axes[0].collections) == 2
    assert len(axes[0].lines) == 7
    assert [label.get_text() for label in axes[0].get_xticklabels()] == ["f1", "f2"]
    plt.close(figure)


def test_comparison_biological_effects_hide_intervals_and_idp_labels_above_ten():
    n_idps = 11
    df = pd.DataFrame(
        {
            "IDP": [f"f{i}" for i in range(n_idps)],
            "age_est": np.arange(n_idps, dtype=float),
            "age_ciL": np.arange(n_idps, dtype=float) - 0.5,
            "age_ciU": np.arange(n_idps, dtype=float) + 0.5,
        }
    )
    results = {"method": SimpleNamespace(mixed_effects=df)}

    figure = PlotComparisonResults.plot_compare_long_biological_effects(results)[0][1]
    ax = figure.axes[0]

    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    plt.close(figure)