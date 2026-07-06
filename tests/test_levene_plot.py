import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest

from DiagnoseHarmonisation.DiagnosticFunctions import Levenes_Test
from DiagnoseHarmonisation import PlotDiagnosticResults as PDR
from DiagnoseHarmonisation import PlotComparisonResults as PCR


def test_levene_plot_smoke():
    rng = np.random.default_rng(0)
    n = 60
    # create 3 batches with different variances on a subset of features
    n_features = 20
    batch = np.array([0] * 20 + [1] * 20 + [2] * 20)

    # baseline data: unit variance
    data = rng.normal(size=(n, n_features))
    # increase variance in batch 1 for first 5 features
    data[20:40, :5] *= 3.0
    # decrease variance in batch 2 for features 5:10
    data[40:60, 5:10] *= 0.5

    # run Levene test implementation
    levene_results = Levenes_Test(data, batch, centre="median")
    assert isinstance(levene_results, dict)
    assert len(levene_results) > 0

    feature_names = [f"f{i}" for i in range(n_features)]

    figs = PDR.Levenes_Test(levene_results, feature_names=feature_names, alpha=0.05, show=False)
    # wrapped function returns list of (caption, Figure)
    assert isinstance(figs, list)
    assert all(isinstance(x[0], str) for x in figs)
    # matplotlib Figure type check (weak import to avoid strict backend issues)
    for cap, fig in figs:
        assert hasattr(fig, "savefig")


def test_levene_plot_uses_p_value_key_and_feature_label_policy():
    n_features = 24
    stats = np.linspace(0.1, 3.0, n_features)
    pvals = np.linspace(0.001, 0.2, n_features)
    levene_results = {("A", "B"): {"statistic": stats, "p_value": pvals}}
    feature_names = [f"f{i+1}" for i in range(n_features)]

    figs = PDR.Levenes_Test(levene_results, feature_names=feature_names, alpha=0.05, show=False)
    assert len(figs) == 1
    _, fig = figs[0]
    ax = fig.axes[0]

    rendered_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert all(label == "" for label in rendered_labels)


def test_levene_plot_diagonal_labels_for_20_or_fewer_features():
    n_features = 20
    stats = np.linspace(0.1, 2.0, n_features)
    pvals = np.linspace(0.01, 0.3, n_features)
    levene_results = {("A", "B"): {"stat": stats, "p_value": pvals}}
    feature_names = [f"f{i+1}" for i in range(n_features)]

    figs = PDR.Levenes_Test(levene_results, feature_names=feature_names, alpha=0.05, show=False)
    assert len(figs) == 1
    _, fig = figs[0]
    ax = fig.axes[0]

    first_rotation = ax.get_xticklabels()[0].get_rotation()
    assert int(first_rotation) == 45


def test_levene_with_residuals_uses_square_axes_and_p_value_key():
    n_features = 12
    raw = {("A", "B"): {"stat": np.ones(n_features), "p_value": np.full(n_features, 0.04)}}
    resid = {("A", "B"): {"stat": np.ones(n_features) * 0.8, "p_value": np.full(n_features, 0.06)}}

    figs = PDR.Levenes_Test_with_residuals(raw, resid, feature_names=[f"f{i+1}" for i in range(n_features)], show=False)
    assert len(figs) == 1
    _, fig = figs[0]
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert round(float(ax.get_box_aspect()), 2) == 1.0


def test_comparison_feature_label_policy_thresholds():
    labels_small, rot_small = PCR._feature_labels(20)
    assert len(labels_small) == 20
    assert rot_small == 45
    assert all(label != "" for label in labels_small)

    labels_large, rot_large = PCR._feature_labels(21)
    assert len(labels_large) == 21
    assert rot_large == 0
    assert all(label == "" for label in labels_large)
