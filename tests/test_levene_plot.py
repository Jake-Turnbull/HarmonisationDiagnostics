import numpy as np
import matplotlib
matplotlib.use("Agg")
import pytest

from DiagnoseHarmonisation.DiagnosticFunctions import Levene_Test
from DiagnoseHarmonisation import PlotDiagnosticResults as PDR


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
    levene_results = Levene_Test(data, batch, centre="median")
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

    # optionally: ensure that at least one comparison found significant features
    any_sig = False
    for res in levene_results.values():
        pvals = None
        for k in ("pvalue", "p_val", "pvalues", "p"):
            if k in res:
                pvals = np.asarray(res[k])
                break
        if pvals is None:
            continue
        if (pvals < 0.05).any():
            any_sig = True
            break
