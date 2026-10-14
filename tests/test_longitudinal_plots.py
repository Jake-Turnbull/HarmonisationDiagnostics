import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import PlotDiagnosticResults as PDR
from DiagnoseHarmonisation import PlotComparisonResults as PCR
from DiagnoseHarmonisation.DiagnosticReport import (
    _run_single_longitudinal_method_diagnostics,
)


def _additive_df(n):
    return pd.DataFrame({"Feature": [f"feat_{i}" for i in range(n)], "p-value": np.linspace(0.001, 0.9, n)})


@pytest.mark.parametrize("n,named", [(10, True), (25, False)])
def test_additive_plot_tick_rule(n, named):
    fig, _ = PDR.plot_AddMultEffects(_additive_df(n), labels=["Additive"], figsize=(7.2, 20)), None
    fig = fig if not isinstance(fig, tuple) else fig[0]
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any(l.startswith("feat_") for l in labels) == named
    assert fig.get_size_inches()[1] <= 6.0


def test_comparison_zscore_has_pdf_lines():
    rng = np.random.default_rng(0)
    feats = [f"f{i}" for i in range(5)]
    batch = np.array(["a", "b"] * 20)
    datasets = {"Raw": rng.normal(size=(40, 5)), "M": rng.normal(size=(40, 5))}
    covariates = {"age": list(rng.normal(size=40))}
    figs = PCR.plot_compare_long_zscore_distributions(datasets, batch, feats, covariates=covariates)
    assert len(figs) == 2
    for _, fig in figs:
        assert len(fig.axes[0].lines) >= 2


def test_additive_effect_recorded_in_comparison():
    rng = np.random.default_rng(1)
    n_sub, tps = 12, ["t1", "t2"]
    subjects = np.repeat([f"s{i}" for i in range(n_sub)], len(tps))
    timepoints = np.tile(tps, n_sub)
    batch = np.tile(["a", "b"], n_sub)
    data = rng.normal(size=(n_sub * 2, 3))
    res = _run_single_longitudinal_method_diagnostics(
        "Raw", data, batch, subjects, timepoints, None, ["f1", "f2", "f3"]
    )
    assert not any("AdditiveEffect_long" in e for e in res.errors)
    assert isinstance(res.additive_effects, pd.DataFrame) and not res.additive_effects.empty
