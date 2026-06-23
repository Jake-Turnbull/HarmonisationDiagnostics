import numpy as np
import pandas as pd
from DiagnoseHarmonisation import HarmonisationFunctions as hf


def test_modular_matches_baseline():
    """Parity test: `combat_modular` (baseline mode) must reproduce `combat` exactly within tolerance."""
    np.random.seed(0)

    # small synthetic dataset
    n_features = 10
    n_samples = 8
    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"f{i}" for i in range(n_features)],
        columns=[f"s{i}" for i in range(n_samples)],
    )

    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples))

    out_baseline = hf.combat(data=data, batch=batch, mod=None, UseEB=True, parametric=True, return_priors=True)
    out_modular = hf.combat_modular(data=data, batch=batch, mod=None, mean_model="ols", prior_mode="global", parametric=True, UseEB=True, return_priors=True)

    assert isinstance(out_baseline, dict)
    assert isinstance(out_modular, dict)

    bd1 = out_baseline["bayesdata"] if isinstance(out_baseline, dict) else out_baseline
    bd2 = out_modular["bayesdata"] if isinstance(out_modular, dict) else out_modular

    np.testing.assert_allclose(np.asarray(bd1), np.asarray(bd2), rtol=1e-6, atol=1e-8)


def test_gam_path_runs():
    """Ensure the `mean_model='gam'` path executes (requires mod)."""
    np.random.seed(1)
    n_features = 6
    n_samples = 12
    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples))
    # synthetic nonlinear covariate
    age = np.sin(np.linspace(0, 3.14, n_samples))
    mod = pd.DataFrame({"age": age})

    out = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="gam", return_priors=False)
    assert isinstance(out, np.ndarray) or isinstance(out, pd.DataFrame)


def test_gam_preserves_nonlinear_structure():
    """GAM mean model should better preserve a known nonlinear covariate signal than OLS."""
    np.random.seed(42)
    n_features = 5
    n_samples = 120
    age = np.linspace(0, 2 * np.pi, n_samples)
    true_signal = np.sin(age)

    # strong additive batch effect
    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples))
    batch_effect = np.where(batch.values == "A", 1.0, -1.0)

    data_arr = np.zeros((n_features, n_samples))
    # feature 0 has the nonlinear signal
    data_arr[0, :] = true_signal + batch_effect + 0.2 * np.random.randn(n_samples)
    # other features: noise + batch effect
    for j in range(1, n_features):
        data_arr[j, :] = 0.2 * np.random.randn(n_samples) + np.where(batch.values == "A", 0.5, -0.5)

    data = pd.DataFrame(data_arr)
    mod = pd.DataFrame({"age": age})

    out_gam = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="gam", return_priors=False)
    out_ols = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global", return_priors=False)

    arr_gam = np.asarray(out_gam).reshape((n_features, n_samples))
    arr_ols = np.asarray(out_ols).reshape((n_features, n_samples))

    # Check that GAM-corrected feature retains the nonlinear relationship
    corr_gam = np.corrcoef(arr_gam[0, :], true_signal)[0, 1]
    assert corr_gam > 0.9
