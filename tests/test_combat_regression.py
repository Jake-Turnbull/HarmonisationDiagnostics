import numpy as np
import pandas as pd
import pytest
from DiagnoseHarmonisation import HarmonisationFunctions as hf


def test_modular_matches_baseline():
    """Global modular fast-path should preserve legacy bayesdata while adding metadata."""
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
    out_modular = hf.combat_modular(data=data, batch=batch, mod=None, mean_model="ols", prior_mode="global", parametric=True, UseEB=True)

    assert isinstance(out_baseline, dict)
    assert isinstance(out_modular, dict)
    assert "design_diagnostics" in out_modular
    assert out_modular["priors"]["priors_used"]["mode"] == "global"

    bd1 = out_baseline["bayesdata"] if isinstance(out_baseline, dict) else out_baseline
    bd2 = out_modular["bayesdata"] if isinstance(out_modular, dict) else out_modular

    np.testing.assert_allclose(np.asarray(bd1), np.asarray(bd2), rtol=1e-6, atol=1e-8)


def test_gam_path_runs():
    """Ensure mean_model='gam' executes and returns modular diagnostics."""
    np.random.seed(1)
    n_features = 6
    n_samples = 24
    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A" if i % 2 == 0 else "B" for i in range(n_samples)])
    age = np.linspace(18.0, 78.0, n_samples)
    mod = pd.DataFrame({"age": age})

    with pytest.warns(RuntimeWarning, match="spline basis expansion"):
        out = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="gam", return_priors=False)

    assert isinstance(out, dict)
    assert out["design_diagnostics"]["mean_model"] == "gam"
    assert np.asarray(out["bayesdata"]).shape == (n_features, n_samples)


def test_gam_preserves_nonlinear_structure():
    """GAM path should preserve strong nonlinear structure under updated modular output contract."""
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

    out_gam = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="gam")
    out_ols = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    arr_gam = np.asarray(out_gam["bayesdata"]).reshape((n_features, n_samples))
    arr_ols = np.asarray(out_ols["bayesdata"]).reshape((n_features, n_samples))

    # Check that GAM-corrected feature retains the nonlinear relationship
    corr_gam = np.corrcoef(arr_gam[0, :], true_signal)[0, 1]
    corr_ols = np.corrcoef(arr_ols[0, :], true_signal)[0, 1]
    assert corr_gam > 0.9
    assert np.isfinite(corr_ols)


def test_modular_prunes_redundant_dummy_covariates():
    """Modular path should tolerate redundant one-hot covariate columns and report pruning."""
    np.random.seed(7)
    n_features = 8
    n_samples = 20
    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    # Deterministic non-confounded batch pattern.
    batch = pd.Series(["A", "B"] * (n_samples // 2))

    # Binary covariate with intentionally redundant encoding.
    sex = np.array(([0, 0, 1, 1] * (n_samples // 4)))
    mod = pd.DataFrame(
        {
            "sex_0": (sex == 0).astype(float),
            "sex_1": (sex == 1).astype(float),
            "sex_dup": (sex == 1).astype(float),  # exact duplicate of sex_1
            "all_zero": np.zeros(n_samples),       # constant redundant column
        }
    )

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="local",
        return_priors=False,
    )

    assert isinstance(out, dict)
    diag = out["design_diagnostics"]
    dropped = diag["dropped_covariate_columns"]

    assert diag["design_rank"] == diag["design_n_columns"]
    assert len(dropped) > 0
    assert any(v in ("redundant_or_constant", "linearly_dependent_with_batch") for v in dropped.values())

    arr = np.asarray(out["bayesdata"])
    assert arr.shape == (n_features, n_samples)


def test_local_prior_similarity_metadata_present():
    """Local prior mode should expose similarity/weight metadata in priors."""
    np.random.seed(13)
    n_features = 9
    n_samples = 40

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=None,
        mean_model="ols",
        prior_mode="local",
        prior_weight_methods=["correlation_similarity", "magnitude_similarity"],
        prior_weight_opts={"method_weights": {"correlation_similarity": 0.5, "magnitude_similarity": 0.5}},
    )

    local = out["priors"]["local_priors"]
    weights = np.asarray(local["weights"])

    assert set(local["weight_methods"]) == {"correlation_similarity", "magnitude_similarity"}
    assert weights.shape == (2, n_features, n_features)


def test_covbat_mode_supported_in_fast_path():
    """covbat_mode should run via fast-path and return valid modular dict output."""
    np.random.seed(21)
    n_features = 10
    n_samples = 30

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(np.random.choice(["A", "B", "C"], size=n_samples))

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=None,
        mean_model="ols",
        prior_mode="global",
        covbat_mode=True,
    )

    assert isinstance(out, dict)
    assert out["design_diagnostics"]["mode"] == "legacy_fast_path"
    assert np.asarray(out["bayesdata"]).shape == (n_features, n_samples)
