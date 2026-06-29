from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def test_combat_modular_returns_rich_dict_fast_path() -> None:
    np.random.seed(0)
    n_features = 10
    n_samples = 8

    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"f{i}" for i in range(n_features)],
        columns=[f"s{i}" for i in range(n_samples)],
    )
    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples), index=data.columns)

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=pd.DataFrame({"age": np.linspace(20.0, 70.0, n_samples)}),
        mean_model="ols",
        prior_mode="global",
        UseEB=True,
        parametric=True,
        return_priors=False,
    )

    assert isinstance(out, dict)
    assert "bayesdata" in out
    assert "priors" in out
    assert "design_diagnostics" in out
    assert out["design_diagnostics"]["mode"] == "legacy_fast_path"
    assert out["priors"]["priors_used"]["mode"] == "global"
    assert np.asarray(out["bayesdata"]).shape == (n_features, n_samples)


def test_combat_modular_local_priors_expose_similarity_outputs() -> None:
    np.random.seed(11)
    n_features = 12
    n_samples = 60

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    mod = pd.DataFrame(
        {
            "age": np.linspace(20.0, 80.0, n_samples),
            "sex": ["F", "M"] * (n_samples // 2),
        }
    )

    methods = ["correlation_similarity", "variance_similarity"]
    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="local",
        prior_weight_methods=methods,
        prior_weight_opts={
            "method_weights": {"correlation_similarity": 0.7, "variance_similarity": 0.3},
        },
    )

    assert isinstance(out, dict)
    priors = out["priors"]
    assert "local_priors" in priors
    local = priors["local_priors"]

    n_batch = len(priors["levels"])
    weights = np.asarray(local["weights"])
    assert weights.shape == (n_batch, n_features, n_features)
    assert set(local["weight_methods"]) == set(methods)

    row_sums = weights.sum(axis=2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6, atol=1e-6)


def test_combat_modular_covbat_fast_path_changes_output() -> None:
    np.random.seed(9)
    n_features = 14
    n_samples = 50

    batch = np.array(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    cov_profile = np.linspace(-1.0, 1.0, n_features)[:, None]
    base = np.random.randn(n_features, n_samples)
    data = base + np.where(batch == "A", cov_profile, -cov_profile)
    data = pd.DataFrame(data)

    out_no_covbat = hf.combat_modular(
        data=data,
        batch=pd.Series(batch),
        mod=None,
        mean_model="ols",
        prior_mode="global",
        covbat_mode=False,
    )
    out_covbat = hf.combat_modular(
        data=data,
        batch=pd.Series(batch),
        mod=None,
        mean_model="ols",
        prior_mode="global",
        covbat_mode=True,
    )

    arr_no_covbat = np.asarray(out_no_covbat["bayesdata"])
    arr_covbat = np.asarray(out_covbat["bayesdata"])

    assert arr_no_covbat.shape == arr_covbat.shape
    assert np.linalg.norm(arr_no_covbat - arr_covbat) > 0.0


def test_combat_modular_covariate_diagnostics_capture_pruning() -> None:
    np.random.seed(7)
    n_features = 8
    n_samples = 20

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A", "B"] * (n_samples // 2))

    sex = np.array(([0, 0, 1, 1] * (n_samples // 4)))
    mod = pd.DataFrame(
        {
            "sex_0": (sex == 0).astype(float),
            "sex_1": (sex == 1).astype(float),
            "sex_dup": (sex == 1).astype(float),
            "all_zero": np.zeros(n_samples),
        }
    )

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="local",
    )

    diag = out["design_diagnostics"]
    dropped = diag["dropped_covariate_columns"]

    assert diag["design_rank"] == diag["design_n_columns"]
    assert len(diag["kept_covariate_columns"]) > 0
    assert any(reason in ("redundant_or_constant", "linearly_dependent_with_batch") for reason in dropped.values())
    assert np.asarray(out["bayesdata"]).shape == (n_features, n_samples)


def test_combat_modular_gam_path_reports_design_encoding() -> None:
    np.random.seed(1)
    n_features = 6
    n_samples = 24

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A" if i % 2 == 0 else "B" for i in range(n_samples)])
    mod = pd.DataFrame({"age": np.linspace(20.0, 80.0, n_samples)})

    with pytest.warns(RuntimeWarning, match="spline basis expansion"):
        out = hf.combat_modular(
            data=data,
            batch=batch,
            mod=mod,
            mean_model="gam",
        )

    diag = out["design_diagnostics"]
    assert diag["mean_model"] == "gam"
    assert np.asarray(out["bayesdata"]).shape == (n_features, n_samples)
    assert diag["encoding"].get("age") in {"spline", "numeric_linear_nonspline", "numeric_fallback"}
