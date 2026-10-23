from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def test_combat_modular_returns_rich_dict_from_modular_path() -> None:
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
    assert out["design_diagnostics"]["mode"] == "modular"
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

    for key in ("gamma_bar_local", "t2_local", "a_prior_local", "b_prior_local"):
        values = np.asarray(local[key])
        assert values.shape == (n_batch, n_features)
        assert np.isfinite(values).all()

    assert np.isfinite(weights).all()
    assert (weights >= 0.0).all()
    for key in ("gamma_star", "delta_star"):
        assert np.isfinite(np.asarray(priors[key])).all()
    assert np.isfinite(np.asarray(out["bayesdata"])).all()


def test_construct_local_priors_include_self_and_negative_correlations() -> None:
    s_data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    gamma_hat = np.array([[1.0, 2.0, 4.0]])
    delta_hat = np.array([[1.0, 2.0, 4.0]])

    local = hf._construct_local_priors(
        s_data=s_data,
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        var_pooled=np.ones(3),
        batches=[np.arange(s_data.shape[1])],
        weight_methods=["correlation_similarity"],
        weight_opts={"min_effective": 1},
    )

    weights = local["weights"][0]
    assert np.all(weights > 0.0)
    assert np.all(np.diag(weights) > 0.0)
    assert weights[0, 1] > weights[0, 2]
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert not np.asarray(local["used_global_fallback"]).any()


def test_construct_local_priors_signed_weights_and_corrected_moments() -> None:
    s_data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    gamma_hat = np.array([[1.0, 2.0, 4.0]])
    delta_hat = np.array([[1.0, 2.0, 4.0]])
    local = hf._construct_local_priors(
        s_data=s_data,
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        var_pooled=np.ones(3),
        batches=[np.arange(s_data.shape[1])],
        weight_methods=["correlation_similarity", "variance_similarity"],
        weight_opts={
            "method_weights": {"correlation_similarity": 2.0, "variance_similarity": 6.0},
            "min_effective_features": 1,
        },
    )

    weights = np.asarray(local["weights"])
    scale_weights = np.asarray(local["scale_weights"])
    assert weights.shape == (1, 3, 3)
    assert scale_weights.shape == (1, 3, 3)
    assert np.isfinite(weights).all()
    assert np.isfinite(scale_weights).all()
    assert (weights >= 0.0).all()
    assert (scale_weights >= 0.0).all()
    np.testing.assert_allclose(weights.sum(axis=2), 1.0)
    np.testing.assert_allclose(scale_weights.sum(axis=2), 1.0)
    assert local["method_weights"] == {
        "correlation_similarity": 0.25,
        "variance_similarity": 0.75,
    }
    assert weights[0, 0, 0] > weights[0, 0, 2] > 0.0
    assert weights[0, 0, 1] > weights[0, 0, 2]

    for feature_index in range(3):
        mean_weights = weights[0, feature_index]
        gamma_mean = np.sum(mean_weights * gamma_hat[0])
        gamma_variance = np.sum(mean_weights * (gamma_hat[0] - gamma_mean) ** 2)
        gamma_variance /= 1.0 - np.sum(mean_weights**2)
        scale_feature_weights = scale_weights[0, feature_index]
        delta_mean = np.sum(scale_feature_weights * delta_hat[0])
        delta_variance = np.sum(scale_feature_weights * (delta_hat[0] - delta_mean) ** 2)
        delta_variance /= 1.0 - np.sum(scale_feature_weights**2)
        expected_a = (2.0 * delta_variance + delta_mean**2) / delta_variance
        expected_b = (delta_mean * delta_variance + delta_mean**3) / delta_variance
        np.testing.assert_allclose(local["gamma_bar_local"][0, feature_index], gamma_mean)
        np.testing.assert_allclose(local["t2_local"][0, feature_index], gamma_variance)
        np.testing.assert_allclose(local["a_prior_local"][0, feature_index], expected_a)
        np.testing.assert_allclose(local["b_prior_local"][0, feature_index], expected_b)

    np.testing.assert_allclose(local["effective_sample_size"], 1.0 / np.sum(weights**2, axis=2))
    np.testing.assert_allclose(local["scale_effective_sample_size"], 1.0 / np.sum(scale_weights**2, axis=2))
    assert not np.asarray(local["used_global_fallback"]).any()


def test_construct_local_priors_mean_weights_do_not_change_scale_priors() -> None:
    common_args = {
        "s_data": np.array(
            [
                [0.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
            ]
        ),
        "gamma_hat": np.array([[0.0, 2.0, 5.0]]),
        "delta_hat": np.array([[1.0, 3.0, 8.0]]),
        "var_pooled": np.ones(3),
        "batches": [np.arange(4)],
        "weight_methods": ["custom"],
    }
    first = hf._construct_local_priors(
        **common_args,
        weight_opts={
            "min_effective": 1,
            "similarity_matrix": np.array(
                [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.2, 0.75]]
            ),
        },
    )
    second = hf._construct_local_priors(
        **common_args,
        weight_opts={
            "min_effective": 1,
            "similarity_matrix": np.array(
                [[0.1, 0.8, 0.1], [0.7, 0.1, 0.2], [0.2, 0.1, 0.7]]
            ),
        },
    )

    assert not np.allclose(first["weights"], second["weights"])
    np.testing.assert_allclose(first["scale_weights"], second["scale_weights"])
    assert not np.allclose(first["gamma_bar_local"], second["gamma_bar_local"])
    assert not np.allclose(first["t2_local"], second["t2_local"])
    np.testing.assert_allclose(first["a_prior_local"], second["a_prior_local"])
    np.testing.assert_allclose(first["b_prior_local"], second["b_prior_local"])


def test_construct_local_priors_signed_mean_similarity_does_not_change_scale_priors() -> None:
    centered = np.array([-1.5, -0.5, 0.5, 1.5])
    s_data = np.array([centered, centered + 3.0, centered + 7.0])
    flipped_s_data = s_data.copy()
    flipped_s_data[1] *= -1.0
    common_args = {
        "gamma_hat": np.array([[0.0, 2.0, 8.0]]),
        "delta_hat": np.array([[1.0, 3.0, 8.0]]),
        "var_pooled": np.ones(3),
        "batches": [np.arange(s_data.shape[1])],
        "weight_methods": ["directional_bias"],
        "weight_opts": {"min_effective": 1, "mean_min_n": 0},
    }

    first = hf._construct_local_priors(s_data=s_data, **common_args)
    second = hf._construct_local_priors(s_data=flipped_s_data, **common_args)

    np.testing.assert_allclose(np.var(s_data, axis=1, ddof=1), np.var(flipped_s_data, axis=1, ddof=1))
    assert not np.allclose(first["weights"], second["weights"])
    np.testing.assert_allclose(first["scale_weights"], second["scale_weights"])
    np.testing.assert_allclose(first["a_prior_local"], second["a_prior_local"])
    np.testing.assert_allclose(first["b_prior_local"], second["b_prior_local"])


def test_construct_local_priors_variance_similarity_does_not_change_mean_priors() -> None:
    centered = np.array([-1.5, -0.5, 0.5, 1.5])
    s_data = np.array([centered, 2.0 * centered, 3.0 * centered])
    changed_variances = np.array([centered, 4.0 * centered, 7.0 * centered])
    common_args = {
        "gamma_hat": np.array([[0.0, 2.0, 8.0]]),
        "delta_hat": np.array([[1.0, 3.0, 8.0]]),
        "var_pooled": np.ones(3),
        "batches": [np.arange(s_data.shape[1])],
        "weight_methods": ["spatial_proximity", "variance_similarity"],
        "weight_opts": {
            "min_effective": 1,
            "method_weights": {"spatial_proximity": 0.5, "variance_similarity": 0.5},
            "feature_coords": np.array([[0.0], [1.0], [3.0]]),
        },
    }

    first = hf._construct_local_priors(s_data=s_data, **common_args)
    second = hf._construct_local_priors(s_data=changed_variances, **common_args)

    assert not np.allclose(first["scale_weights"], second["scale_weights"])
    np.testing.assert_allclose(first["weights"], second["weights"])
    np.testing.assert_allclose(first["gamma_bar_local"], second["gamma_bar_local"])
    np.testing.assert_allclose(first["t2_local"], second["t2_local"])


def test_construct_local_priors_work_with_bimodal_data() -> None:

    # Use gaussian noise as epsilon
    n_samples = 10000
    n_features = 40
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n_samples, n_features))
    batch = np.array(["A"] * int((n_samples/2)) + ["B"] * int((n_samples/2)))

    # Add sex and age with fixed effects
    sex = np.array([0] * int(n_samples/2) + [1] * int(n_samples/2)).reshape(-1, 1)
    # Shuffle sex to avoid any ordering effects
    rng.shuffle(sex)
    age = np.linspace(20, 60, int(n_samples)).reshape(-1, 1)

    # Add bimodal batch effect
    n_features_per_distribution = n_features // 2

    ground_truth = data
    ground_truth += (sex-np.mean(sex, axis=0)) * 0.5 + (age-np.mean(age, axis=0)) * 0.3

    # Add scale effect to the first half of the features for the first batch (i.e., multiply by a factor)
    data[:int(n_samples/2), :n_features_per_distribution] = data[:int(n_samples/2), :n_features_per_distribution] * 2.0
    data[:int(n_samples/2):, n_features_per_distribution:] = data[:int(n_samples/2):, n_features_per_distribution:] * 0.5

    # Add location-specific shifts to the data for the first batch plus random noise
    for i in range(int(n_samples/2)):
        data[i, :n_features_per_distribution] += 10 + rng.normal(scale=0.3, size=n_features_per_distribution)
        data[i, n_features_per_distribution:] -= 3.0 - rng.normal(scale=0.1, size=n_features_per_distribution)

    # keep one half the same:
    data[int(n_samples/2):, :n_features_per_distribution] -= 1.5
    data[int(n_samples/2):, n_features_per_distribution:] += 1.5

    data[:, :2] += (sex-np.mean(sex, axis=0)) * 0.5 + (age-np.mean(age, axis=0)) * 0.3
    # Try Combat and test RMSE, assert lower when using modular priors
    h1 = hf.combat_modular(
        data=data,
        batch=batch,
        covariates=np.hstack([sex, age]),
    )
    h2 = hf.combat_modular(
        data=data,
        batch=batch,
        covariates=np.hstack([sex, age]),
        prior_mode = "local",
        prior_weight_methods = ["magnitude_similarity","variance_similarity"]
    )

    # Save heatmap of priors used for modular priors
    local_priors=h2["priors"]
    local_priors = local_priors["local_priors"]
    # print local priors in assertion for debugging
    assert(False, f"Debugging local priors {local_priors.keys()}")

    gamma_bar_used = local_priors["gamma_bar_local"]
    t2_used = local_priors["t2_local"]
    a_prior_used = local_priors["a_prior_local"]
    b_prior_used = local_priors["b_prior_local"]

    # Plot heatmaps of the local priors
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Gamma Bar Local")
    plt.imshow(gamma_bar_used, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title("T2 Local")
    plt.imshow(t2_used, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("A Prior Local")
    plt.imshow(a_prior_used, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("B Prior Local")
    plt.imshow(b_prior_used, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # save the figure to testresults
    plt.savefig("testresults/local_priors_heatmap.png")

    # Get the non-local priors (global priors) used in the modular approach
    global_priors_used = h2["priors"]
    gamma_bar_global = global_priors_used["gamma_bar"]
    t2_global = global_priors_used["t2"]
    a_prior_global = global_priors_used["a_prior"]
    b_prior_global = global_priors_used["b_prior"]

    # The priors are not batch by features only batch specific, extend across features after checking size
    if gamma_bar_global.ndim == 1:
        gamma_bar_global = np.tile(gamma_bar_global[:, None], (1, data.shape[0]))
    if t2_global.ndim == 1:
        t2_global = np.tile(t2_global[:, None], (1, data.shape[0]))
    if a_prior_global.ndim == 1:
        a_prior_global = np.tile(a_prior_global[:, None], (1, data.shape[0]))
    if b_prior_global.ndim == 1:
        b_prior_global = np.tile(b_prior_global[:, None], (1, data.shape[0]))


    # Plot heatmaps of the global priors
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.title("Gamma Bar Global")
    plt.imshow(gamma_bar_global, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title("T2 Global")
    plt.imshow(t2_global, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("A Prior Global")
    plt.imshow(a_prior_global, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("B Prior Global")
    plt.imshow(b_prior_global, aspect='auto', cmap='viridis')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # save the figure to testresults
    plt.savefig("testresults/global_priors_heatmap.png")

    # unpack the results for comparison
    data1 = h1["bayesdata"]
    data2 = h2["bayesdata"]

    rmse1 = np.sqrt(np.mean((data1 - ground_truth) ** 2))
    rmse2 = np.sqrt(np.mean((data2 - ground_truth) ** 2))

    # Try mean squared error as well
    mse1 = np.mean((data1 - ground_truth) ** 2)
    mse2 = np.mean((data2 - ground_truth) ** 2)
    assert mse2 < mse1


    assert rmse2 < rmse1
    assert rmse2 >= 0


    

def test_construct_local_priors_degenerate_and_single_feature_fallbacks() -> None:
    degenerate = hf._construct_local_priors(
        s_data=np.ones((3, 4)),
        gamma_hat=np.array([[1.0, 2.0, 3.0]]),
        delta_hat=np.array([[1.0, 2.0, 3.0]]),
        var_pooled=np.ones(3),
        batches=[np.arange(4)],
        weight_methods=["custom"],
        weight_opts={"similarity_matrix": np.zeros((3, 3)), "min_effective_features": 1},
    )
    np.testing.assert_allclose(degenerate["weights"], np.full((1, 3, 3), 1.0 / 3.0))

    global_priors = (
        np.array([0.25]),
        np.array([0.5]),
        np.array([3.0]),
        np.array([2.0]),
    )
    single = hf._construct_local_priors(
        s_data=np.array([[1.0, 2.0, 3.0]]),
        gamma_hat=np.array([[2.0]]),
        delta_hat=np.array([[1.5]]),
        var_pooled=np.ones(1),
        batches=[np.arange(3)],
        weight_methods=["correlation_similarity"],
        global_priors=global_priors,
    )
    np.testing.assert_array_equal(single["weights"], np.ones((1, 1, 1)))
    np.testing.assert_allclose(single["gamma_bar_local"], [[0.25]])
    np.testing.assert_allclose(single["t2_local"], [[0.5]])
    np.testing.assert_allclose(single["a_prior_local"], [[3.0]])
    np.testing.assert_allclose(single["b_prior_local"], [[2.0]])
    np.testing.assert_array_equal(single["effective_sample_size"], [[1.0]])
    np.testing.assert_array_equal(single["used_global_fallback"], [[True]])


def test_construct_local_priors_signed_covariance_and_directional_floor() -> None:
    s_data = np.array(
        [
            [-4.0, -2.0, -3.0, -3.0],
            [-3.0, -1.0, -2.0, -2.0],
            [6.0, 8.0, 7.0, 7.0],
        ]
    )
    common_args = {
        "s_data": s_data,
        "gamma_hat": np.array([[-3.0, -2.0, 7.0]]),
        "delta_hat": np.array([[1.0, 2.0, 3.0]]),
        "var_pooled": np.ones(3),
        "batches": [np.arange(s_data.shape[1])],
    }
    covariance_local = hf._construct_local_priors(
        **common_args,
        weight_methods=["covariance_similarity"],
        weight_opts={"similarity_floor": 0.01, "min_effective_features": 1},
    )
    covariance_weights = covariance_local["weights"][0]
    assert covariance_weights[0, 0] >= covariance_weights[0, 1] > 0.0

    directional_local = hf._construct_local_priors(
        **common_args,
        weight_methods=["directional_bias"],
        weight_opts={
            "similarity_floor": 0.01,
            "opposing_similarity": 0.02,
            "min_effective_features": 1,
        },
    )
    directional_weights = directional_local["weights"][0]
    assert directional_weights[0, 0] > directional_weights[0, 1] > directional_weights[0, 2] > 0.0

    magnitude_local = hf._construct_local_priors(
        **common_args,
        weight_methods=["magnitude_similarity"],
        weight_opts={"similarity_floor": 0.01, "min_effective_features": 1},
    )
    np.testing.assert_allclose(magnitude_local["weights"], directional_local["weights"])


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


def test_combat_modular_covariate_removal_matrix() -> None:
    np.random.seed(42)
    n_features = 5
    n_samples = 100

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    mod = pd.DataFrame({"age": np.linspace(20.0, 70.0, n_samples)})
    # Add age, sex and headsize as covariates
    mod["sex"] = [1, 0] * (n_samples // 2)  # 1 for male, 0 for female
    mod["headsize"] = np.random.randn(n_samples)
    
    # Add effects to the data based on covariates
    data += mod["age"].values * 0.1  # Add age effect
    data += (mod["sex"].map({0: 1, 1: -1}).values) * 0.2  # Add sex effect
    data += mod["headsize"].values * 0.3  # Add headsize effect

    # Create a covariate removal matrix that removes headsize effect for all features:
    covariate_removal_matrix = np.array([[0, 0, 1]])  # Keep age and sex, remove headsize


    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="global",
        RegressCovariates=True,
        CovariateRemovalMatrix=covariate_removal_matrix,
    )

    bayesdata = out["bayesdata"]
    correlations = np.corrcoef(bayesdata.T, mod["headsize"], rowvar=False)
    original_correlations = np.corrcoef(data.T, mod["headsize"], rowvar=False)
    

    assert np.mean(np.abs(correlations)) < np.mean(np.abs(original_correlations)), "Headsize effect was not removed as expected."
