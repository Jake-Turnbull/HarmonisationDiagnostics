from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def _make_data(n_features=10, n_samples=60, seed=0):
    rng = np.random.default_rng(seed)
    batch = np.array(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    age = np.linspace(20.0, 80.0, n_samples)
    base = rng.normal(size=(n_features, n_samples))
    batch_effect = np.where(batch == "A", 1.5, -1.5)
    data = base + batch_effect + 0.05 * age
    return pd.DataFrame(data), pd.Series(batch), pd.DataFrame({"age": age})


def test_combat_apply_round_trip_legacy_combat():
    data, batch, mod = _make_data()
    trained = hf.combat(data=data, batch=batch, mod=mod, return_priors=True)

    result = hf.combat_apply(new_data=data, new_batch=batch, new_mod=mod, trained_output=trained)
    np.testing.assert_allclose(
        np.asarray(result["bayesdata"]), np.asarray(trained["bayesdata"]), atol=1e-6
    )


def test_combat_apply_round_trip_modular_fast_path():
    data, batch, mod = _make_data(seed=1)
    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    result = hf.combat_apply(new_data=data, new_batch=batch, new_mod=mod, trained_output=trained)
    np.testing.assert_allclose(
        np.asarray(result["bayesdata"]), np.asarray(trained["bayesdata"]), atol=1e-6
    )


def test_combat_apply_round_trip_modular_local_priors_categorical_covariate():
    n_samples = 60
    n_features = 8
    rng = np.random.default_rng(2)
    batch = pd.Series(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    sex = pd.Series((["M", "F"] * (n_samples // 2)))
    data = pd.DataFrame(rng.normal(size=(n_features, n_samples)))
    mod = pd.DataFrame({"sex": sex})

    trained = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="local",
    )

    result = hf.combat_apply(new_data=data, new_batch=batch, new_mod=mod, trained_output=trained)
    np.testing.assert_allclose(
        np.asarray(result["bayesdata"]), np.asarray(trained["bayesdata"]), atol=1e-6
    )


def test_combat_apply_gam_spline_covariate_idempotent():
    data, batch, mod = _make_data(n_samples=48, seed=3)
    with pytest.warns(RuntimeWarning, match="spline basis expansion"):
        trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="gam")

    result = hf.combat_apply(new_data=data, new_batch=batch, new_mod=mod, trained_output=trained)
    np.testing.assert_allclose(
        np.asarray(result["bayesdata"]), np.asarray(trained["bayesdata"]), atol=1e-4
    )


def test_combat_apply_rejects_wrong_covariate_column_count():
    data, batch, mod = _make_data(seed=4)
    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    bad_mod = mod.copy()
    bad_mod["extra"] = 1.0
    with pytest.raises(ValueError, match="covariate column"):
        hf.combat_apply(new_data=data, new_batch=batch, new_mod=bad_mod, trained_output=trained)


def test_combat_apply_rejects_unseen_batch_label():
    data, batch, mod = _make_data(seed=5)
    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    new_batch = batch.copy()
    new_batch.iloc[0] = "C"
    with pytest.raises(ValueError, match="not seen during training"):
        hf.combat_apply(new_data=data, new_batch=new_batch, new_mod=mod, trained_output=trained)


def test_combat_apply_rejects_unseen_category():
    n_samples = 40
    n_features = 6
    rng = np.random.default_rng(6)
    batch = pd.Series(["A"] * (n_samples // 2) + ["B"] * (n_samples // 2))
    sex = pd.Series((["M", "F"] * (n_samples // 2)))
    data = pd.DataFrame(rng.normal(size=(n_features, n_samples)))
    mod = pd.DataFrame({"sex": sex})

    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="local")

    new_mod = mod.copy()
    new_mod.loc[0, "sex"] = "U"
    with pytest.raises(ValueError, match="not seen during training"):
        hf.combat_apply(new_data=data, new_batch=batch, new_mod=new_mod, trained_output=trained)


def test_combat_apply_transposed_input_and_wrong_feature_count():
    data, batch, mod = _make_data(seed=7)
    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    # Samples-as-rows orientation should be auto-detected.
    result = hf.combat_apply(new_data=data.T, new_batch=batch, new_mod=mod, trained_output=trained)
    assert np.asarray(result["bayesdata"]).shape == data.T.shape

    bad_data = data.iloc[:-1, :]
    with pytest.raises(ValueError, match="feature"):
        hf.combat_apply(new_data=bad_data, new_batch=batch, new_mod=mod, trained_output=trained)


def test_combat_apply_refine_eb_runs_and_differs_on_shifted_data():
    data, batch, mod = _make_data(seed=8)
    trained = hf.combat_modular(data=data, batch=batch, mod=mod, mean_model="ols", prior_mode="global")

    shifted = data + 5.0
    result_plain = hf.combat_apply(new_data=shifted, new_batch=batch, new_mod=mod, trained_output=trained)
    result_refined = hf.combat_apply(
        new_data=shifted, new_batch=batch, new_mod=mod, trained_output=trained, refine_eb=True
    )

    assert result_refined["refine_eb"] is True
    assert np.asarray(result_refined["bayesdata"]).shape == np.asarray(result_plain["bayesdata"]).shape
    assert not np.allclose(
        np.asarray(result_plain["bayesdata"]), np.asarray(result_refined["bayesdata"])
    )


def test_combat_apply_missing_standardisation_raises():
    with pytest.raises(ValueError, match="standardisation"):
        hf.combat_apply(new_data=np.zeros((5, 5)), new_batch=np.array(["A"] * 5), trained_output={"bayesdata": None})
