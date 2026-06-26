from __future__ import annotations

import time

import numpy as np
import pandas as pd

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def test_combat_modular_baseline_parity() -> None:
    np.random.seed(0)
    n_features = 10
    n_samples = 8

    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"f{i}" for i in range(n_features)],
        columns=[f"s{i}" for i in range(n_samples)],
    )
    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples), index=data.columns)

    out_legacy = hf.combat(
        data=data,
        batch=batch,
        mod=None,
        UseEB=True,
        parametric=True,
        return_priors=True,
    )
    out_modular = hf.combat_modular(
        data=data,
        batch=batch,
        mod=None,
        mean_model="ols",
        prior_mode="global",
        parametric=True,
        UseEB=True,
        return_priors=True,
    )

    arr_legacy = np.asarray(out_legacy["bayesdata"])
    arr_modular = np.asarray(out_modular["bayesdata"])
    np.testing.assert_allclose(arr_legacy, arr_modular, rtol=1e-6, atol=1e-8)


def test_combat_modular_gam_path_executes() -> None:
    np.random.seed(1)
    n_features = 6
    n_samples = 12

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(["A" if i % 2 == 0 else "B" for i in range(n_samples)])
    age = np.linspace(20.0, 80.0, n_samples)
    mod = pd.DataFrame({"age": age})

    out = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="gam",
        return_priors=False,
    )

    arr = np.asarray(out)
    assert arr.shape == (n_features, n_samples)


def test_combat_modular_gam_preserves_nonlinear_signal() -> None:
    np.random.seed(42)
    n_features = 5
    n_samples = 120

    age = np.linspace(0.0, 2.0 * np.pi, n_samples)
    true_signal = np.sin(age)
    batch = pd.Series(np.random.choice(["A", "B"], size=n_samples))
    batch_effect = np.where(batch.values == "A", 1.0, -1.0)

    data_arr = np.zeros((n_features, n_samples), dtype=float)
    data_arr[0, :] = true_signal + batch_effect + 0.2 * np.random.randn(n_samples)
    for j in range(1, n_features):
        data_arr[j, :] = 0.2 * np.random.randn(n_samples) + np.where(batch.values == "A", 0.5, -0.5)

    data = pd.DataFrame(data_arr)
    mod = pd.DataFrame({"age": age})

    out_gam = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="gam",
        return_priors=False,
    )
    out_ols = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="ols",
        prior_mode="global",
        return_priors=False,
    )

    arr_gam = np.asarray(out_gam)
    arr_ols = np.asarray(out_ols)

    corr_gam = float(np.corrcoef(arr_gam[0, :], true_signal)[0, 1])
    corr_ols = float(np.corrcoef(arr_ols[0, :], true_signal)[0, 1])

    assert corr_gam > 0.9
    assert np.isfinite(corr_ols)


def test_combat_modular_runtime_smoke() -> None:
    np.random.seed(7)
    n_features = 100
    n_samples = 80

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    batch = pd.Series(np.random.choice(["A", "B", "C"], size=n_samples))

    t0 = time.perf_counter()
    _ = hf.combat(data=data, batch=batch, mod=None, return_priors=False)
    legacy_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = hf.combat_modular(
        data=data,
        batch=batch,
        mod=None,
        mean_model="ols",
        prior_mode="global",
        return_priors=False,
    )
    modular_seconds = time.perf_counter() - t1

    assert legacy_seconds >= 0.0
    assert modular_seconds >= 0.0
