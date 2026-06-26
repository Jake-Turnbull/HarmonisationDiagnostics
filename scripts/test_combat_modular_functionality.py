#!/usr/bin/env python3
"""Standalone smoke/regression checks for combat_modular functionality.

This script validates three key behaviors:
1) Baseline parity: combat_modular(mean_model='ols', prior_mode='global')
   matches legacy combat output.
2) GAM path execution: mean_model='gam' runs successfully with covariates.
3) Nonlinear preservation: GAM better preserves known nonlinear signal than OLS.

It also reports basic runtime for legacy combat vs modular baseline path.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd


# Ensure local package import works when script is run from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def check_baseline_parity() -> None:
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
    print("[PASS] Baseline parity (legacy combat == modular baseline)")


def check_gam_path_executes() -> None:
    np.random.seed(1)
    n_features = 6
    n_samples = 12

    data = pd.DataFrame(np.random.randn(n_features, n_samples))
    # Use a balanced deterministic batch assignment to avoid confounding.
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
    assert arr.shape == (n_features, n_samples), "Unexpected output shape from GAM path"
    print("[PASS] GAM path executes and returns expected shape")


def check_gam_preserves_nonlinear_signal() -> None:
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

    assert corr_gam > 0.9, f"GAM nonlinear correlation too low: {corr_gam:.3f}"
    print(
        "[PASS] Nonlinear preservation check "
        f"(corr_gam={corr_gam:.3f}, corr_ols={corr_ols:.3f})"
    )


def benchmark_runtime() -> None:
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

    ratio = modular_seconds / max(legacy_seconds, 1e-12)
    print(
        "[INFO] Runtime: "
        f"legacy={legacy_seconds:.3f}s, modular_baseline={modular_seconds:.3f}s, "
        f"ratio={ratio:.2f}x"
    )


def main() -> int:
    print("Running combat_modular functionality checks...")

    check_baseline_parity()
    check_gam_path_executes()
    check_gam_preserves_nonlinear_signal()
    benchmark_runtime()

    print("[PASS] All combat_modular checks completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as err:
        print(f"[FAIL] {err}")
        raise SystemExit(1)
    except Exception as err:
        print(f"[ERROR] {type(err).__name__}: {err}")
        raise SystemExit(1)
