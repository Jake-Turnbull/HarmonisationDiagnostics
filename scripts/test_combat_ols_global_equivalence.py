#!/usr/bin/env python3
"""Verify legacy ComBat and modular OLS/global produce matching data."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DiagnoseHarmonisation import HarmonisationFunctions as hf


def check_equivalence(mod: pd.DataFrame | None) -> None:
    rng = np.random.default_rng(2026)
    n_features = 16
    n_samples = 48
    batch = pd.Series(["site_a", "site_b", "site_c"] * (n_samples // 3))
    data_array = rng.normal(size=(n_features, n_samples))
    batch_offsets = batch.map({"site_a": 0.8, "site_b": -0.4, "site_c": 0.2}).to_numpy()
    data_array += batch_offsets[None, :]
    if mod is not None:
        data_array += 0.025 * mod["age"].to_numpy()[None, :]
    data = pd.DataFrame(data_array)

    legacy = hf.combat(data=data, batch=batch, mod=mod, return_priors=True)
    modular = hf.combat_modular(
        data=data,
        batch=batch,
        mod=mod,
        mean_model="OLS",
        prior_mode="global",
        return_priors=True,
    )

    assert modular["design_diagnostics"]["mode"] == "modular"
    np.testing.assert_allclose(
        np.asarray(modular["bayesdata"]),
        np.asarray(legacy["bayesdata"]),
        rtol=1e-6,
        atol=1e-8,
    )


def main() -> int:
    check_equivalence(mod=None)
    check_equivalence(pd.DataFrame({"age": np.linspace(20.0, 80.0, 48)}))
    print("[PASS] combat() and combat_modular(OLS, global) outputs match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
