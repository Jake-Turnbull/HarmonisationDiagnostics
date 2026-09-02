"""Tests for DiagnosticFunctions.calculate_pc_associations.

These specifically target the bug in the old PCA correlation heatmap, where
Pearson correlation with a factorized batch/categorical label depends on the
arbitrary numeric codes assigned to each category. The omnibus R2 computed
here must be invariant to relabelling/permuting those codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import DiagnosticFunctions


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_batch_separated_pc(rng, n_per_batch=50, n_batches=3, effect=5.0, noise=0.5):
    batches = np.repeat(np.arange(n_batches), n_per_batch)
    pc1 = batches * effect + rng.normal(0, noise, size=batches.shape[0])
    pc_scores = pc1.reshape(-1, 1)
    return pc_scores, batches


def test_batch_relabelling_gives_identical_r2(rng):
    pc_scores, batch_codes = _make_batch_separated_pc(rng)
    batch_letters = np.array(["A", "B", "C"])[batch_codes]

    r2_letters = DiagnosticFunctions.calculate_pc_associations(pc_scores, batch=batch_letters)["r2_matrix"]
    r2_numeric = DiagnosticFunctions.calculate_pc_associations(pc_scores, batch=batch_codes + 1)["r2_matrix"]

    np.testing.assert_allclose(
        r2_letters.loc["batch"].to_numpy(dtype=float),
        r2_numeric.loc["batch"].to_numpy(dtype=float),
        atol=1e-10,
    )


def test_batch_code_permutation_gives_identical_r2(rng):
    pc_scores, batch_codes = _make_batch_separated_pc(rng)
    permuted_map = {0: 1, 1: 3, 2: 2}
    permuted_codes = np.vectorize(permuted_map.get)(batch_codes)

    r2_original = DiagnosticFunctions.calculate_pc_associations(pc_scores, batch=batch_codes)["r2_matrix"]
    r2_permuted = DiagnosticFunctions.calculate_pc_associations(pc_scores, batch=permuted_codes)["r2_matrix"]

    np.testing.assert_allclose(
        r2_original.loc["batch"].to_numpy(dtype=float),
        r2_permuted.loc["batch"].to_numpy(dtype=float),
        atol=1e-10,
    )


def test_continuous_predictor_r2_matches_pearson_r_squared(rng):
    n = 200
    age = rng.normal(50, 10, size=n)
    pc1 = 0.8 * age + rng.normal(0, 5, size=n)
    pc_scores = pc1.reshape(-1, 1)
    covariates = pd.DataFrame({"age": age})

    result = DiagnosticFunctions.calculate_pc_associations(
        pc_scores, covariates=covariates, variable_types={"age": "continuous"}
    )
    r2 = result["r2_matrix"].loc["age", "PC1"]
    expected_r2 = np.corrcoef(age, pc1)[0, 1] ** 2

    assert r2 == pytest.approx(expected_r2, abs=1e-8)


def test_unrelated_predictor_gives_near_zero_r2(rng):
    n = 300
    pc1 = rng.normal(0, 1, size=n)
    unrelated = rng.normal(0, 1, size=n)
    pc_scores = pc1.reshape(-1, 1)
    covariates = pd.DataFrame({"unrelated": unrelated})

    result = DiagnosticFunctions.calculate_pc_associations(
        pc_scores, covariates=covariates, variable_types={"unrelated": "continuous"}
    )
    r2 = result["r2_matrix"].loc["unrelated", "PC1"]
    assert r2 < 0.05


def test_strongly_separated_batch_gives_large_r2(rng):
    pc_scores, batch_codes = _make_batch_separated_pc(rng, effect=10.0, noise=0.2)
    result = DiagnosticFunctions.calculate_pc_associations(pc_scores, batch=batch_codes)
    r2 = result["r2_matrix"].loc["batch", "PC1"]
    assert r2 > 0.9


def test_constant_covariate_handled_gracefully(rng):
    n = 50
    pc1 = rng.normal(0, 1, size=n)
    pc_scores = pc1.reshape(-1, 1)
    constant_covariate = pd.DataFrame({"constant": np.full(n, 7.0)})

    result = DiagnosticFunctions.calculate_pc_associations(
        pc_scores,
        covariates=constant_covariate,
        variable_types={"constant": "categorical"},
    )
    r2 = result["r2_matrix"].loc["constant", "PC1"]
    assert np.isnan(r2)


def test_tidy_output_shape_and_columns(rng):
    n = 60
    pc_scores = rng.normal(0, 1, size=(n, 2))
    batch_codes = np.repeat([0, 1], n // 2)
    age = rng.normal(50, 10, size=n)
    covariates = pd.DataFrame({"age": age})

    result = DiagnosticFunctions.calculate_pc_associations(
        pc_scores, covariates=covariates, batch=batch_codes, variable_types={"age": "continuous"}
    )
    tidy = result["tidy"]
    assert list(tidy.columns) == ["PC", "Variable", "Type", "R2"]
    assert set(tidy["Variable"]) == {"batch", "age"}
    assert set(tidy["PC"]) == {"PC1", "PC2"}
    assert result["variable_types"]["batch"] == "categorical"
    assert result["variable_types"]["age"] == "continuous"
    assert result["r2_matrix"].shape == (2, 2)
