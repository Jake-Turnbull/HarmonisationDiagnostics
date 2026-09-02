"""Tests for DiagnosticFunctions.calculate_batch_covariate_confounding.

This diagnostic is deliberately separate from calculate_pc_associations: it
quantifies batch-covariate imbalance (continuous covariate ~ batch omnibus R2,
categorical covariate vs batch Cramer's V) rather than PC-vs-metadata
association. Batch must always be treated as nominal categorical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import DiagnosticFunctions


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_continuous_covariate_batch_relabelling_invariant(rng):
    n_per_batch = 60
    batches = np.repeat(np.arange(3), n_per_batch)
    age = batches * 8.0 + rng.normal(0, 2, size=batches.shape[0])
    covariates = pd.DataFrame({"age": age})

    letters = np.array(["A", "B", "C"])[batches]
    result_letters = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, letters, variable_types={"age": "continuous"}
    )
    result_numeric = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches + 1, variable_types={"age": "continuous"}
    )

    r2_letters = result_letters["tidy"].set_index("Variable").loc["age", "Value"]
    r2_numeric = result_numeric["tidy"].set_index("Variable").loc["age", "Value"]
    assert r2_letters == pytest.approx(r2_numeric, abs=1e-10)
    assert result_letters["tidy"].set_index("Variable").loc["age", "Statistic"] == "R2"


def test_categorical_covariate_cramers_v_batch_relabelling_invariant(rng):
    n = 180
    batches = np.repeat(np.arange(3), n // 3)
    # sex strongly associated with batch
    sex = np.where(batches == 0, "M", np.where(batches == 1, "F", "M"))
    covariates = pd.DataFrame({"sex": sex})

    permuted_map = {0: 2, 1: 0, 2: 1}
    permuted_batches = np.vectorize(permuted_map.get)(batches)

    result_original = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches, variable_types={"sex": "categorical"}
    )
    result_permuted = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, permuted_batches, variable_types={"sex": "categorical"}
    )

    v_original = result_original["tidy"].set_index("Variable").loc["sex", "Value"]
    v_permuted = result_permuted["tidy"].set_index("Variable").loc["sex", "Value"]
    assert v_original == pytest.approx(v_permuted, abs=1e-10)
    assert result_original["tidy"].set_index("Variable").loc["sex", "Statistic"] == "CramersV"


def test_unrelated_continuous_covariate_gives_near_zero_r2(rng):
    n = 300
    batches = rng.integers(0, 3, size=n)
    unrelated = rng.normal(0, 1, size=n)
    covariates = pd.DataFrame({"unrelated": unrelated})

    result = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches, variable_types={"unrelated": "continuous"}
    )
    value = result["tidy"].set_index("Variable").loc["unrelated", "Value"]
    assert value < 0.05


def test_strongly_separated_categorical_covariate_gives_large_cramers_v():
    batches = np.repeat(["site1", "site2"], 100)
    sex = np.where(batches == "site1", "M", "F")  # perfectly confounded
    covariates = pd.DataFrame({"sex": sex})

    result = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches, variable_types={"sex": "categorical"}
    )
    value = result["tidy"].set_index("Variable").loc["sex", "Value"]
    assert value > 0.95


def test_constant_covariate_handled_gracefully():
    n = 60
    batches = np.repeat([0, 1, 2], n // 3)
    covariates = pd.DataFrame({"constant": np.full(n, 3.0)})

    result = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches, variable_types={"constant": "continuous"}
    )
    value = result["tidy"].set_index("Variable").loc["constant", "Value"]
    assert np.isnan(value)


def test_tidy_output_shape_and_columns():
    n = 90
    batches = np.repeat([0, 1, 2], n // 3)
    covariates = pd.DataFrame(
        {"age": np.random.default_rng(1).normal(50, 10, size=n), "sex": np.tile(["M", "F", "M"], n // 3)}
    )

    result = DiagnosticFunctions.calculate_batch_covariate_confounding(
        covariates, batches, variable_types={"age": "continuous", "sex": "categorical"}
    )
    tidy = result["tidy"]
    assert list(tidy.columns) == ["Variable", "Type", "Statistic", "Value"]
    assert set(tidy["Variable"]) == {"age", "sex"}
    assert result["variable_types"] == {"age": "continuous", "sex": "categorical"}
