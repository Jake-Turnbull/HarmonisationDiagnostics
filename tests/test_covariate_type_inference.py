from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import DiagnosticFunctions, DiagnosticReport
from DiagnoseHarmonisation.HarmonisationFunctions import _infer_covariate_type


def test_integer_cardinality_uses_fraction_and_level_cap():
    assert _infer_covariate_type(pd.Series(np.repeat(np.arange(10), 20))) == "categorical"
    assert _infer_covariate_type(pd.Series(np.repeat(np.arange(3), 20))) == "categorical"
    assert _infer_covariate_type(pd.Series(np.repeat(np.arange(11), 20))) == "continuous"
    assert _infer_covariate_type(pd.Series(np.repeat(np.arange(3), 17))) == "continuous"


def test_numeric_dtype_overrides_integer_valued_samples():
    integer_valued_float = pd.Series(np.tile(np.arange(10, dtype=float), 10))
    low_cardinality_integer = pd.Series(np.tile([0, 1, 2], 40))

    assert _infer_covariate_type(integer_valued_float) == "continuous"
    assert _infer_covariate_type(low_cardinality_integer) == "categorical"
    assert _infer_covariate_type(pd.Series(np.tile([0, 1], 20))) == "binary"
    assert _infer_covariate_type(pd.Series(pd.Categorical(np.tile(["a", "b", "c"], 20)))) == "categorical"


def test_report_covariate_encoding_preserves_mixed_dataframe_dtypes():
    covariates = pd.DataFrame(
        {
            "integer_measure": np.tile(np.arange(10, dtype=float), 10),
            "coded_group": np.tile([0, 1, 2], 34)[:100],
            "label": pd.Categorical(np.tile(["a", "b"], 50)),
        }
    )

    legacy_types, variable_types, names = DiagnosticReport._resolve_report_covariate_types(covariates)
    numeric = DiagnosticReport.covariate_to_numeric(covariates, legacy_types)

    assert names == ["integer_measure", "coded_group", "label"]
    assert variable_types == {
        "integer_measure": "continuous",
        "coded_group": "categorical",
        "label": "binary",
    }
    assert legacy_types == [3, 2, 0]
    np.testing.assert_allclose(numeric[:, 0], covariates["integer_measure"].to_numpy())
    assert set(np.unique(numeric[:, 2])) == {0.0, 1.0}


def test_batch_confounding_and_pc_associations_infer_integer_measure_as_continuous():
    n = 200
    batch = np.repeat(["A", "B"], n // 2)
    measure = np.tile(np.arange(10), n // 10).astype(float)
    coded_group = np.tile([0, 1, 2], n // 3 + 1)[:n]
    covariates = pd.DataFrame({"measure": measure, "coded_group": coded_group})

    confounding = DiagnosticFunctions.calculate_batch_covariate_confounding(covariates, batch)
    associations = DiagnosticFunctions.calculate_pc_associations(
        pc_scores=np.column_stack((measure, coded_group)),
        covariates=covariates,
    )

    assert confounding["variable_types"] == {"measure": "continuous", "coded_group": "categorical"}
    assert associations["variable_types"] == {"measure": "continuous", "coded_group": "categorical"}
    assert confounding["tidy"].set_index("Variable").loc["measure", "Statistic"] == "R2"


def test_robust_ols_builds_continuous_and_categorical_terms(monkeypatch):
    n = 120
    measure = np.tile(np.arange(10, dtype=float), n // 10)
    category = np.repeat([0, 1, 2], n // 3)
    batch = np.tile(["A", "B"], n // 2)
    data = np.column_stack((0.3 * measure + category, measure - category)).astype(float)
    covariates = np.column_stack((measure, category))

    design_matrices = []
    original_lstsq = DiagnosticFunctions.lstsq

    def capture_design(design, response, rcond=None):
        design_matrices.append(design.copy())
        return original_lstsq(design, response, rcond=rcond)

    monkeypatch.setattr(DiagnosticFunctions, "lstsq", capture_design)
    continuous_result = DiagnosticFunctions.RobustOLS(
        data,
        measure.reshape(-1, 1),
        batch,
        ["measure"],
    )
    categorical_result = DiagnosticFunctions.RobustOLS(
        data,
        category.reshape(-1, 1),
        batch,
        ["category"],
    )

    assert continuous_result.shape == data.shape
    assert categorical_result.shape == data.shape
    assert design_matrices[0].shape[1] == 3
    assert design_matrices[1].shape[1] == 4
