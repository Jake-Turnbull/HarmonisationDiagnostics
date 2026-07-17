"""
Basic numerical test to ensure that the linear modelling harmonisation is working as expected.

    Use linear models to estimate and remove batch effects from data, optionally adjusting for covariates (which are preserved by default).

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The data to be harmonised (samples x features).
    batch : array-like, shape (n_samples,)
        Batch labels for each sample.
    covariates : array-like, shape (n_samples, n_covariates), optional
        Covariate values for each sample. Can be None if no covariates are used
    covariate_names : sequence of str, optional
        Names for the covariates. If not provided, defaults to "cov1", "cov2", etc.
    covariate_types : sequence or mapping, optional
        Covariate types, either as a sequence (in order) or a mapping from name to type. Types can be "binary", "categorical", or "continuous". If not provided, inferred from data
    feature_names : sequence of str, optional
        Names for the features. If not provided, defaults to "feature_1", "feature_2", etc.
    model_type : str, default 'auto'
        Type of model to fit. Options are:
        - 'auto': mixedlm when batch_as_random or subject_col is supplied, otherwise OLS.
        - 'ols': batch is a fixed categorical effect.
        - 'mixedlm': batch is a random intercept unless subject_col is supplied, in which case subject is the random intercept and batch is fixed.
    batch_as_random : bool, default False
        If True, treat batch as a random effect in mixedlm. Ignored if model_type is 'ols'.
    subject_col : array-like, shape (n_samples,), optional
        Subject labels for each sample. If provided, subjects are treated as random intercepts in mixedlm, and batch is treated as a fixed effect.
    interactions : sequence of (str, str), optional
        Pairs of covariate names to include as interaction terms in the model.
    residuals : str, default 'Batch_only'
        Which residuals to return. Options are:
        - 'Batch_only': returns y minus only the estimated batch effect.
        - 'Full': returns ordinary model residuals (y - fitted).

    Model selection:
      * model_type='ols': batch is a fixed categorical effect.
      * model_type='mixedlm': batch is a random intercept unless subject_col is
        supplied, in which case subject is the random intercept and batch is fixed.
      * model_type='auto': mixedlm when batch_as_random or subject_col is supplied,
        otherwise OLS.
"""

def test_linear_model_harmonisation():
    import numpy as np
    from DiagnoseHarmonisation.HarmonisationFunctions import linearmodelling_harmonisation

    # Create synthetic data with batch effects and covariates
    np.random.seed(0)
    n_samples = 100
    n_features = 6
    n_batches = 3

    # Generate random data
    data = np.random.randn(n_samples, n_features)

    # Assign batches
    batch = np.random.choice(n_batches, size=n_samples)

    # Create covariates (e.g., age and sex)
    age = np.random.randint(20, 60, size=n_samples)

    sex = np.random.choice(['M', 'F'], size=n_samples)

    # Introduce batch effects
    for b in range(n_batches):
        data[batch == b] += b * 0.5  # Add batch effect
    
    # Harmonise the data using linear modelling
    harmonised_data = linearmodelling_harmonisation(
        data=data,
        batch=batch,
        covariates=np.column_stack((age, sex)),
        covariate_names=['age', 'sex'],
        covariate_types={'age': 'continuous', 'sex': 'categorical'},
        model_type='auto',
        batch_as_random=False,
        subject_col=None,
        interactions=None,
        residuals='Batch_only'
    )

    harmonised_data = harmonised_data['Residuals']  # Extract the residuals from the result

    # Residuals is stored as a dataframe, convert to numpy array for shape comparison
    harmonised_data = harmonised_data.to_numpy()
    # Check that the harmonised data has the same shape as the original data
    assert harmonised_data.shape == data.shape, "Harmonised data shape mismatch."

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation.HarmonisationFunctions import (
    linearmodelling_harmonisation,
)


@pytest.fixture
def synthetic_harmonisation_data():
    """Create reproducible data with batch and covariate effects."""
    rng = np.random.default_rng(0)

    n_samples = 120
    n_features = 6
    n_batches = 3

    batch = np.repeat(np.arange(n_batches), n_samples // n_batches)
    rng.shuffle(batch)

    age = rng.integers(20, 70, size=n_samples)
    sex = rng.choice(["M", "F"], size=n_samples)
    scanner = rng.choice(["A", "B", "C"], size=n_samples)

    # Repeated subjects for subject-level mixed models.
    # Each subject has three observations.
    subject = np.repeat(np.arange(n_samples // 3), 3)

    data = rng.normal(size=(n_samples, n_features))

    # Add known covariate effects.
    data += 0.03 * age[:, None]
    data += (sex == "M")[:, None] * 0.25

    # Add batch effects.
    batch_effects = np.array([0.0, 0.75, -0.5])
    data += batch_effects[batch][:, None]

    covariates_array = np.column_stack((age, sex, scanner))
    covariates_df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "scanner": scanner,
        }
    )

    return {
        "data": data,
        "batch": batch,
        "age": age,
        "sex": sex,
        "scanner": scanner,
        "subject": subject,
        "covariates_array": covariates_array,
        "covariates_df": covariates_df,
        "feature_names": [f"volume_{i + 1}" for i in range(n_features)],
    }


def assert_valid_harmonisation_result(result, original_data):
    """Apply common checks to every successful configuration."""
    required_keys = {
        "Residuals",
        "Method",
        "Predicted_effect",
        "model_fit",
        "batch",
        "Covariates",
        "Covariate_types",
        "Model statistics",
        "Batch_effect",
        "Fitted_values",
        "Warnings",
        "Design_formula",
    }

    assert required_keys.issubset(result.keys())

    assert isinstance(result["Residuals"], pd.DataFrame)
    assert isinstance(result["Predicted_effect"], pd.DataFrame)
    assert isinstance(result["Model statistics"], pd.DataFrame)
    assert isinstance(result["Batch_effect"], pd.DataFrame)
    assert isinstance(result["Fitted_values"], pd.DataFrame)

    assert result["Residuals"].shape == original_data.shape
    assert result["Batch_effect"].shape == original_data.shape
    assert result["Fitted_values"].shape == original_data.shape

    assert len(result["Model statistics"]) == original_data.shape[1]

    assert result["Method"] in {"OLS", "MixedLM", "Mixed"}

    assert (
        result["model_fit"]["n_features_succeeded"]
        + result["model_fit"]["n_features_failed"]
        == original_data.shape[1]
    )

    successful = result["Model statistics"].query("status == 'ok'")
    assert len(successful) > 0, result["Model statistics"].get("error")

    assert np.isfinite(
        result["Residuals"].to_numpy()[np.isfinite(original_data)]
    ).all()


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param(
            {
                "model_type": "auto",
                "batch_as_random": False,
                "subject_col": None,
                "residuals": "Batch_only",
            },
            id="auto-fixed-batch",
        ),
        pytest.param(
            {
                "model_type": "ols",
                "batch_as_random": False,
                "subject_col": None,
                "residuals": "Batch_only",
            },
            id="explicit-ols",
        ),
        pytest.param(
            {
                "model_type": "ols",
                "batch_as_random": False,
                "subject_col": None,
                "residuals": "Full",
            },
            id="ols-full-residuals",
        ),
        pytest.param(
            {
                "model_type": "auto",
                "batch_as_random": True,
                "subject_col": None,
                "residuals": "Batch_only",
            },
            id="auto-random-batch",
        ),
        pytest.param(
            {
                "model_type": "mixedlm",
                "batch_as_random": True,
                "subject_col": None,
                "residuals": "Batch_only",
            },
            id="explicit-random-batch",
        ),
        pytest.param(
            {
                "model_type": "mixedlm",
                "batch_as_random": False,
                "subject_col": "use_subject",
                "residuals": "Batch_only",
            },
            id="subject-random-intercept",
        ),
    ],
)
def test_model_configurations(
    synthetic_harmonisation_data,
    configuration,
):
    values = synthetic_harmonisation_data
    configuration = configuration.copy()

    if configuration["subject_col"] == "use_subject":
        configuration["subject_col"] = values["subject"]

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types={
            "age": "continuous",
            "sex": "binary",
            "scanner": "categorical",
        },
        feature_names=values["feature_names"],
        interactions=None,
        min_group_n=3,
        maxiter=200,
        **configuration,
    )

    assert_valid_harmonisation_result(result, values["data"])

    expected_method = (
        "MixedLM"
        if configuration["model_type"] == "mixedlm"
        or configuration["batch_as_random"]
        or configuration["subject_col"] is not None
        else "OLS"
    )

    assert result["Method"] == expected_method


@pytest.mark.parametrize(
    "covariates,covariate_names,covariate_types",
    [
        pytest.param(
            "dataframe",
            None,
            {
                "age": "continuous",
                "sex": "binary",
                "scanner": "categorical",
            },
            id="dataframe-explicit-types",
        ),
        pytest.param(
            "array",
            ["age", "sex", "scanner"],
            {
                "age": "continuous",
                "sex": "binary",
                "scanner": "categorical",
            },
            id="array-explicit-types",
        ),
        pytest.param(
            "dataframe",
            None,
            None,
            id="dataframe-inferred-types",
        ),
        pytest.param(
            "array",
            ["age", "sex", "scanner"],
            None,
            id="array-inferred-types",
        ),
        pytest.param(
            None,
            None,
            None,
            id="no-covariates",
        ),
    ],
)
def test_covariate_input_variations(
    synthetic_harmonisation_data,
    covariates,
    covariate_names,
    covariate_types,
):
    values = synthetic_harmonisation_data

    if covariates == "dataframe":
        covariate_values = values["covariates_df"]
    elif covariates == "array":
        covariate_values = values["covariates_array"]
    else:
        covariate_values = None

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=covariate_values,
        covariate_names=covariate_names,
        covariate_types=covariate_types,
        model_type="ols",
        batch_as_random=False,
        residuals="Batch_only",
    )

    assert_valid_harmonisation_result(result, values["data"])

    if covariate_values is None:
        assert result["Covariates"].empty
        assert len(result["Covariate_types"]) == 0
    else:
        assert list(result["Covariates"].columns) == [
            "age",
            "sex",
            "scanner",
        ]


@pytest.mark.parametrize(
    "interactions",
    [
        pytest.param(None, id="no-interactions"),
        pytest.param(
            [("age", "sex")],
            id="continuous-binary-interaction",
        ),
        pytest.param(
            [("age", "scanner")],
            id="continuous-categorical-interaction",
        ),
        pytest.param(
            [("sex", "scanner")],
            id="binary-categorical-interaction",
        ),
        pytest.param(
            [("age", "sex"), ("age", "scanner")],
            id="multiple-interactions",
        ),
    ],
)
def test_interaction_variations(
    synthetic_harmonisation_data,
    interactions,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types={
            "age": "continuous",
            "sex": "binary",
            "scanner": "categorical",
        },
        model_type="ols",
        interactions=interactions,
        residuals="Batch_only",
    )

    assert_valid_harmonisation_result(result, values["data"])

    if interactions:
        assert ":" in result["Design_formula"]


@pytest.mark.parametrize(
    "standardize_continuous",
    [True, False],
)
def test_continuous_standardisation(
    synthetic_harmonisation_data,
    standardize_continuous,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types={
            "age": "continuous",
            "sex": "binary",
            "scanner": "categorical",
        },
        model_type="ols",
        standardize_continuous=standardize_continuous,
    )

    assert_valid_harmonisation_result(result, values["data"])

    scaling = result["Covariate_scaling"]

    if standardize_continuous:
        assert "age" in scaling.index
        assert np.isfinite(scaling.loc["age", "mean"])
        assert np.isfinite(scaling.loc["age", "std"])


def test_batch_only_removes_estimated_batch_effect(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types={
            "age": "continuous",
            "sex": "binary",
            "scanner": "categorical",
        },
        model_type="ols",
        residuals="Batch_only",
    )

    expected = (
        pd.DataFrame(values["data"])
        - result["Batch_effect"].set_axis(
            range(values["data"].shape[1]),
            axis=1,
        )
    )

    np.testing.assert_allclose(
        result["Residuals"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_full_residuals_equal_data_minus_fitted_values(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types={
            "age": "continuous",
            "sex": "binary",
            "scanner": "categorical",
        },
        model_type="ols",
        residuals="Full",
    )

    expected = values["data"] - result["Fitted_values"].to_numpy()

    np.testing.assert_allclose(
        result["Residuals"].to_numpy(),
        expected,
        rtol=1e-7,
        atol=1e-7,
    )


def test_inferred_covariate_types(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        covariate_types=None,
        model_type="ols",
    )

    inferred = result["Covariate_types"].to_dict()

    assert inferred["age"] == "continuous"
    assert inferred["sex"] in {"binary", "categorical"}
    assert inferred["scanner"] == "categorical"


def test_dataframe_feature_names_are_preserved(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    data_df = pd.DataFrame(
        values["data"],
        columns=values["feature_names"],
    )

    result = linearmodelling_harmonisation(
        data=data_df,
        batch=values["batch"],
        covariates=values["covariates_df"],
        model_type="ols",
    )

    assert list(result["Residuals"].columns) == values["feature_names"]
    assert list(result["Model statistics"].index) == values["feature_names"]


def test_return_models(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    result = linearmodelling_harmonisation(
        data=values["data"],
        batch=values["batch"],
        covariates=values["covariates_df"],
        model_type="ols",
        return_models=True,
    )

    assert "Models" in result
    assert len(result["Models"]) == values["data"].shape[1]


def test_missing_values_are_dropped_per_feature(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data
    data = values["data"].copy()
    covariates = values["covariates_df"].copy()

    data[0, 0] = np.nan
    data[1, 1] = np.nan
    covariates.loc[2, "age"] = np.nan

    result = linearmodelling_harmonisation(
        data=data,
        batch=values["batch"],
        covariates=covariates,
        model_type="ols",
        missing="drop",
    )

    stats = result["Model statistics"]

    assert stats.loc["feature_1", "n_missing"] >= 2
    assert stats.loc["feature_2", "n_missing"] >= 2
    assert np.isnan(result["Residuals"].iloc[0, 0])
    assert np.isnan(result["Residuals"].iloc[1, 1])


def test_missing_values_raise_error(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data
    data = values["data"].copy()
    data[0, 0] = np.nan

    with pytest.raises(ValueError, match="Missing values"):
        linearmodelling_harmonisation(
            data=data,
            batch=values["batch"],
            covariates=values["covariates_df"],
            model_type="ols",
            missing="raise",
        )


@pytest.mark.parametrize(
    "invalid_arguments,error_match",
    [
        (
            {"model_type": "invalid"},
            "model_type",
        ),
        (
            {"residuals": "invalid"},
            "residuals",
        ),
        (
            {"missing": "invalid"},
            "missing",
        ),
    ],
)
def test_invalid_options_raise(
    synthetic_harmonisation_data,
    invalid_arguments,
    error_match,
):
    values = synthetic_harmonisation_data

    with pytest.raises(ValueError, match=error_match):
        linearmodelling_harmonisation(
            data=values["data"],
            batch=values["batch"],
            covariates=values["covariates_df"],
            **invalid_arguments,
        )


def test_batch_length_mismatch_raises(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    with pytest.raises(ValueError, match="Length of batch"):
        linearmodelling_harmonisation(
            data=values["data"],
            batch=values["batch"][:-1],
            covariates=values["covariates_df"],
        )


def test_covariate_length_mismatch_raises(
    synthetic_harmonisation_data,
):
    values = synthetic_harmonisation_data

    with pytest.raises(ValueError, match="same number of rows"):
        linearmodelling_harmonisation(
            data=values["data"],
            batch=values["batch"],
            covariates=values["covariates_df"].iloc[:-1],
        )