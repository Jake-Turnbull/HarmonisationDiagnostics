import numpy as np
import pandas as pd

from DiagnoseHarmonisation.Imputations import (
    PCA_imputation,
    knn_imputation,
    mean_imputation,
    median_imputation,
    regression_imputation_covariates,
    regression_imputation_feature,
)


def _sample_df_with_nans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0],
            "f2": [2.0, np.nan, 6.0, 8.0, 10.0],
            "f3": [1.0, 2.0, 3.0, np.nan, 5.0],
        }
    )


def _regression_friendly_df() -> pd.DataFrame:
    # Perfect linear relationships so regression-based imputers have learnable signal.
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x2 = 2.0 * x1
    x3 = x1 + x2
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    df.loc[2, "x1"] = np.nan
    df.loc[4, "x2"] = np.nan
    df.loc[1, "x3"] = np.nan
    return df


def test_mean_imputation_fills_nans_with_column_means():
    df = _sample_df_with_nans()
    out = mean_imputation(df)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0

    expected_f1_mean = np.nanmean(df["f1"].values)
    expected_f2_mean = np.nanmean(df["f2"].values)
    expected_f3_mean = np.nanmean(df["f3"].values)

    assert np.isclose(out.loc[2, "f1"], expected_f1_mean)
    assert np.isclose(out.loc[1, "f2"], expected_f2_mean)
    assert np.isclose(out.loc[3, "f3"], expected_f3_mean)


def test_median_imputation_fills_nans_with_column_medians():
    df = _sample_df_with_nans()
    out = median_imputation(df)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0

    expected_f1_median = np.nanmedian(df["f1"].values)
    expected_f2_median = np.nanmedian(df["f2"].values)
    expected_f3_median = np.nanmedian(df["f3"].values)

    assert np.isclose(out.loc[2, "f1"], expected_f1_median)
    assert np.isclose(out.loc[1, "f2"], expected_f2_median)
    assert np.isclose(out.loc[3, "f3"], expected_f3_median)


def test_knn_imputation_returns_dataframe_without_nans():
    df = _sample_df_with_nans()
    out = knn_imputation(df, n_neighbors=2)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(df.index)
    assert out.isna().sum().sum() == 0


def test_regression_imputation_feature_single_pass_fills_missing():
    df = _regression_friendly_df()
    out = regression_imputation_feature(df.copy(), iter=False)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0


def test_regression_imputation_feature_iterative_fills_missing():
    df = _regression_friendly_df()
    out = regression_imputation_feature(df.copy(), iter=True)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0


def test_regression_imputation_covariates_single_pass_fills_missing():
    # Construct features as linear functions of covariates.
    cov1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cov2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    y1 = 2.0 * cov1 + 3.0 * cov2
    y2 = -1.0 * cov1 + 0.5 * cov2

    df = pd.DataFrame({"cov1": cov1, "cov2": cov2, "y1": y1, "y2": y2})
    df.loc[1, "y1"] = np.nan
    df.loc[4, "y2"] = np.nan

    out = regression_imputation_covariates(df.copy(), covariates=["cov1", "cov2"], iter=False)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0


def test_regression_imputation_covariates_iterative_fills_missing():
    cov1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cov2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    y1 = 2.0 * cov1 + 3.0 * cov2
    y2 = -1.0 * cov1 + 0.5 * cov2

    df = pd.DataFrame({"cov1": cov1, "cov2": cov2, "y1": y1, "y2": y2})
    df.loc[1, "y1"] = np.nan
    df.loc[4, "y2"] = np.nan

    out = regression_imputation_covariates(df.copy(), covariates=["cov1", "cov2"], iter=True)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == df.shape
    assert out.isna().sum().sum() == 0


def test_pca_imputation_returns_expected_shapes_and_no_nans():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "f2": [2.0, np.nan, 6.0, 8.0, 10.0, 12.0],
            "f3": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )

    np.random.seed(0)
    impute_replace, impute_noreplace, pcaU, pcaS, pcaV = PCA_imputation(df, Npca=2, Nrand=0)

    assert isinstance(impute_replace, pd.DataFrame)
    assert isinstance(impute_noreplace, pd.DataFrame)
    assert isinstance(pcaU, pd.DataFrame)
    assert isinstance(pcaS, pd.DataFrame)
    assert isinstance(pcaV, pd.DataFrame)

    assert impute_replace.shape == df.shape
    assert impute_noreplace.shape == df.shape
    assert pcaU.shape == (df.shape[0], 2)
    assert pcaS.shape == (2, 2)
    assert pcaV.shape == (df.shape[1], 2)

    assert impute_replace.isna().sum().sum() == 0
    assert impute_noreplace.isna().sum().sum() == 0
