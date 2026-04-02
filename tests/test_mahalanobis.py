from DiagnoseHarmonisation import DiagnosticFunctions
import numpy as np

def make_toy_data():
    rng = np.random.default_rng(0)
    Data = rng.normal(size=(80, 20))  # 12 samples, 4 features
    batch = np.array(["Siemens"] * 20 + ["Philips"] * 20 + ["GE"] * 20 + ["Magnetom"] * 20)
    covariates = rng.normal(size=(80, 2))  # two covariates
    return Data, batch, covariates

def test_returns_expected_keys_without_covariates():
    Data, batch, _ = make_toy_data()
    result = DiagnosticFunctions.Mahalanobis_Distance(Data=Data, batch=batch)
    assert isinstance(result, dict)
    for key in ["pairwise_raw", "centroid_raw", "batches"]:
        assert key in result
    # Residual keys should be None
    assert result["pairwise_resid"] is None
    assert result["centroid_resid"] is None

def test_returns_expected_keys_with_covariates():
    Data, batch, covariates = make_toy_data()
    result = DiagnosticFunctions.Mahalanobis_Distance(Data=Data, batch=batch, covariates=covariates)
    for key in ["pairwise_raw", "pairwise_resid", "centroid_raw", "centroid_resid", "batches"]:
        assert key in result
    # Should give some distances
    assert all(isinstance(v, float) for v in result["pairwise_raw"].values())
    assert all(isinstance(v, float) for v in result["pairwise_resid"].values())

def test_distances_positive():
    Data, batch, covariates = make_toy_data()
    result = DiagnosticFunctions.Mahalanobis_Distance(Data=Data, batch=batch, covariates=covariates)
    for d in list(result["pairwise_raw"].values()) + list(result["pairwise_resid"].values()):
        assert d >= 0

# Additional tests for then plotting functions to ensure they run without error
from DiagnoseHarmonisation import PlotDiagnosticResults
import matplotlib.pyplot as plt

def test_mahalanobis_distance_plot():
    Data, batch, covariates = make_toy_data()
    result = DiagnosticFunctions.Mahalanobis_Distance(Data=Data, batch=batch, covariates=covariates)
    fig, axes = PlotDiagnosticResults.mahalanobis_distance_plot(result,show=True)
    plt.close("all")
    print("Tested Mahalanobis distance plot function successfully.")