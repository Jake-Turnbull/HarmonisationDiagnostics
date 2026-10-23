"""Tests for native local-prior similarity comparisons in the SW benchmark."""

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation import HarmonisationFunctions as hf
from scripts import Bimodal_batch_simulation as simulation


@pytest.mark.parametrize("scenario", simulation.ALL_SCENARIOS)
def test_scenario_generation_is_deterministic(scenario):
    first = simulation.simulate_bimodal_dataset(20, 8, seed=7, scenario=scenario)
    second = simulation.simulate_bimodal_dataset(20, 8, seed=7, scenario=scenario)

    assert first["batch_sizes"] == {"0": 10, "1": 10}
    assert first["datasets"]["additive_and_multiplicative"]["observed"].shape == (20, 8)
    assert np.array_equal(first["additive_effects"], second["additive_effects"])
    assert np.array_equal(first["similarities"]["oracle"], second["similarities"]["oracle"])


def test_fit_uses_native_methods_and_truth_oracle(monkeypatch):
    calls = []
    original = hf._construct_local_priors

    def record_methods(*args, **kwargs):
        calls.append(list(kwargs["weight_methods"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(hf, "_construct_local_priors", record_methods)
    bundle = simulation.simulate_bimodal_dataset(20, 8, seed=7, scenario="bimodal")
    data = bundle["datasets"]["additive_and_multiplicative"]
    results, weights = simulation._fit_dataset(
        data["observed"], data["clean"], bundle["batch"], bundle["covariates"],
        data["true_gamma"], data["true_delta"], bundle["noise_sd"]
    )

    assert calls == [["magnitude_similarity"], ["correlation_similarity"],
                     ["variance_similarity"], ["custom"]]
    assert {row["method"] for row in results if row["method"].startswith("SW-")} == {
        "SW-mean-difference", "SW-correlation", "SW-variance-difference", "SW-oracle"
    }
    assert set(weights) == {"mean_difference", "correlation", "variance_difference", "oracle"}
    assert all(matrix.shape == (8, 8) for matrix in weights.values())
    assert all(np.isfinite(matrix).all() for matrix in weights.values())


def test_true_effect_oracle_is_symmetric_and_diagonal_one():
    gamma = np.array([[-2.0, -1.0, 1.0, 2.0], [2.0, 1.0, -1.0, -2.0]])
    delta = np.exp(np.array([[-0.8, -0.4, 0.4, 0.8], [0.8, 0.4, -0.4, -0.8]]))
    similarity = simulation._oracle_similarity(gamma, delta)

    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity[0, 1] > similarity[0, 2]


def test_family_figures_are_separate_and_include_all_native_methods(tmp_path):
    representatives = {}
    rows = []
    scenario_family = {"simple": "alignment", "bimodal": "effect_shapes"}
    for scenario, family in scenario_family.items():
        for n_samples in (20, 40):
            bundle = simulation.simulate_bimodal_dataset(
                n_samples, 8, seed=5, scenario=scenario
            )
            data = bundle["datasets"]["additive_and_multiplicative"]
            _, weights = simulation._fit_dataset(
                data["observed"], data["clean"], bundle["batch"], bundle["covariates"],
                data["true_gamma"], data["true_delta"], bundle["noise_sd"]
            )
            rep = representatives.setdefault((scenario, 8), {})
            sample_key = "low" if n_samples == 20 else "high"
            rep[sample_key] = bundle
            rep[f"{sample_key}_weights"] = weights
            for method, similarity in [
                ("Raw", "none"), ("ComBat", "none"),
                ("SW-mean-difference", "mean_difference"),
                ("SW-correlation", "correlation"),
                ("SW-variance-difference", "variance_difference"),
                ("SW-oracle", "oracle"),
            ]:
                for seed in (1, 2):
                    rows.append({
                        "scenario": scenario,
                        "scenario_family": family,
                        "n_samples": n_samples,
                        "n_features": 8,
                        "seed": seed,
                        "experiment": "combined",
                        "method": method,
                        "similarity": similarity,
                        "rmse_to_clean": 1.0 + seed * 0.1,
                        "mae_to_clean": 0.5 + seed * 0.1,
                    })

    simulation._make_figures(pd.DataFrame(rows), representatives, tmp_path)

    for family in ("alignment", "effect_shapes"):
        for stem in (
            "figure1_feature_correlations_and_effects",
            "figure2_similarity_matrices_low_n",
            "figure2_similarity_matrices_high_n",
            "figure3_rmse_by_sample_size",
            "figure4_paired_rmse_improvement",
        ):
            path = tmp_path / f"{family}_{stem}.png"
            assert path.is_file()
            assert path.stat().st_size > 0
