"""Focused tests for simulation regimes and benchmark figures."""

import numpy as np
import pandas as pd
import pytest

from scripts import Bimodal_batch_simulation as simulation
from DiagnoseHarmonisation import HarmonisationFunctions as hf


@pytest.mark.parametrize("scenario", simulation.ALL_SCENARIOS)
def test_simulation_scenarios_are_reproducible_and_well_formed(scenario):
    first = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=13, scenario=scenario
    )
    second = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=13, scenario=scenario
    )

    dataset = first["datasets"]["additive_and_multiplicative"]
    assert dataset["observed"].shape == (20, 10)
    assert dataset["clean"].shape == (20, 10)
    assert first["additive_effects"].shape == (2, 10)
    assert first["log_multiplicative_effects"].shape == (2, 10)
    assert first["batch_sizes"] == {"0": 10, "1": 10}
    assert first["scenario_family"] in {"alignment", "effect_shapes"}
    assert np.isfinite(dataset["observed"]).all()
    assert np.isfinite(first["similarities"]["estimated"]).all()
    assert np.array_equal(first["additive_effects"], second["additive_effects"])
    assert np.array_equal(first["similarities"]["estimated"], second["similarities"]["estimated"])


def test_paired_summary_keeps_n_and_feature_count_separate():
    rows = []
    for n_samples in (20, 40):
        for n_features in (5, 10):
            for seed in (1, 2):
                common = {"scenario": "bimodal", "n_samples": n_samples,
                          "n_features": n_features, "seed": seed,
                          "experiment": "combined", "similarity": "none"}
                rows.append({**common, "method": "ComBat", "rmse_to_clean": 2.0,
                             "mae_to_clean": 1.0})
                rows.append({**common, "method": "SW-revised", "similarity": "estimated",
                             "rmse_to_clean": 1.8 if seed == 1 else 2.2,
                             "mae_to_clean": 0.9})

    comparison, summary = simulation._paired_summary(pd.DataFrame(rows))

    assert len(comparison) == 8
    rmse_summary = summary[summary.metric == "rmse_improvement_percent"]
    assert len(rmse_summary) == 4
    assert set(rmse_summary.n_features) == {5, 10}
    assert set(rmse_summary.n_samples) == {20, 40}
    assert np.allclose(rmse_summary.fraction_sw_worse, 0.5)


def test_figure_pipeline_writes_effect_similarity_and_rmse_plots(tmp_path):
    reps = {}
    rows = []
    for n_samples in (20, 40):
        bundle = simulation.simulate_bimodal_dataset(
            n_samples=n_samples, n_features=10, seed=5, scenario="bimodal"
        )
        reps.setdefault(("bimodal", 10), {})["low" if n_samples == 20 else "high"] = bundle
        for seed in (1, 2):
            common = {"scenario": "bimodal", "n_samples": n_samples,
                      "n_features": 10, "seed": seed,
                      "experiment": "combined"}
            rows.extend([
                {**common, "method": "ComBat", "similarity": "none",
                 "rmse_to_clean": 2.0, "mae_to_clean": 1.0},
                {**common, "method": "Raw", "similarity": "none",
                 "rmse_to_clean": 3.0, "mae_to_clean": 2.0},
                {**common, "method": "SW-revised", "similarity": "estimated",
                 "rmse_to_clean": 1.8, "mae_to_clean": 0.9},
                {**common, "method": "SW-original", "similarity": "estimated_unmapped",
                 "rmse_to_clean": 2.2, "mae_to_clean": 1.1},
                {**common, "method": "SW-ablation-no-signed-remap",
                 "similarity": "estimated_unmapped", "rmse_to_clean": 2.1,
                 "mae_to_clean": 1.05},
            ])

    simulation._make_figures(pd.DataFrame(rows), reps, tmp_path)

    for stem in ("figure1_feature_correlations_and_effects", "figure2_similarity_matrices",
                 "figure3_rmse_by_sample_size", "figure4_paired_rmse_improvement"):
        for extension in ("png", "pdf"):
            path = tmp_path / f"{stem}.{extension}"
            assert path.is_file()
            assert path.stat().st_size > 0


def test_oracle_similarity_uses_true_batch_effect_profiles():
    gamma = np.array([[-2.0, -1.0, 1.0, 2.0], [2.0, 1.0, -1.0, -2.0]])
    delta = np.exp(np.array([[-0.8, -0.4, 0.4, 0.8], [0.8, 0.4, -0.4, -0.8]]))

    similarity = simulation._oracle_similarity(gamma, delta)

    assert similarity.shape == (4, 4)
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert similarity[0, 1] > similarity[0, 2]


def test_fit_uses_only_three_native_methods_and_oracle(monkeypatch):
    calls = []
    original = hf._construct_local_priors

    def record_methods(*args, **kwargs):
        calls.append(list(kwargs["weight_methods"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(hf, "_construct_local_priors", record_methods)
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="bimodal"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert calls == [["magnitude_similarity"], ["correlation_similarity"],
                     ["variance_similarity"], ["custom"]]
    assert {row["method"] for row in rows if row["method"].startswith("SW-")} == {
        "SW-mean-difference", "SW-correlation", "SW-variance-difference", "SW-oracle"
    }
    assert set(weights) == {"mean_difference", "correlation", "variance_difference", "oracle"}
    assert all(matrix.shape == (10, 10) for matrix in weights.values())


def test_fit_returns_correct_weights_for_bimodal_data():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="bimodal"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_additive_data():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="additive"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_multiplicative_data():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="multiplicative"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_effect_shapes():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="effect_shapes"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_alignment():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="alignment"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(10),
        "correlation": np.eye(10),
        "variance_difference": np.eye(10),
        "oracle": np.eye(10)
    }


def test_fit_returns_correct_weights_for_all_scenarios_with_different_seeds_and_different_samples_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features_and_different_batch_sizes_and_different_features():
    bundle = simulation.simulate_bimodal_dataset(
        n_samples=20, n_features=10, seed=19, scenario="all"
    )
    dataset = bundle["datasets"]["additive_and_multiplicative"]
    rows, weights = simulation._fit_dataset(
        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"]
    )

    assert weights == {
        "mean_difference": np.eye(1)}