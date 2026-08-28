"""
Basic simulation test for `long_combat`, the R longCombat-style harmonisation
function in DiagnoseHarmonisation.HarmonisationFunctions.

Simulates a simple longitudinal dataset (multiple subjects, repeated timepoints,
batch effects, and a continuous covariate) and checks that `long_combat` runs
end-to-end and returns sensibly shaped, finite outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation.HarmonisationFunctions import long_combat


def _simulate_longitudinal_data(seed: int = 0):
    rng = np.random.default_rng(seed)

    n_subjects = 40
    n_timepoints = 3
    n_features = 5
    batch_levels = ["site_A", "site_B", "site_C"]

    subject_ids = np.repeat(np.arange(n_subjects), n_timepoints)
    timepoints = np.tile(np.arange(n_timepoints), n_subjects)

    # Assign each subject to a single batch (site scanned once, all timepoints there).
    subject_batch = rng.choice(batch_levels, size=n_subjects)
    batch = subject_batch[subject_ids]

    # Random per-subject random intercept and a continuous covariate (e.g. age).
    subject_intercept = rng.normal(0, 1.0, size=n_subjects)
    age = rng.normal(50, 10, size=n_subjects)

    batch_effect_map = {"site_A": 0.0, "site_B": 2.0, "site_C": -1.5}

    n_obs = n_subjects * n_timepoints
    data = np.zeros((n_obs, n_features))
    for f in range(n_features):
        noise = rng.normal(0, 1.0, size=n_obs)
        age_effect = 0.05 * (f + 1) * age[subject_ids]
        batch_effect = np.array([batch_effect_map[b] for b in batch]) * (f + 1)
        time_effect = 0.5 * timepoints
        data[:, f] = (
            10.0
            + subject_intercept[subject_ids]
            + age_effect
            + batch_effect
            + time_effect
            + noise
        )

    data_df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(n_features)])
    batch_series = pd.Series(batch, name="batch")
    model_inputs = pd.DataFrame({
        "subject_id": subject_ids,
        "timepoint": timepoints,
        "age": age[subject_ids],
    })

    return data_df, batch_series, model_inputs


def test_long_combat_runs_and_returns_expected_structure():
    data_df, batch_series, model_inputs = _simulate_longitudinal_data()

    output = long_combat(
        data=data_df,
        batch=batch_series,
        model_inputs=model_inputs,
        verbose=False,
    )

    assert not output["failed_features"]

    bayesdata = output["Bayesdata"]
    assert isinstance(bayesdata, pd.DataFrame)
    assert bayesdata.shape == data_df.shape
    assert list(bayesdata.columns) == list(data_df.columns)
    assert np.all(np.isfinite(bayesdata.to_numpy()))

    n_batch = batch_series.nunique()
    n_features = data_df.shape[1]
    for key in ("gamma_hat", "delta_hat", "gamma_star", "delta_star"):
        arr = output[key]
        assert arr.shape == (n_batch, n_features)
        assert np.all(np.isfinite(arr))

    assert np.all(output["delta_hat"] > 0)
    assert np.all(output["delta_star"] > 0)

    effects = output["Effects"]
    assert effects["sigma"].shape[0] == n_features
    assert np.all(effects["sigma"] > 0)
    assert effects["fitted"].shape == data_df.shape
    assert effects["batch_effects_adjusted"].shape == (n_batch, n_features)


def test_long_combat_reduces_batch_effect_variance():
    data_df, batch_series, model_inputs = _simulate_longitudinal_data(seed=1)

    output = long_combat(
        data=data_df,
        batch=batch_series,
        model_inputs=model_inputs,
        verbose=False,
    )
    bayesdata = output["Bayesdata"]

    # Batch-wise means of harmonised data should be closer together than raw data.
    raw_means = data_df.groupby(batch_series.to_numpy()).mean()
    harmonised_means = bayesdata.groupby(batch_series.to_numpy()).mean()

    raw_spread = raw_means.to_numpy().std(axis=0).mean()
    harmonised_spread = harmonised_means.to_numpy().std(axis=0).mean()

    assert harmonised_spread < raw_spread


def test_long_combat_without_eb_uses_raw_estimates():
    data_df, batch_series, model_inputs = _simulate_longitudinal_data(seed=2)

    output = long_combat(
        data=data_df,
        batch=batch_series,
        model_inputs=model_inputs,
        verbose=False,
        UseEB=False,
    )

    assert output["eb_hist"]["mode"] == "disabled_use_raw"
    np.testing.assert_allclose(output["gamma_star"], output["gamma_hat"])
    np.testing.assert_allclose(output["delta_star"], output["delta_hat"])


def test_long_combat_requires_subject_id_column():
    data_df, batch_series, model_inputs = _simulate_longitudinal_data(seed=3)
    model_inputs_missing = model_inputs.drop(columns=["subject_id"])

    with pytest.raises(ValueError, match="subject_id"):
        long_combat(
            data=data_df,
            batch=batch_series,
            model_inputs=model_inputs_missing,
            verbose=False,
        )


def test_long_combat_rejects_missing_data():
    data_df, batch_series, model_inputs = _simulate_longitudinal_data(seed=4)
    data_with_nan = data_df.copy()
    data_with_nan.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        long_combat(
            data=data_with_nan,
            batch=batch_series,
            model_inputs=model_inputs,
            verbose=False,
        )


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _simulate_longitudinal_trajectory_data(seed: int = 123):
    """
    Simulate longitudinal data while retaining each stage of data generation.

    Each subject has:
      - four timepoints,
      - an individual random intercept,
      - an individual trajectory/slope,
      - a subject-level continuous covariate,
      - a batch independently/randomly assigned at every timepoint.

    Returns the clean data both before and after adding the covariate so that
    the effect of long_combat can be inspected visually.
    """
    rng = np.random.default_rng(seed)

    n_subjects = 40
    n_timepoints = 4
    n_features = 5
    batch_levels = np.array(["site_A", "site_B", "site_C"])

    subject_ids = np.repeat(np.arange(n_subjects), n_timepoints)
    timepoints = np.tile(np.arange(n_timepoints), n_subjects)
    n_obs = len(subject_ids)

    # Batch is deliberately assigned independently at EACH observation rather
    # than fixing a subject to one site.
    batch = rng.choice(batch_levels, size=n_obs)

    # Subject-specific properties.
    subject_intercept = rng.normal(0.0, 2.0, size=n_subjects)
    subject_slope = rng.normal(0.75, 0.25, size=n_subjects)

    # Subject-level continuous covariate.
    age = rng.normal(50.0, 10.0, size=n_subjects)

    batch_effect_map = {
        "site_A": 0.0,
        "site_B": 4.0,
        "site_C": -3.0,
    }

    clean_no_covariate = np.zeros((n_obs, n_features))
    clean_with_covariate = np.zeros((n_obs, n_features))
    pre_harmonisation = np.zeros((n_obs, n_features))

    for f in range(n_features):
        # Keep noise relatively small so individual trajectories are visible.
        noise = rng.normal(0.0, 0.35, size=n_obs)

        intercept = (
            10.0
            + 2.0 * f
            + subject_intercept[subject_ids]
        )

        # Allow both the average longitudinal trend and subject-specific
        # trajectory to vary slightly by feature.
        trajectory = (
            subject_slope[subject_ids]
            * (1.0 + 0.15 * f)
            * timepoints
        )

        clean_no_covariate[:, f] = (
            intercept
            + trajectory
            + noise
        )

        # Centre age so it shifts subjects without introducing a huge offset.
        covariate_effect = (
            0.08
            * (f + 1)
            * (age[subject_ids] - 50.0)
        )

        clean_with_covariate[:, f] = (
            clean_no_covariate[:, f]
            + covariate_effect
        )

        batch_effect = np.array(
            [batch_effect_map[b] for b in batch]
        ) * (1.0 + 0.4 * f)

        pre_harmonisation[:, f] = (
            clean_with_covariate[:, f]
            + batch_effect
        )

    columns = [f"feature_{i + 1}" for i in range(n_features)]

    clean_no_covariate_df = pd.DataFrame(
        clean_no_covariate,
        columns=columns,
    )

    clean_with_covariate_df = pd.DataFrame(
        clean_with_covariate,
        columns=columns,
    )

    pre_harmonisation_df = pd.DataFrame(
        pre_harmonisation,
        columns=columns,
    )

    batch_series = pd.Series(batch, name="batch")

    model_inputs = pd.DataFrame({
        "subject_id": subject_ids,
        "timepoint": timepoints,
        "age": age[subject_ids],
    })

    return (
        clean_no_covariate_df,
        clean_with_covariate_df,
        pre_harmonisation_df,
        batch_series,
        model_inputs,
    )


def test_long_combat_subject_trajectories_visualisation():
    (
        clean_no_covariate,
        clean_with_covariate,
        pre_harmonisation,
        batch_series,
        model_inputs,
    ) = _simulate_longitudinal_trajectory_data()

    output = long_combat(
        data=pre_harmonisation,
        batch=batch_series,
        model_inputs=model_inputs,
        verbose=False,
    )

    assert not output["failed_features"]

    post_harmonisation = output["Bayesdata"]

    # Basic checks before generating the diagnostic figures.
    assert post_harmonisation.shape == pre_harmonisation.shape
    assert np.all(np.isfinite(post_harmonisation.to_numpy()))

    # ---------------------------------------------------------------
    # Plot only a subset so individual trajectories remain readable.
    # ---------------------------------------------------------------
    subjects_to_plot = [0, 1, 2, 3, 4, 5]

    output_dir = (
        Path(__file__).resolve().parent
        / "TestResults"
        / "longcombat"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (
            "Clean: no batch or covariate",
            clean_no_covariate,
        ),
        (
            "Clean: covariate, no batch",
            clean_with_covariate,
        ),
        (
            "Pre-harmonisation: covariate + batch",
            pre_harmonisation,
        ),
        (
            "Post-harmonisation",
            post_harmonisation,
        ),
    ]

    batch_markers = {
        "site_A": "o",
        "site_B": "s",
        "site_C": "^",
    }

    for feature in pre_harmonisation.columns:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
            sharex=True,
            sharey=True,
        )
        axes = axes.ravel()

        for ax, (title, dataset) in zip(axes, datasets):
            for subject in subjects_to_plot:
                mask = (
                    model_inputs["subject_id"].to_numpy()
                    == subject
                )

                subject_time = (
                    model_inputs.loc[mask, "timepoint"].to_numpy()
                )
                subject_values = dataset.loc[mask, feature].to_numpy()

                # Draw the subject trajectory.
                ax.plot(
                    subject_time,
                    subject_values,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"Subject {subject}",
                )

                # For pre-harmonisation, overlay markers identifying the
                # randomly assigned batch at each timepoint.
                if title.startswith("Pre-harmonisation"):
                    subject_batches = batch_series.loc[mask].to_numpy()

                    for t, value, batch in zip(
                        subject_time,
                        subject_values,
                        subject_batches,
                    ):
                        ax.scatter(
                            t,
                            value,
                            marker=batch_markers[batch],
                            s=60,
                            edgecolors="black",
                            zorder=3,
                        )
                else:
                    ax.scatter(
                        subject_time,
                        subject_values,
                        s=30,
                        zorder=3,
                    )

            ax.set_title(title)
            ax.set_xlabel("Timepoint")
            ax.set_ylabel(feature)
            ax.set_xticks([0, 1, 2, 3])
            ax.grid(alpha=0.25)

        # Subject legend.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(subjects_to_plot),
            bbox_to_anchor=(0.5, 0.98),
        )

        # Explicit batch marker legend.
        batch_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label=batch,
            )
            for batch, marker in batch_markers.items()
        ]

        axes[2].legend(
            handles=batch_handles,
            title="Batch",
            loc="best",
        )

        fig.suptitle(
            f"long_combat trajectory diagnostic — {feature}",
            fontsize=14,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        figure_path = output_dir / f"{feature}_trajectories.png"
        fig.savefig(
            figure_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Verify that all expected figures were actually produced.
    for feature in pre_harmonisation.columns:
        expected_file = output_dir / f"{feature}_trajectories.png"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0