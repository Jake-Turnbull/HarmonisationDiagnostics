#!/usr/bin/env python3
"""Two-batch ComBat and SW-ComBat stress-test simulation."""

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "sample_sizes": [20,30,40,50,60,70,80,90, 100,120,140,160,180,200],
    "seeds": list(range(1, 6)),
    "feature_sizes": [50, 100],
    "n_batches": 2,
    "additive_mean": 20.0,
    "additive_sd": 5.0,
    "multiplicative_log_sd": 0.5,
    "scenario_families": {
        "alignment": ["simple", "opposing", "correlated_independent", "correlated_shared"],
        "effect_shapes": ["unimodal", "bimodal", "multimodal", "overlapping", "heavy_tailed"],
    },
}
ALL_SCENARIOS = [scenario for family in DEFAULT_CONFIG["scenario_families"].values()
                 for scenario in family]


def _balanced_batch_labels(n_samples, n_batches, rng):
    if n_batches != 2:
        raise ValueError("This stress-test simulator is fixed to exactly two batches")
    labels = np.tile(np.arange(n_batches), (n_samples + n_batches - 1) // n_batches)[:n_samples]
    rng.shuffle(labels)
    return labels


def _feature_groups(n_features, n_groups=5):
    return np.array_split(np.arange(n_features), n_groups)


def simulate_bimodal_dataset(n_samples=100, n_features=100, n_batches=2, seed=1,
                             additive_mean=20.0, additive_sd=5.0,
                             multiplicative_log_sd=0.5, scenario="simple"):
    """Generate one two-batch additive-plus-multiplicative stress-test dataset."""
    if n_batches != 2:
        raise ValueError("This stress-test simulator is fixed to exactly two batches")
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {ALL_SCENARIOS}")
    if n_features < 5 or n_samples < 2:
        raise ValueError("Require at least 5 features and 2 samples")
    rng = np.random.default_rng(seed)
    batch = _balanced_batch_labels(n_samples, 2, rng)
    age = rng.normal(50.0, 10.0, n_samples)
    sex = rng.binomial(1, 0.5, n_samples).astype(float)
    disease_probability = 1.0 / (1.0 + np.exp(-(-0.8 + 0.035 * (age - 50.0) + 0.15 * sex)))
    disease = rng.binomial(1, disease_probability).astype(float)
    head_size = 1450.0 + 95.0 * sex - 0.9 * (age - 50.0) - 28.0 * disease + rng.normal(0.0, 38.0, n_samples)
    age_z = (age - age.mean()) / (age.std() + 1e-12)

    group_indices = np.empty(n_features, dtype=int)
    for group_idx, feature_indices in enumerate(_feature_groups(n_features)):
        group_indices[feature_indices] = group_idx
    intercept = rng.normal(1000.0, 80.0, n_features)
    age_beta = rng.normal(-4.0, 0.7, n_features)
    sex_beta = rng.normal(2.0, 0.5, n_features)
    disease_beta = rng.normal(-2.5, 0.5, n_features)
    signal = (intercept[None, :] + np.outer(age_z, age_beta)
              + np.outer(sex, sex_beta) + np.outer(disease, disease_beta))
    if scenario in {"correlated_independent", "correlated_shared"}:
        group_latent = rng.normal(0.0, 1.0, size=(n_samples, 5))
        group_loading = rng.normal(14.0, 1.0, size=n_features)
        signal += group_latent[:, group_indices] * group_loading[None, :]
    noise_sd = rng.uniform(7.0, 11.0, n_features)
    residual = rng.normal(0.0, noise_sd, (n_samples, n_features))
    clean = signal + residual

    if scenario == "simple":
        gamma_contrast = rng.normal(additive_mean, additive_sd, n_features)
        log_delta_contrast = rng.normal(
            multiplicative_log_sd, multiplicative_log_sd * 0.2, n_features
        )
        similarity_groups = None
    elif scenario == "opposing":
        signs = np.ones(n_features, dtype=float)
        signs[n_features // 2:] = -1.0
        rng.shuffle(signs)
        gamma_contrast = signs * np.abs(rng.normal(additive_mean, additive_sd, n_features))
        log_delta_contrast = signs * np.abs(
            rng.normal(multiplicative_log_sd, multiplicative_log_sd * 0.2, n_features)
        )
        similarity_groups = None
    elif scenario == "correlated_independent":
        gamma_contrast = rng.normal(0.0, additive_mean, n_features)
        log_delta_contrast = rng.normal(0.0, multiplicative_log_sd, n_features)
        similarity_groups = group_indices
    elif scenario == "correlated_shared":
        group_gamma = rng.normal(0.0, additive_mean, 5)
        group_log_delta = rng.normal(0.0, multiplicative_log_sd, 5)
        gamma_contrast = group_gamma[group_indices] + rng.normal(
            0.0, additive_sd * 0.2, n_features
        )
        log_delta_contrast = group_log_delta[group_indices] + rng.normal(
            0.0, multiplicative_log_sd * 0.1, n_features
        )
        similarity_groups = group_indices
    else:
        centers_by_scenario = {
            "unimodal": [0.3],
            "bimodal": [-1.0, 1.0],
            "multimodal": [-1.4, 0.0, 1.4],
            "overlapping": [-0.2, 0.2],
            "heavy_tailed": [-0.8, 0.8],
        }
        centers = np.asarray(centers_by_scenario[scenario], dtype=float)
        shape_groups = np.arange(n_features) % len(centers)
        locations = centers[shape_groups]
        gamma_noise_sd = additive_sd * (0.14 if scenario != "overlapping" else 0.38)
        gamma_shape = locations + rng.normal(0.0, gamma_noise_sd, n_features)
        if scenario == "heavy_tailed":
            gamma_shape += rng.standard_t(3, n_features) * additive_sd * 0.12
        gamma_contrast = additive_mean * gamma_shape
        log_delta_contrast = multiplicative_log_sd * (
            locations + rng.normal(0.0, 0.12, n_features)
        )
        similarity_groups = shape_groups
        group_indices = shape_groups

    additive = np.stack([-gamma_contrast, gamma_contrast])
    log_delta = np.stack([-log_delta_contrast, log_delta_contrast])
    delta = np.exp(log_delta)
    observed = signal + additive[batch] + np.sqrt(delta[batch]) * residual
    datasets = {
        "additive_and_multiplicative": {
            "observed": observed,
            "clean": clean,
            "true_gamma": additive,
            "true_delta": delta,
        }
    }

    sims = {"oracle": _oracle_similarity(additive, delta)}
    sims["oracle"] = _oracle_similarity(additive, delta)
    return {
        "scenario": scenario,
        "scenario_family": next(
            family for family, scenarios in DEFAULT_CONFIG["scenario_families"].items()
            if scenario in scenarios
        ),
        "datasets": datasets,
        "covariates": {"age": age, "sex": sex, "disease": disease, "head_size": head_size},
        "batch": batch,
        "groups": group_indices,
        "feature_groups": group_indices,
        "signal": signal,
        "residual": residual,
        "noise_sd": noise_sd,
        "additive_effects": additive,
        "multiplicative_effects": delta,
        "log_multiplicative_effects": log_delta,
        "gamma_contrast": gamma_contrast,
        "log_delta_contrast": log_delta_contrast,
        "n_samples": int(n_samples),
        "batch_sizes": {str(level): int(np.sum(batch == level)) for level in np.unique(batch)},
        "similarities": sims,
    }


def _oracle_similarity(truth_gamma, truth_delta):
    gamma_profile = np.asarray(truth_gamma, dtype=float)
    log_delta_profile = np.log(np.asarray(truth_delta, dtype=float))
    profiles = np.concatenate([gamma_profile, log_delta_profile], axis=0).T
    scales = np.std(profiles, axis=0)
    profiles = (profiles - np.mean(profiles, axis=0)) / np.where(scales > 1e-12, scales, 1.0)
    distances = np.sqrt(np.sum((profiles[:, None, :] - profiles[None, :, :]) ** 2, axis=2))
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    similarity = np.exp(-0.5 * (distances / max(bandwidth, 1e-12)) ** 2)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _combat_call(data, batch, design, similarity_method, oracle_similarity=None):
    from DiagnoseHarmonisation import HarmonisationFunctions as H

    method = "custom" if similarity_method == "oracle" else similarity_method
    weight_opts = {"similarity_floor": 1e-8, "min_effective": 2}
    if similarity_method == "oracle":
        weight_opts["similarity_matrix"] = oracle_similarity
    return H.combat_modular(
        data,
        batch,
        design,
        prior_mode="local",
        prior_weight_methods=[method],
        prior_weight_opts=weight_opts,
        covbat_mode=False,
    )


def _age_coefficients(data, covariates):
    age = np.asarray(covariates["age"])
    age_z = (age - age.mean()) / (age.std() + 1e-12)
    head = np.asarray(covariates["head_size"])
    head_z = (head - head.mean()) / (head.std() + 1e-12)
    design = np.column_stack([np.ones(len(age)), age_z, covariates["sex"], covariates["disease"], head_z])
    return np.linalg.lstsq(design, data, rcond=None)[0][1]


def _metric_row(method, result, clean, covariates, truth_gamma, truth_delta, noise_sd):
    corrected = np.asarray(result["bayesdata"])
    priors = result.get("priors", {})
    gamma_star = np.asarray(priors.get("gamma_star", result.get("gamma_star", [])))
    delta_star = np.asarray(priors.get("delta_star", result.get("delta_star", [])))
    truth_gamma_std = truth_gamma / np.asarray(noise_sd)[None, :]
    age_error = _age_coefficients(corrected, covariates) - _age_coefficients(clean, covariates)
    local = priors.get("local_priors", {})
    weights = np.asarray(local.get("weights", []))
    ess = np.asarray(local.get("effective_sample_size", []))
    fallback = np.asarray(local.get("used_global_fallback", []), dtype=float)
    t2_local = np.asarray(local.get("t2_local", []))
    t2_global = np.asarray(priors.get("t2", []))
    variance_ratio = float(np.mean(t2_local / (t2_global[:, None] + 1e-12))) if t2_local.size else 1.0
    effective_features = float(np.mean(ess)) if ess.size else float(clean.shape[1])
    error = corrected - clean
    return {
        "method": method,
        "rmse_to_clean": float(np.sqrt(np.mean(error ** 2))),
        "mae_to_clean": float(np.mean(np.abs(error))),
        "additive_parameter_rmse": float(np.sqrt(np.mean((gamma_star - truth_gamma_std) ** 2))) if gamma_star.shape == truth_gamma_std.shape else np.nan,
        "multiplicative_parameter_rmse": float(np.sqrt(np.mean((delta_star - truth_delta) ** 2))) if delta_star.shape == truth_delta.shape else np.nan,
        "age_coefficient_error": float(np.sqrt(np.mean(age_error ** 2))),
        "effective_feature_count": effective_features,
        "local_global_prior_variance_ratio": variance_ratio,
        "global_fallback_frequency": float(np.mean(fallback)) if fallback.size else 0.0,
        "mean_batch_weight_ess": float(np.mean(ess)) if ess.size else np.nan,
        "mean_weight_concentration": float(np.mean(np.sum(weights ** 2, axis=-1))) if weights.size else np.nan,
    }


def _fit_dataset(data, clean, batch, covariates, truth_gamma, truth_delta, noise_sd):
    from DiagnoseHarmonisation import HarmonisationFunctions as H

    covariate_names = ("age", "sex", "head_size", "disease")
    design = np.column_stack([np.asarray(covariates[name]) for name in covariate_names])
    results = []
    raw_row = _metric_row(
        "Raw", {"bayesdata": data}, clean, covariates, truth_gamma, truth_delta, noise_sd
    )
    raw_row["similarity"] = "none"
    results.append(raw_row)

    combat = H.combat_modular(data, batch, design, covbat_mode=False)
    combat_row = _metric_row(
        "ComBat", combat, clean, covariates, truth_gamma, truth_delta, noise_sd
    )
    combat_row["similarity"] = "none"
    results.append(combat_row)

    oracle_similarity = _oracle_similarity(truth_gamma, truth_delta)
    methods = [
        ("SW-mean-difference", "magnitude_similarity", "mean_difference"),
        ("SW-correlation", "correlation_similarity", "correlation"),
        ("SW-variance-difference", "variance_similarity", "variance_difference"),
        ("SW-oracle", "oracle", "oracle"),
    ]
    weight_snapshots = {}
    for method_name, similarity_method, similarity_label in methods:
        result = _combat_call(
            data, batch, design, similarity_method, oracle_similarity=oracle_similarity
        )
        row = _metric_row(
            method_name, result, clean, covariates, truth_gamma, truth_delta, noise_sd
        )
        row["similarity"] = similarity_label
        results.append(row)

        local_priors = result["priors"]["local_priors"]
        source_key = "scale_weights" if similarity_method == "variance_similarity" else "weights"
        weight_snapshots[similarity_label] = np.mean(local_priors[source_key], axis=0)

    return results, weight_snapshots


def _paired_summary(metrics):
    join_keys = ["scenario", "n_samples", "n_features", "seed", "experiment"]
    combat = metrics[metrics.method == "ComBat"][
        join_keys + ["rmse_to_clean", "mae_to_clean"]
    ]
    comparison = metrics[metrics.method.str.startswith("SW-")].merge(
        combat,
        on=join_keys,
        suffixes=("_sw", "_combat"),
        validate="many_to_one",
    )
    comparison["rmse_difference_sw_minus_combat"] = comparison.rmse_to_clean_sw - comparison.rmse_to_clean_combat
    comparison["mae_difference_sw_minus_combat"] = comparison.mae_to_clean_sw - comparison.mae_to_clean_combat
    comparison["rmse_improvement_percent"] = 100.0 * (
        comparison.rmse_to_clean_combat - comparison.rmse_to_clean_sw
    ) / np.maximum(comparison.rmse_to_clean_combat, 1e-12)
    comparison["mae_improvement_percent"] = 100.0 * (
        comparison.mae_to_clean_combat - comparison.mae_to_clean_sw
    ) / np.maximum(comparison.mae_to_clean_combat, 1e-12)
    comparison["sw_worse_than_combat"] = comparison["rmse_difference_sw_minus_combat"] > 0
    records = []
    group_cols = ["scenario", "n_samples", "n_features", "experiment", "similarity", "method"]
    for keys, group in comparison.groupby(group_cols, dropna=False):
        records.append(dict(
            zip(group_cols, keys),
            metric="rmse_improvement_percent",
            mean_improvement_percent=float(group["rmse_improvement_percent"].mean()),
            ci95_low=float(group["rmse_improvement_percent"].mean() - (
                1.96 * group["rmse_improvement_percent"].std(ddof=1) /
                np.sqrt(group["rmse_improvement_percent"].count())
                if group["rmse_improvement_percent"].count() > 1 else 0.0
            )),
            ci95_high=float(group["rmse_improvement_percent"].mean() + (
                1.96 * group["rmse_improvement_percent"].std(ddof=1) /
                np.sqrt(group["rmse_improvement_percent"].count())
                if group["rmse_improvement_percent"].count() > 1 else 0.0
            )),
            fraction_sw_worse=float(group["sw_worse_than_combat"].mean()),
            n_seeds=int(group["rmse_improvement_percent"].count()),
        ))
        vals = group["mae_improvement_percent"].dropna().to_numpy()
        mean = float(vals.mean()) if len(vals) else np.nan
        half = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        records.append(dict(zip(group_cols, keys), metric="mae_improvement_percent",
                            mean_improvement_percent=mean, ci95_low=mean-half,
                            ci95_high=mean+half, fraction_sw_worse=np.nan, n_seeds=len(vals)))
    return comparison, pd.DataFrame(records)


def _save_figure(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _make_figures(metrics, representatives, output):
    for family in ("alignment", "effect_shapes"):
        family_scenarios = DEFAULT_CONFIG["scenario_families"][family]
        family_metrics = metrics[metrics["scenario"].isin(family_scenarios)]
        family_representatives = {
            key: value for key, value in representatives.items() if key[0] in family_scenarios
        }
        if not family_metrics.empty and family_representatives:
            _make_family_figures(family_metrics, family_representatives, output, family)


def _make_family_figures(metrics, representatives, output, family):
    scenarios = list(dict.fromkeys(metrics["scenario"].tolist()))
    feature_sizes = sorted({key[1] for key in representatives})
    primary_features = feature_sizes[0]

    figure, axes = plt.subplots(len(scenarios), 3,
                                figsize=(12, 3.2 * len(scenarios)), squeeze=False)
    for row_idx, scenario in enumerate(scenarios):
        reps = representatives[(scenario, primary_features)]
        bundle = reps["low"]
        clean = bundle["datasets"]["additive_and_multiplicative"]["clean"]
        correlation = np.corrcoef(clean, rowvar=False)
        image = axes[row_idx, 0].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        batch_counts = "/".join(str(count) for count in bundle["batch_sizes"].values())
        axes[row_idx, 0].set_title(
            f"{scenario}: feature correlation\nN={bundle['n_samples']} ({batch_counts} per batch)"
        )
        figure.colorbar(image, ax=axes[row_idx, 0], fraction=0.046)
        feature = np.arange(clean.shape[1])
        for batch_idx, color in enumerate(("#176B87", "#D95D39")):
            axes[row_idx, 1].scatter(
                feature, bundle["additive_effects"][batch_idx], s=12,
                color=color, label=f"Batch {batch_idx}",
            )
            axes[row_idx, 2].scatter(
                feature, bundle["log_multiplicative_effects"][batch_idx], s=12,
                color=color, label=f"Batch {batch_idx}",
            )
        axes[row_idx, 1].axhline(0, color="black", linewidth=0.7)
        axes[row_idx, 1].set(title="Additive batch effects", xlabel="Feature", ylabel="Gamma")
        axes[row_idx, 2].axhline(0, color="black", linewidth=0.7)
        axes[row_idx, 2].set(title="Log multiplicative effects", xlabel="Feature", ylabel="Log delta")
        if row_idx == 0:
            axes[row_idx, 1].legend(frameon=False, fontsize=8)
            axes[row_idx, 2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output, f"{family}_figure1_feature_correlations_and_effects")

    method_labels = ["mean_difference", "correlation", "variance_difference", "oracle"]
    method_titles = ["Mean-difference weights", "Correlation weights",
                     "Variance-difference scale weights", "True-effect oracle weights"]
    for sample_key, size_label in (("low", "low_n"), ("high", "high_n")):
        figure, axes = plt.subplots(len(scenarios), len(method_labels),
                                    figsize=(3.2 * len(method_labels), 3.0 * len(scenarios)),
                                    squeeze=False)
        for row_idx, scenario in enumerate(scenarios):
            reps = representatives[(scenario, primary_features)]
            bundle = reps[sample_key]
            for col_idx, (method_label, method_title) in enumerate(zip(method_labels, method_titles)):
                matrix = reps[f"{sample_key}_weights"][method_label]
                image = axes[row_idx, col_idx].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
                axes[row_idx, col_idx].set_title(f"{scenario}\n{method_title}", fontsize=9)
                axes[row_idx, col_idx].set_xlabel("Feature")
                axes[row_idx, col_idx].set_ylabel(
                    f"Feature\nN={bundle['n_samples']} ({'/'.join(map(str, bundle['batch_sizes'].values()))}/batch)"
                )
                figure.colorbar(image, ax=axes[row_idx, col_idx], fraction=0.046)
        figure.tight_layout()
        _save_figure(figure, output, f"{family}_figure2_similarity_matrices_{size_label}")

    methods = [
        ("Raw", "none", "Raw"),
        ("ComBat", "none", "ComBat"),
        ("SW-mean-difference", "mean_difference", "SW mean difference"),
        ("SW-correlation", "correlation", "SW correlation"),
        ("SW-variance-difference", "variance_difference", "SW variance difference"),
        ("SW-oracle", "oracle", "SW true-effect oracle"),
    ]
    nrows = len(scenarios)
    ncols = len(feature_sizes)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.0 * nrows),
                                squeeze=False, sharex="col")
    for row_idx, scenario in enumerate(scenarios):
        for col_idx, n_features in enumerate(feature_sizes):
            ax = axes[row_idx, col_idx]
            subset = metrics[(metrics.scenario == scenario) & (metrics.n_features == n_features)]
            for method, similarity, label in methods:
                rows = subset[(subset.method == method) & (subset.similarity == similarity)]
                if rows.empty:
                    continue
                summary = rows.groupby("n_samples")["rmse_to_clean"].agg(["mean", "std", "count"])
                x = summary.index.to_numpy(dtype=int)
                y = summary["mean"].to_numpy()
                half = 1.96 * summary["std"].fillna(0).to_numpy() / np.sqrt(summary["count"].to_numpy())
                ax.errorbar(x, y, yerr=half, marker="o", linewidth=1.4, capsize=2, label=label)
            ax.set_title(f"{scenario}, {n_features} features")
            ax.set_ylabel("RMSE to clean data")
            ax.set_xticks(sorted(subset.n_samples.unique()))
            batch_labels = [f"{n}\n{n // 2}/{n - n // 2}" for n in sorted(subset.n_samples.unique())]
            ax.set_xticklabels(batch_labels)
            ax.grid(axis="y", alpha=0.2)
            if row_idx == nrows - 1:
                ax.set_xlabel("Total N (samples per batch)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output, f"{family}_figure3_rmse_by_sample_size")

    comparison, _ = _paired_summary(metrics)
    sw_methods = [(method, similarity, label) for method, similarity, label in methods
                  if method.startswith("SW-")]
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.0 * nrows),
                                squeeze=False, sharex="col", sharey=True)
    for row_idx, scenario in enumerate(scenarios):
        for col_idx, n_features in enumerate(feature_sizes):
            ax = axes[row_idx, col_idx]
            subset = comparison[(comparison.scenario == scenario) &
                                (comparison.n_features == n_features)]
            for method, similarity, label in sw_methods:
                rows = subset[(subset.method == method) & (subset.similarity == similarity)]
                if rows.empty:
                    continue
                summary = rows.groupby("n_samples")["rmse_improvement_percent"].agg(
                    ["mean", "std", "count"]
                )
                x = summary.index.to_numpy(dtype=int)
                y = summary["mean"].to_numpy()
                half = 1.96 * summary["std"].fillna(0).to_numpy() / np.sqrt(summary["count"].to_numpy())
                ax.errorbar(x, y, yerr=half, marker="o", linewidth=1.4, capsize=2, label=label)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
            ax.set_title(f"{scenario}, {n_features} features")
            ax.set_ylabel("RMSE improvement vs ComBat (%)")
            ax.set_xticks(sorted(subset.n_samples.unique()))
            batch_labels = [f"{n}\n{n // 2}/{n - n // 2}" for n in sorted(subset.n_samples.unique())]
            ax.set_xticklabels(batch_labels)
            ax.grid(axis="y", alpha=0.2)
            if row_idx == nrows - 1:
                ax.set_xlabel("Total N (samples per batch)")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    _save_figure(figure, output, f"{family}_figure4_paired_rmse_improvement")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "bimodal_validation_outputs")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=DEFAULT_CONFIG["sample_sizes"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_CONFIG["seeds"])
    parser.add_argument("--features", type=int, nargs="+", default=DEFAULT_CONFIG["feature_sizes"])
    parser.add_argument("--scenario-family", choices=["alignment", "effect_shapes", "all"],
                        default="all", help="Scenario family to run; defaults to both families.")
    parser.add_argument("--scenarios", nargs="+", choices=ALL_SCENARIOS, default=None)
    args = parser.parse_args()
    if any(sample_size < 2 for sample_size in args.sample_sizes):
        parser.error("sample sizes must be at least 2")
    if any(feature_size < 5 for feature_size in args.features):
        parser.error("feature counts must be at least 5")
    selected_families = list(DEFAULT_CONFIG["scenario_families"])
    if args.scenario_family != "all":
        selected_families = [args.scenario_family]
    family_scenarios = [scenario for family in selected_families
                        for scenario in DEFAULT_CONFIG["scenario_families"][family]]
    if args.scenarios is not None:
        invalid = sorted(set(args.scenarios) - set(family_scenarios))
        if invalid:
            parser.error(f"scenarios {invalid} are outside selected family {args.scenario_family!r}")
        scenarios = args.scenarios
    else:
        scenarios = family_scenarios
    args.output.mkdir(parents=True, exist_ok=True)
    config = {"sample_sizes": args.sample_sizes, "seeds": args.seeds, "feature_sizes": args.features,
              "n_batches": 2, "scenario_family": args.scenario_family, "scenarios": scenarios,
              "batch_sizes_by_n": {str(n): [n // 2, n - n // 2] for n in args.sample_sizes},
              "additive_mean": DEFAULT_CONFIG["additive_mean"],
              "additive_sd": DEFAULT_CONFIG["additive_sd"],
              "multiplicative_log_sd": DEFAULT_CONFIG["multiplicative_log_sd"],
              "experiment": "additive_and_multiplicative"}
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows = []
    representatives = {}
    experiment = "additive_and_multiplicative"
    low_n, high_n = min(args.sample_sizes), max(args.sample_sizes)
    for scenario in scenarios:
        for n_samples in args.sample_sizes:
            for n_features in args.features:
                for seed in args.seeds:
                    print(f"scenario={scenario}, n={n_samples}, features={n_features}, seed={seed}", flush=True)
                    bundle = simulate_bimodal_dataset(n_samples, n_features, 2, seed, scenario=scenario)
                    dataset = bundle["datasets"][experiment]
                    fit_rows, method_weights = _fit_dataset(
                        dataset["observed"], dataset["clean"], bundle["batch"], bundle["covariates"],
                        dataset["true_gamma"], dataset["true_delta"], bundle["noise_sd"])
                    all_rows.extend({"scenario": scenario, "scenario_family": bundle["scenario_family"],
                                     "n_samples": n_samples, "batch_sizes": json.dumps(bundle["batch_sizes"]),
                                     "n_features": n_features, "seed": seed,
                                     "experiment": experiment, **row} for row in fit_rows)
                    reps = representatives.setdefault((scenario, n_features), {})
                    if n_samples == low_n:
                        reps["low"] = bundle
                        reps["low_weights"] = method_weights
                    if n_samples == high_n:
                        reps["high"] = bundle
                        reps["high_weights"] = method_weights
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.output / "raw_metrics.csv", index=False)
    paired, summary = _paired_summary(metrics)
    paired.to_csv(args.output / "paired_differences.csv", index=False)
    summary.to_csv(args.output / "paired_summary_ci95.csv", index=False)
    for (scenario, n_features), reps in representatives.items():
        bundle = reps["low"]
        dataset = bundle["datasets"][experiment]
        np.savez_compressed(
            args.output / f"representative_truth_{scenario}_f{n_features}.npz",
            clean=dataset["clean"], signal=bundle["signal"], residual=bundle["residual"],
            additive_effects=bundle["additive_effects"],
            multiplicative_effects=bundle["multiplicative_effects"],
            log_multiplicative_effects=bundle["log_multiplicative_effects"],
            feature_groups=bundle["feature_groups"], batch=bundle["batch"],
            n_samples=bundle["n_samples"], batch_sizes=bundle["batch_sizes"],
        )
        for sample_key in ("low", "high"):
            sample_bundle = reps[sample_key]
            weights = reps[f"{sample_key}_weights"]
            pd.DataFrame(sample_bundle["similarities"]["oracle"]).to_csv(
                args.output / f"similarity_{scenario}_f{n_features}_{sample_key}_oracle_true_effects.csv",
                index=False,
            )
            for method_name, matrix in weights.items():
                pd.DataFrame(matrix).to_csv(
                    args.output / f"similarity_{scenario}_f{n_features}_{sample_key}_{method_name}_prior_weights.csv",
                    index=False,
                )
    if representatives:
        _make_figures(metrics, representatives, args.output)
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*unstable below.*")
        main()