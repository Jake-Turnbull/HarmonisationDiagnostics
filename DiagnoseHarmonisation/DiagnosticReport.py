# Diagnostic report generation using DiagnosticFunctions 
from ensurepip import version
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Any
from DiagnoseHarmonisation import PlotComparisonResults

from DiagnoseHarmonisation import DiagnosticFunctions
from DiagnoseHarmonisation import PlotDiagnosticResults
from DiagnoseHarmonisation.LoggingTool import StatsReporter

# Helper function 
def covariate_to_numeric(covariates) -> np.ndarray | None:
    """
    Convert categorical covariates to numeric codes for downstream analyses.

    Args:
        covariates (np.ndarray or pd.DataFrame): Covariate matrix with categorical variables.

    Returns:
        np.ndarray | None: Covariates converted to a numeric array, or `None` if
        no covariates were provided.

    Notes:
        If `covariates` is a DataFrame, each categorical column is factorized
        independently.
        If `covariates` is a NumPy array, each categorical column is factorized
        independently.
        Numeric columns are left unchanged.
    """
    # Check covariate columns independently and factorize if they are categorical (string/object), otherwise keep as is. This allows for mixed covariate types.
    if covariates is None:
        return None
    for i in range(covariates.shape[1]):
        if covariates[:, i].dtype.kind in {"U", "S", "O"}:  # string/object categorical
            covariates[:, i], unique = pd.factorize(covariates[:, i])
        elif covariates[:, i].dtype.kind in {"i", "f"}:  # numeric, keep as is
            pass
    covariate_numeric = covariates.astype(float)  # ensure all numeric for functions that require numeric input
    
    return covariate_numeric


def _normalize_data_matrix(data, feature_names=None) -> tuple[np.ndarray, list | None]:
    """Normalize feature input to a numeric 2D matrix and aligned feature names."""
    inferred_feature_names = None
    if isinstance(data, pd.DataFrame):
        inferred_feature_names = [str(col) for col in data.columns]
        data_numeric = data.apply(pd.to_numeric, errors="coerce")
        matrix = data_numeric.to_numpy(dtype=float)
    else:
        matrix = np.asarray(data)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.ndim != 2:
            raise ValueError(f"data must be 2D (samples x features). Got shape {matrix.shape}.")
        matrix = pd.DataFrame(matrix).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"data must be 2D (samples x features). Got shape {matrix.shape}.")

    if feature_names is None and inferred_feature_names is not None:
        feature_names = inferred_feature_names

    if feature_names is not None and len(feature_names) != matrix.shape[1]:
        raise ValueError(
            f"feature_names length mismatch: expected {matrix.shape[1]}, got {len(feature_names)}."
        )

    return matrix, feature_names


def _normalize_batch_vector(batch, n_samples: int) -> np.ndarray:
    """Normalize batch labels to a 1D vector with n_samples entries."""
    if isinstance(batch, pd.Series):
        batch_arr = batch.to_numpy()
    elif isinstance(batch, pd.DataFrame):
        if batch.shape[1] != 1:
            raise ValueError(
                "batch DataFrame must have exactly one column. "
                f"Got {batch.shape[1]} columns."
            )
        batch_arr = batch.iloc[:, 0].to_numpy()
    else:
        batch_arr = np.asarray(batch)

    if batch_arr.ndim != 1:
        raise ValueError("batch must be a 1D vector of length n_samples.")
    if batch_arr.shape[0] != n_samples:
        raise ValueError(
            f"batch length mismatch: expected {n_samples} samples, got {batch_arr.shape[0]}."
        )
    return batch_arr


def _normalize_covariates_input(covariates, n_samples: int):
    """Normalize covariates to 2D array-like with n_samples rows or None."""
    if covariates is None:
        return None

    if isinstance(covariates, dict):
        cov_arr = np.column_stack(list(covariates.values()))
    elif isinstance(covariates, pd.DataFrame):
        cov_arr = covariates.to_numpy()
    else:
        cov_arr = np.asarray(covariates)

    if cov_arr.ndim == 1:
        cov_arr = cov_arr.reshape(-1, 1)
    if cov_arr.ndim != 2:
        raise ValueError("covariates must be a 2D array-like with n_samples rows.")
    if cov_arr.shape[0] != n_samples:
        raise ValueError(
            f"covariates row mismatch: expected {n_samples}, got {cov_arr.shape[0]}."
        )
    return cov_arr


def _generate_harmonisation_advice(
    cohens_d_results,
    mahalanobis_results,
    lmm_results_df,
    variance_summary_df,
    covariance_results,
    batch_sizes,
):
    """
    Turn the existing diagnostic outputs into short harmonisation advice.

    The thresholds here are intentionally heuristic so the report can give
    practical guidance without introducing new statistical tests.
    """
    abs_d = np.abs(np.asarray(cohens_d_results, dtype=float))
    if abs_d.size == 0:
        max_large_effect_fraction = 0.0
        median_abs_d = 0.0
    else:
        max_large_effect_fraction = float(np.nanmax(np.mean(abs_d >= 0.5, axis=1)))
        median_abs_d = float(np.nanmedian(abs_d))

    pairwise_mahal = np.asarray(
        list((mahalanobis_results or {}).get("pairwise_raw", {}).values()),
        dtype=float,
    )
    centroid_mahal = np.asarray(
        list(
            ((mahalanobis_results or {}).get("centroid_resid") or
             (mahalanobis_results or {}).get("centroid_raw") or {}).values()
        ),
        dtype=float,
    )
    max_pairwise_mahal = float(np.nanmax(pairwise_mahal)) if pairwise_mahal.size else 0.0
    max_centroid_mahal = float(np.nanmax(centroid_mahal)) if centroid_mahal.size else 0.0

    icc_values = np.array([], dtype=float)
    if lmm_results_df is not None and "ICC" in lmm_results_df:
        icc_values = pd.to_numeric(lmm_results_df["ICC"], errors="coerce").dropna().to_numpy(dtype=float)
    median_icc = float(np.nanmedian(icc_values)) if icc_values.size else 0.0
    high_icc_fraction = float(np.mean(icc_values >= 0.1)) if icc_values.size else 0.0

    mean_signal_details = {
        "cohens_d": (max_large_effect_fraction >= 0.2) or (median_abs_d >= 0.35),
        "mahalanobis": (max_pairwise_mahal >= 1.0) or (max_centroid_mahal >= 1.0),
        "lmm": (median_icc >= 0.1) or (high_icc_fraction >= 0.2),
    }
    has_mean_differences = any(mean_signal_details.values())

    has_scale_differences = False
    if variance_summary_df is not None and not variance_summary_df.empty:
        median_logs = pd.to_numeric(
            variance_summary_df["Median log ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        has_scale_differences = bool(
            np.any(np.abs(median_logs) >= np.log(1.25))
        )

    normalized_covariance = (covariance_results or {}).get("pairwise_frobenius_normalized")
    covariance_strength = 0.0
    if normalized_covariance is not None:
        if isinstance(normalized_covariance, pd.DataFrame):
            covariance_array = normalized_covariance.to_numpy(dtype=float)
        else:
            covariance_array = np.asarray(normalized_covariance, dtype=float)

        if covariance_array.ndim == 2 and covariance_array.size > 0:
            upper_idx = np.triu_indices_from(covariance_array, k=1)
            upper_values = covariance_array[upper_idx]
        else:
            upper_values = covariance_array.ravel()

        if upper_values.size:
            covariance_strength = float(np.nanmax(upper_values))
    has_covariance_differences = covariance_strength >= 0.3

    largest_batch = max(batch_sizes, key=batch_sizes.get)
    smallest_batch = min(batch_sizes, key=batch_sizes.get)
    largest_batch_n = batch_sizes[largest_batch]
    smallest_batch_n = batch_sizes[smallest_batch]
    has_large_batch_imbalance = smallest_batch_n > 0 and (largest_batch_n > (2 * smallest_batch_n))

    mean_signal_labels = [
        label.replace("_", " ")
        for label, present in mean_signal_details.items()
        if present
    ]
    advice_lines = []

    if has_mean_differences:
        if mean_signal_labels:
            advice_lines.append(
                "Strong mean-shift signals were detected from \n "
                + ", ".join(mean_signal_labels)
                + ".\n"
            )
    else:
        advice_lines.append(
            "Mean-shift diagnostics were not especially strong, so any harmonisation choice should be made cautiously.\n"
        )

    if has_large_batch_imbalance and has_mean_differences and has_scale_differences and has_covariance_differences:
        advice_lines.append(
            f"{largest_batch} is much larger than the other batches (n={largest_batch_n} vs smallest n={smallest_batch_n}), \n"
            f"and the diagnostics suggest differences in mean, scale, and covariance structure. \n"
            f"CovBat with {largest_batch} as the reference batch looks like the strongest candidate.\n"
        )
    else:
        if has_mean_differences and not has_scale_differences and not has_covariance_differences:
            advice_lines.append(
                "The residual batch effects look mainly additive, so a regression-based harmonisation approach or ComBat would be a sensible first choice.\n"
            )

        if has_scale_differences:
            advice_lines.append(
                "Scale differences were also detected, so ComBat is likely a better fit than a mean-only regression adjustment.\n"
            )

        if has_covariance_differences:
            advice_lines.append(
                "Covariance structure differences were detected between batches, so CovBat could be a good alternative when multivariate structure needs to be aligned.\n"
            )

        if has_large_batch_imbalance:
            advice_lines.append(
                f"Batch sizes are imbalanced and {largest_batch} is the largest batch (n={largest_batch_n}), "
                f"so using {largest_batch} as the ComBat reference batch may help avoid over-correcting that cohort.\n"
            )

    if not any(
        [
            has_mean_differences,
            has_scale_differences,
            has_covariance_differences,
            has_large_batch_imbalance,
        ]
    ):
        advice_lines.append(
            "The diagnostics do not indicate a strong harmonisation target pattern, so a lighter-touch adjustment or no harmonisation may be reasonable depending on the study goal.\n"
        )

    return {
        "advice_lines": advice_lines,
        "has_mean_differences": has_mean_differences,
        "has_scale_differences": has_scale_differences,
        "has_covariance_differences": has_covariance_differences,
        "has_large_batch_imbalance": has_large_batch_imbalance,
        "largest_batch": largest_batch,
        "largest_batch_n": largest_batch_n,
        "smallest_batch": smallest_batch,
        "smallest_batch_n": smallest_batch_n,
        "mean_signal_details": mean_signal_details,
        "covariance_strength": covariance_strength,
    }


@dataclass
class CrossSectionalDiagnosticResult:
    """Container for one method's diagnostics in a comparison report.

    The comparison workflow fills this structure incrementally as each test
    succeeds or fails, then uses the collected fields to build the scorecard,
    summary advice, and per-method export files.
    """

    method_name: str
    data: np.ndarray
    zscore_raw: np.ndarray | None = None
    zscore_residual: np.ndarray | None = None
    cohens_d: np.ndarray | None = None
    cohens_d_pairlabels: list | None = None
    mahalanobis: dict[str, Any] | None = None
    lmm_results: pd.DataFrame | None = None
    lmm_summary: dict[str, Any] | None = None
    variance_ratios: Any | None = None
    variance_pairlabels: list | None = None
    variance_summary: pd.DataFrame | None = None
    levene_raw: dict | None = None
    levene_residual: dict | None = None
    ks_results: dict[str, Any] | None = None
    covariance_results: dict[str, Any] | None = None
    pca_results: dict[str, Any] | None = None
    umap_results: dict[str, Any] | None = None
    summary_metrics: dict[str, float] | None = None
    errors: list[str] | None = None


def validate_comparison_datasets(
    datasets: dict[str, np.ndarray],
    batch,
    covariates=None,
    feature_names=None,
) -> dict[str, np.ndarray]:
    """Validate and normalize the datasets used by the comparison report.

    The function enforces a non-empty mapping of method name to 2D data array,
    checks that every method has the same shape, and validates that batch,
    covariate, and feature-name dimensions are compatible with the data.
    """
    if not isinstance(datasets, dict) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty dictionary mapping method name to data array.")

    normalized: dict[str, np.ndarray] = {}
    seen_names = set()
    expected_shape = None
    inferred_feature_names = None

    for method_name, data in datasets.items():
        if not isinstance(method_name, str) or method_name.strip() == "":
            raise ValueError("Each dataset key must be a non-empty method name string.")
        clean_name = method_name.strip()
        if clean_name in seen_names:
            raise ValueError(f"Duplicate method name detected after stripping whitespace: {clean_name}")
        seen_names.add(clean_name)

        arr, inferred_names = _normalize_data_matrix(data, feature_names=None)
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"Dataset '{clean_name}' must be non-empty. Got shape {arr.shape}.")

        if inferred_names is not None:
            if inferred_feature_names is None:
                inferred_feature_names = inferred_names
            elif inferred_feature_names != inferred_names:
                raise ValueError(
                    "All DataFrame datasets must use identical feature columns and order. "
                    f"Mismatch found for '{clean_name}'."
                )

        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"All datasets must have identical shape. Expected {expected_shape}, got {arr.shape} for '{clean_name}'."
            )

        normalized[clean_name] = arr

    n_samples, n_features = expected_shape

    _normalize_batch_vector(batch, n_samples)

    _normalize_covariates_input(covariates, n_samples)

    if feature_names is None and inferred_feature_names is not None:
        feature_names = inferred_feature_names

    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length mismatch: expected {n_features}, got {len(feature_names)}."
        )

    return normalized


def _compute_variance_summary(variance_ratios: np.ndarray, pair_labels: list[str]) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(pair_labels):
        ratios = np.asarray(variance_ratios[i], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)
        iqr_log = np.nanpercentile(log_ratios, [25, 75])
        rows.append(
            {
                "Comparison": label,
                "Median log ratio": float(np.nanmedian(log_ratios)),
                "Mean log ratio": float(np.nanmean(log_ratios)),
                "IQR lower": float(iqr_log[0]),
                "IQR upper": float(iqr_log[1]),
                "Prop > 0": float(np.nanmean(np.where(np.isnan(log_ratios), False, log_ratios > 0))),
            }
        )
    return pd.DataFrame(rows)


def _compute_covariance_frobenius_summary(
    data: np.ndarray,
    batch: np.ndarray,
    max_components: int = 50,
) -> dict[str, Any]:
    """Compute covariance Frobenius comparisons without plotting."""
    if data.ndim != 2:
        raise ValueError("data must be 2D")

    n_features = data.shape[1]
    k = min(n_features, max_components)
    batch_arr = np.asarray(batch)
    unique_batches = np.unique(batch_arr)
    batch_list = list(unique_batches)
    G = len(batch_list)

    cov_matrices = {}
    for b in batch_list:
        idx = np.where(batch_arr == b)[0]
        if len(idx) < 2:
            cov_matrices[b] = np.full((k, k), np.nan)
        else:
            cov_matrices[b] = np.cov(data[idx, :k], rowvar=False, ddof=1)

    pooled_cov = np.cov(data[:, :k], rowvar=False, ddof=1)
    pooled_frob = float(np.linalg.norm(pooled_cov, ord="fro"))

    pairwise = np.full((G, G), np.nan)
    for i, bi in enumerate(batch_list):
        for j, bj in enumerate(batch_list):
            Ci = cov_matrices[bi]
            Cj = cov_matrices[bj]
            if np.isnan(Ci).all() or np.isnan(Cj).all():
                continue
            pairwise[i, j] = np.linalg.norm(Ci - Cj, ord="fro")

    pairwise_norm = pairwise.copy()
    if pooled_frob > 0 and not np.isnan(pooled_frob):
        pairwise_norm = pairwise / pooled_frob

    return {
        "Number of features": k,
        "cov_matrices": cov_matrices,
        "pairwise_frobenius": pd.DataFrame(pairwise, index=batch_list, columns=batch_list),
        "pairwise_frobenius_normalized": pd.DataFrame(pairwise_norm, index=batch_list, columns=batch_list),
        "pooled_frobenius": pooled_frob,
    }


def extract_summary_metrics(result: CrossSectionalDiagnosticResult) -> dict[str, float]:
    metrics = {
        "median_abs_cohens_d": np.nan,
        "prop_large_abs_cohens_d": np.nan,
        "max_mahalanobis": np.nan,
        "median_icc": np.nan,
        "prop_high_icc": np.nan,
        "median_delta_r2": np.nan,
        "median_abs_log_variance_ratio": np.nan,
        "prop_significant_levene": np.nan,
        "prop_significant_ks": np.nan,
        "max_frobenius_normalized": np.nan,
        "max_abs_batch_pc_correlation": np.nan,
    }

    if result.cohens_d is not None:
        abs_d = np.abs(np.asarray(result.cohens_d, dtype=float)).ravel()
        if abs_d.size:
            metrics["median_abs_cohens_d"] = float(np.nanmedian(abs_d))
            metrics["prop_large_abs_cohens_d"] = float(np.nanmean(abs_d >= 0.5))

    if result.mahalanobis is not None:
        pairwise = (result.mahalanobis.get("pairwise_resid") or result.mahalanobis.get("pairwise_raw") or {})
        vals = np.asarray(list(pairwise.values()), dtype=float)
        if vals.size:
            metrics["max_mahalanobis"] = float(np.nanmax(vals))

    if result.lmm_results is not None and not result.lmm_results.empty:
        icc = pd.to_numeric(result.lmm_results.get("ICC"), errors="coerce").to_numpy(dtype=float)
        if icc.size:
            metrics["median_icc"] = float(np.nanmedian(icc))
            metrics["prop_high_icc"] = float(np.nanmean(icc >= 0.1))

        delta = pd.to_numeric(result.lmm_results.get("delta_R2"), errors="coerce").to_numpy(dtype=float)
        if delta.size:
            metrics["median_delta_r2"] = float(np.nanmedian(delta))

    if result.variance_ratios is not None:
        ratios = np.asarray(result.variance_ratios, dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)
        if log_ratios.size:
            metrics["median_abs_log_variance_ratio"] = float(np.nanmedian(np.abs(log_ratios)))

    levene_source = result.levene_residual if result.levene_residual is not None else result.levene_raw
    if isinstance(levene_source, dict) and len(levene_source) > 0:
        pvals = []
        for comp_result in levene_source.values():
            p = comp_result.get("p_value")
            if p is not None:
                pvals.extend(np.asarray(p, dtype=float).ravel().tolist())
        if len(pvals) > 0:
            pvals_arr = np.asarray(pvals, dtype=float)
            metrics["prop_significant_levene"] = float(np.nanmean(pvals_arr < 0.05))

    if isinstance(result.ks_results, dict) and len(result.ks_results) > 0:
        pvals = []
        for key, comp_result in result.ks_results.items():
            if key == "params" or not isinstance(comp_result, dict):
                continue
            p = comp_result.get("p_value_fdr")
            if p is None:
                p = comp_result.get("p_value")
            if p is not None:
                pvals.extend(np.asarray(p, dtype=float).ravel().tolist())
        if len(pvals) > 0:
            pvals_arr = np.asarray(pvals, dtype=float)
            metrics["prop_significant_ks"] = float(np.nanmean(pvals_arr < 0.05))

    if result.covariance_results is not None:
        norm_mat = result.covariance_results.get("pairwise_frobenius_normalized")
        if norm_mat is not None:
            norm_arr = norm_mat.to_numpy(dtype=float) if isinstance(norm_mat, pd.DataFrame) else np.asarray(norm_mat, dtype=float)
            if norm_arr.ndim == 2 and norm_arr.size > 0:
                iu = np.triu_indices_from(norm_arr, k=1)
                upper = norm_arr[iu]
                if upper.size:
                    metrics["max_frobenius_normalized"] = float(np.nanmax(upper))

    if result.pca_results is not None:
        corr_dict = result.pca_results.get("pc_correlations", {})
        batch_corr = (corr_dict.get("batch") or {}).get("correlation")
        if batch_corr is not None:
            arr = np.asarray(batch_corr, dtype=float)
            if arr.size:
                metrics["max_abs_batch_pc_correlation"] = float(np.nanmax(np.abs(arr)))

    return metrics


def summarise_method_performance(
    results: dict[str, CrossSectionalDiagnosticResult],
    scoring_config: dict | None = None,
) -> pd.DataFrame:
    """Turn per-method diagnostics into a comparable scorecard.

    The summary combines the extracted metrics into category-level scores for
    additive, multiplicative, linear-modelling, distributional, and PCA
    behaviour. Optional scoring configuration can reweight the metrics or mark
    specific metrics as higher-is-better.
    """

    default_weights = {
        "median_abs_cohens_d": 0.20,
        "prop_large_abs_cohens_d": 0.10,
        "max_mahalanobis": 0.15,
        "median_icc": 0.15,
        "median_abs_log_variance_ratio": 0.15,
        "prop_significant_ks": 0.10,
        "max_frobenius_normalized": 0.10,
        "max_abs_batch_pc_correlation": 0.05,
    }

    metric_categories = {
        "additive": ["median_abs_cohens_d", "prop_large_abs_cohens_d", "max_mahalanobis"],
        "multiplicative": ["median_abs_log_variance_ratio", "prop_significant_levene"],
        "linear_modeling": ["median_icc", "prop_high_icc", "median_delta_r2"],
        "distributional": ["prop_significant_ks", "max_frobenius_normalized"],
        "principal_component_analysis": ["max_abs_batch_pc_correlation"],
    }

    cfg = scoring_config or {}
    weights = cfg.get("weights", default_weights)
    higher_is_better = set(cfg.get("higher_is_better", []))

    rows = []
    for method_name, method_result in results.items():
        metrics = method_result.summary_metrics or extract_summary_metrics(method_result)
        row = {"method": method_name}
        row.update(metrics)
        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame(columns=["method", "overall_score", "overall_rank"])

    summary_df = pd.DataFrame(rows)

    for metric, weight in weights.items():
        if metric not in summary_df.columns:
            summary_df[metric] = np.nan

        vals = pd.to_numeric(summary_df[metric], errors="coerce").to_numpy(dtype=float)
        good = ~np.isnan(vals)
        score_col = f"score_{metric}"
        scores = np.full_like(vals, np.nan)

        if np.sum(good) > 0:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if np.isclose(vmax, vmin):
                scores[good] = 1.0
            else:
                if metric in higher_is_better:
                    scores[good] = (vals[good] - vmin) / (vmax - vmin)
                else:
                    scores[good] = (vmax - vals[good]) / (vmax - vmin)

        summary_df[score_col] = scores

    category_scores = {}
    for category, metrics_in_category in metric_categories.items():
        score_columns = [f"score_{metric}" for metric in metrics_in_category if f"score_{metric}" in summary_df.columns]
        if score_columns:
            category_scores[f"{category}_score"] = summary_df[score_columns].mean(axis=1, skipna=True)
        else:
            category_scores[f"{category}_score"] = np.nan

    for column_name, values in category_scores.items():
        summary_df[column_name] = values

    weighted_scores = []
    for _, row in summary_df.iterrows():
        category_values = [row.get(f"{category}_score") for category in metric_categories]
        category_values = [float(value) for value in category_values if pd.notna(value)]
        weighted_scores.append(float(np.mean(category_values)) if category_values else np.nan)

    summary_df["batch_removal_score"] = weighted_scores
    summary_df["covariate_preservation_score"] = summary_df["linear_modeling_score"]
    summary_df["overall_score"] = summary_df["batch_removal_score"]
    summary_df["overall_rank"] = summary_df["overall_score"].rank(method="dense", ascending=False)

    summary_df = summary_df.sort_values(["overall_rank", "overall_score"], ascending=[True, False]).reset_index(drop=True)
    return summary_df


def generate_comparison_advice(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Generate a short natural-language recommendation from the scorecard.

    The advice selects the best overall method, identifies the strongest method
    for each diagnostic theme, and adds a short note when the diagnostics favor
    different methods in different domains.
    """

    if summary_df is None or summary_df.empty:
        return {
            "best_overall": None,
            "best_by_metric": {},
            "summary_text": "No methods were successfully scored.",
        }

    best_overall = str(summary_df.sort_values("overall_score", ascending=False).iloc[0]["method"])

    metric_map = {
        "mean_shift": "median_abs_cohens_d",
        "variance_alignment": "median_abs_log_variance_ratio",
        "batch_random_effect": "median_icc",
        "distribution_similarity": "prop_significant_ks",
        "covariance_alignment": "max_frobenius_normalized",
        "pc_batch_association": "max_abs_batch_pc_correlation",
    }

    best_by_metric = {}
    for label, metric in metric_map.items():
        if metric in summary_df.columns:
            metric_vals = pd.to_numeric(summary_df[metric], errors="coerce")
            if metric_vals.notna().any():
                best_idx = metric_vals.idxmin()
                best_by_metric[label] = str(summary_df.loc[best_idx, "method"])
            else:
                best_by_metric[label] = "No valid result"
        else:
            best_by_metric[label] = "Not computed"

    unique_winners = {v for v in best_by_metric.values() if isinstance(v, str) and v not in {"No valid result", "Not computed"}}
    if len(unique_winners) > 1:
        summary_text = (
            f"Best overall method is {best_overall}, but diagnostics disagree across domains. "
            "This indicates trade-offs between harmonisation targets rather than a universally best method."
        )
    else:
        summary_text = f"Best overall method is {best_overall} and it is consistently strong across the available diagnostics."

    return {
        "best_overall": best_overall,
        "best_by_metric": best_by_metric,
        "summary_text": summary_text,
    }


def _sanitize_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name).strip())
    return clean.strip("_") or "method"


def _format_list_block(title: str, items: list[str]) -> str:
    return title + ":\n" + "\n".join(f"- {item}" for item in items)


def _save_comparison_results(
    result: CrossSectionalDiagnosticResult,
    batch: np.ndarray,
    save_dir: Path,
    report_date: str,
    report_name: str | None,
    feature_names: list | None,
    save_data_name: str | None = None,
) -> dict[str, str]:
    """Write all available per-method diagnostics to disk.

    The helper mirrors the standard single-report export behavior, but prefixes
    files with the method name so the raw outputs from each harmonisation
    strategy stay separate inside the comparison report output directory.
    """

    from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results

    prefix = _sanitize_name(result.method_name)
    if save_data_name:
        prefix = f"{_sanitize_name(save_data_name)}_{prefix}"

    saved_paths: dict[str, str] = {}

    if result.zscore_raw is not None:
        z_raw_df = pd.DataFrame(result.zscore_raw, columns=feature_names)
        saved_paths["zscore_raw"] = save_test_results(
            z_raw_df,
            test_name=f"{prefix}_ZScore_Raw",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.zscore_residual is not None:
        z_resid_df = pd.DataFrame(result.zscore_residual, columns=feature_names)
        saved_paths["zscore_residual"] = save_test_results(
            z_resid_df,
            test_name=f"{prefix}_ZScore_Residual",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.cohens_d is not None and result.cohens_d_pairlabels is not None:
        cohens_dict = {}
        for i, label in enumerate(result.cohens_d_pairlabels):
            cohens_dict[str(label)] = np.asarray(result.cohens_d[i], dtype=float)
        saved_paths["cohens_d"] = save_test_results(
            cohens_dict,
            test_name=f"{prefix}_CohensD",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.mahalanobis is not None:
        maha_dict = {
            "pairwise_raw": result.mahalanobis.get("pairwise_raw", {}),
            "pairwise_resid": result.mahalanobis.get("pairwise_resid", {}),
            "centroid_raw": result.mahalanobis.get("centroid_raw", {}),
            "centroid_resid": result.mahalanobis.get("centroid_resid", {}),
        }
        saved_paths["mahalanobis"] = save_test_results(
            maha_dict,
            test_name=f"{prefix}_Mahalanobis",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.lmm_results is not None:
        saved_paths["lmm"] = save_test_results(
            result.lmm_results,
            test_name=f"{prefix}_LMM",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.variance_ratios is not None and result.variance_pairlabels is not None:
        vr_dict = {}
        for i, label in enumerate(result.variance_pairlabels):
            vr_dict[str(label)] = np.asarray(result.variance_ratios[i], dtype=float)
        saved_paths["variance_ratios"] = save_test_results(
            vr_dict,
            test_name=f"{prefix}_VarianceRatios",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )
    if result.variance_summary is not None:
        saved_paths["variance_summary"] = save_test_results(
            result.variance_summary,
            test_name=f"{prefix}_VarianceSummary",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.levene_raw is not None:
        saved_paths["levene_raw"] = save_test_results(
            result.levene_raw,
            test_name=f"{prefix}_LeveneRaw",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )
    if result.levene_residual is not None:
        saved_paths["levene_residual"] = save_test_results(
            result.levene_residual,
            test_name=f"{prefix}_LeveneResidual",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.ks_results is not None:
        saved_paths["ks"] = save_test_results(
            result.ks_results,
            test_name=f"{prefix}_KS",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date,
            report_name=report_name,
        )

    if result.covariance_results is not None:
        cov_raw = result.covariance_results.get("pairwise_frobenius")
        if cov_raw is not None:
            saved_paths["covariance_frobenius"] = save_test_results(
                cov_raw,
                test_name=f"{prefix}_CovarianceFrobenius",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        cov_norm = result.covariance_results.get("pairwise_frobenius_normalized")
        if cov_norm is not None:
            saved_paths["covariance_frobenius_normalized"] = save_test_results(
                cov_norm,
                test_name=f"{prefix}_CovarianceFrobeniusNormalized",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

    if result.pca_results is not None and "scores" in result.pca_results:
        pca_scores = np.asarray(result.pca_results["scores"], dtype=float)
        pca_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(pca_scores.shape[1])])
        saved_paths["pca_scores"] = save_test_results(
            pca_df,
            test_name=f"{prefix}_PCAScores",
            save_root=save_dir,
            feature_names=list(pca_df.columns),
            report_date=report_date,
            report_name=report_name,
        )

        explained = result.pca_results.get("explained_variance")
        if explained is not None:
            explained_arr = np.asarray(explained, dtype=float)
            explained_df = pd.DataFrame(
                {
                    "PC": [f"PC{i+1}" for i in range(explained_arr.shape[0])],
                    "explained_variance_ratio": explained_arr,
                    "cumulative_explained_variance_ratio": np.cumsum(explained_arr),
                }
            )
            saved_paths["pca_explained_variance"] = save_test_results(
                explained_df,
                test_name=f"{prefix}_PCAExplainedVariance",
                save_root=save_dir,
                feature_names=list(explained_df.columns),
                report_date=report_date,
                report_name=report_name,
            )

        corr_dict = result.pca_results.get("pc_correlations", {})
        if isinstance(corr_dict, dict) and len(corr_dict) > 0:
            corr_rows = []
            for variable_name, corr_values in corr_dict.items():
                corr = np.asarray(corr_values.get("correlation", []), dtype=float)
                pval = np.asarray(corr_values.get("p_value", []), dtype=float)
                n_pc = int(max(corr.shape[0], pval.shape[0]))
                for i in range(n_pc):
                    corr_rows.append(
                        {
                            "variable": str(variable_name),
                            "PC": f"PC{i+1}",
                            "correlation": float(corr[i]) if i < corr.shape[0] else np.nan,
                            "p_value": float(pval[i]) if i < pval.shape[0] else np.nan,
                        }
                    )
            if corr_rows:
                corr_df = pd.DataFrame(corr_rows)
                saved_paths["pca_correlations"] = save_test_results(
                    corr_df,
                    test_name=f"{prefix}_PCACorrelations",
                    save_root=save_dir,
                    feature_names=list(corr_df.columns),
                    report_date=report_date,
                    report_name=report_name,
                )

        if pca_scores.ndim == 2 and pca_scores.shape[0] == result.data.shape[0]:
            n_pc = min(pca_scores.shape[1], 20)
            batch_arr = np.asarray(batch)
            unique_batches = np.unique(batch_arr)
            scree_rows = []
            for batch_name in unique_batches:
                idx = np.where(batch_arr == batch_name)[0]
                if idx.size < 2:
                    continue
                var = np.var(pca_scores[idx, :n_pc], axis=0, ddof=1)
                total = np.nansum(var)
                frac = var / total if total > 0 else np.zeros_like(var)
                cum = np.cumsum(frac)
                for i in range(n_pc):
                    scree_rows.append(
                        {
                            "batch": str(batch_name),
                            "PC": f"PC{i+1}",
                            "variance_fraction": float(frac[i]),
                            "cumulative_variance_fraction": float(cum[i]),
                        }
                    )
            if scree_rows:
                scree_df = pd.DataFrame(scree_rows)
                saved_paths["batch_scree"] = save_test_results(
                    scree_df,
                    test_name=f"{prefix}_BatchScree",
                    save_root=save_dir,
                    feature_names=list(scree_df.columns),
                    report_date=report_date,
                    report_name=report_name,
                )

    if result.umap_results is not None and "embedding" in result.umap_results:
        embedding = np.asarray(result.umap_results["embedding"], dtype=float)
        if embedding.ndim == 2 and embedding.shape[1] >= 2:
            embedding_df = pd.DataFrame(
                {
                    "UMAP1": embedding[:, 0],
                    "UMAP2": embedding[:, 1],
                }
            )
            saved_paths["umap_embedding"] = save_test_results(
                embedding_df,
                test_name=f"{prefix}_UMAPEmbedding",
                save_root=save_dir,
                feature_names=["UMAP1", "UMAP2"],
                report_date=report_date,
                report_name=report_name,
            )

    if result.summary_metrics is not None:
        metrics_df = pd.DataFrame([result.summary_metrics])
        saved_paths["metrics"] = save_test_results(
            metrics_df,
            test_name=f"{prefix}_SummaryMetrics",
            save_root=save_dir,
            feature_names=list(metrics_df.columns),
            report_date=report_date,
            report_name=report_name,
        )

    return saved_paths


def _run_single_method_diagnostics(
    method_name: str,
    data: np.ndarray,
    batch: np.ndarray,
    covariates_numeric,
    covariate_names,
    covariate_types,
    feature_names,
    ratio_type: str,
    compute_umap: bool = True,
) -> CrossSectionalDiagnosticResult:
    """Execute the full diagnostic suite for one harmonisation method.

    The function runs the same core tests used by the single-dataset report,
    captures any failures as strings in the result object, and returns the
    collected outputs for downstream comparison, plotting, and export.
    """

    result = CrossSectionalDiagnosticResult(method_name=method_name, data=data, errors=[])

    # Z-score and residual data
    try:
        result.zscore_raw = DiagnosticFunctions.robust_z_score(data)
    except Exception as exc:
        result.errors.append(f"zscore_raw failed: {exc}")

    data_resid = data
    if covariates_numeric is not None:
        try:
            data_resid = DiagnosticFunctions.RobustOLS(
                data,
                covariates_numeric,
                batch,
                covariate_names or [f"covariate_{i+1}" for i in range(covariates_numeric.shape[1])],
                covariate_types,
            )
            result.zscore_residual = DiagnosticFunctions.robust_z_score(data_resid)
        except Exception as exc:
            result.errors.append(f"residualization failed: {exc}")
            data_resid = data

    try:
        cohens_d, pairlabels = DiagnosticFunctions.Cohens_D(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
        )
        result.cohens_d = cohens_d
        result.cohens_d_pairlabels = pairlabels
    except Exception as exc:
        result.errors.append(f"Cohens_D failed: {exc}")

    try:
        result.mahalanobis = DiagnosticFunctions.Mahalanobis_Distance(data, batch, covariates=covariates_numeric)
    except Exception as exc:
        result.errors.append(f"Mahalanobis_Distance failed: {exc}")

    has_covariates = covariates_numeric is not None and np.asarray(covariates_numeric).shape[1] > 0
    if has_covariates:
        try:
            lmm_df, lmm_summary = DiagnosticFunctions.Run_LMM_cross_sectional(
                data,
                batch,
                covariates=covariates_numeric,
                feature_names=feature_names,
                covariate_names=covariate_names,
                min_group_n=2,
            )
            result.lmm_results = lmm_df
            result.lmm_summary = lmm_summary
        except Exception as exc:
            result.errors.append(f"Run_LMM_cross_sectional failed: {exc}")
    else:
        result.lmm_summary = {
            "status": "skipped_missing_covariates",
            "reason": "No covariates provided; skipping LMM diagnostics.",
        }

    try:
        variance_ratios, pair_labels = DiagnosticFunctions.Variance_Ratios(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
            mode=ratio_type,
        )
        result.variance_ratios = variance_ratios
        result.variance_pairlabels = pair_labels
        result.variance_summary = _compute_variance_summary(variance_ratios, pair_labels)
    except Exception as exc:
        result.errors.append(f"Variance_Ratios failed: {exc}")

    try:
        result.levene_raw = DiagnosticFunctions.Levenes_Test(data, batch, centre="median")
    except Exception as exc:
        result.errors.append(f"Levenes_Test (raw) failed: {exc}")

    if covariates_numeric is not None:
        try:
            result.levene_residual = DiagnosticFunctions.Levenes_Test(data_resid, batch, centre="median")
        except Exception as exc:
            result.errors.append(f"Levenes_Test (residual) failed: {exc}")

    try:
        result.ks_results = DiagnosticFunctions.KS_Test(
            data,
            batch,
            feature_names=feature_names,
            covariates=covariates_numeric,
            do_fdr=True,
            residualize_covariates=True,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
        )
    except Exception as exc:
        result.errors.append(f"KS_Test failed: {exc}")

    try:
        result.covariance_results = _compute_covariance_frobenius_summary(data, batch)
    except Exception as exc:
        result.errors.append(f"Covariance Frobenius summary failed: {exc}")

    try:
        pca_variable_names = ["batch"]
        if covariates_numeric is not None:
            if covariate_names is None:
                pca_variable_names = ["batch"] + [f"covariate_{i+1}" for i in range(covariates_numeric.shape[1])]
            else:
                pca_variable_names = ["batch"] + list(covariate_names)
        explained_variance, scores, correlations, pca_obj = DiagnosticFunctions.PC_Correlations(
            data,
            batch,
            N_components=20,
            covariates=covariates_numeric,
            variable_names=pca_variable_names,
        )
        result.pca_results = {
            "explained_variance": explained_variance,
            "scores": scores,
            "pc_correlations": correlations,
            "pca": pca_obj,
        }
    except Exception as exc:
        result.errors.append(f"PC_Correlations failed: {exc}")

    if compute_umap:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
            embedding = reducer.fit_transform(data)
            result.umap_results = {
                "embedding": embedding,
                "n_neighbors": 10,
                "min_dist": 0.1,
            }
        except Exception as exc:
            result.errors.append(f"UMAP embedding failed: {exc}")

    result.summary_metrics = extract_summary_metrics(result)
    return result

# Min report (for quick surface level checks;)
def CrossSectionalReportMin(data,
                             batch,
                             covariates=None,
                             covariate_names=None,
                             save_data: bool = True,
                             save_data_name: str | None = None,
                             save_dir: str | os.PathLike | None = None,
                             feature_names: list | None = None,
                             report_name: str | None = None,
                             SaveArtifacts: bool = False,
                             rep= None,
                             show: bool = False,
                             timestamped_reports: bool = True,
                             covariate_types: list | None = None,
                             ratio_type: str = "rest"
                             ) -> StatsReporter:
    """
    Create a minimal cross-sectional diagnostic report for quick checks.

    This version keeps the report lightweight by running a reduced subset of
    diagnostics and visualizations. For a more comprehensive analysis, use
    `CrossSectionalReport`.

    Args:
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        feature_names (list, optional): Names of features.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
        timestamped_reports (bool, optional): Whether to append a timestamp to the report filename.
        covariate_types (list, optional): Types of covariates (e.g., 'categorical', 'numeric').
        ratio_type (str, optional): Variance-ratio comparison mode passed to `Variance_Ratios`.

    Returns:
        StatsReporter: The report object containing the generated figures, text,
        and saved artifact references.
    """

    data, feature_names = _normalize_data_matrix(data, feature_names=feature_names)
    batch = _normalize_batch_vector(batch, data.shape[0])
    covariates = _normalize_covariates_input(covariates, data.shape[0])

    if save_dir is None:
        save_dir = Path.cwd()
        # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "CrossSectionalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialised HTML report at: \n"
                            f"{rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()
        report = ctx
    else:
        report = report_ctx
    # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger
        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 150
        report.text_simple("This is a minimal diagnostic report that includes only Z-score visualization, Cohen's D test for mean differences, Variance ratio test for variance differences between batches,\n\n "
        "and ICC/R² from LMM diagnostics. For a more comprehensive report with additional tests and visualizations, please use the full CrossSectionalReport function.")


        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        # Print version info from _version.py
        from DiagnoseHarmonisation._version import version
        report.log_text(f"DiagnoseHarmonisation version: {version}")
            
        # Get todays date for saving results
        report_date = datetime.now().date().isoformat()

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")

        # Samples per batch
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

        # Check for missing data (NaNs) in the dataset, if high proportion log a warning:
        # Create array of size data, if NaN, mark as 1, else 0. Then sum per feature and per batch to get proportion of missing data.
        nan_mask = np.isnan(data)
        # Check proportion of missing data per batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_nans = nan_mask[batch_mask, :]
            prop_nans = np.mean(batch_nans)
            if prop_nans > 0.1:  # arbitrary threshold for logging warning
                logger.warning(f"Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
                report.text_simple(f"Warning: Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
            else:
                report.text_simple(f"Batch {b} has a proportion of missing data: {prop_nans:.2%}")

        # Replace NaNs with structured noise of mean and variance of each batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_data = data[batch_mask, :]
            batch_mean = np.nanmean(batch_data, axis=0)
            batch_std = np.nanstd(batch_data, axis=0)
            # For each Nan, fill with random normal noise with batch mean and std:
            for i in range(batch_data.shape[0]):
                for j in range(batch_data.shape[1]):
                    if np.isnan(batch_data[i, j]):
                        batch_data[i, j] = np.random.normal(loc=batch_mean[j], scale=batch_std[j])
            
            # Replace in original data
            data[batch_mask, :] = batch_data


        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple(
            " The order of tests is as follows: Multivariate distribution comparisson, "
            "Additive tests, Multiplicative tests, Model fit"
        )
        report.text_simple(line_break_in_text)

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        report.text_simple("Z-score normalization (median-centred) visualization across batches,\n" \
        "Here, we convert each feature to a median absolute deviation (MAD) and express each observation as a histogram.\n " \
        "As the normalisation is done globally, batchwise histograms that appear differently (width or location) indicate batch differences in mean and/or variance across features. ")
        
        zscored_data = DiagnosticFunctions.robust_z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)

        covariates_numeric = covariates
        # if dataframe or dictionary, convert to numeric array:
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                covariates_numeric = covariate_to_numeric(covariates.values)
            elif isinstance(covariates, dict):
                covariates_numeric = covariate_to_numeric(np.column_stack(list(covariates.values())))
            elif isinstance(covariates, np.ndarray):
                covariates_numeric = covariate_to_numeric(covariates)
            else:
                raise ValueError("Covariates must be a numpy array, pandas DataFrame, or dictionary of arrays")
        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch, covariates=covariates,covariate_names=covariate_names, covariate_types=covariate_types)
        report.text_simple("Cohen's D test for mean differences completed")

        # Plot (PlotDiagnosticResults should call rep.log_plot internally; our report.log_section ensures plots are attached)
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results, pair_labels=pairlabels, rep=report)
        report.log_text("Cohen's D plot added to report")

        # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict = {}
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results
        if save_data:
            save_test_results(data_dict,
            test_name="Cohens_D",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date, 
            report_name=report_name,
            )

        # Use the same code from CrossSectionalReport, report LMM diagnostics, Variance ratio
        # run LMM diagnostics
        if covariates is not None and covariates.shape[1] > 0:
            lmm_results_df, lmm_summary = DiagnosticFunctions.Run_LMM_cross_sectional(
                data,
                batch,
                covariates=covariates,
                feature_names=feature_names,
                covariate_names=covariate_names,
                min_group_n=2,
            )
        else:
            lmm_results_df = pd.DataFrame()
            lmm_summary = {
                "n_features": data.shape[1],
                "status": "skipped_missing_covariates",
                "reason": "No covariates were provided; skipping LMM diagnostics.",
            }
            report.text_simple("Warning: no covariates provided; LMM diagnostics were skipped.")

        report.text_simple("LMM diagnostics completed.")
        report.log_text("LMM results table added to report")

        # add summary text
        report.text_simple(
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
        )

        # list common notes
        note_lines = []
        numeric_items = [
            (tag, count)
            for tag, count in lmm_summary.items()
            if isinstance(count, (int, float, np.integer, np.floating))
        ]
        for tag, count in sorted(numeric_items, key=lambda x: -float(x[1]))[:10]:
            if tag == 'n_features':
                continue
            note_lines.append(f"{tag}: {count}")
        if "reason" in lmm_summary:
            note_lines.append(f"reason: {lmm_summary['reason']}")
        report.text_simple("LMM diagnostics notes (top):\n" + "\n".join(note_lines))
        data_dict = {}
        # Save DF if needed
        if save_data:
            data_dict['LMM_results_df'] = lmm_results_df
            data_dict['LMM_summary'] = lmm_summary
        
        # Save LMM results as csv
        save_test_results(data_dict,
        test_name="LMM_Results",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date,
        report_name=report_name
        )

        report.text_simple("Histogram of ICC (proportion of variance explained by batch):")
        # How to interpret ICC:
        report.text_simple("Intraclass Correlation Coefficient (ICC) is the ratio of variance due to batch effects to the total variance (batch + residual). \n" 
        "It quantifies the extent to which batch membership explains variability in the data.")
        report.text_simple(
            "Interpretation of ICC values:\n"
            "- ICC close to 0: Little to no variance explained by batch; suggests minimal batch effect.\n"
            "- ICC around 0.1-0.3: Small batch effect; may be acceptable depending on context.\n"
            "- ICC around 0.3-0.5: Moderate batch effect; consider further investigation or correction.\n"
            "- ICC above 0.5: Strong batch effect; likely requires correction to avoid confounding.\n"
        )
        try:
            icc_nonan = lmm_results_df['ICC'].dropna()
            if len(icc_nonan) > 0:
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(icc_nonan)), icc_nonan)
                plt.xlabel("Feature index")
                plt.ylabel("ICC")
                plt.title("ICC values per feature")
                report.log_plot(plt, caption="ICC values per feature")
                plt.close()

        except Exception:
            logger.exception("Could not produce ICC histogram")

        
        # Plot conditional and marginal R^2 per feature, indicate what each means for interpretation
        report.text_simple("Marginal R² represents the variance explained by fixed effects (covariates)\n"
                           "while Conditional R² represents the variance explained by both fixed and random effects (batch + covariates).")
        if {"R2_marginal", "R2_conditional"}.issubset(lmm_results_df.columns):
            lmm_r = lmm_results_df[['R2_marginal', 'R2_conditional']].dropna()
        else:
            lmm_r = pd.DataFrame()
        if len(lmm_r) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(lmm_r['R2_marginal'].values, label='Marginal R²', alpha=0.7)
            plt.plot(lmm_r['R2_conditional'].values, label='Conditional R²', alpha=0.7)
            plt.xlabel("Feature index")
            plt.ylabel("R² value")
            plt.title("Marginal and Conditional R² values per feature")
            plt.legend()
            report.log_plot(plt, caption="Marginal and Conditional R² values per feature")
            plt.close()

        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # Variance ratio
        mode = ratio_type
        report.log_section("variance_ratio", "Variance ratio test (F-test) for variance differences between batches")
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratios, pair_labels = DiagnosticFunctions.Variance_Ratios(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
            mode=mode
        )

        report.log_text("Variance ratio test completed")

        # save variance ratios raw:
        if save_data:
            save_test_results(
                variance_ratios,
                test_name="Variance_Ratios_Raw",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        # Summarise variance ratio results
        data_dict = {}
        summary_rows = []

        # variance_ratios is (num_pairs x num_features)
        n_pairs = variance_ratios.shape[0]

        for i in range(n_pairs):
            ratios = np.array(variance_ratios[i], dtype=float)

            # Safe log: treat non-positive values as NaN for log stats
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)

            mean_log = np.nanmean(log_ratios)
            median_log = np.nanmedian(log_ratios)
            iqr_log = np.nanpercentile(log_ratios, [25, 75])
            # Proportion > 0: treat NaNs as False
            prop_higher = np.nanmean(np.where(np.isnan(log_ratios), False, log_ratios > 0))

            # exponentiate summary stats where meaningful
            median_ratio = np.exp(median_log) if not np.isnan(median_log) else np.nan
            mean_ratio = np.exp(mean_log) if not np.isnan(mean_log) else np.nan

            label = pair_labels[i]
            # Try to split label into two parts like "A / B" -> b1, b2. Otherwise keep label as-is.
            if isinstance(label, str) and " / " in label:
                b1, b2 = [s.strip() for s in label.split(" / ", 1)]
            else:
                # fallback: present full label in Batch 1, leave Batch 2 empty
                b1 = label
                b2 = ""

            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })

            # sanitize label for keys (replace spaces and parentheses)
            safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_vs_")

            data_dict[f"VarianceRatio_{safe_label}"] = ratios
            data_dict[f"MedianLogVarianceRatio_{safe_label}"] = median_log
            data_dict[f"MeanLogVarianceRatio_{safe_label}"] = mean_log
            data_dict[f"IQRLowerLogVarianceRatio_{safe_label}"] = iqr_log[0]
            data_dict[f"IQRUpperLogVarianceRatio_{safe_label}"] = iqr_log[1]
            data_dict[f"PropHigherLogVarianceRatio_{safe_label}"] = prop_higher
            data_dict[f"MedianVarianceRatioExp_{safe_label}"] = median_ratio
            data_dict[f"MeanVarianceRatioExp_{safe_label}"] = mean_ratio

            # human-readable report line
            report.text_simple(
                f"Variance ratio {label}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in {b1}"
            )

        # Save summary as well
        if save_data:
            save_test_results(
                data_dict,
                test_name="Variance_Ratio_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        summary_df = pd.DataFrame(summary_rows)
        report.text_simple("Variance ratio test summaries (per batch pair):")
        # Plot using your plot function
        PlotDiagnosticResults.variance_ratio_plot(variance_ratios, pair_labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")
    
        report.text_simple(line_break_in_text)
    
        report.text_simple(line_break_in_text)
        # add summary text

        report.log_section("report_conclusion", "Minimal Cross-Sectional Report")
        report.text_simple(
            "This concludes the minimal cross-sectional diagnostic report. "
            "For a more comprehensive analysis with additional tests and visualizations, "
            "please use the full CrossSectionalReport function."
        )

        report.text_simple("Summary:")
                # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict = {}
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        # Report LMM diagnostics summary
        report.text_simple(
            f"LMM diagnostics summary:\n"
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
        )
        # Report variance ratio summary
        report.text_simple("Variance ratio test summaries (per batch pair):")
        for _, row in summary_df.iterrows():
            report.text_simple(
                f"Batch {row['Batch 1']} vs Batch {row['Batch 2']}: median log ratio={row['Median log ratio']:.3f} "
                f"(IQR {row['IQR lower']:.3f}–{row['IQR upper']:.3f}), "
                f"{row['Prop > 0']*100:.1f}% of features higher in batch {row['Batch 1']}"
            )
        # Finalise report:
    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report
            report_ctx.__exit__(None, None, None)

# Full cross-sectional report with all tests and visualizations:
def CrossSectionalReport(
    data,
    batch,
    covariates=None,
    covariate_names=None,
    save_data: bool = True,
    save_data_name: str | None = None,
    save_dir: str | os.PathLike | None = None,
    feature_names: list | None = None,
    report_name: str | None = None,
    SaveArtifacts: bool = False,
    rep= None,
    show: bool = False,
    timestamped_reports: bool = True,
    covariate_types: list | None = None,
    ratio_type: str = "rest",
    UMAP_embedding = True, # whether to include UMAP embedding visualizations in the report
    UMAP_tuning = 'auto', # can also be batch or none (for default umap with no tuning)
    Random_state = None # random state for reproducibility of UMAP embeddings
) -> StatsReporter:
    """
    Create a full cross-sectional diagnostic report for batch effects.

    The report combines summary text, statistical tests, and visualizations for
    mean, variance, covariance, clustering, and distributional differences
    across batches.

    Args:
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        feature_names (list, optional): Names of features.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
        timestamped_reports (bool, optional): Whether to append a timestamp to the report filename.
        covariate_types (list, optional): Types of covariates used by the report's numeric and categorical workflows.
        ratio_type (str, optional): Variance-ratio comparison mode passed to `Variance_Ratios`.

    Returns:
        StatsReporter: The report object containing the generated narrative,
        figures, and saved outputs.

    Notes:
        `covariate_types` should align with `covariate_names` so the report can
        decide when to factorize categorical covariates and when to keep numeric
        covariates unchanged.
        If `covariate_types` is not provided, the function infers categorical
        versus numeric handling from the supplied data.




    """


    data, feature_names = _normalize_data_matrix(data, feature_names=feature_names)
    batch = _normalize_batch_vector(batch, data.shape[0])
    covariates = _normalize_covariates_input(covariates, data.shape[0])

    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "CrossSectionalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"initialised HTML report at: \n"
                            rf"{rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()
        report = ctx
    else:
        report = report_ctx
    # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 150

        report.make_title("Cross-sectional harmonisation diagnostic report")
        report.set_report_title("Cross-sectional harmonisation diagnostic report")

        report.log_section("Preamble", "Report preamble and overview")

        report.text_simple("This is the full diagnostic cross-sectional report that includes a comprehensive"\
                           "set of tests and visualizations to assess batch effects in the dataset. \n\n")
        report.text_simple("For full documentation and interpretation of each test, please refer to the online documentation at:"\
                            " https://jake-turnbull.github.io/HarmonisationDiagnostics/")

        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        # Print version info from _version.py
        from DiagnoseHarmonisation._version import version
        report.log_text(f"DiagnoseHarmonisation version: {version}")

            
        # Get todays date for saving results
        report_date = datetime.now().date().isoformat()

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                report.text_simple(
                    f"Batch categories detected: {list(set(batch))}. "
                    "Numeric codes will be used for tests, but string labels will be kept "
                    "for plots and summaries where possible."
                )
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")

        # Samples per batch
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

                # Check for missing data (NaNs) in the dataset, if high proportion log a warning:
        # Create array of size data, if NaN, mark as 1, else 0. Then sum per feature and per batch to get proportion of missing data.
        nan_mask = np.isnan(data)
        # Check proportion of missing data per batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_nans = nan_mask[batch_mask, :]
            prop_nans = np.mean(batch_nans)
            if prop_nans > 0.001:  # arbitrary threshold for logging warning
                logger.warning(f"Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
                report.text_simple("Missing data will be replaced with batch-specific structured noise (random normal)")
                logger.warning("If this is unwanted, or if batch size is too small to reliably estimate batch-specific noise, consider removing these samples prior to analysis.")
            else:
                report.text_simple(f"Batch {b} has a proportion of missing data: {prop_nans:.2%}")

        

        # Replace NaNs with structured noise of mean and variance of each batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_data = data[batch_mask, :]
            batch_mean = np.nanmean(batch_data, axis=0)
            batch_std = np.nanstd(batch_data, axis=0)
            # For each Nan, fill with random normal noise with batch mean and std:
            for i in range(batch_data.shape[0]):
                for j in range(batch_data.shape[1]):
                    if np.isnan(batch_data[i, j]):
                        batch_data[i, j] = np.random.normal(loc=batch_mean[j], scale=batch_std[j])
            
            # Replace in original data
            data[batch_mask, :] = batch_data

        # Check if any columns still have NaNs, if this has happened, its because all data had missing data for that batch, so we fill with global mean and std:
        nan_mask_after = np.isnan(data)
        if np.any(nan_mask_after):
            global_mean = np.nanmean(data, axis=0)
            global_std = np.nanstd(data, axis=0)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isnan(data[i, j]):
                        data[i, j] = np.random.normal(loc=global_mean[j], scale=global_std[j])
                        # Warn user that this has happened in the report and that this feature will not be reliable for diagnostics:
                        # Check if covariate names given to give index and name of feature with all NaNs:
                        if feature_names is not None and j < len(feature_names):
                            feature_name = feature_names[j]
                        else:                           
                            feature_name = f"index {j}"
                        logger.warning(f"Feature {feature_name} has all missing data for at least one batch. NaNs in this feature have been replaced with global mean and std, which may not be reliable for diagnostics.")

        # Final check; if there are still NaNs it is because whole column for all batches is NaN, so we set all values to 1 and log a warning that this feature will not be reliable for diagnostics:
        nan_mask_final = np.isnan(data)
        if np.any(nan_mask_final):
            for j in range(data.shape[1]):
                if np.any(nan_mask_final[:, j]):
                    data[:, j] = 1.0
                    if feature_names is not None and j < len(feature_names):
                        feature_name = feature_names[j]
                    else:                           
                        feature_name = f"index {j}"
                    logger.warning(f"Feature {feature_name} has all missing data across all batches. NaNs in this feature have been replaced with 1. This feature will not be reliable for diagnostics.")

        report.text_simple(
            "Missing data (NaNs) have been replaced with batch-specific structured "
            "noise (random normal with batch mean and std) for the purposes of diagnostics. \n"
        )

        report.text_simple(line_break_in_text)
        report.text_simple("\n\n")

        # Repeat data replacement process for covariates if needed:
        if covariates is not None:
            covariates_numeric = covariates
            # if dataframe or dictionary, convert to numeric array:
            if isinstance(covariates, pd.DataFrame):
                covariates_numeric = covariate_to_numeric(covariates.values)
            elif isinstance(covariates, dict):
                covariates_numeric = covariate_to_numeric(np.column_stack(list(covariates.values())))
            elif isinstance(covariates, np.ndarray):
                covariates_numeric = covariate_to_numeric(covariates)
            else:
                raise ValueError("Covariates must be a numpy array, pandas DataFrame, or dictionary of arrays")
            
        #  Check for NaNs and replace with batch mean (for numeric) or mode (for categorical) as appropriate:
        if covariates is not None:
            n_total = covariates_numeric.shape[0]
            cat_threshold = int(np.ceil(0.05 * n_total))  # 5% of total dataset size

            for col_idx in range(covariates_numeric.shape[1]):
                col = covariates_numeric[:, col_idx]
                nan_mask_cov = np.isnan(col)
                if not np.any(nan_mask_cov):
                    report.text_simple(f"Covariate column {col_idx} has no missing data.")
                    continue

                # Determine if column is categorical:
                # categorical if values are integers and number of unique non-NaN values <= 5% of n_total
                col_nonan = col[~nan_mask_cov]
                unique_vals = np.unique(col_nonan)
                is_integer_valued = np.all(col_nonan == np.floor(col_nonan))
                is_categorical = is_integer_valued and (len(unique_vals) <= cat_threshold)

                for b in unique_batches:
                    batch_mask_cov = (batch == b) & nan_mask_cov
                    if not np.any(batch_mask_cov):
                        continue

                    batch_col_nonan = col[(batch == b) & ~nan_mask_cov]

                    if is_categorical:
                        # Replace with batch-specific mode
                        if len(batch_col_nonan) > 0:
                            vals, counts_cov = np.unique(batch_col_nonan, return_counts=True)
                            mode_val = vals[np.argmax(counts_cov)]
                        else:
                            # fallback to global mode if no non-NaN values in this batch
                            vals, counts_cov = np.unique(col_nonan, return_counts=True)
                            mode_val = vals[np.argmax(counts_cov)]
                        covariates_numeric[batch_mask_cov, col_idx] = mode_val
                        report.text_simple(
                            f"Covariate column {col_idx} (categorical): replaced {np.sum(batch_mask_cov)} "
                            f"NaNs in batch '{b}' with mode={mode_val}"
                        )
                    else:
                        # Replace with batch-specific Gaussian noise
                        if len(batch_col_nonan) > 1:
                            b_mean = np.nanmean(batch_col_nonan)
                            b_std = np.nanstd(batch_col_nonan)
                        elif len(batch_col_nonan) == 1:
                            b_mean = batch_col_nonan[0]
                            b_std = np.nanstd(col_nonan) if len(col_nonan) > 1 else 0.0
                        else:
                            # fallback to global stats
                            b_mean = np.nanmean(col_nonan)
                            b_std = np.nanstd(col_nonan)
                        n_missing = np.sum(batch_mask_cov)
                        fill_vals = np.random.normal(loc=b_mean, scale=max(b_std, 1e-8), size=n_missing)
                        covariates_numeric[batch_mask_cov, col_idx] = fill_vals
                        report.text_simple(
                            f"Covariate column {col_idx} (continuous): replaced {n_missing} "
                            f"NaNs in batch '{b}' with Gaussian noise (mean={b_mean:.4f}, std={b_std:.4f})"
                        )

        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple("This pipeline breaks down batch analysis into the following tests:\n" \
        "1. Multivariate visualisation of batch differences (MAD histograms)," \
        "2. Univariate additive tests (Cohen's D for mean differences) " \
        "3. Multivariate additive tests (Mahalanobis distance) \n "
        "4. LMM diagnostics, unique variance explained by batch and model fit comparisson\n"\
        "5. Multiplicative tests (Variance ratio test for variance differences between batches) \n" \
        "6. Correlation between batch, covariates and principal components\n" \
        "7. PCA eigenvalues by batch and overall difference in covariance structure\n"\
        "8. UMAP visualization of batch and covariate clusters\n"
        "9. Population similarity between batches using univariate two sample Kolmogorov-Smirnov test and multivariate MMD test\n" 
        )
        report.text_simple(line_break_in_text)

        # If random state not provided, leave as None, otherwise set random seed for reproducibility

        if Random_state is not None:
            np.random.seed(Random_state)
            logger.info(
                f"Random state set to {Random_state} for reproducibility of UMAP embeddings "
                "and other random processes."
            )

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        logger.info("Generating Z-score normalization visualization")

        report.text_simple("Z-score normalization (median-centred) visualization across batches,\n" \
        "Here, we convert each feature to a median absolute deviation (MAD) and express each observation as a histogram.\n " \
        "As the normalisation is done globally, batchwise histograms that appear differently (width or location) indicate batch differences in mean and/or variance across features. ")


        zscored_data = DiagnosticFunctions.robust_z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)

        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch, covariates=covariates_numeric,covariate_names=covariate_names, covariate_types=covariate_types)
        report.text_simple("Cohen's D test for mean differences completed")

        # Plot (PlotDiagnosticResults should call rep.log_plot internally; our report.log_section ensures plots are attached)
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results, pair_labels=pairlabels, rep=report)
        report.log_text("Cohen's D plot added to report")
        data_dict = {}
        # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results
        
        if save_data:
            save_test_results(data_dict,
            test_name="Cohens_D",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date, 
            report_name=report_name,
            )
        report.text_simple("Cohen's D test summaries added to report and saved as csv if requested")
        report.text_simple(line_break_in_text)

        # Mahalanobis
        report.log_section("mahalanobis", "Mahalanobis distance test")
        logger.info("Doing Mahalanobis distance test for multivariate mean differences")
        mahalanobis_results = DiagnosticFunctions.Mahalanobis_Distance(data, batch, covariates=covariates_numeric)
        report.log_text("Mahalanobis distance test for multivariate mean differences completed")
        PlotDiagnosticResults.mahalanobis_distance_plot(mahalanobis_results, rep=report)
        report.log_text("Mahalanobis distance plot added to report")

        # Summaries from mahalanobis_results
        pairwise_distances = mahalanobis_results.get("pairwise_raw", {})
        for (b1, b2), dist in pairwise_distances.items():
            report.text_simple(f"Mahalanobis distance between {b1} and {b2}: {dist:.4f}")

        centroid_distances = mahalanobis_results.get("centroid_raw", {})
        for b, dist in centroid_distances.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid: {dist:.4f}")

        centroid_resid_distance = mahalanobis_results.get("centroid_resid", {})
        for b, dist in centroid_resid_distance.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid after residualising by covariates: {dist:.4f}")
        data_dict = {}
        if save_data:
            for b, dist in centroid_distances.items():
                data_dict[f"Mahonalobis_Centroid_Batch{b}"] = dist
            for b, dist in centroid_resid_distance.items():
                data_dict[f"Mahonalobis_Centroid_Resid_Batch{b}"] = dist
        
        save_test_results(data_dict,
        test_name="Mahalanobis_Distance",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date, 
        report_name=report_name,
        )
        report.text_simple("Mahalanobis distance test summaries added to report and saved as csv if requested")
        report.text_simple(line_break_in_text)
        # ---------------------
        # Mixed model tests
        # ---------------------
        logger.info("Beginning LMM diagnostics")
        report.log_section("lmm_diagnostics", "Linear mixed effects diagnostics (batch + covariates)")
        report.text_simple(
            "Fitting per-feature LMMs (random intercept for batch). "
            "Where LMM fails or batch variance is zero we fallback to OLS fixed-effects."
        )

        from DiagnoseHarmonisation import temp
        # run LMM diagnostics
        if covariates is not None and covariates.shape[1] > 0:
            lmm_results_df, lmm_summary = DiagnosticFunctions.Run_LMM_cross_sectional(
                data,
                batch,
                covariates=covariates,
                feature_names=feature_names,
                covariate_names=covariate_names,
                min_group_n=2,
            )
        else:
            lmm_results_df = pd.DataFrame()
            lmm_summary = {
                "n_features": data.shape[1],
                "status": "skipped_missing_covariates",
                "reason": "No covariates were provided; skipping LMM diagnostics.",
            }
            report.text_simple("Warning: no covariates provided; LMM diagnostics were skipped.")

        report.text_simple("LMM diagnostics completed.")
        report.log_text("LMM results table added to report")

        report.text_simple(
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
    )

        # list common notes
        note_lines = []
        numeric_items = [
            (tag, count)
            for tag, count in lmm_summary.items()
            if isinstance(count, (int, float, np.integer, np.floating))
        ]
        for tag, count in sorted(numeric_items, key=lambda x: -float(x[1]))[:10]:
            if tag == 'n_features':
                continue
            note_lines.append(f"{tag}: {count}")
        if "reason" in lmm_summary:
            note_lines.append(f"reason: {lmm_summary['reason']}")
        report.text_simple("LMM diagnostics notes (top):\n" + "\n".join(note_lines))
        data_dict = {}
        # Save DF if needed
        if save_data:
            data_dict = lmm_results_df
        
        # Save LMM results as csv
        save_test_results(data_dict,
        test_name="LMM_fitting_results",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date,
        report_name=report_name
        )

        report.text_simple("Histogram of ICC (proportion of variance explained by batch):")
        # How to interpret ICC:
        report.text_simple("Intraclass Correlation Coefficient (ICC) is the ratio of variance due to batch effects to the total variance (batch + residual). \n" 
        "It quantifies the extent to which batch membership explains variability in the data.")
        report.text_simple(
            "Interpretation of ICC values:\n"
            "- ICC close to 0: Little to no variance explained by batch; suggests minimal batch effect.\n"
            "- ICC around 0.1-0.3: Small batch effect; may be acceptable depending on context.\n"
            "- ICC around 0.3-0.5: Moderate batch effect; consider further investigation or correction.\n"
            "- ICC above 0.5: Strong batch effect; likely requires correction to avoid confounding.\n"
        )
        
        # Plot conditional and marginal R^2 per feature, indicate what each means for interpretation
        report.text_simple("Marginal R² represents the variance explained by fixed effects (covariates)\n"
                           "while Conditional R² represents the variance explained by both fixed and random effects (batch + covariates).")
        if {"R2_marginal", "R2_conditional"}.issubset(lmm_results_df.columns):
            lmm_r = lmm_results_df[['R2_marginal', 'R2_conditional']].dropna()
            lmm_figs = PlotDiagnosticResults.LMM_Diagnostics_Plot(
                lmm_results_df,
                feature_order="original",
                include_delta_r2=True,
                include_status_summary=True,
            )

            for caption, fig in lmm_figs:
                report.log_plot(fig, caption=caption)
                plt.close(fig)
        else:
            lmm_r = pd.DataFrame()

        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # Variance ratio
        report.log_section("variance_ratio", "Variance ratio test (F-test) for variance differences between batches")
        logger.info("Variance ratio test between each unique batch pair")
        mode = ratio_type
        variance_ratios, pair_labels = DiagnosticFunctions.Variance_Ratios(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
            mode = mode
        )


        # Summarise variance ratio results
        data_dict = {}
        Ratios={}
        summary_rows = []

        # variance_ratios is (num_pairs x num_features)
        n_pairs = variance_ratios.shape[0]

        for i in range(n_pairs):
            ratios = np.array(variance_ratios[i], dtype=float)

            # Safe log: treat non-positive values as NaN for log stats
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)

            mean_log = np.nanmean(log_ratios)
            median_log = np.nanmedian(log_ratios)
            iqr_log = np.nanpercentile(log_ratios, [25, 75])
            # Proportion > 0: treat NaNs as False
            prop_higher = np.nanmean(np.where(np.isnan(log_ratios), False, log_ratios > 0))

            # exponentiate summary stats where meaningful
            median_ratio = np.exp(median_log) if not np.isnan(median_log) else np.nan
            mean_ratio = np.exp(mean_log) if not np.isnan(mean_log) else np.nan

            label = pair_labels[i]
            # Try to split label into two parts like "A / B" -> b1, b2. Otherwise keep label as-is.
            if isinstance(label, str) and " / " in label:
                b1, b2 = [s.strip() for s in label.split(" / ", 1)]
            else:
                # fallback: present full label in Batch 1, leave Batch 2 empty
                b1 = label
                b2 = ""

            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })

            # sanitize label for keys (replace spaces and parentheses)
            safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_vs_")

            Ratios[f"VarianceRatio_{safe_label}"] = ratios
            data_dict[f"MedianLogVarianceRatio_{safe_label}"] = median_log
            data_dict[f"MeanLogVarianceRatio_{safe_label}"] = mean_log
            data_dict[f"IQRLowerLogVarianceRatio_{safe_label}"] = iqr_log[0]
            data_dict[f"IQRUpperLogVarianceRatio_{safe_label}"] = iqr_log[1]
            data_dict[f"PropHigherLogVarianceRatio_{safe_label}"] = prop_higher
            data_dict[f"MedianVarianceRatioExp_{safe_label}"] = median_ratio
            data_dict[f"MeanVarianceRatioExp_{safe_label}"] = mean_ratio

            # human-readable report line
            report.text_simple(
                f"Variance ratio {label}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in {b1}"
            )
            report.log_text("Variance ratio test completed")

        # save variance ratios raw:
        if save_data:
            save_test_results(
                Ratios,
                test_name="Variance_Ratios_Raw",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        # Save summary as well
        if save_data:
            save_test_results(
                data_dict,
                test_name="Variance_Ratio_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        summary_df = pd.DataFrame(summary_rows)
        report.text_simple("Variance ratio test summaries (per batch pair):")
        # Plot using your plot function
        PlotDiagnosticResults.variance_ratio_plot(variance_ratios, pair_labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")

        # ---------------------
        # Levenes test for variance differences
        # ---------------------
        report.log_section("levenes_test", "Levene's test for variance differences between batches")
        logger.info("Levene's test for variance differences between batches")
        # Raw Levene test
        levene_results_raw = DiagnosticFunctions.Levenes_Test(data, batch, centre='median')

        # Residualise covariates (if provided) and run Levene on residuals
        levene_results_resid = None
        if covariates_numeric is not None:
            try:
                data_resid = DiagnosticFunctions.RobustOLS(data, covariates_numeric, batch, covariate_names or [], covariate_types)
                levene_results_resid = DiagnosticFunctions.Levenes_Test(data_resid, batch, centre='median')
            except Exception:
                logger.exception("Failed to compute residualised Levene results; continuing with raw results only")

        report.text_simple(
            "Levene's test for variance differences between batches completed "
            "(raw and residualised if covariates supplied)"
        )
        # Use the combined plotting function to show raw vs residual side-by-side
        PlotDiagnosticResults.Levenes_Test_with_residuals(levene_results_raw, levene_results_resid, feature_names=feature_names, rep=report)
        report.log_text("Levene's test plots (raw + residual) added to report")

        # Save summary arrays for downstream saving
        data_dict = {}
        # levene_results_raw is a dict keyed by (b1,b2)
        for comp, res in levene_results_raw.items():
            key = f"LeveneRaw_{comp[0]}_vs_{comp[1]}"
            data_dict[f"{key}_stat"] = np.asarray(res.get('stat') if 'stat' in res else res.get('statistic'))
            for pk in ('pvalue', 'p_val', 'pvalues', 'p'):
                if pk in res:
                    data_dict[f"{key}_pval"] = np.asarray(res[pk])
                    break
        if levene_results_resid is not None:
            for comp, res in levene_results_resid.items():
                key = f"LeveneResid_{comp[0]}_vs_{comp[1]}"
                data_dict[f"{key}_stat"] = np.asarray(res.get('stat') if 'stat' in res else res.get('statistic'))
                for pk in ('pvalue', 'p_val', 'pvalues', 'p'):
                    if pk in res:
                        data_dict[f"{key}_pval"] = np.asarray(res[pk])
                        break

        if save_data:
            save_test_results(
                data_dict,
                test_name="Levenes_Test_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )
        report.text_simple("Levene's test summaries added to report and saved as csv if requested")
        report.text_simple(line_break_in_text)


        # ---------------------
        # PCA and clustering
        # ---------------------
        report.log_section("pca", "PCA & covariate correlations")
        logger.info("Running PCA")
        if covariates is not None:
            if covariate_names is None or len(covariate_names) != covariates.shape[1]:
                logger.warning("Variable names not provided or do not match number of covariates. Using defaults.")
                covariate_names = ["batch"] + [f"covariate_{i+1}" for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            covariate_names = ["batch"]

        variable_names = ["batch"] + covariate_names
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch, covariates=covariates_numeric, variable_names=variable_names
        )

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")
    
        PlotDiagnosticResults.PC_corr_plot(
            score, batch, covariates=covariates_numeric, variable_names=covariate_names,
            PC_correlations=True, rep=report, show=False
        )
        report.log_text("PCA correlation plot added to report")

        # Demean the data before PCA to avoid mean differences dominating first PC (i.e don't force PC'S > 1 to be orthogonal to mean)
        #data_demeaned = data - np.mean(data, axis=0)
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch,N_components=20, covariates=covariates_numeric, variable_names=variable_names
        )

        data_dict = {}
        # save just the scores, not the full PCA object
        # Give each PC a name like PC1, PC2, ... as is more intuitive
        if save_data:
            n_pcs = score.shape[1]
            for pc_idx in range(n_pcs):
                pc_name = f"PC{pc_idx + 1}"    
                data_dict[pc_name] = score[:, pc_idx]
            # Create dummy index to replace feature names as in the dictionary, each PC is length of subjects:
            feature_names = [f"Feature_{idx+1}" for idx in range(n_pcs)]

            save_test_results(data_dict,
                test_name="PCA_Scores_demeaned",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        report.log_section("Eigenvalue_Scree", "PCA Eigenvalues and Covariance Structure")  
        report.text_simple(
            "Using the computed PCA from earlier, visualise the eigenvalues associated "
            "with each principal component (PC) \n"
        )
        report.text_simple(
            "Display as an Eigenvalue Spectrum comparisson across batches and display "
            "Fronenius norm of the covariance matrices across batches \n"
        )
        report.text_simple(line_break_in_text)

        logger.info("Generating PCA Eigenvalue Scree Plot:")
        report.text_simple("The scree plot displays the eigenvalues associated with each principal component (PC). \n")
        report.text_simple("Using this test, we can see if the variance by batch is the same across all PCs \n")
        report.text_simple(
            "In short, the steepness and shape of the batchwise plots can help to differ "
            "batchwise differences in the variance structure across features \n"
        )
        spectra_res = PlotDiagnosticResults.plot_eigen_spectra_and_cumulative(score, batch, rep=report)
        logger.info("PCA Eigenvalue Scree Plot added to report")
        report.text_simple(line_break_in_text)


        # Change Frobenius norm plot to be based on the covariance matrices of the data for each batch, rather than the PCA scores, as this is more interpretable in terms of the original features and their covariance structure (which is what we want to compare across batches)
        logger.info("Generating Frobenius Norm Plot")
        report.text_simple(
            "The Frobenius norm plot displays the pairwise Frobenius norms of the "
            "covariance matrices between batches. \n"
        )
        report.text_simple("Using this test, we can see if the covariance structure by batch is the same across all batches \n")
        report.text_simple(
            "In short, larger Frobenius norms between batches indicate greater "
            "differences in covariance structure across features \n"
        )
        report.text_simple(
            "We calculate the overall covariance matrix for each batch and then compute "
            "the pairwise Frobenius norms between these covariance matrices. \n"
        )
        covres = PlotDiagnosticResults.plot_covariance_frobenius(data, batch, rep=report)

        logger.info("Frobenius norm plot between covariance matrices added to report")
        report.text_simple(line_break_in_text)

        report.log_section("Clustering", "Clustering and visualiation of batch and covariate clusters")
        logger.info("Beginning cluster visulisation of batch and covariate clusters")

        report.text_simple("Using UMAP and PCA to visualise clustering of samples by batch and covariates. \n" \
        "If samples cluster by batch more strongly than covariates, this indicates strong batch effects. \n" \
        "We include both PCA and UMAP visualisations to show both linear and non-linear clustering patterns. \n" \
        "If you see clear clustering by batch in either PCA or UMAP, this suggests strong batch effects that may require correction. \n" \
        "If you see clustering by covariates, this suggests that covariates are driving some of the variance in the data, which may be important to account for in harmonisation. \n" \
        "If you see no clear clustering by either batch or covariates, this suggests minimal batch effects and that the data may be relatively homogeneous across batches. \n")


        if len(data) > 1000:
            logger.info("Large dataset detected, this could make UMAP very slow, especially if not using GPU.")

        PlotDiagnosticResults.clustering_analysis_all(score,
        data,
        batch,
        covariates=covariates,
        rep=report,
        variable_names=covariate_names,
        UMAP_embedding=UMAP_embedding,
        UMAP_tuning=UMAP_tuning)
        logger.info("Clustering visualizations added to report")


        # ---------------------
        # Distribution tests (KS)
        # ---------------------
        report.log_section("ks", "Two-sample Kolmogorov-Smirnov tests")
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_Test(data, batch, feature_names=None, covariates=covariates_numeric, do_fdr=True,residualize_covariates=True,
                                                 covariate_names=covariate_names,covariate_types=covariate_types)
        
        report.log_text("Two-sample Kolmogorov-Smirnov test completed")

        for key, value in ks_results.items():
            if key != "params":
                logger.info(f"Key: {key}, Value type: {type(value)}")

        report.text_simple(
            "- each value is a dict with:\n"
            "    'statistic': np.array of D statistics (length n_features)\n"
            "    'p_value': np.array of p-values (nan where test not run)\n"
            "    'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)\n"
            "    'n_group1': array of sample counts per feature for group1\n"
            "    'n_group2': array of counts for group2\n"
        )
        report.text_simple("The KS test compares the distribution of each feature between batches. \n" \
        "A significant KS test (low p-value) indicates that the distribution of that feature differs between the groups being compared (either batch vs overall, or batch vs batch) being compared. \n" \
        "The D statistic indicates the magnitude of the distribution difference, with higher values indicating greater differences. \n" \
        "By examining the KS test results across features, we can identify which features show the most significant distribution differences between batches, which can inform our choice of harmonisation method and whether to apply it globally or on specific features. \n")

        report.text_simple("Users should look both at the plot by P-value magnitude and in the distributions of D statistics and p-values across features. \n" \
                           "If specific clusters of features show significant KS differences, this may indicate that certain types of features are more affected by batch effects and may benefit from targeted harmonisation approaches. \n" \
                           "If KS differences are widespread across many features, this may indicate more global batch effects that could benefit from global harmonisation approaches. \n" \
                           "If KS differences are minimal, this may indicate minimal batch effects and that harmonisation may not be necessary. \n")
        data_dict = {}
        if save_data:
            for key, value in ks_results.items():
                if key != "params":
                    data_dict[f"KS_Stat_{key}"] = value["statistic"]
                    data_dict[f"KS_PValue_{key}"] = value["p_value"]
                    if value.get("p_value_fdr") is not None:
                        data_dict[f"KS_PValueFDR_{key}"] = value["p_value_fdr"]
                    
            save_test_results(data_dict,
                test_name="KS_Test",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        PlotDiagnosticResults.KS_plot(ks_results, rep=report)
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")

        # Save data dictionary as csv if requested 
        report.log_section("Summary","Summary of Diagnostic Report and Advice")
        report.text_simple("Summary of diagnostic findings and advice for harmonisation:")
        report.text_simple("Based on the diagnostic tests performed, we can summarise the major findings regarding batch differences in the data. \n" \
                           "We can also provide advice on which harmonisation methods may be most appropriate given the observed batch effects. \n")

        batch_sizes = {b: np.sum(batch == b) for b in unique_batches}
        advice_summary = _generate_harmonisation_advice(
            cohens_d_results=cohens_d_results,
            mahalanobis_results=mahalanobis_results,
            lmm_results_df=lmm_results_df,
            variance_summary_df=summary_df,
            covariance_results=covres,
            batch_sizes=batch_sizes,
        )

        for advice_line in advice_summary["advice_lines"]:
            report.text_simple(advice_line)
        
        return data_dict if save_data else None

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report
            report_ctx.__exit__(None, None, None)

def CrossSectionalComparisonReport(
    datasets: dict[str, np.ndarray],
    batch,
    covariates=None,
    covariate_names=None,
    save_data: bool = True,
    save_data_name: str | None = None,
    feature_names=None,
    save_dir: str | os.PathLike | None = None,
    report_name: str | None = None,
    include_raw: bool = True,
    raw_name: str = "Raw",
    scoring_config: dict | None = None,
    rep = None,
    SaveArtifacts: bool = False,
    show: bool = False,
    timestamped_reports: bool = True,
    covariate_types=None,
    ratio_type: str = "rest",
    UMAP_embedding: bool = True,
    UMAP_tuning: str = "auto",
    plot_covariate_embeddings: bool = True,
    allow_many_covariate_embeddings: bool = False,
) -> StatsReporter:
    """
    Create a comparative diagnostic report for multiple harmonisation methods.

    The comparison report runs the same diagnostic suite on each candidate
    dataset, then aggregates the resulting metrics into a method scorecard and
    a short recommendation summary. It is intended for side-by-side evaluation
    of raw and harmonised outputs that share the same sample order, batch
    labels, and optional covariates.

    The report reuses the same per-method diagnostic pipeline as the single
    cross-sectional workflow through the following helpers:

    - ``validate_comparison_datasets``: checks that all methods are compatible.
    - ``_run_single_method_diagnostics``: runs the full diagnostic suite.
    - ``summarise_method_performance``: builds the comparison scorecard.
    - ``generate_comparison_advice``: turns the scorecard into a recommendation.
    - ``_save_comparison_results``: exports per-method CSV artifacts.

    Args:
        datasets: Mapping of method name to data matrix `(n_samples, n_features)`.
        batch: Batch vector of length `n_samples`.
        covariates: Optional covariate matrix `(n_samples, n_covariates)`.
        covariate_names: Optional covariate names.
        save_data: Whether to save per-method per-test CSV outputs.
        save_data_name: Optional prefix to include in per-method saved CSV names.
        feature_names: Optional feature names.
        save_dir: Directory for report and CSV outputs.
        report_name: HTML report name.
        scoring_config: Optional scoring configuration.
        rep: Optional existing `StatsReporter` instance.
        SaveArtifacts: Whether to save report artifacts.
        show: Whether to show plots interactively.
        timestamped_reports: Whether to timestamp the report filename.
        covariate_types: Optional covariate type codes.
        ratio_type: Variance-ratio mode.

    Returns:
        StatsReporter: Report object containing method-wise diagnostics,
        side-by-side plots, scorecard, and advice.
    """

    if not isinstance(datasets, dict) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty dictionary mapping method name to data array.")

    first_dataset = next(iter(datasets.values()))
    first_matrix, inferred_feature_names = _normalize_data_matrix(first_dataset, feature_names=feature_names)
    if feature_names is None and inferred_feature_names is not None:
        feature_names = inferred_feature_names

    normalized_datasets = validate_comparison_datasets(
        datasets=datasets,
        batch=batch,
        covariates=covariates,
        feature_names=feature_names,
    )
    if not include_raw and raw_name in normalized_datasets:
        normalized_datasets = {k: v for k, v in normalized_datasets.items() if k != raw_name}
        if len(normalized_datasets) == 0:
            raise ValueError("No datasets remain after excluding raw dataset. Provide at least one method dataset.")
    batch_arr = _normalize_batch_vector(batch, first_matrix.shape[0])
    covariates = _normalize_covariates_input(covariates, first_matrix.shape[0])

    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "CrossSectionalComparisonReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        rp = report_obj.write_report()
        report_obj.log_text(f"initialised HTML report at: \n"\
                            f"{rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    if created_local_report:
        ctx = report_ctx.__enter__()
        report = ctx
    else:
        report = report_ctx

    try:
        from DiagnoseHarmonisation import PlotComparisonResults

        logger = report.logger
        _configure_report(report)
        # add title and description
        report.make_title("Cross-sectional comparison of harmonisation methods")
        report.set_report_title("Cross-sectional comparison of harmonisation methods")

        report.log_section("Preamble", "Report preamble and overview")
        report.text_simple(
            "This report compares multiple harmonisation methods using the same"\
            " cross-sectional diagnostics and summarises best-performing methods"\
            " by diagnostic and overall score. " )
        report.text_simple(
            "The report is intended for side-by-side evaluation of raw and harmonised outputs but should be interpreted with caution.\n" \
            "As the best performing method by the metrics here may not be the best method for your downstream analysis. \n" \
            ""\
            "For more information on usage, please see the documentation at https://jake-turnbull.github.io/HarmonisationDiagnostics/"
        )

        line_break_in_text = "-" * 150
        report.log_section("comparison_overview", "Multi-method comparison overview")
        report.text_simple(
            "This report compares multiple harmonisation methods using the same"
            " cross-sectional diagnostics and summarises best-performing methods"
            " by diagnostic and overall score."
        )
        covariates_numeric = None
        if covariates is not None:
            covariates_numeric = covariate_to_numeric(np.asarray(covariates).copy())
            if covariates_numeric is not None and covariates_numeric.shape[1] == 0:
                covariates_numeric = None
        if covariates_numeric is None:
            report.text_simple(
                "Warning: no covariates were provided. LMM diagnostics are skipped; all other diagnostics continue."
            )

        shape = next(iter(normalized_datasets.values())).shape
        n_samples, n_features = shape
        batch_counts = dict(zip(*np.unique(batch_arr, return_counts=True)))
        covariate_labels = (
            [str(name) for name in covariate_names]
            if covariate_names is not None
            else ([f"covariate_{i+1}" for i in range(covariates_numeric.shape[1])] if covariates is not None else [])
        )
        dataset_overview_lines = [
            f"Validated methods: {', '.join(normalized_datasets.keys())}",
            f"Total samples: {n_samples}",
            f"Features: {n_features}",
            f"Samples in each batch: {batch_counts}",
            f"Missing data per method: {', '.join(f'{name}={int(np.isnan(data).sum())}' for name, data in normalized_datasets.items())}",
            f"Covariates: {', '.join(covariate_labels) if covariate_labels else 'None'}",
        ]
        report.text_simple("Comparison dataset characteristics:")
        report.text_simple("\n".join(dataset_overview_lines))
        report.text_simple(line_break_in_text)

        method_results: dict[str, CrossSectionalDiagnosticResult] = {}
        saved_paths: dict[str, dict[str, str]] = {}
        report_date = datetime.now().date().isoformat()

        for method_name, data in normalized_datasets.items():
            report.log_section(_sanitize_name(method_name), f"Diagnostics for {method_name}")
            logger.info(f"Running diagnostics for method: {method_name}")

            method_result = _run_single_method_diagnostics(
                method_name=method_name,
                data=data,
                batch=batch_arr,
                covariates_numeric=covariates_numeric,
                covariate_names=covariate_names,
                covariate_types=covariate_types,
                feature_names=feature_names,
                ratio_type=ratio_type,
                compute_umap=UMAP_embedding,
            )
            method_results[method_name] = method_result

            if method_result.errors:
                report.text_simple(f"Warnings for {method_name}:")
                for err in method_result.errors:
                    report.text_simple(f"- {err}")

            metrics = method_result.summary_metrics or {}
            report.text_simple(f"Summary metrics for {method_name}:")
            metric_lines = [f"- {k}: {v}" for k, v in metrics.items()]
            report.text_simple("\n".join(metric_lines))
            if isinstance(method_result.lmm_summary, dict) and method_result.lmm_summary.get("status") == "skipped_missing_covariates":
                report.text_simple(f"LMM status for {method_name}: skipped (missing covariates).")

            if save_data:
                saved_paths[method_name] = _save_comparison_results(
                    result=method_result,
                    batch=batch_arr,
                    save_dir=save_dir,
                    report_date=report_date,
                    report_name=report_name,
                    feature_names=feature_names,
                    save_data_name=save_data_name,
                )

            report.text_simple(line_break_in_text)

        summary_df = summarise_method_performance(method_results, scoring_config=scoring_config)
        advice = generate_comparison_advice(summary_df)

        report.log_section("scorecard", "Method scorecard")
        if summary_df.empty:
            report.text_simple("No methods could be scored.")
        else:
            scorecard_sections = {
                "additive": ["method", "median_abs_cohens_d", "prop_large_abs_cohens_d", "max_mahalanobis", "additive_score"],
                "multiplicative": ["method", "median_abs_log_variance_ratio", "prop_significant_levene", "multiplicative_score"],
                "linear modelling": ["method", "median_icc", "prop_high_icc", "median_delta_r2", "linear_modeling_score"],
                "distributional": ["method", "prop_significant_ks", "max_frobenius_normalized", "distributional_score"],
                "principal component analysis": ["method", "max_abs_batch_pc_correlation", "principal_component_analysis_score"],
            }

            report.text_simple("Method scorecard by diagnostic category (higher is better within each block):")
            for section_name, columns in scorecard_sections.items():
                available_cols = [col for col in columns if col in summary_df.columns]
                if not available_cols:
                    continue
                report.text_simple(section_name.title())
                report.text_simple(summary_df[available_cols].to_string(index=False))
                report.text_simple(line_break_in_text)

            report.text_simple("Interpretation guide:")
            report.text_simple(
                "Additive metrics reward smaller absolute mean-shift and distance effects; multiplicative metrics reward stable variance structure; \n"\
                "linear-modelling metrics reward better covariate preservation; distributional metrics reward closer overall distributions; \n"\
                "and PCA metrics reward weaker batch-aligned structure in the principal components."
            )

        report.log_section("comparison_advice", "Best method summary")
        report.text_simple(f"Best overall method: {advice.get('best_overall')}")
        for diag_name, method_name in advice.get("best_by_metric", {}).items():
            report.text_simple(f"Best {diag_name}: {method_name}")
        report.text_simple(advice.get("summary_text", ""))

        def _log_figures(fig_tuples):
            for caption, fig in fig_tuples:
                try:
                    report.log_plot(fig, caption=caption)
                finally:
                    plt.close(fig)

        _log_figures(PlotComparisonResults.plot_compare_zscore_distributions(method_results, batch_arr))
        _log_figures(PlotComparisonResults.plot_compare_cohens_d(method_results))
        _log_figures(PlotComparisonResults.plot_compare_variance_ratios(method_results))
        _log_figures(PlotComparisonResults.plot_compare_lmm_icc(method_results))
        _log_figures(PlotComparisonResults.plot_compare_mahalanobis(method_results))
        _log_figures(PlotComparisonResults.plot_compare_ks(method_results))
        _log_figures(PlotComparisonResults.plot_compare_covariance(method_results))
        _log_figures(PlotComparisonResults.plot_compare_batch_scree(method_results, batch_arr))
        _log_figures(
            PlotComparisonResults.plot_compare_pca_embeddings(
                method_results,
                batch_arr,
                covariates=covariates_numeric,
                plot_covariate_embeddings=plot_covariate_embeddings,
                allow_many_covariates=allow_many_covariate_embeddings,
            )
        )
        if UMAP_embedding:
            _log_figures(
                PlotComparisonResults.Plot_compare_UMAP_embeddings(
                    method_results,
                    batch_arr,
                    covariates=covariates_numeric,
                    plot_covariate_embeddings=plot_covariate_embeddings,
                    allow_many_covariates=allow_many_covariate_embeddings,
                )
            )
        _log_figures(PlotComparisonResults.plot_method_scorecard(summary_df))

        if save_data and summary_df is not None and not summary_df.empty:
            from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results

            scorecard_name = _sanitize_name(save_data_name) + "_Scorecard" if save_data_name else "Comparison_Scorecard"
            scorecard_path = save_test_results(
                summary_df,
                test_name=scorecard_name,
                save_root=save_dir,
                feature_names=list(summary_df.columns),
                report_date=report_date,
                report_name=report_name,
            )
            report.text_simple(f"Saved scorecard CSV: {scorecard_path}")

        report.comparison_results = method_results
        report.comparison_scorecard = summary_df
        report.comparison_advice = advice
        report.comparison_saved_paths = saved_paths
        return report

    finally:    
        if created_local_report:
            report_ctx.__exit__(None, None, None)

        


# Longitudinal testing:
from typing import Optional, Union
def LongitudinalReport(data, batch,
                          subject_ids,
                          timepoints,
                          covariates=None,
                          covariate_names=None,
                          features = None,
                          save_data: bool = False,
                          save_data_name: Optional[str]= None,
                          save_dir: Optional[Union[str,os.PathLike]] = None,
                          report_name: Optional[str] = None,
                          SaveArtifacts: bool = False,
                          rep= None,
                          show: bool = False,
                          timestamped_reports: bool = True):
    """
    Create a diagnostic report for dataset differences across batches in longitudinal data.

    Args: 
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        subject_ids (list or np.ndarray): Subject IDs for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
    
    Outputs:
        Generates an HTML report with diagnostic plots and statistics for longitudinal data.
        If `save_data` is True, also returns a dictionary and csv with input data and results.
        If SaveArtifacts is True, saves intermediate plots to `save_dir`.
    Note:
        This function is designed for repeated data where we do not expect to see a longitudinal trent over time.
        If need arises, we will revise this to include an additional function where we would expect to see a longitudinal trend and want to test for that explicitly.
    
    """
    from pprint import pformat
    from DiagnoseHarmonisation import DiagnosticFunctionsLong

    # Check inputs and revert to defaults as needed 

    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "LongitudinalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"initialised HTML report at: \n"
                            f"{rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    def _plot_figsize(kind: str, n_items: int | None = None) -> tuple[float, float]:
        """Consistent figure sizing for the report runner."""
        n = int(n_items or 0)
        if kind == "raw":
            return (7.4, 5.2)
        if kind == "summary_hbar":
            return (7.2, max(5.2, 0.24 * n + 2.7))
        if kind == "summary_box":
            return (7.2, max(5.8, 0.18 * n + 2.8))
        if kind == "effects":
            return (7.8, 6.6)
        if kind == "biological":
            return (7.8, 6.6)
        return (6.5, 6.5)

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    if covariates is not None:
        covariate_names = list(covariates.keys())
    else:
        covariate_names = []

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()  # type: ignore
        report = ctx
    else:
        report = report_ctx
        # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        report.make_title("Longitudinal Data Diagnostic Report")
        report.set_report_title("Longitudinal Data Diagnostic Report")
        report.log_section("Preamble", "Report preamble and overview")
        report.text_simple(
            "This report evaluates longitudinal data for batch effects and covariate preservation. \n" \
            "The report is intended for repeated measures data where we do not expect to see a longitudinal trend over time. \n" \
            "If you expect to see a longitudinal trend, please use the appropriate longitudinal analysis function. \n" \
            "For more information on usage, please see the documentation at https://jake-turnbull.github.io/HarmonisationDiagnostics/"
        )

        line_break_in_text = "-" * 150
        unique_subjects = set(subject_ids)
        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of measures: {data.shape[0]}\n"
            f"Unique subjects: {len(set(subject_ids))}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        report.text_simple(line_break_in_text)

        # Ensure batch is numeric array where needed
        #logger.info("Checking data format")
        #if isinstance(batch, (list, np.ndarray)):
        #    batch = np.array(batch)
        #    if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
        #        logger.info(f"Original batch categories: {list(set(batch))}")
        #        logger.info("Creating numeric codes for batch categories")
        #        batch_numeric, unique = pd.factorize(batch)
        #        logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
        #        # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        #else:
        #    raise ValueError("Batch must be a list or numpy array")
        
        
        # Check that covariates are an array if provided (.shape[1] throwing error with a list), convert to array if needed
        if covariates is not None:
            if isinstance(covariates, list):
                covariates = np.array(covariates)
            elif isinstance(covariates, dict):
                pass
            elif not isinstance(covariates, np.ndarray):
                raise ValueError(f"Covariates must be a numpy array or list if provided, covariates type: {type(covariates)}")

                    
        # Check if there is only one covariate and convert to 2D array if that is the case (avoid shape issue in next call):
        try:
            if covariates is not None and covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
        except AttributeError:
            pass

        # Prepare save-data dict if requested
        if save_data:
            data_dict = {}
            data_dict["batch"] = batch
            if covariates is not None:
                for i in range(covariates.shape[1]):
                    if covariate_names is not None and i < len(covariate_names):
                        cov_name = covariate_names[i]
                    else:
                        cov_name = f"covariate_{i+1}"
                    data_dict[cov_name] = covariates[:, i]
            if save_data_name is None:
                save_data_name = "DiagnosticReport_InputData.csv"
        else:
            data_dict = None
        # Check batch, subject_ids, and data dimensions
        #if not (len(batch) == len(subject_ids) == data.shape[0]):
        #    raise ValueError("Length of batch and subject_ids must match number of samples in data")
        #if len(covariates) is not None and len(covariates) != data.shape[0]:
        #    raise ValueError("Number of rows in covariates must match number of samples in data")
        #if len(covariate_names) is not None and len(covariate_names) != covariates.shape[1]:
        #    raise ValueError("Length of covariate_names must match number of columns in covariates")
        

        report.log_section("Introduction", "Longitudinal Data Diagnostic Report Introduction")
        report.text_simple(
        """
        Longitudinal data contain repeated measurements collected from the same individuals over time. 
        
        Such datasets require evaluation of measurement stability, batch-related variability, preservation of biological signal, and retention of between-subject differences.

        This report provides a set of complementary diagnostics designed to evaluate these properties before and after harmonisation.

        ────────────────────────────────────────────────────────

        SUBJECT-LEVEL VARIABILITY

        • Subject order consistency
        Preservation of subject ranking across visits

        • Within-subject variability
        Longitudinal stability within individual subjects

        ────────────────────────────────────────────────────────

        BATCH VARIABILITY

        • Additive batch effects
        Batch-related mean shifts

        • Pairwise batch differences
        Post-hoc comparisons between batches

        • Multiplicative batch effects
        Batch-related variance differences

        • Multivariate batch differences
        Mahalanobis distance from a reference distribution

        ────────────────────────────────────────────────────────

        BETWEEN-SUBJECT VARIABILITY

        • Intra-Class Correlation (ICC)
        Preservation of between-subject differences

        ────────────────────────────────────────────────────────

        BIOLOGICAL VARIABILITY

        • Covariate significance
        • Effect sizes (β coefficients)
        • 95% confidence intervals

        ────────────────────────────────────────────────────────

        Together, these diagnostics help determine whether harmonisation reduces unwanted batch effects while preserving meaningful biological and subject-specific variation.
        """
        )

        # --------------------------------------------------------------
        # Raw IDP distributions across sites (first plot in report)
        # --------------------------------------------------------------
        report.log_section(
            "Raw_IDP_across_sites",
            "Raw imaging-derived phenotypes across sites"
        )

        site_counts = pd.Series(batch).astype(str).value_counts()
        site_subject_counts = (
            pd.DataFrame({
                "batch": pd.Series(batch).astype(str),
                "subject": pd.Series(subject_ids).astype(str),
            })
            .drop_duplicates()
            .groupby("batch")["subject"]
            .nunique()
        )

        report.text_simple(
            f"""
            📋 OVERVIEW
            ────────────────────────────────────────────────────────

            This section shows the raw distributions of each IDP across
            acquisition sites before harmonisation.

            Site sample sizes:
            • sites: {site_counts.size}
            • datapoints per site: min={site_counts.min()}, median={int(site_counts.median())}, max={site_counts.max()}
            • unique subjects per site: min={site_subject_counts.min()}, median={int(site_subject_counts.median())}, max={site_subject_counts.max()}

            The plot adapts automatically to the dataset:
            • vertical boxplots when the number of sites is small
            • horizontal boxplots when there are many sites
            • only the top 10 most site-sensitive IDPs when there are
              many features
            • a PCA summary panel when the full feature set is too large
              to display directly

            📈 INTERPRETATION
            ────────────────────────────────────────────────────────

            Differences in medians, spread, or outliers across sites
            suggest site-related effects in the raw data.
            """
        )

        raw_df = pd.DataFrame(data, columns=features)
        raw_df["batch"] = batch
        raw_df["subject"] = subject_ids

        PlotDiagnosticResults.plot_RawIDPBoxplotsAcrossSites(
            raw_df,
            batch_col="batch",
            subject_col="subject",
            idp_cols=list(features),
            figsize_per_panel=_plot_figsize("raw"),
            rep=report,
            show=False,
        )

        report.log_text("Raw IDP boxplots across sites added to report")
        report.text_simple("────────────────────────────────────────────")

        report.log_section(
            "subject_order_consistency",
            "Subject-level variability: Subject order consistency analysis"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates whether subjects preserve their
        relative ranking across longitudinal visits.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        For each pair of visits, Spearman rank correlation (ρ)
        is computed using subjects measured at both timepoints.

        Statistical significance is assessed using permutation
        testing, where subject labels are randomly shuffled to
        generate a null distribution of correlation values.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Higher correlation values indicate stronger preservation
        of subject ordering across visits and greater longitudinal
        consistency.

        A significant permutation test (p < 0.05) suggests that
        the observed consistency is unlikely to arise by chance.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        This metric is most informative in test–retest and
        travelling-subject designs.

        In longitudinal disease studies, true biological
        progression may also influence subject ordering over
        time and should therefore be considered when
        interpreting results.
        """
        )
        
        # Subject-level: Subject order consistency
        subjorder = DiagnosticFunctionsLong.SubjectOrder_long(idp_matrix=data,
                                                          subjects=subject_ids,
                                                          timepoints=timepoints,
                                                          idp_names=features,
                                                          nPerm=100)
        print("\nSUBJECT ORDER CONSISTENCY: RANK CORRELATIONS WITH PERMUTATION TESTS")
        print(subjorder)
        report.text_simple(
            "────────────────────────────────────────────"
            )
        PlotDiagnosticResults.plot_SubjectOrder(subjorder,                 
                              ncols=2,
                              figsize_per_plot=(8,6),
                              limit_idps=None,
                              sample_method='random',
                              random_state=42,
                              rep=report) 
        report.log_text("Subject order consistency plots added to report")

        # Subject-level: Within Subject Consistency 
        report.log_section(
            "Within_subject_variability",
            "Subject-level variability: Within-subject variability analysis"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis quantifies how much individual subjects
        vary across repeated measurements.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        For datasets containing more than two visits,
        within-subject variability is summarised using the
        Coefficient of Variation (CV).

            CV = standard deviation / mean

        For datasets containing exactly two visits,
        variability is summarised using the Relative Percent
        Difference (RPD).

        All variability measures are computed at the subject
        level and subsequently summarised across subjects and
        features.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Lower variability values indicate greater longitudinal
        stability within subjects.

        Higher variability values indicate greater change across
        repeated measurements.

        In general, reduced variability following harmonisation
        suggests improved measurement consistency.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        This metric should be interpreted alongside measures of
        between-subject variability to ensure that reductions are
        not caused by over-smoothing or loss of meaningful
        biological signal.

        In short-term test–retest studies, variability primarily
        reflects measurement noise.

        In longitudinal studies, variability may reflect both
        true biological change and technical variation and should
        therefore be interpreted with appropriate caution.
        """
        )

        wsv = DiagnosticFunctionsLong.WithinSubjVar_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            idp_names=features,
                          )
        print("\nWITHIN SUBJECT VARIABILITY: BETWEEN TIMEPOINTS")
        print(wsv)

        report.text_simple(
            "────────────────────────────────────────────"
            )
        
        PlotDiagnosticResults.plot_WithinSubjVar(
            wsv,
            subject_col='subject',
            rep=report,
            debug=False
            )
        report.log_text("Within subject variability plots added to report")

        # Batch-level: Additive batch effects
        report.log_section(
            "Additive_batch_effects_mixed_models",
            "Batch variability (Univariate): Additive batch effect analysis (mean shift)"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates whether feature means differ
        systematically across batches after accounting for
        repeated measurements within subjects.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        For each feature, a linear mixed-effects model is fitted
        with batch included as a fixed effect and subject
        included as a random effect.

        A Kenward–Roger F-test is then used to compare the full
        model against a nested model without the batch term.

        The resulting p-value quantifies the contribution of
        batch membership to variation in feature values.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Non-significant p-values suggest no evidence of residual
        additive batch effects (mean shifts between batches).

        Significant p-values indicate that batch membership
        explains additional variation in feature means and that
        batch-related differences remain present.

        When comparing harmonisation strategies, a reduction in
        the number of significant features suggests improved
        removal of additive batch effects.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        This analysis evaluates differences in feature means
        only.

        Variance differences and multivariate batch structure
        are assessed separately in subsequent analyses.
        """
        )

        addeff,model_defs_add = DiagnosticFunctionsLong.AdditiveEffect_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batch_name=batch,
            idp_names=features,
            covariates=covariates,
            #fix_eff=["age", "sex"],   # fixed effects
            #ran_eff=["subjects"],            # random intercepts
            do_zscore=True,                  # z-score predictors AND response per feature
            reml=False,
            verbose=True)
        print("\nRESULTS: ADDITIVE EFFECTS")
        print(addeff)
        #report.log_text(pformat(model_defs_add, width=60, sort_dicts=False))
        report.text_simple(
            "────────────────────────────────────────────"
            )
        
        PlotDiagnosticResults.plot_AddMultEffects(addeff,
                                     feature_col='Feature',
                                     p_col='p-value',
                                     labels=['Additive batch effect'],
                                     p_thr=0.05,
                                     annot_fmt="{:.3f}",
                                     value_scale='p',
                                     figsize=_plot_figsize("summary_hbar", len(addeff)),
                                     rep=report)
        report.log_text("Additive batch effect plot added to report")
        
        # Batch-level: Pairwise batch comparison
        report.text_simple(
        """
        PAIRWISE BATCH COMPARISONS
        ────────────────────────────────────────────────────────

        Following the mixed-effects analysis, post-hoc pairwise
        comparisons are performed between batches to identify
        which batch pairs differ significantly in feature means.

        For each feature, the number of significant batch pairs
        is reported following multiple-comparison correction.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        A larger number of significant batch pairs indicates
        stronger residual batch-related mean differences.

        When comparing harmonisation strategies, a reduction in
        the number of significant batch pairs suggests improved
        removal of additive batch effects and greater agreement
        between batches.
        """
        )


        mf,model_defs = DiagnosticFunctionsLong.MixedEffects_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batches=batch,
            idp_names=features,
            covariates=covariates,  # optional
            p_corr=1
            #fix_eff=["age","sex"],   # batch is included automatically
            #ran_eff=["subjects"],
            #force_categorical=["sex"],
            #force_numeric=["age"],
            #zscore_var=["age"]
            ) 
        print("\nMIXED EFFECTS OUTPUTS:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(mf) 
        #report.log_text(pformat(model_defs, width=60, sort_dicts=False))

        n_batches = len(np.unique(batch))
        total_pairs = n_batches * (n_batches - 1) // 2
        report.log_text(f"Total number of pairs {total_pairs} for {len(np.unique(batch))} batches")

        if len(features) < 30:
            report.text_simple(
            "────────────────────────────────────────────"
            )
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['n_is_batchSig'],
                        plot_type='bar',
                        seed=123,
                        figsize=_plot_figsize("summary_hbar", len(features)), rep=report)
        else:
            report.text_simple(
            "────────────────────────────────────────────"
            )
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                      idp_col='IDP',
                      metrics=['n_is_batchSig'],
                      plot_type='box',
                      limit_idps=2,
                      seed=123,
                      figsize=_plot_figsize("summary_box", len(features)), rep=report)
        report.log_text("Pairwise batch variability plots added to report")

        # Multiplicative batch effects
        report.log_section(
            "Multiplicative_batch_effects_Fligner_Killeen",
            "Batch variability (Univariate): Multiplicative batch effect analysis (variance scaling)"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates whether feature variability
        differs systematically across batches.

        Differences in variance are consistent with
        multiplicative (scaling) batch effects.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        After accounting for covariate effects, feature
        variances are compared across batches using the
        Fligner–Killeen test.

        The Fligner–Killeen test is a robust non-parametric
        test for homogeneity of variances and is less sensitive
        to departures from normality than many classical
        variance tests.

        For each feature, a p-value is computed to assess
        whether variance differs significantly between batches.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Non-significant p-values suggest no evidence of
        multiplicative batch effects.

        Significant p-values indicate residual variance
        differences across batches and suggest that scaling-
        related batch effects remain present.

        When comparing harmonisation strategies, a reduction
        in the number of significant features suggests improved
        removal of multiplicative batch effects.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        This analysis evaluates variance differences only.

        Mean shifts and multivariate batch structure are
        assessed separately in other sections of the report.
        """
        )

        muleff,model_defs_mul = DiagnosticFunctionsLong.MultiplicativeEffect_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batch_name=batch,
            idp_names=features,
            covariates=covariates,
            #fix_eff=["age", "sex"],   # fixed effects
            #ran_eff=["subjects"],            # random intercepts
            do_zscore=True,                  # z-score predictors AND response per feature
            verbose=True)
        print("\nRESULTS: MULTIPLICATIVE EFFECTS")
        print(muleff)
        #report.log_text(pformat(model_defs_mul, width=60, sort_dicts=False))
        report.text_simple(
            "────────────────────────────────────────────"
            )
        PlotDiagnosticResults.plot_AddMultEffects(muleff,
                                     feature_col='Feature',
                                     p_col='p-value',
                                     labels=['Multiplicative batch effect'],
                                     p_thr=0.05,
                                     annot_fmt="{:.3f}",
                                     value_scale='p',
                                     figsize=_plot_figsize("summary_hbar", len(muleff)),
                                     rep=report)
        report.log_text("Multiplicative batch effect plots added to report")
       
       # Multivariate site differences using Mahalanobis distances
        report.log_section(
            "Multivariate_batch_difference_reference",
            "Batch variability (Multivariate): Difference from reference distribution"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates how different each batch is from
        a reference distribution when all features are considered
        simultaneously.

        Unlike univariate analyses, this approach captures
        relationships between features and assesses the overall
        multivariate structure of the data.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        For each batch, Mahalanobis distance is computed relative
        to a reference distribution.

        Mahalanobis distance quantifies how far a batch lies from
        the reference while accounting for correlations between
        features.

        Distances are calculated using the full covariance
        structure of the dataset.

        Both batch-specific distances and the overall average
        distance are reported.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Lower Mahalanobis distances indicate that a batch more
        closely resembles the reference distribution.

        Higher distances indicate greater multivariate
        differences and suggest stronger residual batch effects.

        When comparing harmonisation strategies, reductions in
        distance indicate improved alignment of batch
        distributions in multivariate feature space.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        Distance values depend on the dimensionality and
        covariance structure of the dataset.

        Distances should therefore be compared within the same
        dataset and analytical framework rather than across
        different studies.
        """
        )
        md = DiagnosticFunctionsLong.MultiVariateBatchDifference_long(
           idp_matrix=data,
           batch=batch,
           idp_names=features)
        print("\nMULTIVARIATE PAIRWISE SITE DIFFERENCES:")
        print(md)
        report.text_simple(
            "────────────────────────────────────────────"
            )
        PlotDiagnosticResults.plot_MultivariateBatchDifference(md, figsize=_plot_figsize("summary_hbar", len(md)), rep=report) 
        report.log_text("Multivariate batch variability plots added to report")


        report.log_section(
            "Between_subject_variability_mixed_models",
            "Between-subject variability (Univariate): Cross-subject variability analysis"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates how much of the total variation
        in each feature can be attributed to differences between
        subjects.

        Preservation of between-subject variability is important
        because biologically meaningful differences should remain
        detectable after harmonisation.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        For each feature, a linear mixed-effects model is fitted
        with subject included as a random effect.

        The model decomposes total variance into:

        • Between-subject variance
        • Within-subject variance

        From these variance components, the Intra-Class
        Correlation (ICC) is computed.

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        ICC represents the proportion of total variance explained
        by differences between subjects.

        Higher ICC values indicate stronger subject-specific
        signal relative to residual variability.

        When evaluating harmonisation strategies, preservation
        or improvement of ICC suggests that meaningful
        between-subject variation has been retained while
        unwanted variability has been reduced.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        ICC values depend on study design, feature properties,
        and model specification.

        Comparisons should therefore be made within the same
        dataset and analytical framework.
        """
        )

        #report.log_text(pformat(model_defs, width=60, sort_dicts=False))
        if len(features) < 30:
            report.text_simple(
            "────────────────────────────────────────────"
            )
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['ICC'],
                        plot_type='bar',
                        seed=123,
                        figsize=_plot_figsize("summary_hbar", len(features)), rep=report)
        else:
            report.text_simple(
            "────────────────────────────────────────────"
            )
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                      idp_col='IDP',
                      metrics=['ICC'],
                      plot_type='box',
                      limit_idps=2,
                      seed=123,
                      figsize=_plot_figsize("summary_box", len(features)))
        report.log_text("ICC plot added to report")  
        
        # Biological variability
        report.log_section(
            "Biological_variability_mixed_models",
            "Biological variability analysis"
        )

        report.text_simple(
        """
        📋 OVERVIEW
        ────────────────────────────────────────────────────────

        This analysis evaluates whether biologically meaningful
        covariates remain associated with imaging-derived
        phenotypes after accounting for repeated measurements
        and batch-related effects.

        Preservation of biological associations is a key
        requirement of successful harmonisation.

        ⚙️ METHOD
        ────────────────────────────────────────────────────────

        Linear mixed-effects models are fitted for each feature
        with biological covariates included as fixed effects.

        For each covariate–feature combination, the following
        statistics are reported:

        • Statistical significance (p-value)
        • Effect size (β coefficient)
        • 95% confidence interval

        📈 INTERPRETATION
        ────────────────────────────────────────────────────────

        Significant associations indicate that variation in the
        feature is related to the biological covariate.

        Preservation of significant effects and stable effect
        sizes following harmonisation suggests that meaningful
        biological signal has been retained.

        Substantial attenuation or loss of biological
        associations may indicate over-correction or removal of
        true biological variability.

        Strong residual batch effects may also obscure biological
        relationships and reduce interpretability.

        ⚠️ LIMITATIONS
        ────────────────────────────────────────────────────────

        Biological effects should not be evaluated using
        p-values alone.

        Effect size magnitude, effect direction, and confidence
        intervals should all be considered when assessing the
        impact of harmonisation.
        """
        )
        inferred_fix = list(covariates.keys())
        report.text_simple(
            "────────────────────────────────────────────"
            )
        PlotDiagnosticResults.plot_MixedEffectsPart2(mf,
                      idp_col='IDP',
                      fix_eff=inferred_fix,
                      p_thr=0.05,
                      figsize=_plot_figsize("biological", len(features)),
                      rep=report)
        report.log_text("Biological variability plots added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")
        
        report.log_section(
            "REFERENCES",
            "REFERENCES"
        )

        report.text_simple(
        """
        The following references provide methodological
        background for selected diagnostics included in this
        report.

        ────────────────────────────────────────────────────────

        Subject Order Consistency

        Warrington et al. 
        Imaging Neuroscience (2023)

        https://doi.org/10.1162/imag_a_00042

        ────────────────────────────────────────────────────────

        Multivariate Batch Differences

        Beer et al.
        NeuroImage (2022)

        https://doi.org/10.1016/j.neuroimage.2022.119768

        ────────────────────────────────────────────────────────

        Mahalanobis Distance metric 

        Parekh et al. 2022
        NeuroImage (2022)

        https://doi.org/10.1016/j.neuroimage.2022.119768 

        ────────────────────────────────────────────────────────


        Application of these metrics to Real Longitudinal Data and 
        comparison of statistical and image-based harmonisation 
        methods.

        Bhalerao et al. 2026

        https://doi.org/10.64898/2026.04.21.26351106 

        ────────────────────────────────────────────────────────

        These references are provided to support interpretation
        of the diagnostic metrics and their application to
        harmonisation studies.
        """
        )

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore
