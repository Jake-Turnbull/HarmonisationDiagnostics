from __future__ import annotations
from pathlib import Path
from typing import Any
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

from DiagnoseHarmonisation import PlotDiagnosticResults


def _method_items(results: dict[str, Any]) -> list[tuple[str, Any]]:
    return list(results.items())


def _safe_series(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _grid_shape(n_items: int, max_cols: int = 3) -> tuple[int, int]:
    cols = min(max_cols, max(1, n_items))
    rows = int(np.ceil(n_items / cols))
    return rows, cols


def _make_method_grid(n_items: int, square_size: float = 4.8, max_cols: int = 3):
    rows, cols = _grid_shape(n_items, max_cols=max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * square_size, rows * square_size), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes:
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
    return fig, flat_axes, rows, cols


def _title(text: str, width: int = 34) -> str:
    return textwrap.fill(str(text), width=width)


def _batch_palette(batch: np.ndarray, cmap_name: str = "tab10") -> tuple[np.ndarray, Any]:
    unique_batches = np.unique(batch)
    cmap = plt.get_cmap(cmap_name, len(unique_batches))
    return unique_batches, cmap


def _hide_unused_axes(axes, start_idx: int):
    for ax in axes[start_idx:]:
        ax.axis("off")


def _safe_covariate_matrix(covariates):
    if covariates is None:
        return None
    cov = np.asarray(covariates)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    if cov.ndim != 2:
        return None
    return cov


def _feature_labels(n_features: int):
    labels = [f"f{i+1}" for i in range(n_features)]
    if n_features > 20:
        return ["" for _ in labels], 0
    return labels, 45


def _add_right_colorbar(fig, ax, mappable, label: str | None = None,shrink=0.5,labelsize=3):
    cbar = fig.colorbar(mappable, ax=ax, location="right", fraction=0.04, pad=0.01, shrink=shrink)

    cbar.ax.tick_params(labelsize=labelsize)
    return cbar


def _weighted_covariate_pc_correlation(pca_results, max_pcs: int = 3) -> float:
    if not isinstance(pca_results, dict):
        return np.nan
    corr_dict = pca_results.get("pc_correlations", {}) or {}
    explained = np.asarray(pca_results.get("explained_variance", []), dtype=float)
    if explained.size == 0:
        return np.nan
    k = min(max_pcs, explained.shape[0])
    weights = explained[:k]
    weight_sum = np.nansum(weights)
    if k == 0 or not np.isfinite(weight_sum) or weight_sum <= 0:
        return np.nan
    weights = weights / weight_sum

    values = []
    for name, payload in corr_dict.items():
        if str(name).lower() == "batch":
            continue
        corr = np.asarray((payload or {}).get("correlation", []), dtype=float)
        if corr.shape[0] < k:
            continue
        values.append(float(np.nansum(np.abs(corr[:k]) * weights)))
    return float(np.nanmean(values)) if values else np.nan


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.nan
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _numeric_embedding_values(values, is_categorical: bool) -> np.ndarray:
    series = pd.Series(values)
    if is_categorical or not pd.api.types.is_numeric_dtype(series):
        cats = series.astype("category")
        codes = cats.cat.codes.to_numpy(dtype=float)
        codes[codes < 0] = np.nan
        return codes
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _embedding_corr_text(x_label: str, y_label: str, embedding: np.ndarray, values: np.ndarray) -> str:
    r_x = _safe_pearson_corr(values, embedding[:, 0])
    r_y = _safe_pearson_corr(values, embedding[:, 1])
    return f"r({x_label})={r_x:.2f}\nr({y_label})={r_y:.2f}"


def _add_embedding_corr_label(ax, x_label: str, y_label: str, embedding: np.ndarray, values: np.ndarray):
    ax.text(
        0.03,
        0.97,
        _embedding_corr_text(x_label, y_label, embedding, values),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
    )


def _draw_effect_size_guides(ax, values: np.ndarray):
    max_abs = float(np.nanmax(np.abs(values))) if values.size else np.nan
    if not np.isfinite(max_abs):
        return

    thresholds = [0.2, 0.5, 0.8]
    if max_abs <= thresholds[0]:
        ax.axhline(max_abs, color="purple", linestyle="--", linewidth=1)
        ax.text(
            0.98,
            max_abs,
            f"max |d|={max_abs:.2f}",
            color="purple",
            fontsize=6,
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )
        return

    active_thresholds = [thr for thr in thresholds if thr <= max_abs]
    if not active_thresholds:
        active_thresholds = [max_abs]

    for idx, thr in enumerate(active_thresholds):
        color = ["green", "orange", "red"][min(idx, 2)]
        ax.axhline(thr, color=color, linestyle="--", linewidth=1)
        ax.text(
            0.98,
            thr,
            f"{thr:.1f}",
            color=color,
            fontsize=6,
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )

    if max_abs < thresholds[-1]:
        ax.axhline(max_abs, color="purple", linestyle="--", linewidth=1)
        ax.text(
            0.98,
            max_abs,
            f"max |d|={max_abs:.2f}",
            color="purple",
            fontsize=6,
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
        )


def plot_compare_zscore_distributions(
    results,
    batch,
    use_residual: bool = False,
    probability_distribution: bool = True,
    base_cmap: str = "tab10",
    fontsize: float = 10,
):
    figs = []
    items = _method_items(results)
    if len(items) == 0:
        return figs

    batch_arr = np.asarray(batch)
    unique_batches, cmap = _batch_palette(batch_arr, base_cmap)

    fig, axes, _, _ = _make_method_grid(len(items))
    shared_bins = np.linspace(-5, 5, 60)
    bin_width = float(shared_bins[1] - shared_bins[0])
    n_valid = 0
    z_key = "zscore_residual" if use_residual else "zscore_raw"
    figure_caption = "Comparison: residual z-score distributions" if use_residual else "Comparison: raw z-score distributions"
    density_mode = bool(probability_distribution)

    for ax, (method, res) in zip(axes, items):
        z = getattr(res, z_key, None)
        if z is None:
            missing_label = "residual" if use_residual else "raw"
            ax.set_title(_title(f"{method} (no {missing_label} z-score)"))
            ax.axis("off")
            continue

        z = np.asarray(z, dtype=float)
        if z.ndim != 2 or z.shape[0] != batch_arr.shape[0]:
            ax.set_title(_title(f"{method} (invalid shape)"))
            ax.axis("off")
            continue

        for b in unique_batches:
            mask = batch_arr == b
            vals = _safe_series(z[mask, :])
            if vals.size == 0:
                continue
            color = cmap(np.where(unique_batches == b)[0][0])
            histogram_kwargs = {"bins": shared_bins, "alpha": 0.45, "label": str(b), "color": color}
            if density_mode:
                weights = np.ones(vals.shape[0], dtype=float) / float(vals.shape[0])
                histogram_kwargs["weights"] = weights
            ax.hist(vals, **histogram_kwargs)
            # Use a gaussian kernel density estimate to plot a smooth curve of the distribution
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(vals)
                x = np.linspace(-5, 5, 100)
                scale = bin_width if density_mode else float(vals.size) * bin_width
                ax.plot(x, kde(x) * scale, color=color, linewidth=1.5, alpha=0.7)
            except Exception:
                # KDE can fail for near-constant vectors; keep the histogram only.
                pass
            
        ax.set_xlim([-6, 6])
        ax.invert_xaxis()
        panel_suffix = "Cov_adjusted" if use_residual else "Cov_unadjusted"
        ax.set_title(_title(f"{method} ({panel_suffix})"), fontsize=fontsize)
        ax.set_xlabel("Robust z-score", fontsize=fontsize)
        ax.set_ylabel("Proportion within batch" if density_mode else "Frequency", fontsize=fontsize)
        ax.legend(fontsize=7, frameon=False, title="Batch", title_fontsize=8)
        n_valid += 1

    _hide_unused_axes(axes, len(items))

    if n_valid == 0:
        plt.close(fig)
        return figs

    fig.tight_layout()
    figs.append((figure_caption, fig))
    return figs


def plot_compare_cohens_d(results, shared_y: bool = False, fontsize: float = 10):
    figs = []
    rows = []
    featurewise = []

    for method, res in _method_items(results):
        if res.cohens_d is None:
            continue

        d_arr = np.asarray(res.cohens_d, dtype=float)
        abs_d = np.abs(d_arr).ravel()
        if abs_d.size == 0:
            continue
        rows.append(
            {
                "method": method,
                "mean_abs_d": float(np.nanmean(abs_d)),
                "median_abs_d": float(np.nanmedian(abs_d)),
                "prop_large_abs_d": float(np.nanmean(abs_d >= 0.5)),
            }
        )

        if d_arr.ndim == 2 and d_arr.shape[0] > 0:
            featurewise.append((method, np.nanmedian(np.abs(d_arr), axis=0)))

    if len(rows) == 0:
        return figs

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.25, df["mean_abs_d"], width=0.25, label="Mean |d|")
    ax.bar(x, df["median_abs_d"], width=0.25, label="Median |d|")
    ax.bar(x + 0.25, df["prop_large_abs_d"], width=0.25, label="Prop |d| >= 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    ax.set_ylabel("Value", fontsize=fontsize)
    ax.set_title("Cohen's d summary", fontsize=fontsize)
    ax.legend()
    fig.tight_layout()
    figs.append(("Comparison: Cohen's d summary", fig))

    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        if shared_y:
            shared_y_max = max((np.nanmax(np.abs(vec)) if np.asarray(vec).size else 1.0 for _, vec in featurewise), default=1.0)
            shared_y_max = max(1.0, shared_y_max) * 1.1

        for ax2, (method, vec) in zip(axes, featurewise):
            vec = np.asarray(vec, dtype=float)
            x2 = np.arange(vec.size)
            ax2.scatter(x2, vec, color="C0", s=12)
            _draw_effect_size_guides(ax2, vec)

            if shared_y:
                ax2.set_ylim(-0.1, shared_y_max)
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature", fontsize=fontsize)
            ax2.set_ylabel("Median |d| across comparisons", fontsize=fontsize)
            labels, rotation = _feature_labels(vec.size)
            ax2.set_xticks(np.arange(vec.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: Cohen's d feature-wise", fig2))

    return figs


def plot_compare_variance_ratios(results, shared_y: bool = False, fontsize: float = 10):
    figs = []
    rows = []
    featurewise = []

    for method, res in _method_items(results):
        vr = res.variance_ratios
        if vr is None:
            continue
        vals = np.asarray(vr, dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_vals = np.where(vals > 0, np.log(vals), np.nan)
        if log_vals.size == 0:
            continue

        rows.append(
            {
                "method": method,
                "mean_abs_log_ratio": float(np.nanmean(np.abs(log_vals))),
                "median_abs_log_ratio": float(np.nanmedian(np.abs(log_vals))),
                "max_abs_log_ratio": float(np.nanmax(np.abs(log_vals))),
            }
        )

        vr_arr = np.asarray(vr, dtype=float)
        if vr_arr.ndim == 2 and vr_arr.shape[0] > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                feature_vec = np.nanmedian(np.where(vr_arr > 0, np.log(vr_arr), np.nan), axis=0)
            featurewise.append((method, feature_vec))

    if len(rows) == 0:
        return figs

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.15, df["mean_abs_log_ratio"], width=0.3, label="Mean |log ratio|")
    ax.bar(x + 0.15, df["median_abs_log_ratio"], width=0.3, label="Median |log ratio|")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=20, ha="right")
    ax.set_ylabel("Value", fontsize=fontsize)
    ax.set_title("Variance-ratio summary", fontsize=fontsize)
    ax.legend()
    fig.tight_layout()
    figs.append(("Comparison: variance ratios summary", fig))

    # Find maximum absolute log ratio across all methods for setting y-limits in feature-wise plots
    max_log_ratio = max((row["max_abs_log_ratio"] for row in rows), default=0)
    
    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        # Calculate a common limit only when explicitly requested.
        for ax2, (method, vec) in zip(axes, featurewise):
            vec = np.asarray(vec, dtype=float)
            x2 = np.arange(vec.size)
            ax2.plot(x2, vec, "b-", linewidth=1)
            ax2.plot(x2, vec, "r.", markersize=2)
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature", fontsize=fontsize)
            ax2.set_ylabel("Median log variance ratio", fontsize=fontsize)
            labels, rotation = _feature_labels(vec.size)
            ax2.set_xticks(np.arange(vec.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)
            if shared_y:
                ax2.set_ylim(-max_log_ratio * 1.1, max_log_ratio * 1.1)  # Set y-limits based on max log ratio across all methods

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: variance ratios feature-wise", fig2))

    return figs


def plot_compare_lmm_icc(results, fontsize: float = 10):
    figs = []
    icc_items = []
    r2_items = []

    for method, res in _method_items(results):
        if res.lmm_results is None or res.lmm_results.empty:
            continue
        df = res.lmm_results.copy()
        icc = pd.to_numeric(df.get("ICC"), errors="coerce").to_numpy(dtype=float)
        r2_cov = pd.to_numeric(df.get("R2_covariates"), errors="coerce").to_numpy(dtype=float)
        r2_cov_batch = pd.to_numeric(df.get("R2_covariates_batch"), errors="coerce").to_numpy(dtype=float)
        if icc.size:
            icc_items.append((method, icc))
        if r2_cov.size and r2_cov_batch.size:
            r2_items.append((method, r2_cov, r2_cov_batch))

    if icc_items:
        fig_icc, axes, _, _ = _make_method_grid(len(icc_items))
        for ax, (method, icc) in zip(axes, icc_items):
            x = np.arange(len(icc))
            df_method = next((res.lmm_results for m, res in _method_items(results) if m == method), None)
            if isinstance(df_method, pd.DataFrame):
                if "pval_LRT_random_mixture_fdr" in df_method:
                    pvals = pd.to_numeric(df_method["pval_LRT_random_mixture_fdr"], errors="coerce").to_numpy(dtype=float)
                elif "pval_LRT_random_mixture" in df_method:
                    pvals = pd.to_numeric(df_method["pval_LRT_random_mixture"], errors="coerce").to_numpy(dtype=float)
                else:
                    pvals = np.full_like(icc, np.nan, dtype=float)
                if not np.isfinite(pvals).any():
                    if "pval_LRT_random_fdr" in df_method:
                        pvals = pd.to_numeric(df_method["pval_LRT_random_fdr"], errors="coerce").to_numpy(dtype=float)
                    elif "pval_LRT_random" in df_method:
                        pvals = pd.to_numeric(df_method["pval_LRT_random"], errors="coerce").to_numpy(dtype=float)
                    else:
                        pvals = np.full_like(icc, np.nan, dtype=float)
            else:
                pvals = np.full_like(icc, np.nan, dtype=float)

            # If pvals are none, model failed to converge or there is a problem. Plot all points in gray. If pvals are available, color significant points in red and non-significant in blue.
            nan_mask = ~np.isfinite(pvals)
            sig_mask = np.isfinite(pvals) & (pvals < 0.05)
            colors = np.where(nan_mask, "lightgray", np.where(sig_mask, "C3", "C2"))
            ax.scatter(x, icc, c=colors, s=22, zorder=3)
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(0.98, 0.5, "ICC=0.5", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color="black")
            ax.set_title(_title(method))
            ax.set_xlabel("Feature", fontsize=fontsize)
            ax.set_ylabel("ICC", fontsize=fontsize)
            ax.set_ylim(bottom=0)
            labels, rotation = _feature_labels(len(icc))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=5, label="p < 0.05"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", markersize=5, label="p >= 0.05"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=5, label="p unavailable"),
                ],
                frameon=False,
                fontsize=6,
                loc="best",
            )
        _hide_unused_axes(axes, len(icc_items))
        fig_icc.tight_layout()
        figs.append(("Comparison: LMM ICC per feature", fig_icc))

    if r2_items:
        fig_r2, axes, _, _ = _make_method_grid(len(r2_items))
        for ax, (method, r2_cov, r2_cov_batch) in zip(axes, r2_items):
            x = np.arange(len(r2_cov))
            ax.plot(x, r2_cov, label="Covariates R2", linewidth=1.2)
            ax.plot(x, r2_cov_batch, label="Covariates + Batch R2", linewidth=1.2)
            ax.set_title(_title(method))
            ax.set_xlabel("Feature", fontsize=fontsize)
            ax.set_ylabel("R2", fontsize=fontsize)
            ax.set_ylim(bottom=0)
            labels, rotation = _feature_labels(len(r2_cov))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax.legend(fontsize=7, frameon=False)
        _hide_unused_axes(axes, len(r2_items))
        fig_r2.tight_layout()
        figs.append(("Comparison: covariates and covariates+batch R2", fig_r2))

    return figs


def plot_compare_lmm_biological_effects(results, fontsize: float = 10):
    figs = []
    summary_rows = []
    ols_items = []
    lmm_items = []

    for method, res in _method_items(results):
        if res.lmm_results is None or res.lmm_results.empty:
            continue

        bio_df = PlotDiagnosticResults.summarise_lmm_biological_effects(res.lmm_results)
        if not bio_df.empty:
            ols_items.append((method, bio_df[["covariate", "mean_ols_partial_r2"]].copy()))
            lmm_items.append((method, bio_df[["covariate", "mean_batch_adjusted_partial_r2"]].copy()))

        r2_cov = pd.to_numeric(res.lmm_results.get("R2_covariates"), errors="coerce").to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "median_r2_covariates": float(np.nanmedian(r2_cov)) if r2_cov.size else np.nan,
                "weighted_covariate_pc_correlation_top3": _weighted_covariate_pc_correlation(res.pca_results),
            }
        )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - 0.18, df["median_r2_covariates"], width=0.36, label="Median covariates R2")
        ax.bar(
            x + 0.18,
            df["weighted_covariate_pc_correlation_top3"],
            width=0.36,
            label="Weighted covariate-PC correlation",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df["method"], rotation=20, ha="right")
        ax.set_ylabel("Value", fontsize=fontsize)
        ax.set_title("Biological preservation summary", fontsize=fontsize)
        ax.legend(frameon=False)
        fig.tight_layout()
        figs.append(("Comparison: biological preservation summary", fig))

    for caption, items, value_col, title in [
        ("Comparison: covariate variance from OLS", ols_items, "mean_ols_partial_r2", "OLS partial R2")
    ]:
        if not items:
            continue
        fig_grid, axes, _, _ = _make_method_grid(len(items), square_size=5.2)
        for ax, (method, df_cov) in zip(axes, items):

            
            # df_cov = df_cov.sort_values(by=value_col, ascending=True, na_position="last") # Commented out to keep original order
            ax.barh(df_cov["covariate"].astype(str), df_cov[value_col], color="C1" if "OLS" in caption else "C2", alpha=0.85)
            ax.set_title(_title(method), fontsize=fontsize)
            ax.set_xlabel(title, fontsize=fontsize)
            ax.grid(axis="x", alpha=0.2)
            ax.tick_params(axis='y', labelsize=fontsize)
            # Make text for batch on axis bold if exists in covariates
            for label in ax.get_yticklabels():
                if "batch" in label.get_text().lower():
                    label.set_fontweight("bold")
                label.set_fontsize(fontsize)
            

        _hide_unused_axes(axes, len(items))
        fig_grid.tight_layout()
        figs.append((caption, fig_grid))

    return figs


def plot_compare_pca_r2_heatmaps(results, max_pcs: int = 7, fontsize: float = 10):
    """
    Grid of omnibus R^2 heatmaps (metadata variable x PC) per method, built from
    each method's `pc_associations` (see `DiagnosticFunctions.calculate_pc_associations`).

    Unlike a Pearson correlation matrix, R^2 is invariant to arbitrary batch/
    categorical label encoding, so this replaces the old PCA correlation heatmap.
    `max_pcs` is hard-capped at 5 to keep the heatmap readable when many
    low-variance PCs are retained.
    """
    figs = []
    items = []
    k_pcs = min(max(int(max_pcs), 1), 5)
    for method, res in _method_items(results):
        pc_associations = getattr(res, "pc_associations", None) or {}
        r2_matrix = pc_associations.get("r2_matrix")
        if not isinstance(r2_matrix, pd.DataFrame) or r2_matrix.empty:
            continue
        k = min(k_pcs, r2_matrix.shape[1])
        explained = np.asarray((res.pca_results or {}).get("explained_variance", []), dtype=float)
        items.append((method, r2_matrix.iloc[:, :k], explained[:k]))

    if not items:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items), square_size=5.8, max_cols=3)
    last_im = None
    for ax, (method, matrix, explained) in zip(axes, items):
        values = matrix.to_numpy(dtype=float)
        last_im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        ax.set_title(_title(method), fontsize=fontsize)
        ax.set_xticks(np.arange(matrix.shape[1]))
        col_labels = [
            f"{col}\n({explained[i]:.1f}%)" if i < explained.shape[0] else str(col)
            for i, col in enumerate(matrix.columns)
        ]
        ax.set_xticklabels(
            col_labels,
            fontsize=7,
            rotation=45 if matrix.shape[1] > 8 else 0,
            ha="right" if matrix.shape[1] > 8 else "center",
        )
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([str(name) for name in matrix.index], fontsize=7)
        ax.set_xlabel("Principal component (% variance explained)")
        _add_right_colorbar(fig, ax, last_im, label="Omnibus R2", shrink=1.0, labelsize=6)

        if values.shape[0] * values.shape[1] <= 24:
            for (i, j), value in np.ndenumerate(values):
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)

    _hide_unused_axes(axes, len(items))
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.92, wspace=0.40, hspace=0.40)
    figs.append(("Comparison: variance associated with metadata (Omnibus R2)", fig))
    return figs


def plot_compare_mahalanobis(results):
    figs = []
    for method, res in _method_items(results):
        if res.mahalanobis is None:
            continue
        out = PlotDiagnosticResults.mahalanobis_distance_plot(res.mahalanobis, rep=None, show=False)
        if out is None:
            continue
        fig = out[0] if isinstance(out, tuple) else out
        if fig is not None:
            fig.suptitle(_title(f"Mahalanobis comparison: {method}", width=55))
            figs.append((f"Comparison: Mahalanobis ({method})", fig))
    return figs


def plot_compare_ks(results, fontsize: float = 10):
    figs = []
    rows = []
    featurewise = []

    for method, res in _method_items(results):
        if not isinstance(res.ks_results, dict):
            continue

        pvals = []
        per_feature = []
        for key, value in res.ks_results.items():
            if key == "params" or not isinstance(value, dict):
                continue
            p = value.get("p_value_fdr")
            if p is None:
                p = value.get("p_value")
            if p is not None:
                parr = np.asarray(p, dtype=float)
                pvals.extend(parr.ravel().tolist())
                per_feature.append(parr)

        if len(pvals) == 0:
            continue

        p = np.asarray(pvals, dtype=float)
        rows.append({"method": method, "prop_significant_ks": float(np.nanmean(p < 0.05))})

        if per_feature:
            stacked = np.vstack(per_feature)
            min_p = np.nanmin(stacked, axis=0)
            featurewise.append((method, min_p))

    if len(rows) == 0:
        return figs

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["method"], df["prop_significant_ks"])
    ax.set_ylabel("Proportion p < 0.05", fontsize=fontsize)
    ax.set_title("KS summary", fontsize=fontsize)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figs.append(("Comparison: KS summary", fig))

    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        for ax2, (method, min_p) in zip(axes, featurewise):
            min_p = np.asarray(min_p, dtype=float)
            x2 = np.arange(min_p.size)
            y = -np.log10(np.clip(min_p, 1e-300, 1.0))
            colors = np.where(y >= -np.log10(0.05), "C3", "C2") # Swap colours, green when not significant, red when significant
            ax2.scatter(x2, y, c=colors, s=14, zorder=3)
            sig_line = -np.log10(0.05)
            ax2.axhline(sig_line, color="black", linestyle="--", linewidth=2.0)
            ax2.text(0.98, sig_line, "p = 0.05", transform=ax2.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color="black")
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature", fontsize=fontsize)
            ax2.set_ylabel("-log10(min adjusted p)", fontsize=fontsize)
            labels, rotation = _feature_labels(min_p.size)
            ax2.set_xticks(np.arange(min_p.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: KS feature-wise", fig2))

    return figs


def plot_compare_covariance(results, batch=None, fontsize: float = 10):
    figs = []
    mats = []
    names = []
    batch_labels = []
    vmax = 0.0

    # Ground-truth batch labels, in the same sorted order used to build the
    # pairwise Frobenius matrices, so labels never fall back to plain indices.
    original_batch_labels = None
    if batch is not None:
        original_batch_labels = [str(b) for b in np.unique(np.asarray(batch))]

    for method, res in _method_items(results):
        cov = res.covariance_results or {}
        mat = cov.get("pairwise_frobenius_normalized")
        if mat is None:
            continue
        if isinstance(mat, pd.DataFrame):
            arr = mat.to_numpy(dtype=float)
            labels = [str(name) for name in mat.index]
        else:
            arr = np.asarray(mat, dtype=float)
            labels = [str(i) for i in range(arr.shape[0])]
        if arr.ndim != 2:
            continue
        if original_batch_labels is not None and len(original_batch_labels) == arr.shape[0]:
            labels = original_batch_labels
        mats.append(arr)
        names.append(method)
        batch_labels.append(labels)
        vmax = max(vmax, float(np.nanmax(arr)))

    if len(mats) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(mats))

    for ax, method, arr, labels in zip(axes, names, mats, batch_labels):
        im = ax.imshow(arr, vmin=0.0, vmax=vmax if vmax > 0 else None, aspect="equal")
        ax.set_title(_title(method))
        ax.set_xlabel("Batch", fontsize=fontsize)
        ax.set_ylabel("Batch", fontsize=fontsize)
        # Make the x and y ticks correspond to the batch names
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        # Add numbers to each tile in the heatmap, but only if the matrix is small enough
        if arr.shape[0] <= 10 and arr.shape[1] <= 10:
            for (i, j), val in np.ndenumerate(arr):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if val > vmax / 2 else "black")

    _hide_unused_axes(axes, len(mats))
    for ax in axes[:len(mats)]:
        _add_right_colorbar(fig, ax, im, label="Normalized Frobenius distance")
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=0.30, hspace=0.35)
    figs.append(("Comparison: covariance Frobenius", fig))
    return figs


def plot_compare_batch_scree(results, batch, fontsize: float = 10):
    figs = []
    items = [(m, r) for m, r in _method_items(results) if isinstance(r.pca_results, dict) and "scores" in r.pca_results]
    if len(items) == 0:
        return figs

    batch_arr = np.asarray(batch)
    unique_batches = np.unique(batch_arr)
    fig, axes, _, _ = _make_method_grid(len(items))

    for ax, (method, res) in zip(axes, items):
        scores = np.asarray(res.pca_results["scores"], dtype=float)
        if scores.ndim != 2 or scores.shape[1] < 2:
            ax.set_title(_title(f"{method} (insufficient PCs)"))
            ax.axis("off")
            continue
        k = min(scores.shape[1], 20)
        for b in unique_batches:
            idx = np.where(batch_arr == b)[0]
            if idx.size < 2:
                continue
            var = np.var(scores[idx, :k], axis=0, ddof=1)
            denom = np.nansum(var)
            frac = var / denom if denom > 0 else np.zeros_like(var)
            ax.plot(np.arange(1, k + 1), frac, marker="o", markersize=2, linewidth=1, label=str(b))
        ax.set_title(_title(method), fontsize=fontsize)
        ax.set_xlabel("PC index", fontsize=fontsize)
        ax.set_ylabel("Fraction variance", fontsize=fontsize)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=6, frameon=False)

    _hide_unused_axes(axes, len(items))
    fig.tight_layout()
    figs.append(("Comparison: batch scree plots", fig))
    return figs


def _plot_embedding_grid(
    embedding_by_method: dict[str, np.ndarray],
    color_values: np.ndarray,
    title: str,
    is_categorical: bool,
    color_label: str,
    base_cmap: str = "tab10",
):
    figs = []
    items = list(embedding_by_method.items())
    if len(items) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items))
    mappables: list[Any] = []

    for ax, (method, emb) in zip(axes, items):
        emb = np.asarray(emb, dtype=float)
        if emb.ndim != 2 or emb.shape[1] < 2:
            ax.set_title(_title(f"{method} (embedding unavailable)"))
            ax.axis("off")
            mappables.append(None)
            continue
        values = _numeric_embedding_values(color_values, is_categorical)
        if is_categorical:
            series = pd.Series(color_values).astype("category")
            categories = list(series.cat.categories)
            codes = series.cat.codes.to_numpy(dtype=int)
            cmap = plt.get_cmap(base_cmap, max(2, len(categories)))
            for idx, category in enumerate(categories):
                mask = codes == idx
                ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7, color=cmap(idx), label=str(category))
            if len(categories) > 0:
                ax.legend(fontsize=6, frameon=False, loc="best")
            mappables.append(None)
        else:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=values, cmap="viridis", s=8, alpha=0.7)
            mappables.append(sc)
        ax.set_title(_title(method))
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        # Ensure readability in X and Y ticks for when values are large in magnitude
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
        _add_embedding_corr_label(ax, "Dim 1", "Dim 2", emb, values)

    _hide_unused_axes(axes, len(items))
    for ax, mappable in zip(axes[:len(items)], mappables):
        if mappable is not None:
            _add_right_colorbar(fig, ax, mappable, label=color_label)
    fig.suptitle(_title(title, width=70))
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    figs.append((title, fig))
    return figs


def plot_compare_pca_embeddings(
    results,
    batch,
    covariates=None,
    covariate_names=None,
    plot_covariate_embeddings: bool = True,
    allow_many_covariates: bool = False,
    base_cmap: str = "tab10",
    fontsize: float = 10,
):
    figs = []
    items = [(m, r) for m, r in _method_items(results) if isinstance(r.pca_results, dict) and "scores" in r.pca_results]
    if len(items) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items), square_size=4.8)

    batch_arr = np.asarray(batch)
    unique_batches, batch_cmap = _batch_palette(batch_arr, base_cmap)
    embeddings = {}
    for ax, (method, res) in zip(axes, items):
        score = np.asarray(res.pca_results["scores"], dtype=float)
        if score.ndim != 2 or score.shape[1] < 2:
            ax.set_title(_title(f"{method} (no PC1/PC2)"))
            ax.axis("off")
            continue
        embeddings[method] = score[:, :2]
        for b in unique_batches:
            idx = batch_arr == b
            color = batch_cmap(np.where(unique_batches == b)[0][0])
            ax.scatter(score[idx, 0], score[idx, 1], s=8, alpha=0.6, label=str(b), color=color)
        ax.set_title(_title(method), fontsize=fontsize)
        explained = np.asarray(res.pca_results.get("explained_variance", []), dtype=float)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)" if explained.shape[0] > 0 else "PC1", fontsize=fontsize)
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)" if explained.shape[0] > 1 else "PC2", fontsize=fontsize)
        ax.legend(fontsize=7, frameon=False, loc="best")

    _hide_unused_axes(axes, len(items))
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=0.30, hspace=0.35)
    figs.append(("Comparison: PCA embeddings", fig))

    cov = _safe_covariate_matrix(covariates)
    if not plot_covariate_embeddings or cov is None:
        return figs

    n_cov = cov.shape[1]
    if n_cov > 5 and not allow_many_covariates:
        return figs

    for i in range(n_cov):
        col = cov[:, i]
        series = pd.Series(col)
        n_unique = series.nunique(dropna=True)
        is_categorical = n_unique <= 12
        if covariate_names is not None and i < len(covariate_names):
            cov_name = covariate_names[i]
        else:
            cov_name = f"covariate_{i+1}"
        figs.extend(
            _plot_embedding_grid(
                embeddings,
                color_values=col,
                title=f"Comparison: PCA embeddings coloured by {cov_name}",
                is_categorical=is_categorical,
                color_label=cov_name,
                base_cmap=base_cmap,
            )
        )
    return figs


def Plot_compare_UMAP_embeddings(
    results,
    batch,
    covariates=None,
    covariate_names=None,
    n_neighbours=10,
    min_dist=0.1,
    random_state=None,
    plot_covariate_embeddings: bool = True,
    allow_many_covariates: bool = False,
    base_cmap: str = "tab10",
    fontsize: float = 10,
):
    figs = []
    try:
        import umap  # type: ignore
    except Exception:
        return figs

    items = _method_items(results)
    if len(items) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items), square_size=4.8)

    batch_arr = np.asarray(batch)
    unique_batches, batch_cmap = _batch_palette(batch_arr, base_cmap)
    embeddings = {}

    for ax, (method, res) in zip(axes, items):
        data = np.asarray(res.data, dtype=float)
        if data.ndim != 2:
            ax.set_title(_title(f"{method} (no data)"))
            ax.axis("off")
            continue
        reducer = umap.UMAP(
            n_neighbors=n_neighbours,
            min_dist=min_dist,
            random_state=random_state,
        )
        emb = reducer.fit_transform(data)
        embeddings[method] = emb
        for b in unique_batches:
            idx = batch_arr == b
            color = batch_cmap(np.where(unique_batches == b)[0][0])
            ax.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.6, label=str(b), color=color)
        ax.set_title(_title(method), fontsize=fontsize)
        ax.set_xlabel("UMAP1", fontsize=fontsize)
        ax.set_ylabel("UMAP2", fontsize=fontsize)
        ax.legend(fontsize=6, frameon=False, loc="best")

    _hide_unused_axes(axes, len(items))
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92, wspace=0.30, hspace=0.35)
    figs.append(("Comparison: UMAP embeddings", fig))

    cov = _safe_covariate_matrix(covariates)
    if not plot_covariate_embeddings or cov is None:
        return figs

    n_cov = cov.shape[1]
    if n_cov > 5 and not allow_many_covariates:
        return figs

    for i in range(n_cov):
        col = cov[:, i]
        series = pd.Series(col)
        n_unique = series.nunique(dropna=True)
        is_categorical = n_unique <= 12
        if covariate_names is not None and i < len(covariate_names):
            cov_name = covariate_names[i]
        else:
            cov_name = f"covariate_{i+1}"
        figs.extend(
            _plot_embedding_grid(
                embeddings,
                color_values=col,
                title=f"Comparison: UMAP embeddings coloured by {cov_name}",
                is_categorical=is_categorical,
                color_label=cov_name,
                base_cmap=base_cmap,
            )
        )

    return figs


def plot_method_scorecard(summary_df):
    figs = []
    if summary_df is None or summary_df.empty:
        return figs

    df = summary_df.copy()
    if "method" not in df or "overall_score" not in df:
        return figs

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["method"], df["overall_score"])
    ax.set_ylabel("Overall score")
    ax.set_title("Method scorecard")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figs.append(("Comparison: method scorecard", fig))
    return figs


def _safe_log10_p(pvals: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.clip(arr, eps, 1.0))

def plot_compare_long_subject_order(
    results,
    summary_stat: str = "median",
    summary_savepath: str | None = None,
):
    """
    Compare longitudinal subject-order consistency across methods.

    Subject-order consistency is calculated separately for each
    dataset before this function is called.

    For each method, Spearman rho is summarised across all
    TimeA-TimeB comparisons within each IDP. This produces one
    subject-order consistency value per IDP.

    The distribution of IDP-level values is then compared across
    methods using boxplots.

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

        Each result.subject_order is expected to contain:

            TimeA
            TimeB
            IDP
            nPairs
            SpearmanRho
            NullMeanRho
            pValue

    summary_stat : {"median", "mean"}, default "median"
        Statistic used to summarise Spearman rho across
        TimeA-TimeB comparisons within each IDP.

    Returns
    -------
    list
        List of (caption, figure) tuples.
    """

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    summary_stat = (
        str(summary_stat)
        .lower()
        .strip()
    )

    if summary_stat not in (
        "median",
        "mean",
    ):
        raise ValueError(
            "summary_stat must be 'median' or 'mean'."
        )


    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []

    items = _method_items(
        results
    )

    if len(items) == 0:
        return figs, pd.DataFrame()

    # ============================================================
    # EXTRACT IDP-LEVEL SUBJECT-ORDER VALUES
    # ============================================================

    methods = []

    method_values = {}

    summary_rows = []


    for method, res in items:

        # --------------------------------------------------------
        # Get subject-order results
        # --------------------------------------------------------

        df = getattr(
            res,
            "subject_order",
            None,
        )


        if (
            not isinstance(
                df,
                pd.DataFrame,
            )
            or df.empty
        ):
            continue


        tmp = df.copy()


        # --------------------------------------------------------
        # Check required columns
        # --------------------------------------------------------

        required_columns = [
            "IDP",
            "SpearmanRho",
        ]


        missing_columns = [
            column
            for column in required_columns
            if column not in tmp.columns
        ]


        if missing_columns:
            continue


        # --------------------------------------------------------
        # Clean IDP names
        # --------------------------------------------------------

        tmp["IDP"] = (
            tmp["IDP"]
            .astype(str)
            .str.strip()
        )


        # --------------------------------------------------------
        # Convert Spearman rho to numeric
        # --------------------------------------------------------

        tmp["SpearmanRho"] = pd.to_numeric(
            tmp["SpearmanRho"],
            errors="coerce",
        )


        # --------------------------------------------------------
        # Preserve IDP order
        # --------------------------------------------------------

        idp_order = list(
            pd.Index(
                tmp["IDP"]
            ).unique()
        )


        # --------------------------------------------------------
        # Keep valid Spearman rho values
        # --------------------------------------------------------

        valid_rho = tmp[
            np.isfinite(
                tmp["SpearmanRho"]
            )
        ].copy()


        if valid_rho.empty:
            continue


        # ========================================================
        # SUMMARISE ACROSS TIMEPOINT PAIRS WITHIN EACH IDP
        # ========================================================

        if summary_stat == "median":

            idp_summary = (
                valid_rho
                .groupby(
                    "IDP",
                    sort=False,
                )["SpearmanRho"]
                .median()
            )


        else:

            idp_summary = (
                valid_rho
                .groupby(
                    "IDP",
                    sort=False,
                )["SpearmanRho"]
                .mean()
            )


        # --------------------------------------------------------
        # Restore original IDP order
        # --------------------------------------------------------

        idp_summary = (
            idp_summary
            .reindex(
                idp_order
            )
        )


        # --------------------------------------------------------
        # Convert to numeric array
        # --------------------------------------------------------

        values = (
            idp_summary
            .to_numpy(
                dtype=float
            )
        )


        values = values[
            np.isfinite(
                values
            )
        ]


        if len(values) == 0:
            continue


        # --------------------------------------------------------
        # Store method-level values
        # --------------------------------------------------------

        methods.append(
            method
        )


        method_values[
            method
        ] = values


        # ========================================================
        # SUMMARY STATISTICS ACROSS IDPs
        # ========================================================

        median_rho = float(
            np.median(
                values
            )
        )


        q1_rho = float(
            np.percentile(
                values,
                25,
            )
        )


        q3_rho = float(
            np.percentile(
                values,
                75,
            )
        )


        iqr_rho = (
            q3_rho
            - q1_rho
        )


        mean_rho = float(
            np.mean(
                values
            )
        )


        sd_rho = (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else np.nan
        )


        summary_rows.append(
            {
                "Method":
                    method,

                "Median rho":
                    median_rho,

                "Q1 rho":
                    q1_rho,

                "Q3 rho":
                    q3_rho,

                "IQR rho":
                    iqr_rho,

                "Mean rho":
                    mean_rho,

                "SD rho":
                    sd_rho,

                "N IDPs":
                    int(
                        len(values)
                    ),
            }
        )


    # ============================================================
    # CHECK VALID METHODS
    # ============================================================

    if len(methods) == 0:
        return figs,pd.DataFrame()


    # ============================================================
    # CREATE FIGURE
    # ============================================================

    fig_width = max(
        8.5,
        2.0 * len(methods),
    )


    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            6.8,
        ),
        facecolor="white",
    )


    # ============================================================
    # X POSITIONS
    # ============================================================

    positions = np.arange(
        1,
        len(methods) + 1,
        dtype=float,
    )


    # ============================================================
    # REPRODUCIBLE POINT JITTER
    # ============================================================

    rng = np.random.default_rng(
        12345
    )


    # ============================================================
    # PLOT METHODS
    # ============================================================

    for position, method in zip(
        positions,
        methods,
    ):

        values = method_values[
            method
        ]


        # --------------------------------------------------------
        # Boxplot
        # --------------------------------------------------------

        bp = ax.boxplot(

            [
                values
            ],

            positions=[
                position
            ],

            widths=
                0.55,

            patch_artist=
                True,

            showfliers=
                False,

            showmeans=
                True,

            medianprops={
                "color":
                    "black",

                "linewidth":
                    1.8,
            },

            whiskerprops={
                "color":
                    "black",

                "linewidth":
                    1.2,
            },

            capprops={
                "color":
                    "black",

                "linewidth":
                    1.2,
            },

            boxprops={
                "linewidth":
                    1.2,
            },

            meanprops={
                "marker":
                    "D",

                "markersize":
                    6,

                "markerfacecolor":
                    "black",

                "markeredgecolor":
                    "black",
            },
        )


        bp[
            "boxes"
        ][0].set_facecolor(
            "C0"
        )


        bp[
            "boxes"
        ][0].set_alpha(
            0.45
        )


        # --------------------------------------------------------
        # Individual IDP points
        # --------------------------------------------------------

        n_values = len(
            values
        )


        if n_values <= 20:

            jitter_width = 0.10
            point_size = 35


        elif n_values <= 100:

            jitter_width = 0.08
            point_size = 24


        elif n_values <= 500:

            jitter_width = 0.065
            point_size = 16


        else:

            jitter_width = 0.045
            point_size = 8


        jitter = rng.uniform(
            -jitter_width,
            jitter_width,
            size=n_values,
        )


        ax.scatter(

            np.full(
                n_values,
                position,
                dtype=float,
            )
            + jitter,

            values,

            s=
                point_size,

            facecolors=
                "white",

            edgecolors=
                "black",

            linewidths=
                0.5,

            alpha=
                0.75,

            zorder=
                4,
        )


    # ============================================================
    # ZERO REFERENCE LINE
    # ============================================================

    ax.axhline(
        y=0.0,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        alpha=0.45,
        zorder=1,
    )


    # ============================================================
    # X AXIS
    # ============================================================

    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(
        methods,
        fontsize=11,
        fontweight="bold",
    )


    ax.set_xlim(
        0.4,
        positions[-1]
        + 0.6,
    )


    ax.set_xlabel(
        "Harmonisation method",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )


    # ============================================================
    # Y AXIS
    # ============================================================

    ax.set_ylim(
        -1.02,
        1.02,
    )


    ax.set_ylabel(
        "Spearman rank correlation (ρ)",
        fontsize=12,
        fontweight="bold",
    )


    # ============================================================
    # TITLE
    # ============================================================

    ax.set_title(
        "Subject-order consistency",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )


    # ============================================================
    # GRID
    # ============================================================

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
        linewidth=0.8,
    )


    # ============================================================
    # SPINES
    # ============================================================

    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    # ============================================================
    # FIGURE NOTE
    # ============================================================

    fig.text(
        0.10,
        0.025,
        (
            "Open circles represent individual IDPs. "
            "For each IDP, subject-order correlations across "
            "available acquisition pairs were summarised using "
            f"the {summary_stat}."
        ),
        fontsize=8.2,
        alpha=0.65,
    )


    # ============================================================
    # SUMMARY TABLE
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    # ------------------------------------------------------------
    # Median (Q1-Q3)
    # ------------------------------------------------------------

    summary_df[
        "Median (Q1-Q3)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Median rho']:.2f} "
            f"("
            f"{row['Q1 rho']:.2f}-"
            f"{row['Q3 rho']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Median rho"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q1 rho"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q3 rho"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    # ------------------------------------------------------------
    # Mean (SD)
    # ------------------------------------------------------------

    summary_df[
        "Mean (SD)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Mean rho']:.2f} "
            f"("
            f"{row['SD rho']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Mean rho"
                    ]
                )
                and
                pd.notna(
                    row[
                        "SD rho"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    # ============================================================
    # DISPLAY SUMMARY
    # ============================================================

    display_summary = (
        summary_df[
            [
                "Method",
                "Median (Q1-Q3)",
                "Mean (SD)",
                "N IDPs",
            ]
        ]
        .copy()
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "SUBJECT-ORDER CONSISTENCY COMPARISON"
    )


    print(
        "=" * 70
    )


    print(
        display_summary.to_string(
            index=False
        )
    )

    # ============================================================
    # SAVE SUMMARY TABLE
    # ============================================================

    if summary_savepath is not None:

        summary_path = Path(
            summary_savepath
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        print(
            f"Subject-order summary saved to: "
            f"{summary_path}"
        )

    # ============================================================
    # LAYOUT
    # ============================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.14,
    )


    # ============================================================
    # RETURN FIGURE
    # ============================================================

    figs.append(
        (
            "Longitudinal comparison: "
            "subject-order consistency",

            fig,
        )
    )


    return figs,summary_df


def plot_compare_long_within_subject_variability(
    results,
    summary_stat: str = "median",
):
    """
    Compare longitudinal within-subject variability across methods.

    Within-subject variability is calculated separately for each
    dataset before this function is called.

    The input for each method is expected to contain one row per
    subject, with IDPs stored as columns. Each subject x IDP value
    represents the subject-level summary of pairwise relative
    percent difference (RPD) across repeated observations.

    For each method:

        1. Identify the IDP columns.
        2. Summarise subject-level RPD values across subjects
           separately for each IDP.
        3. This produces one RPD value per IDP.
        4. Compare the resulting IDP-level distributions across
           methods using boxplots.

    Each open circle in the figure represents one IDP.

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

        Each result.within_subject_variability is expected to
        contain:

            subject
            n_obs
            metric_type
            <IDP 1>
            <IDP 2>
            ...
            <IDP N>

    summary_stat : {"median", "mean"}, default "median"
        Statistic used to summarise subject-level RPD values
        across subjects for each IDP.

    Returns
    -------
    tuple
        figs : list
            List of (caption, figure) tuples.

        summary_df : pd.DataFrame
            Method-level descriptive statistics across IDPs.
    """

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    summary_stat = (
        str(summary_stat)
        .lower()
        .strip()
    )


    if summary_stat not in (
        "median",
        "mean",
    ):

        raise ValueError(
            "summary_stat must be 'median' or 'mean'."
        )


    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []

    items = _method_items(
        results
    )


    if len(items) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # STORAGE
    # ============================================================

    methods = []

    method_values = {}

    summary_rows = []


    # ============================================================
    # PROCESS EACH METHOD
    # ============================================================

    for method, res in items:

        # --------------------------------------------------------
        # Get within-subject variability results
        # --------------------------------------------------------

        df = getattr(
            res,
            "within_subject_variability",
            None,
        )


        if (
            not isinstance(
                df,
                pd.DataFrame,
            )
            or df.empty
        ):

            continue


        tmp = df.copy()


        # ========================================================
        # IDENTIFY IDP COLUMNS
        #
        # Non-IDP metadata columns are excluded.
        # ========================================================

        excluded_columns = {
            "subject",
            "n_obs",
            "metric_type",
        }


        idp_columns = [
            column
            for column in tmp.columns
            if column not in excluded_columns
        ]


        if len(idp_columns) == 0:

            continue


        # ========================================================
        # CONVERT IDP VALUES TO NUMERIC
        # ========================================================

        idp_data = (
            tmp[
                idp_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )


        # ========================================================
        # ONE WITHIN-SUBJECT VARIABILITY VALUE PER IDP
        #
        # Rows:
        #     subjects
        #
        # Columns:
        #     IDPs
        #
        # Each cell:
        #     subject-level median pairwise RPD
        #
        # The values are summarised across subjects separately
        # for each IDP.
        # ========================================================

        if summary_stat == "median":

            idp_summary = (
                idp_data
                .median(
                    axis=0,
                    skipna=True,
                )
            )


        else:

            idp_summary = (
                idp_data
                .mean(
                    axis=0,
                    skipna=True,
                )
            )


        # ========================================================
        # CONVERT TO ARRAY
        # ========================================================

        values = (
            idp_summary
            .to_numpy(
                dtype=float
            )
        )


        values = values[
            np.isfinite(
                values
            )
        ]


        if len(values) == 0:

            continue


        # ========================================================
        # STORE METHOD VALUES
        # ========================================================

        methods.append(
            method
        )


        method_values[
            method
        ] = values


        # ========================================================
        # SUMMARY STATISTICS ACROSS IDPs
        # ========================================================

        median_rpd = float(
            np.median(
                values
            )
        )


        q1_rpd = float(
            np.percentile(
                values,
                25,
            )
        )


        q3_rpd = float(
            np.percentile(
                values,
                75,
            )
        )


        iqr_rpd = (
            q3_rpd
            - q1_rpd
        )


        mean_rpd = float(
            np.mean(
                values
            )
        )


        sd_rpd = (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else np.nan
        )


        summary_rows.append(
            {
                "Method":
                    method,

                "Median RPD (%)":
                    median_rpd,

                "Q1 RPD (%)":
                    q1_rpd,

                "Q3 RPD (%)":
                    q3_rpd,

                "IQR RPD (%)":
                    iqr_rpd,

                "Mean RPD (%)":
                    mean_rpd,

                "SD RPD (%)":
                    sd_rpd,

                "N IDPs":
                    int(
                        len(values)
                    ),
            }
        )


    # ============================================================
    # CHECK VALID METHODS
    # ============================================================

    if len(methods) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # CREATE FIGURE
    # ============================================================

    fig_width = max(
        8.5,
        2.0 * len(methods),
    )


    fig, ax = plt.subplots(
        figsize=(
            fig_width,
            6.8,
        ),
        facecolor="white",
    )


    # ============================================================
    # X POSITIONS
    # ============================================================

    positions = np.arange(
        1,
        len(methods) + 1,
        dtype=float,
    )


    # ============================================================
    # REPRODUCIBLE POINT JITTER
    # ============================================================

    rng = np.random.default_rng(
        12345
    )


    # ============================================================
    # PLOT METHODS
    # ============================================================

    for position, method in zip(
        positions,
        methods,
    ):

        values = method_values[
            method
        ]


        # --------------------------------------------------------
        # Boxplot
        # --------------------------------------------------------

        bp = ax.boxplot(

            [
                values
            ],

            positions=[
                position
            ],

            widths=
                0.55,

            patch_artist=
                True,

            showfliers=
                False,

            showmeans=
                True,

            medianprops={
                "color":
                    "black",

                "linewidth":
                    1.8,
            },

            whiskerprops={
                "color":
                    "black",

                "linewidth":
                    1.2,
            },

            capprops={
                "color":
                    "black",

                "linewidth":
                    1.2,
            },

            boxprops={
                "linewidth":
                    1.2,
            },

            meanprops={
                "marker":
                    "D",

                "markersize":
                    6,

                "markerfacecolor":
                    "black",

                "markeredgecolor":
                    "black",
            },
        )


        bp[
            "boxes"
        ][0].set_facecolor(
            "C0"
        )


        bp[
            "boxes"
        ][0].set_alpha(
            0.45
        )


        # --------------------------------------------------------
        # Individual IDP points
        # --------------------------------------------------------

        n_values = len(
            values
        )


        if n_values <= 20:

            jitter_width = 0.10

            point_size = 35


        elif n_values <= 100:

            jitter_width = 0.08

            point_size = 24


        elif n_values <= 500:

            jitter_width = 0.065

            point_size = 16


        else:

            jitter_width = 0.045

            point_size = 8


        jitter = rng.uniform(
            -jitter_width,
            jitter_width,
            size=n_values,
        )


        ax.scatter(

            np.full(
                n_values,
                position,
                dtype=float,
            )
            + jitter,

            values,

            s=
                point_size,

            facecolors=
                "white",

            edgecolors=
                "black",

            linewidths=
                0.5,

            alpha=
                0.75,

            zorder=
                4,
        )


    # ============================================================
    # X AXIS
    # ============================================================

    ax.set_xticks(
        positions
    )


    ax.set_xticklabels(
        methods,
        fontsize=11,
        fontweight="bold",
    )


    ax.set_xlim(
        0.4,
        positions[-1]
        + 0.6,
    )


    ax.set_xlabel(
        "Harmonisation method",
        fontsize=11,
        fontweight="bold",
        labelpad=10,
    )


    # ============================================================
    # Y AXIS
    # ============================================================

    ax.set_ylim(
        bottom=0,
    )


    ax.set_ylabel(
        "Relative percent difference (RPD, %)",
        fontsize=12,
        fontweight="bold",
    )


    # ============================================================
    # TITLE
    # ============================================================

    ax.set_title(
        "Within-subject variability",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )


    # ============================================================
    # GRID
    # ============================================================

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.18,
        linewidth=0.8,
    )


    # ============================================================
    # SPINES
    # ============================================================

    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    # ============================================================
    # FIGURE NOTE
    # ============================================================

    fig.text(
        0.10,
        0.025,
        (
            "Open circles represent individual IDPs. "
            "For each IDP, subject-level median pairwise RPD "
            f"values were summarised across subjects using "
            f"the {summary_stat}."
        ),
        fontsize=8.2,
        alpha=0.65,
    )


    # ============================================================
    # SUMMARY DATAFRAME
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    # ============================================================
    # MEDIAN (Q1-Q3)
    # ============================================================

    summary_df[
        "Median (Q1-Q3), RPD %"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Median RPD (%)']:.2f} "
            f"("
            f"{row['Q1 RPD (%)']:.2f}-"
            f"{row['Q3 RPD (%)']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Median RPD (%)"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q1 RPD (%)"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q3 RPD (%)"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    # ============================================================
    # MEAN (SD)
    # ============================================================

    summary_df[
        "Mean (SD), RPD %"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Mean RPD (%)']:.2f} "
            f"("
            f"{row['SD RPD (%)']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Mean RPD (%)"
                    ]
                )
                and
                pd.notna(
                    row[
                        "SD RPD (%)"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    # ============================================================
    # DISPLAY SUMMARY
    # ============================================================

    display_summary = (
        summary_df[
            [
                "Method",
                "Median (Q1-Q3), RPD %",
                "Mean (SD), RPD %",
                "N IDPs",
            ]
        ]
        .copy()
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "WITHIN-SUBJECT VARIABILITY COMPARISON"
    )


    print(
        "=" * 70
    )


    print(
        display_summary.to_string(
            index=False
        )
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    fig.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.90,
        bottom=0.14,
    )


    # ============================================================
    # RETURN
    # ============================================================

    figs.append(
        (
            "Longitudinal comparison: "
            "within-subject variability",

            fig,
        )
    )
    return (
        figs,
        summary_df,
    )


def plot_compare_long_addmul_batch_effects(
    results,
    effect_type: str = "additive",
    feature_col: str = "Feature",
    qvalue_col: str = "q-value",
    fdr_threshold: float = 0.05,
    p_display_floor: float = 0.001,
    figsize: tuple[float, float] = (14, 9),
):
    """
    Compare additive or multiplicative batch effects across
    harmonisation methods.

    The diagnostic is calculated separately for each dataset
    before this function is called.

    This function uses the existing FDR-adjusted p-values
    stored in:

        result.additive_effects

    or:

        result.multiplicative_effects

    depending on effect_type.

    Each panel shows the FDR-adjusted p-value for every IDP
    using -log10(q).

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

    effect_type : {"additive", "multiplicative"}
        Batch-effect diagnostic to compare.

    feature_col : str
        Column containing feature/IDP names.

    qvalue_col : str
        Column containing FDR-adjusted p-values.

    fdr_threshold : float
        FDR significance threshold.

    p_display_floor : float
        Minimum q-value displayed on the plot.

        Values smaller than this are displayed at this value
        to limit the y-axis.

    figsize : tuple
        Figure size.

    Returns
    -------
    figs : list
        List of (caption, figure) tuples.

    summary_df : pd.DataFrame
        Method-level summary of FDR-adjusted batch effects.
    """

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    effect_type = (
        str(effect_type)
        .lower()
        .strip()
    )


    if effect_type not in (
        "additive",
        "multiplicative",
    ):

        raise ValueError(
            "effect_type must be "
            "'additive' or 'multiplicative'."
        )


    if not (
        0 < fdr_threshold < 1
    ):

        raise ValueError(
            "fdr_threshold must be between 0 and 1."
        )


    if not (
        0 < p_display_floor <= 1
    ):

        raise ValueError(
            "p_display_floor must be between 0 and 1."
        )


    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []

    items = _method_items(
        results
    )


    if len(items) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # SELECT RESULT ATTRIBUTE
    # ============================================================

    if effect_type == "additive":

        result_attribute = (
            "additive_effects"
        )

        title_name = (
            "Additive batch effects"
        )

        caption = (
            "Longitudinal comparison: "
            "additive batch effects"
        )


    else:

        result_attribute = (
            "multiplicative_effects"
        )

        title_name = (
            "Multiplicative batch effects"
        )

        caption = (
            "Longitudinal comparison: "
            "multiplicative batch effects"
        )


    # ============================================================
    # LOAD METHOD DATA
    # ============================================================

    data = {}

    summary_rows = []


    for method, res in items:

        # --------------------------------------------------------
        # Get diagnostic result
        # --------------------------------------------------------

        df = getattr(
            res,
            result_attribute,
            None,
        )


        if (
            not isinstance(
                df,
                pd.DataFrame,
            )
            or df.empty
        ):

            continue


        tmp = df.copy()


        # --------------------------------------------------------
        # Check required columns
        # --------------------------------------------------------

        required_columns = {
            feature_col,
            qvalue_col,
        }


        missing_columns = (
            required_columns
            - set(
                tmp.columns
            )
        )


        if missing_columns:

            continue


        # --------------------------------------------------------
        # Keep required columns
        # --------------------------------------------------------

        tmp = tmp[
            [
                feature_col,
                qvalue_col,
            ]
        ].copy()


        # --------------------------------------------------------
        # Clean feature names
        # --------------------------------------------------------

        tmp[
            feature_col
        ] = (
            tmp[
                feature_col
            ]
            .astype(str)
            .str.strip()
        )


        # --------------------------------------------------------
        # Convert q-values to numeric
        # --------------------------------------------------------

        tmp[
            qvalue_col
        ] = pd.to_numeric(
            tmp[
                qvalue_col
            ],
            errors="coerce",
        )


        # --------------------------------------------------------
        # Remove duplicate features
        # --------------------------------------------------------

        tmp = (
            tmp
            .drop_duplicates(
                subset=feature_col,
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )


        data[
            method
        ] = tmp


    # ============================================================
    # CHECK VALID METHODS
    # ============================================================

    if len(data) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # COMMON FEATURE ORDER
    #
    # The first valid method defines the display order.
    # Other methods are reindexed to the same features.
    # ============================================================

    first_method = next(
        iter(data)
    )


    feature_order = (
        data[
            first_method
        ][
            feature_col
        ]
        .tolist()
    )


    for method in data:

        data[
            method
        ] = (
            data[
                method
            ]
            .set_index(
                feature_col
            )
            .reindex(
                feature_order
            )
            .reset_index()
        )


    # ============================================================
    # PREPARE Q-VALUES AND SUMMARY STATISTICS
    # ============================================================

    for method, df in data.items():

        q = (
            df[
                qvalue_col
            ]
            .to_numpy(
                dtype=float
            )
        )


        # --------------------------------------------------------
        # Valid q-values
        # --------------------------------------------------------

        valid_q = (
            np.isfinite(q)
            &
            (q >= 0)
            &
            (q <= 1)
        )


        q_valid = q[
            valid_q
        ]


        # --------------------------------------------------------
        # Significant IDPs
        # --------------------------------------------------------

        significant = (
            q_valid
            < fdr_threshold
        )


        n_features = len(
            df
        )


        n_valid = int(
            len(
                q_valid
            )
        )


        n_significant = int(
            np.sum(
                significant
            )
        )


        prop_significant = (
            float(
                n_significant
                / n_valid
            )
            if n_valid > 0
            else np.nan
        )


        # --------------------------------------------------------
        # Descriptive q-value statistics
        # --------------------------------------------------------

        if n_valid > 0:

            median_q = float(
                np.median(
                    q_valid
                )
            )


            q1_q = float(
                np.percentile(
                    q_valid,
                    25,
                )
            )


            q3_q = float(
                np.percentile(
                    q_valid,
                    75,
                )
            )


            mean_q = float(
                np.mean(
                    q_valid
                )
            )


            sd_q = (
                float(
                    np.std(
                        q_valid,
                        ddof=1,
                    )
                )
                if n_valid > 1
                else np.nan
            )


        else:

            median_q = np.nan

            q1_q = np.nan

            q3_q = np.nan

            mean_q = np.nan

            sd_q = np.nan


        summary_rows.append(
            {
                "Method":
                    method,

                "N IDPs":
                    int(
                        n_features
                    ),

                "N valid q-values":
                    n_valid,

                "N significant IDPs":
                    n_significant,

                "Proportion significant":
                    prop_significant,

                "Median q-value":
                    median_q,

                "Q1 q-value":
                    q1_q,

                "Q3 q-value":
                    q3_q,

                "Mean q-value":
                    mean_q,

                "SD q-value":
                    sd_q,
            }
        )


        # ========================================================
        # CONVERT Q-VALUES TO -log10(q)
        # ========================================================

        q_plot = q.copy()


        valid_plot = (
            np.isfinite(q_plot)
            &
            (q_plot >= 0)
            &
            (q_plot <= 1)
        )


        # --------------------------------------------------------
        # Floor q-values for display
        #
        # q = 0 is also displayed at the floor.
        # --------------------------------------------------------

        q_plot[
            valid_plot
        ] = np.maximum(
            q_plot[
                valid_plot
            ],
            p_display_floor,
        )


        df[
            "neglog10_q"
        ] = np.nan


        df.loc[
            valid_plot,
            "neglog10_q"
        ] = (
            -np.log10(
                q_plot[
                    valid_plot
                ]
            )
        )


    # ============================================================
    # COMMON Y AXIS
    # ============================================================

    threshold_log = (
        -np.log10(
            fdr_threshold
        )
    )


    y_max = (
        -np.log10(
            p_display_floor
        )
    )


    y_max = (
        y_max
        * 1.08
    )


    y_max = max(
        y_max,
        threshold_log
        * 2,
    )


    # ============================================================
    # SUBPLOT LAYOUT
    # ============================================================

    n_methods = len(
        data
    )


    n_cols = min(
        2,
        n_methods,
    )


    n_rows = int(
        np.ceil(
            n_methods
            / n_cols
        )
    )


    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )


    axes = (
        axes
        .flatten()
    )


    # ============================================================
    # PLOT METHODS
    # ============================================================

    for ax, (
        method,
        df,
    ) in zip(
        axes,
        data.items(),
    ):

        x = np.arange(
            len(df)
        )


        y = (
            df[
                "neglog10_q"
            ]
            .to_numpy(
                dtype=float
            )
        )


        valid = np.isfinite(
            y
        )


        # --------------------------------------------------------
        # IDP q-values
        # --------------------------------------------------------

        ax.scatter(
            x[
                valid
            ],
            y[
                valid
            ],
            s=18,
            alpha=0.8,
        )


        # --------------------------------------------------------
        # FDR threshold
        # --------------------------------------------------------

        ax.axhline(
            threshold_log,
            linestyle="--",
            linewidth=1.0,
        )


        # --------------------------------------------------------
        # Method title
        # --------------------------------------------------------

        ax.set_title(
            method,
            fontsize=13,
            fontweight="bold",
        )


        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xlabel(
            "IDPs"
        )


        ax.set_ylabel(
            r"$-\log_{10}(q_{\mathrm{FDR}})$"
        )


        ax.set_xlim(
            -1,
            len(df),
        )


        ax.set_ylim(
            0,
            y_max,
        )


        # --------------------------------------------------------
        # Feature names are omitted from the x-axis
        # --------------------------------------------------------

        ax.set_xticks(
            []
        )


        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        ax.grid(
            axis="y",
            alpha=0.2,
        )


        # --------------------------------------------------------
        # Spines
        # --------------------------------------------------------

        ax.spines[
            "top"
        ].set_visible(
            False
        )


        ax.spines[
            "right"
        ].set_visible(
            False
        )


    # ============================================================
    # FDR THRESHOLD ANNOTATION
    # ============================================================

    axes[
        0
    ].text(
        0.99,
        threshold_log,
        f"FDR = {fdr_threshold:g}",
        transform=(
            axes[
                0
            ]
            .get_yaxis_transform()
        ),
        ha="right",
        va="bottom",
        fontsize=10,
    )


    # ============================================================
    # HIDE UNUSED PANELS
    # ============================================================

    for ax in axes[
        n_methods:
    ]:

        ax.axis(
            "off"
        )


    # ============================================================
    # OVERALL TITLE
    # ============================================================

    fig.suptitle(
        title_name,
        fontsize=16,
        fontweight="bold",
    )


    # ============================================================
    # SUMMARY DATAFRAME
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    # ============================================================
    # FORMATTED SUMMARY COLUMNS
    # ============================================================

    summary_df[
        "Significant IDPs"
    ] = summary_df.apply(

        lambda row: (

            f"{int(row['N significant IDPs'])}"
            f"/"
            f"{int(row['N valid q-values'])}"

            if (
                row[
                    "N valid q-values"
                ] > 0
            )

            else "NA"
        ),

        axis=1,
    )


    summary_df[
        "Significant (%)"
    ] = summary_df[
        "Proportion significant"
    ].apply(

        lambda value: (

            f"{100 * value:.1f}%"

            if pd.notna(
                value
            )

            else "NA"
        )
    )


    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    display_summary = (
        summary_df[
            [
                "Method",
                "N IDPs",
                "N valid q-values",
                "Significant IDPs",
                "Significant (%)",
            ]
        ]
        .copy()
    )


    print(
        "\n" + "=" * 70
    )


    print(
        f"{title_name.upper()} COMPARISON"
    )


    print(
        "=" * 70
    )


    print(
        display_summary.to_string(
            index=False
        )
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )


    # ============================================================
    # RETURN
    # ============================================================

    figs.append(
        (
            caption,
            fig,
        )
    )


    return (
        figs,
        summary_df,
    )

def plot_compare_long_mixed_effects(
    results,
    feature_col: str = "IDP",
    icc_col: str = "ICC",
    batch_sig_col: str = "n_is_batchSig_bonf",
    reference_levels=(0.50, 0.75, 0.90),
    figsize=(14, 9),
):
    """
    Compare mixed-effects diagnostics across methods.

    Two figures are generated:

        1. Intraclass correlation coefficient (ICC)
        2. Number of significant batch effects after
           Bonferroni correction

    Each point represents one IDP.

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

    feature_col : str
        Column containing IDP names.

    icc_col : str
        Column containing ICC values.

    batch_sig_col : str
        Column containing the number of significant batch/site
        effects after Bonferroni correction.

    reference_levels : iterable of float
        Horizontal ICC reference levels.

    figsize : tuple
        Figure size.

    Returns
    -------
    figs : list
        List of (caption, figure) tuples.

    summary_df : pd.DataFrame
        Method-level summary statistics.
    """

    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []

    items = _method_items(
        results
    )


    if len(items) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # LOAD METHOD DATA
    # ============================================================

    data = {}


    for method, res in items:

        df = getattr(
            res,
            "mixed_effects",
            None,
        )


        if (
            not isinstance(
                df,
                pd.DataFrame,
            )
            or df.empty
        ):

            continue


        required_columns = {
            feature_col,
            icc_col,
            batch_sig_col,
        }


        missing_columns = (
            required_columns
            - set(
                df.columns
            )
        )


        if missing_columns:

            continue


        tmp = df[
            [
                feature_col,
                icc_col,
                batch_sig_col,
            ]
        ].copy()


        # --------------------------------------------------------
        # Clean feature names
        # --------------------------------------------------------

        tmp[
            feature_col
        ] = (
            tmp[
                feature_col
            ]
            .astype(str)
            .str.strip()
        )


        # --------------------------------------------------------
        # Convert metrics to numeric
        # --------------------------------------------------------

        tmp[
            icc_col
        ] = pd.to_numeric(
            tmp[
                icc_col
            ],
            errors="coerce",
        )


        tmp[
            batch_sig_col
        ] = pd.to_numeric(
            tmp[
                batch_sig_col
            ],
            errors="coerce",
        )


        # --------------------------------------------------------
        # One row per IDP
        # --------------------------------------------------------

        tmp = (
            tmp
            .drop_duplicates(
                subset=feature_col,
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )


        data[
            method
        ] = tmp


    # ============================================================
    # CHECK VALID METHODS
    # ============================================================

    if len(data) == 0:

        return (
            figs,
            pd.DataFrame(),
        )


    # ============================================================
    # COMMON FEATURE ORDER
    # ============================================================

    first_method = next(
        iter(data)
    )


    feature_order = (
        data[
            first_method
        ][
            feature_col
        ]
        .tolist()
    )


    for method in data:

        data[
            method
        ] = (
            data[
                method
            ]
            .set_index(
                feature_col
            )
            .reindex(
                feature_order
            )
            .reset_index()
        )


    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================

    summary_rows = []


    for method, df in data.items():

        # ========================================================
        # ICC
        # ========================================================

        icc = (
            df[
                icc_col
            ]
            .to_numpy(
                dtype=float
            )
        )


        icc = icc[
            np.isfinite(
                icc
            )
        ]


        # ========================================================
        # BONFERRONI-CORRECTED SIGNIFICANT BATCH EFFECTS
        # ========================================================

        batch_sig = (
            df[
                batch_sig_col
            ]
            .to_numpy(
                dtype=float
            )
        )


        batch_sig = batch_sig[
            np.isfinite(
                batch_sig
            )
        ]


        # ========================================================
        # ICC SUMMARY
        # ========================================================

        if len(icc) > 0:

            median_icc = float(
                np.median(
                    icc
                )
            )


            q1_icc = float(
                np.percentile(
                    icc,
                    25,
                )
            )


            q3_icc = float(
                np.percentile(
                    icc,
                    75,
                )
            )


            mean_icc = float(
                np.mean(
                    icc
                )
            )


            sd_icc = (
                float(
                    np.std(
                        icc,
                        ddof=1,
                    )
                )
                if len(icc) > 1
                else np.nan
            )


            n_icc_050 = int(
                np.sum(
                    icc >= 0.50
                )
            )


            n_icc_075 = int(
                np.sum(
                    icc >= 0.75
                )
            )


            n_icc_090 = int(
                np.sum(
                    icc >= 0.90
                )
            )


        else:

            median_icc = np.nan
            q1_icc = np.nan
            q3_icc = np.nan
            mean_icc = np.nan
            sd_icc = np.nan

            n_icc_050 = 0
            n_icc_075 = 0
            n_icc_090 = 0


        # ========================================================
        # BATCH-EFFECT SUMMARY
        # ========================================================

        if len(batch_sig) > 0:

            median_batch_sig = float(
                np.median(
                    batch_sig
                )
            )


            q1_batch_sig = float(
                np.percentile(
                    batch_sig,
                    25,
                )
            )


            q3_batch_sig = float(
                np.percentile(
                    batch_sig,
                    75,
                )
            )


            mean_batch_sig = float(
                np.mean(
                    batch_sig
                )
            )


            sd_batch_sig = (
                float(
                    np.std(
                        batch_sig,
                        ddof=1,
                    )
                )
                if len(batch_sig) > 1
                else np.nan
            )


            # ----------------------------------------------------
            # IDPs with no residual significant batch effects
            # ----------------------------------------------------

            n_zero_batch_sig = int(
                np.sum(
                    batch_sig == 0
                )
            )


            prop_zero_batch_sig = float(
                n_zero_batch_sig
                / len(
                    batch_sig
                )
            )


        else:

            median_batch_sig = np.nan
            q1_batch_sig = np.nan
            q3_batch_sig = np.nan
            mean_batch_sig = np.nan
            sd_batch_sig = np.nan

            n_zero_batch_sig = 0
            prop_zero_batch_sig = np.nan


        # ========================================================
        # STORE SUMMARY
        # ========================================================

        summary_rows.append(
            {
                "Method":
                    method,

                "N IDPs":
                    int(
                        len(df)
                    ),

                "N valid ICC":
                    int(
                        len(icc)
                    ),

                "Median ICC":
                    median_icc,

                "Q1 ICC":
                    q1_icc,

                "Q3 ICC":
                    q3_icc,

                "IQR ICC":
                    (
                        q3_icc
                        - q1_icc
                        if (
                            np.isfinite(
                                q1_icc
                            )
                            and
                            np.isfinite(
                                q3_icc
                            )
                        )
                        else np.nan
                    ),

                "Mean ICC":
                    mean_icc,

                "SD ICC":
                    sd_icc,

                "N ICC >= 0.50":
                    n_icc_050,

                "N ICC >= 0.75":
                    n_icc_075,

                "N ICC >= 0.90":
                    n_icc_090,

                "N valid batch counts":
                    int(
                        len(batch_sig)
                    ),

                "Median significant batches":
                    median_batch_sig,

                "Q1 significant batches":
                    q1_batch_sig,

                "Q3 significant batches":
                    q3_batch_sig,

                "IQR significant batches":
                    (
                        q3_batch_sig
                        - q1_batch_sig
                        if (
                            np.isfinite(
                                q1_batch_sig
                            )
                            and
                            np.isfinite(
                                q3_batch_sig
                            )
                        )
                        else np.nan
                    ),

                "Mean significant batches":
                    mean_batch_sig,

                "SD significant batches":
                    sd_batch_sig,

                "N IDPs with zero significant batches":
                    n_zero_batch_sig,

                "Proportion IDPs with zero significant batches":
                    prop_zero_batch_sig,
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    # ============================================================
    # ============================================================
    #
    # FIGURE 1
    #
    # ICC
    #
    # ============================================================
    # ============================================================

    all_icc = np.concatenate(
        [
            df[
                icc_col
            ].to_numpy(
                dtype=float
            )
            for df in data.values()
        ]
    )


    all_icc = all_icc[
        np.isfinite(
            all_icc
        )
    ]


    if all_icc.size > 0:

        # --------------------------------------------------------
        # Common y-axis
        # --------------------------------------------------------

        y_min_icc = min(
            0.0,
            float(
                np.min(
                    all_icc
                )
            ),
        )


        y_max_icc = max(
            1.0,
            float(
                np.max(
                    all_icc
                )
            ),
        )


        if y_max_icc > y_min_icc:

            y_padding = (
                0.05
                * (
                    y_max_icc
                    - y_min_icc
                )
            )


            y_max_icc += (
                y_padding
            )


        # --------------------------------------------------------
        # Subplot grid
        # --------------------------------------------------------

        n_methods = len(
            data
        )


        n_cols = min(
            2,
            n_methods,
        )


        n_rows = int(
            np.ceil(
                n_methods
                / n_cols
            )
        )


        fig_icc, axes_icc = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharey=True,
            squeeze=False,
            facecolor="white",
        )


        axes_icc = (
            axes_icc
            .flatten()
        )


        # --------------------------------------------------------
        # Plot
        # --------------------------------------------------------

        for ax, (
            method,
            df,
        ) in zip(
            axes_icc,
            data.items(),
        ):

            x = np.arange(
                len(df)
            )


            y = (
                df[
                    icc_col
                ]
                .to_numpy(
                    dtype=float
                )
            )


            valid = np.isfinite(
                y
            )


            ax.scatter(
                x[
                    valid
                ],
                y[
                    valid
                ],
                s=18,
                alpha=0.8,
            )


            # ----------------------------------------------------
            # ICC reference levels
            # ----------------------------------------------------

            for level in reference_levels:

                ax.axhline(
                    level,
                    linestyle=":",
                    linewidth=0.6,
                    alpha=0.50,
                )


            ax.set_title(
                method,
                fontsize=13,
                fontweight="bold",
            )


            ax.set_xlabel(
                "IDPs"
            )


            ax.set_ylabel(
                "Intraclass correlation coefficient (ICC)"
            )


            ax.set_xlim(
                -1,
                len(df),
            )


            ax.set_ylim(
                y_min_icc,
                y_max_icc,
            )


            ax.set_xticks(
                []
            )


            ax.grid(
                axis="y",
                alpha=0.15,
            )


            ax.spines[
                "top"
            ].set_visible(
                False
            )


            ax.spines[
                "right"
            ].set_visible(
                False
            )


        # --------------------------------------------------------
        # Hide unused panels
        # --------------------------------------------------------

        for ax in axes_icc[
            n_methods:
        ]:

            ax.axis(
                "off"
            )


        fig_icc.suptitle(
            "Intraclass correlation coefficient (ICC)",
            fontsize=16,
            fontweight="bold",
        )


        fig_icc.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96,
            ]
        )


        figs.append(
            (
                "Longitudinal comparison: "
                "intraclass correlation coefficient",

                fig_icc,
            )
        )


    # ============================================================
    # ============================================================
    #
    # FIGURE 2
    #
    # BONFERRONI-CORRECTED SIGNIFICANT BATCH EFFECTS
    #
    # ============================================================
    # ============================================================

    all_batch_sig = np.concatenate(
        [
            df[
                batch_sig_col
            ].to_numpy(
                dtype=float
            )
            for df in data.values()
        ]
    )


    all_batch_sig = all_batch_sig[
        np.isfinite(
            all_batch_sig
        )
    ]


    if all_batch_sig.size > 0:

        # --------------------------------------------------------
        # Common y-axis
        # --------------------------------------------------------

        y_max_batch = float(
            np.max(
                all_batch_sig
            )
        )


        y_max_batch = max(
            1.0,
            y_max_batch,
        )


        y_max_batch = (
            y_max_batch
            * 1.08
        )


        # --------------------------------------------------------
        # Subplot grid
        # --------------------------------------------------------

        n_methods = len(
            data
        )


        n_cols = min(
            2,
            n_methods,
        )


        n_rows = int(
            np.ceil(
                n_methods
                / n_cols
            )
        )


        fig_batch, axes_batch = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharey=True,
            squeeze=False,
            facecolor="white",
        )


        axes_batch = (
            axes_batch
            .flatten()
        )


        # --------------------------------------------------------
        # Plot
        # --------------------------------------------------------

        for ax, (
            method,
            df,
        ) in zip(
            axes_batch,
            data.items(),
        ):

            x = np.arange(
                len(df)
            )


            y = (
                df[
                    batch_sig_col
                ]
                .to_numpy(
                    dtype=float
                )
            )


            valid = np.isfinite(
                y
            )


            ax.scatter(
                x[
                    valid
                ],
                y[
                    valid
                ],
                s=18,
                alpha=0.8,
            )


            # ----------------------------------------------------
            # Zero reference
            # ----------------------------------------------------

            ax.axhline(
                0,
                linestyle=":",
                linewidth=0.8,
                alpha=0.50,
            )


            ax.set_title(
                method,
                fontsize=13,
                fontweight="bold",
            )


            ax.set_xlabel(
                "IDPs"
            )


            ax.set_ylabel(
                "Significant batch effects "
                "(Bonferroni-corrected)"
            )


            ax.set_xlim(
                -1,
                len(df),
            )


            ax.set_ylim(
                0,
                y_max_batch,
            )


            ax.set_xticks(
                []
            )


            ax.grid(
                axis="y",
                alpha=0.15,
            )


            ax.spines[
                "top"
            ].set_visible(
                False
            )


            ax.spines[
                "right"
            ].set_visible(
                False
            )


        # --------------------------------------------------------
        # Hide unused panels
        # --------------------------------------------------------

        for ax in axes_batch[
            n_methods:
        ]:

            ax.axis(
                "off"
            )


        fig_batch.suptitle(
            "Residual batch effects",
            fontsize=16,
            fontweight="bold",
        )


        fig_batch.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96,
            ]
        )


        figs.append(
            (
                "Longitudinal comparison: "
                "Bonferroni-corrected significant batch effects",

                fig_batch,
            )
        )


    # ============================================================
    # FORMATTED SUMMARY COLUMNS
    # ============================================================

    summary_df[
        "Median ICC (Q1-Q3)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Median ICC']:.2f} "
            f"("
            f"{row['Q1 ICC']:.2f}-"
            f"{row['Q3 ICC']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Median ICC"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q1 ICC"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q3 ICC"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    summary_df[
        "Mean ICC (SD)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Mean ICC']:.2f} "
            f"("
            f"{row['SD ICC']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Mean ICC"
                    ]
                )
                and
                pd.notna(
                    row[
                        "SD ICC"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    summary_df[
        "Median significant batches (Q1-Q3)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Median significant batches']:.2f} "
            f"("
            f"{row['Q1 significant batches']:.2f}-"
            f"{row['Q3 significant batches']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Median significant batches"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q1 significant batches"
                    ]
                )
                and
                pd.notna(
                    row[
                        "Q3 significant batches"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    summary_df[
        "Mean significant batches (SD)"
    ] = summary_df.apply(

        lambda row: (

            f"{row['Mean significant batches']:.2f} "
            f"("
            f"{row['SD significant batches']:.2f}"
            f")"

            if (
                pd.notna(
                    row[
                        "Mean significant batches"
                    ]
                )
                and
                pd.notna(
                    row[
                        "SD significant batches"
                    ]
                )
            )

            else "NA"
        ),

        axis=1,
    )


    summary_df[
        "Zero significant batches"
    ] = summary_df.apply(

        lambda row: (

            f"{int(row['N IDPs with zero significant batches'])}"
            f"/"
            f"{int(row['N valid batch counts'])}"
            f" "
            f"("
            f"{100 * row['Proportion IDPs with zero significant batches']:.1f}%"
            f")"

            if (
                row[
                    "N valid batch counts"
                ] > 0
            )

            else "NA"
        ),

        axis=1,
    )


    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    display_summary = (
        summary_df[
            [
                "Method",
                "Median ICC (Q1-Q3)",
                "Mean ICC (SD)",
                "Median significant batches (Q1-Q3)",
                "Mean significant batches (SD)",
                "Zero significant batches",
                "N IDPs",
            ]
        ]
        .copy()
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "MIXED-EFFECTS COMPARISON"
    )


    print(
        "=" * 70
    )


    print(
        display_summary.to_string(
            index=False
        )
    )


    # ============================================================
    # RETURN
    # ============================================================

    return (
        figs,
        summary_df,
    )


def plot_compare_long_biological_effects(
    results,
    covariates,
    feature_col: str = "IDP",
    significance: float = 0.05,
    plot_fdr: bool = False,
    figsize: tuple[float, float] = (14, 9),
    jitter_width: float = 0.18,
    random_seed: int = 42,
):
    """
    Compare biological fixed effects across harmonisation methods.

    Biological fixed effects are calculated separately for each
    dataset before this function is called.

    For each covariate, the mixed-effects results are expected
    to contain:

        <covariate>_est
        <covariate>_pval

    If plot_fdr=True, Benjamini-Hochberg FDR correction is
    applied separately for each method and covariate across IDPs.

    One figure is generated for each biological covariate.

    Within each figure:

        - harmonisation methods are shown as columns on the x-axis;
        - effect estimates are shown on the y-axis;
        - each method has a vertical boxplot;
        - individual IDPs are overlaid;
        - the same IDP receives the same horizontal jitter
          across every harmonisation method;
        - significant IDPs are shown as filled points;
        - non-significant IDPs are shown as open points.

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

    covariates : sequence of str
        Biological covariates to compare.

        Example:

            ["age", "sex"]

    feature_col : str
        Column containing IDP names.

    significance : float
        Significance threshold.

    plot_fdr : bool
        If False, significance is based on the original p-values.

        If True, Benjamini-Hochberg FDR correction is applied
        separately for each method and covariate.

    figsize : tuple
        Base figure size for each covariate.

    jitter_width : float
        Maximum horizontal jitter applied to IDP points around
        each method position.

    random_seed : int
        Seed used for reproducible IDP jitter.

    Returns
    -------
    figs : list
        List of (caption, figure) tuples.

    summary_df : pd.DataFrame
        Method x covariate summary statistics.
    """

    # ============================================================
    # VALIDATE INPUT
    # ============================================================

    if not covariates:

        raise ValueError(
            "covariates must contain at least one covariate."
        )

    if not (
        0 < significance < 1
    ):

        raise ValueError(
            "significance must be between 0 and 1."
        )

    if jitter_width <= 0:

        raise ValueError(
            "jitter_width must be greater than 0."
        )

    covariates = list(
        covariates
    )

    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []

    items = _method_items(
        results
    )

    if len(items) == 0:

        return (
            figs,
            pd.DataFrame(),
        )

    # ============================================================
    # LOAD MIXED-EFFECTS RESULTS
    # ============================================================

    method_data = {}

    for method, res in items:

        df = getattr(
            res,
            "mixed_effects",
            None,
        )

        if (
            not isinstance(
                df,
                pd.DataFrame,
            )
            or df.empty
        ):

            continue

        if feature_col not in df.columns:

            continue

        method_data[
            method
        ] = df.copy()

    # ============================================================
    # CHECK VALID METHODS
    # ============================================================

    if len(method_data) == 0:

        return (
            figs,
            pd.DataFrame(),
        )

    # ============================================================
    # COMMON FEATURE ORDER
    # ============================================================

    first_method = next(
        iter(
            method_data
        )
    )

    feature_order = (
        method_data[
            first_method
        ][
            feature_col
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    feature_order = list(
        dict.fromkeys(
            feature_order
        )
    )

    # ============================================================
    # SHARED IDP JITTER
    #
    # IMPORTANT:
    #
    # One jitter value is generated for each IDP.
    #
    # That same jitter value is then reused for that IDP in every
    # harmonisation method and every covariate.
    #
    # Therefore the horizontal position of an IDP relative to the
    # centre of its method's boxplot is preserved across methods.
    # ============================================================

    rng = np.random.default_rng(
        random_seed
    )

    feature_jitter = rng.uniform(
        -jitter_width,
        jitter_width,
        size=len(
            feature_order
        ),
    )

    # ============================================================
    # SUMMARY STORAGE
    # ============================================================

    summary_rows = []

    # ============================================================
    # ONE FIGURE PER BIOLOGICAL COVARIATE
    # ============================================================

    for covariate in covariates:

        est_col = (
            f"{covariate}_est"
        )

        pval_col = (
            f"{covariate}_pval"
        )

        # ========================================================
        # PREPARE METHOD DATA
        # ========================================================

        data = {}

        for method, df in method_data.items():

            # ----------------------------------------------------
            # Check required columns
            # ----------------------------------------------------

            if (
                est_col not in df.columns
                or
                pval_col not in df.columns
            ):

                continue

            tmp = df[
                [
                    feature_col,
                    est_col,
                    pval_col,
                ]
            ].copy()

            # ----------------------------------------------------
            # Clean feature names
            # ----------------------------------------------------

            tmp[
                feature_col
            ] = (
                tmp[
                    feature_col
                ]
                .astype(str)
                .str.strip()
            )

            # ----------------------------------------------------
            # Convert values to numeric
            # ----------------------------------------------------

            tmp[
                est_col
            ] = pd.to_numeric(
                tmp[
                    est_col
                ],
                errors="coerce",
            )

            tmp[
                pval_col
            ] = pd.to_numeric(
                tmp[
                    pval_col
                ],
                errors="coerce",
            )

            # ----------------------------------------------------
            # One row per IDP
            #
            # Reindexing is important because it guarantees that
            # row i represents the same IDP for every method.
            # ----------------------------------------------------

            tmp = (
                tmp
                .drop_duplicates(
                    subset=feature_col,
                    keep="first",
                )
                .set_index(
                    feature_col
                )
                .reindex(
                    feature_order
                )
                .reset_index()
            )

            # ====================================================
            # OPTIONAL FDR CORRECTION
            # ====================================================

            if plot_fdr:

                pvals = (
                    tmp[
                        pval_col
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                valid_p = (
                    np.isfinite(
                        pvals
                    )
                )

                qvals = np.full(
                    len(tmp),
                    np.nan,
                    dtype=float,
                )

                if np.any(
                    valid_p
                ):

                    (
                        _,
                        qvals_valid,
                        _,
                        _,
                    ) = multipletests(
                        pvals[
                            valid_p
                        ],
                        alpha=significance,
                        method="fdr_bh",
                    )

                    qvals[
                        valid_p
                    ] = qvals_valid

                tmp[
                    "comparison_qvalue"
                ] = qvals

            data[
                method
            ] = tmp

        # ========================================================
        # CHECK COVARIATE
        # ========================================================

        if len(data) == 0:

            continue

        # ========================================================
        # METHODS / X POSITIONS
        # ========================================================

        methods = list(
            data.keys()
        )

        n_methods = len(
            methods
        )

        method_positions = np.arange(
            n_methods,
            dtype=float,
        )

        # ========================================================
        # DETERMINE COMMON Y AXIS
        # ========================================================

        all_estimates = []

        for df in data.values():

            values = (
                df[
                    est_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            if values.size > 0:

                all_estimates.append(
                    values
                )

        if len(
            all_estimates
        ) == 0:

            continue

        all_estimates = np.concatenate(
            all_estimates
        )

        ymin = min(
            float(
                np.min(
                    all_estimates
                )
            ),
            0.0,
        )

        ymax = max(
            float(
                np.max(
                    all_estimates
                )
            ),
            0.0,
        )

        span = (
            ymax
            - ymin
        )

        if span == 0:

            span = 1.0

        padding = (
            0.08
            * span
        )

        ymin -= padding
        ymax += padding

        # ========================================================
        # FIGURE SIZE
        #
        # Allow the figure to become wider automatically when
        # additional harmonisation methods are present.
        # ========================================================

        figure_width = max(
            figsize[0],
            1.8 * n_methods + 4,
        )

        figure_height = figsize[1]

        fig, ax = plt.subplots(
            figsize=(
                figure_width,
                figure_height,
            ),
            facecolor="white",
        )

        # ========================================================
        # PLOT EACH METHOD AS ONE COLUMN
        # ========================================================

        for method_idx, method in enumerate(
            methods
        ):

            df = data[
                method
            ]

            estimates = (
                df[
                    est_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            pvals = (
                df[
                    pval_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            valid_est = np.isfinite(
                estimates
            )

            # ====================================================
            # SIGNIFICANCE
            # ====================================================

            if plot_fdr:

                qvals = (
                    df[
                        "comparison_qvalue"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                significant_mask = (
                    valid_est
                    &
                    np.isfinite(
                        qvals
                    )
                    &
                    (
                        qvals
                        < significance
                    )
                )

                significance_label = (
                    rf"$p_{{\mathrm{{FDR}}}} < "
                    rf"{significance:g}$"
                )

                nonsignificance_label = (
                    rf"$p_{{\mathrm{{FDR}}}} \geq "
                    rf"{significance:g}$"
                )

            else:

                significant_mask = (
                    valid_est
                    &
                    np.isfinite(
                        pvals
                    )
                    &
                    (
                        pvals
                        < significance
                    )
                )

                significance_label = (
                    f"p < {significance:g}"
                )

                nonsignificance_label = (
                    f"p >= {significance:g}"
                )

            nonsignificant_mask = (
                valid_est
                &
                ~significant_mask
            )

            valid_values = estimates[
                valid_est
            ]

            # ====================================================
            # SUMMARY STATISTICS
            # ====================================================

            n_valid = int(
                np.sum(
                    valid_est
                )
            )

            n_significant = int(
                np.sum(
                    significant_mask
                )
            )

            prop_significant = (
                float(
                    n_significant
                    / n_valid
                )
                if n_valid > 0
                else np.nan
            )

            if valid_values.size > 0:

                median_estimate = float(
                    np.median(
                        valid_values
                    )
                )

                q1_estimate = float(
                    np.percentile(
                        valid_values,
                        25,
                    )
                )

                q3_estimate = float(
                    np.percentile(
                        valid_values,
                        75,
                    )
                )

                mean_estimate = float(
                    np.mean(
                        valid_values
                    )
                )

                sd_estimate = (
                    float(
                        np.std(
                            valid_values,
                            ddof=1,
                        )
                    )
                    if valid_values.size > 1
                    else np.nan
                )

            else:

                median_estimate = np.nan
                q1_estimate = np.nan
                q3_estimate = np.nan
                mean_estimate = np.nan
                sd_estimate = np.nan

            summary_rows.append(
                {
                    "Covariate":
                        covariate,

                    "Method":
                        method,

                    "N IDPs":
                        int(
                            len(df)
                        ),

                    "N valid estimates":
                        n_valid,

                    "Median estimate":
                        median_estimate,

                    "Q1 estimate":
                        q1_estimate,

                    "Q3 estimate":
                        q3_estimate,

                    "IQR estimate":
                        (
                            q3_estimate
                            - q1_estimate
                            if (
                                np.isfinite(
                                    q1_estimate
                                )
                                and
                                np.isfinite(
                                    q3_estimate
                                )
                            )
                            else np.nan
                        ),

                    "Mean estimate":
                        mean_estimate,

                    "SD estimate":
                        sd_estimate,

                    "N significant IDPs":
                        n_significant,

                    "Proportion significant":
                        prop_significant,

                    "Correction":
                        (
                            "BH-FDR"
                            if plot_fdr
                            else "None"
                        ),
                }
            )

            # ====================================================
            # VERTICAL BOXPLOT
            # ====================================================

            if valid_values.size > 0:

                ax.boxplot(
                    valid_values,
                    positions=[
                        method_idx
                    ],
                    widths=0.48,
                    vert=True,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(
                        alpha=0.20,
                    ),
                    medianprops=dict(
                        linewidth=1.8,
                    ),
                    whiskerprops=dict(
                        linewidth=1.0,
                    ),
                    capprops=dict(
                        linewidth=1.0,
                    ),
                    flierprops=dict(
                        marker="",
                        markersize=0,
                    ),
                )

            # ====================================================
            # SHARED IDP X POSITIONS
            #
            # Example:
            #
            # IDP A jitter = -0.10
            #
            # Raw:
            #     x = 0 - 0.10
            #
            # ComBat:
            #     x = 1 - 0.10
            #
            # LongComBat:
            #     x = 2 - 0.10
            #
            # Thus the same IDP always occupies the same relative
            # horizontal position within every method.
            # ====================================================

            x_positions = (
                float(
                    method_idx
                )
                + feature_jitter
            )

            # ====================================================
            # NON-SIGNIFICANT IDPs
            # ====================================================

            if np.any(
                nonsignificant_mask
            ):

                ax.scatter(
                    x_positions[
                        nonsignificant_mask
                    ],
                    estimates[
                        nonsignificant_mask
                    ],
                    s=22,
                    facecolors="white",
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.85,
                    zorder=3,
                )

            # ====================================================
            # SIGNIFICANT IDPs
            # ====================================================

            if np.any(
                significant_mask
            ):

                ax.scatter(
                    x_positions[
                        significant_mask
                    ],
                    estimates[
                        significant_mask
                    ],
                    s=24,
                    facecolors="C3",
                    edgecolors="black",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=4,
                )

            # ====================================================
            # SIGNIFICANCE SUMMARY ABOVE EACH METHOD
            # ====================================================

            if np.isfinite(
                prop_significant
            ):

                summary_text = (
                    f"{n_significant}/{n_valid}\n"
                    f"({100 * prop_significant:.1f}%)"
                )

            else:

                summary_text = (
                    "unavailable"
                )

            ax.text(
                method_idx,
                1.01,
                summary_text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # ========================================================
        # ZERO-EFFECT LINE
        # ========================================================

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

        # ========================================================
        # X AXIS
        # ========================================================

        ax.set_xticks(
            method_positions
        )

        ax.set_xticklabels(
            methods,
            rotation=0,
            ha="center",
        )

        ax.set_xlim(
            -0.6,
            n_methods - 0.4,
        )

        ax.set_xlabel(
            "Harmonisation method"
        )

        # ========================================================
        # Y AXIS
        # ========================================================

        ax.set_ylabel(
            rf"{covariate} effect estimate ($\beta$)"
        )

        ax.set_ylim(
            ymin,
            ymax,
        )

        # ========================================================
        # GRID
        # ========================================================

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        # ========================================================
        # SPINES
        # ========================================================

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        # ========================================================
        # LEGEND
        # ========================================================

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="C3",
                markeredgecolor="black",
                markersize=8,
                label=significance_label,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label=nonsignificance_label,
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
        )

        # ========================================================
        # TITLE
        # ========================================================

        if plot_fdr:

            title_suffix = (
                " (FDR-corrected)"
            )

        else:

            title_suffix = (
                ""
            )

        ax.set_title(
            (
                f"{covariate.capitalize()}: "
                f"fixed-effect estimates across "
                f"harmonisation methods"
                f"{title_suffix}"
            ),
            fontsize=16,
            fontweight="bold",
            pad=35,
        )

        # ========================================================
        # LAYOUT
        # ========================================================

        fig.tight_layout()

        # ========================================================
        # STORE FIGURE
        # ========================================================

        figs.append(
            (
                (
                    "Longitudinal comparison: "
                    f"{covariate} biological effect"
                ),
                fig,
            )
        )

    # ============================================================
    # SUMMARY DATAFRAME
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ============================================================
    # FORMATTED SUMMARY COLUMNS
    # ============================================================

    if not summary_df.empty:

        summary_df[
            "Median estimate (Q1-Q3)"
        ] = summary_df.apply(

            lambda row: (

                f"{row['Median estimate']:.3f} "
                f"("
                f"{row['Q1 estimate']:.3f}-"
                f"{row['Q3 estimate']:.3f}"
                f")"

                if (
                    pd.notna(
                        row[
                            "Median estimate"
                        ]
                    )
                    and
                    pd.notna(
                        row[
                            "Q1 estimate"
                        ]
                    )
                    and
                    pd.notna(
                        row[
                            "Q3 estimate"
                        ]
                    )
                )

                else "NA"
            ),

            axis=1,
        )

        summary_df[
            "Mean estimate (SD)"
        ] = summary_df.apply(

            lambda row: (

                f"{row['Mean estimate']:.3f} "
                f"("
                f"{row['SD estimate']:.3f}"
                f")"

                if (
                    pd.notna(
                        row[
                            "Mean estimate"
                        ]
                    )
                    and
                    pd.notna(
                        row[
                            "SD estimate"
                        ]
                    )
                )

                else "NA"
            ),

            axis=1,
        )

        summary_df[
            "Significant IDPs"
        ] = summary_df.apply(

            lambda row: (

                f"{int(row['N significant IDPs'])}"
                f"/"
                f"{int(row['N valid estimates'])}"
                f" "
                f"("
                f"{100 * row['Proportion significant']:.1f}%"
                f")"

                if (
                    row[
                        "N valid estimates"
                    ] > 0
                )

                else "NA"
            ),

            axis=1,
        )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    if not summary_df.empty:

        display_summary = (
            summary_df[
                [
                    "Covariate",
                    "Method",
                    "Median estimate (Q1-Q3)",
                    "Mean estimate (SD)",
                    "Significant IDPs",
                    "N IDPs",
                    "Correction",
                ]
            ]
            .copy()
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "BIOLOGICAL FIXED-EFFECT COMPARISON"
        )

        print(
            "=" * 70
        )

        print(
            display_summary.to_string(
                index=False
            )
        )

    # ============================================================
    # RETURN
    # ============================================================

    return (
        figs,
        summary_df,
    )

def plot_compare_long_multivariate_batch_difference(
    results,
    figsize: tuple[float, float] = (10, 6.8),
):
    """
    Compare multivariate batch differences across methods.

    Mahalanobis distances are calculated separately for each
    dataset before this function is called.

    The function compares:
        1. Unadjusted pairwise batch-to-batch distances
        2. Unadjusted batch-to-global-centroid distances
        3. Covariate-adjusted pairwise distances, if available
        4. Covariate-adjusted centroid distances, if available

    For pairwise comparisons, each point represents one unique
    batch pair.

    For centroid comparisons, each point represents one batch.

    The same batch pair or batch is assigned the same jitter
    position across methods.

    Parameters
    ----------
    results : dict
        Dictionary of LongitudinalDiagnosticResult objects.

        Each result.multivariate_batch_difference is expected
        to contain:
            pairwise_raw
            pairwise_resid
            centroid_raw
            centroid_resid
            batches

    figsize : tuple
        Figure size for each comparison figure.

    Returns
    -------
    figs : list
        List of (caption, figure) tuples.

    summary_df : pd.DataFrame
        Method-level descriptive statistics for pairwise and
        centroid Mahalanobis distances.
    """

    # ============================================================
    # GET METHOD RESULTS
    # ============================================================

    figs = []
    items = _method_items(results)

    if len(items) == 0:
        return figs, pd.DataFrame()

    # ============================================================
    # EXTRACT MAHALANOBIS RESULTS
    # ============================================================

    method_data = {}

    for method, res in items:
        md = getattr(
            res,
            "multivariate_batch_difference",
            None,
        )

        if not isinstance(md, dict) or len(md) == 0:
            continue

        pairwise_raw = md.get("pairwise_raw")
        centroid_raw = md.get("centroid_raw")
        pairwise_resid = md.get("pairwise_resid")
        centroid_resid = md.get("centroid_resid")

        if (
            not isinstance(pairwise_raw, dict)
            or not isinstance(centroid_raw, dict)
        ):
            continue

        method_data[method] = {
            "pairwise_raw": pairwise_raw,
            "centroid_raw": centroid_raw,
            "pairwise_resid": pairwise_resid,
            "centroid_resid": centroid_resid,
        }

    if len(method_data) == 0:
        return figs, pd.DataFrame()

    # ============================================================
    # HELPERS
    # ============================================================

    def _canonical_pair(batch_a, batch_b):
        """Create a consistent label for a batch pair."""
        return tuple(
            sorted(
                (
                    str(batch_a),
                    str(batch_b),
                )
            )
        )

    def _prepare_pairwise(pairwise_dict):
        """
        Convert pairwise-distance dictionary to a Series
        indexed by canonical batch pairs.
        """
        if not isinstance(pairwise_dict, dict):
            return pd.Series(dtype=float)

        values = {}

        for key, value in pairwise_dict.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue

            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(distance):
                continue

            pair = _canonical_pair(
                key[0],
                key[1],
            )

            values[pair] = distance

        return pd.Series(
            values,
            dtype=float,
        )

    def _prepare_centroid(centroid_dict):
        """
        Convert centroid-distance dictionary to a Series
        indexed by batch.
        """
        if not isinstance(centroid_dict, dict):
            return pd.Series(dtype=float)

        values = {}

        for key, value in centroid_dict.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue

            if str(key[1]) != "global":
                continue

            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(distance):
                continue

            batch_name = str(key[0])
            values[batch_name] = distance

        return pd.Series(
            values,
            dtype=float,
        )

    def _descriptive_row(
        method,
        covariate_adjustment,
        distance_type,
        values,
    ):
        """Calculate descriptive statistics for one method."""

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            return {
                "Method": method,
                "Covariate adjustment": covariate_adjustment,
                "Distance type": distance_type,
                "Median distance": np.nan,
                "Q1 distance": np.nan,
                "Q3 distance": np.nan,
                "IQR distance": np.nan,
                "Mean distance": np.nan,
                "SD distance": np.nan,
                "N": 0,
            }

        median_value = float(
            np.median(values)
        )

        q1_value = float(
            np.percentile(
                values,
                25,
            )
        )

        q3_value = float(
            np.percentile(
                values,
                75,
            )
        )

        mean_value = float(
            np.mean(values)
        )

        sd_value = (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if values.size > 1
            else np.nan
        )

        return {
            "Method": method,
            "Covariate adjustment": covariate_adjustment,
            "Distance type": distance_type,
            "Median distance": median_value,
            "Q1 distance": q1_value,
            "Q3 distance": q3_value,
            "IQR distance": q3_value - q1_value,
            "Mean distance": mean_value,
            "SD distance": sd_value,
            "N": int(values.size),
        }

    # ============================================================
    # PREPARE ALL METHOD DATA
    # ============================================================

    prepared = {}

    for method, md in method_data.items():
        prepared[method] = {
            "pairwise_raw": _prepare_pairwise(
                md["pairwise_raw"]
            ),
            "centroid_raw": _prepare_centroid(
                md["centroid_raw"]
            ),
            "pairwise_resid": _prepare_pairwise(
                md["pairwise_resid"]
            ),
            "centroid_resid": _prepare_centroid(
                md["centroid_resid"]
            ),
        }

    # ============================================================
    # SUMMARY STORAGE
    # ============================================================

    summary_rows = []

    # ============================================================
    # FIGURE HELPER
    # ============================================================

    def _plot_distance_comparison(
        data_key,
        distance_type,
        covariate_adjustment,
        title,
        caption,
    ):
        """Plot one Mahalanobis comparison across methods."""

        # --------------------------------------------------------
        # Keep methods containing valid values
        # --------------------------------------------------------

        plot_methods = []

        for method in prepared:
            series = prepared[method][data_key]

            if (
                isinstance(series, pd.Series)
                and not series.empty
            ):
                plot_methods.append(method)

        if len(plot_methods) == 0:
            return None

        # --------------------------------------------------------
        # Determine labels shared across all plotted methods
        # --------------------------------------------------------

        common_labels = None

        for method in plot_methods:
            labels = set(
                prepared[method][data_key]
                .index
                .tolist()
            )

            if common_labels is None:
                common_labels = labels
            else:
                common_labels = common_labels.intersection(
                    labels
                )

        if (
            common_labels is None
            or len(common_labels) == 0
        ):
            return None

        # --------------------------------------------------------
        # Preserve ordering from first valid method
        # --------------------------------------------------------

        first_method = plot_methods[0]

        first_order = (
            prepared[first_method][data_key]
            .index
            .tolist()
        )

        common_order = [
            label
            for label in first_order
            if label in common_labels
        ]

        if len(common_order) == 0:
            return None

        # --------------------------------------------------------
        # Store summary statistics
        # --------------------------------------------------------

        for method in plot_methods:
            values = (
                prepared[method][data_key]
                .reindex(common_order)
                .to_numpy(dtype=float)
            )

            summary_rows.append(
                _descriptive_row(
                    method=method,
                    covariate_adjustment=covariate_adjustment,
                    distance_type=distance_type,
                    values=values,
                )
            )

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig_width = max(
            figsize[0],
            2.0 * len(plot_methods),
        )

        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                figsize[1],
            ),
            facecolor="white",
        )

        positions = np.arange(
            1,
            len(plot_methods) + 1,
            dtype=float,
        )

        # --------------------------------------------------------
        # Shared jitter
        #
        # The same batch pair or batch receives the same
        # horizontal offset in every method.
        # --------------------------------------------------------

        rng = np.random.default_rng(12345)
        n_points = len(common_order)

        if n_points <= 20:
            jitter_width = 0.10
            point_size = 36
        elif n_points <= 100:
            jitter_width = 0.08
            point_size = 24
        else:
            jitter_width = 0.06
            point_size = 16

        shared_jitter = rng.uniform(
            -jitter_width,
            jitter_width,
            size=n_points,
        )

        # --------------------------------------------------------
        # Plot methods
        # --------------------------------------------------------

        for position, method in zip(
            positions,
            plot_methods,
        ):
            values = (
                prepared[method][data_key]
                .reindex(common_order)
                .to_numpy(dtype=float)
            )

            valid = np.isfinite(values)
            valid_values = values[valid]

            # ----------------------------------------------------
            # Boxplot
            # ----------------------------------------------------

            if valid_values.size > 0:
                bp = ax.boxplot(
                    [valid_values],
                    positions=[position],
                    widths=0.55,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    medianprops={
                        "color": "black",
                        "linewidth": 1.8,
                    },
                    whiskerprops={
                        "color": "black",
                        "linewidth": 1.2,
                    },
                    capprops={
                        "color": "black",
                        "linewidth": 1.2,
                    },
                    boxprops={
                        "linewidth": 1.2,
                    },
                    meanprops={
                        "marker": "D",
                        "markersize": 6,
                        "markerfacecolor": "black",
                        "markeredgecolor": "black",
                    },
                )

                bp["boxes"][0].set_facecolor("C0")
                bp["boxes"][0].set_alpha(0.45)

            # ----------------------------------------------------
            # Individual batch-pair / batch points
            # ----------------------------------------------------

            ax.scatter(
                (
                    np.full(
                        np.sum(valid),
                        position,
                        dtype=float,
                    )
                    + shared_jitter[valid]
                ),
                values[valid],
                s=point_size,
                facecolors="white",
                edgecolors="black",
                linewidths=0.6,
                alpha=0.80,
                zorder=4,
            )

        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xticks(positions)

        ax.set_xticklabels(
            plot_methods,
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlim(
            0.4,
            positions[-1] + 0.6,
        )

        ax.set_ylim(
            bottom=0,
        )

        ax.set_xlabel(
            "Harmonisation method",
            fontsize=11,
            fontweight="bold",
            labelpad=10,
        )

        ax.set_ylabel(
            "Mahalanobis distance",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.18,
            linewidth=0.8,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --------------------------------------------------------
        # Figure note
        # --------------------------------------------------------

        if distance_type == "Pairwise":
            note = (
                "Open circles represent matched batch pairs. "
                "The same batch pair has the same horizontal "
                "offset across methods."
            )
        else:
            note = (
                "Open circles represent matched batches. "
                "The same batch has the same horizontal "
                "offset across methods."
            )

        fig.text(
            0.10,
            0.025,
            note,
            fontsize=8.2,
            alpha=0.65,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.97,
            top=0.90,
            bottom=0.14,
        )

        return caption, fig

    # ============================================================
    # UNADJUSTED PAIRWISE DISTANCES
    # ============================================================

    raw_pairwise_fig = _plot_distance_comparison(
        data_key="pairwise_raw",
        distance_type="Pairwise",
        covariate_adjustment="Unadjusted",
        title="Pairwise multivariate batch separation",
        caption=(
            "Longitudinal comparison: "
            "pairwise Mahalanobis batch distances"
        ),
    )

    if raw_pairwise_fig is not None:
        figs.append(raw_pairwise_fig)

    # ============================================================
    # UNADJUSTED CENTROID DISTANCES
    # ============================================================

    raw_centroid_fig = _plot_distance_comparison(
        data_key="centroid_raw",
        distance_type="Centroid",
        covariate_adjustment="Unadjusted",
        title="Batch-to-global multivariate separation",
        caption=(
            "Longitudinal comparison: "
            "Mahalanobis batch-to-global distances"
        ),
    )

    if raw_centroid_fig is not None:
        figs.append(raw_centroid_fig)

    # ============================================================
    # COVARIATE-ADJUSTED PAIRWISE DISTANCES
    # ============================================================

    residual_pairwise_fig = _plot_distance_comparison(
        data_key="pairwise_resid",
        distance_type="Pairwise",
        covariate_adjustment="Adjusted",
        title=(
            "Pairwise multivariate batch separation "
            "(covariate-adjusted)"
        ),
        caption=(
            "Longitudinal comparison: "
            "covariate-adjusted pairwise Mahalanobis "
            "batch distances"
        ),
    )

    if residual_pairwise_fig is not None:
        figs.append(residual_pairwise_fig)

    # ============================================================
    # COVARIATE-ADJUSTED CENTROID DISTANCES
    # ============================================================

    residual_centroid_fig = _plot_distance_comparison(
        data_key="centroid_resid",
        distance_type="Centroid",
        covariate_adjustment="Adjusted",
        title=(
            "Batch-to-global multivariate separation "
            "(covariate-adjusted)"
        ),
        caption=(
            "Longitudinal comparison: "
            "covariate-adjusted Mahalanobis "
            "batch-to-global distances"
        ),
    )

    if residual_centroid_fig is not None:
        figs.append(residual_centroid_fig)

    # ============================================================
    # SUMMARY DATAFRAME
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ============================================================
    # FORMATTED SUMMARY COLUMNS
    # ============================================================

    if not summary_df.empty:
        summary_df[
            "Median (Q1-Q3)"
        ] = summary_df.apply(
            lambda row: (
                f"{row['Median distance']:.3f} "
                f"({row['Q1 distance']:.3f}-"
                f"{row['Q3 distance']:.3f})"
                if (
                    pd.notna(row["Median distance"])
                    and pd.notna(row["Q1 distance"])
                    and pd.notna(row["Q3 distance"])
                )
                else "NA"
            ),
            axis=1,
        )

        summary_df[
            "Mean (SD)"
        ] = summary_df.apply(
            lambda row: (
                f"{row['Mean distance']:.3f} "
                f"({row['SD distance']:.3f})"
                if (
                    pd.notna(row["Mean distance"])
                    and pd.notna(row["SD distance"])
                )
                else "NA"
            ),
            axis=1,
        )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    if not summary_df.empty:
        display_summary = summary_df[
            [
                "Method",
                "Covariate adjustment",
                "Distance type",
                "Median (Q1-Q3)",
                "Mean (SD)",
                "N",
            ]
        ].copy()

        print("\n" + "=" * 70)
        print("MULTIVARIATE BATCH-DIFFERENCE COMPARISON")
        print("=" * 70)

        print(
            display_summary.to_string(
                index=False
            )
        )

    # ============================================================
    # RETURN
    # ============================================================

    return figs, summary_df



def plot_compare_longitudinal_zscore_histograms(
    datasets: dict,
    batch,
    subjects,
    covariates=None,
    idp_names=None,
    probability_distribution: bool = True,
    draw_pdf: bool = True,
    bins: int = 80,
    xlim: tuple = (-8, 8),
    figsize_per_method: tuple = (4.5, 4.2),
    batch_order=None,
):
    """
    Compare robust IDP distributions across batches and methods.

    The single-dataset longitudinal distribution diagnostic is run
    independently for every supplied dataset.

    The comparison figure contains:

        Row 1:
            Original robust z-score distributions.

        Row 2:
            Adjusted robust z-score distributions.

    Each column represents one method.

    The adjusted distributions use the same transformation as the
    single-dataset diagnostic.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping method names to imaging matrices.

        Example:

            {
                "Raw": raw,
                "ComBat": combat,
                "CovBat": covbat,
                "LongComBat": longcombat,
            }

    batch : sequence
        Batch/scanner/site label for each observation.

    subjects : sequence
        Subject identifier for each observation.

    covariates : dict, optional
        Biological covariates.

    idp_names : sequence, optional
        IDP names.

    probability_distribution : bool
        If True, histograms are displayed as densities.

    draw_pdf : bool
        If True, normal density curves are overlaid.

    bins : int
        Number of histogram bins.

    xlim : tuple
        Robust z-score x-axis limits.

    figsize_per_method : tuple
        Approximate size allocated to each method column.

    batch_order : sequence, optional
        Explicit batch ordering.

    Returns
    -------
    figs : list
        List containing:

            (caption, figure)
    """

    # ============================================================
    # IMPORT SINGLE-DATASET PLOTTING FUNCTION
    # ============================================================

    from DiagnoseHarmonisation import PlotDiagnosticResults


    # ============================================================
    # VALIDATE DATASETS
    # ============================================================

    if (
        not isinstance(
            datasets,
            dict,
        )
        or
        len(datasets) == 0
    ):

        raise ValueError(
            "datasets must be a non-empty dictionary."
        )


    method_names = list(
        datasets.keys()
    )


    # ============================================================
    # VALIDATE FIRST DATASET
    # ============================================================

    first_data = np.asarray(
        datasets[
            method_names[0]
        ],
        dtype=float,
    )


    if first_data.ndim != 2:

        raise ValueError(
            "Each dataset must be a 2D array."
        )


    n_samples, n_idps = (
        first_data.shape
    )


    # ============================================================
    # VALIDATE ALL DATASETS
    # ============================================================

    normalized_datasets = {}


    for method, data in datasets.items():

        array = np.asarray(
            data,
            dtype=float,
        )


        if array.ndim != 2:

            raise ValueError(
                f"{method} dataset must be 2D."
            )


        if array.shape != (
            n_samples,
            n_idps,
        ):

            raise ValueError(
                f"{method} has shape {array.shape}; "
                f"expected {(n_samples, n_idps)}."
            )


        normalized_datasets[
            method
        ] = array


    # ============================================================
    # VALIDATE METADATA
    # ============================================================

    if len(
        batch
    ) != n_samples:

        raise ValueError(
            "Length of batch must match number of observations."
        )


    if len(
        subjects
    ) != n_samples:

        raise ValueError(
            "Length of subjects must match number of observations."
        )


    batch_arr = (
        np.asarray(
            batch
        )
        .astype(str)
    )


    subject_arr = (
        np.asarray(
            subjects
        )
        .astype(str)
    )


    # ============================================================
    # VALIDATE COVARIATES
    # ============================================================

    if covariates is not None:

        if not isinstance(
            covariates,
            dict,
        ):

            raise TypeError(
                "covariates must be a dictionary or None."
            )


        for name, values in covariates.items():

            if len(
                values
            ) != n_samples:

                raise ValueError(
                    f"Covariate '{name}' has "
                    f"{len(values)} observations; "
                    f"expected {n_samples}."
                )


    # ============================================================
    # IDP NAMES
    # ============================================================

    if idp_names is None:

        idp_names = [
            f"idp_{i + 1}"
            for i in range(
                n_idps
            )
        ]


    else:

        idp_names = list(
            idp_names
        )


        if len(
            idp_names
        ) != n_idps:

            raise ValueError(
                "idp_names length must match "
                "the number of IDPs."
            )


    # ============================================================
    # BATCH ORDER
    # ============================================================

    if batch_order is None:

        batch_order = (
            pd.Index(
                batch_arr
            )
            .unique()
            .tolist()
        )


    else:

        batch_order = [
            str(
                value
            )
            for value in batch_order
        ]


    # ============================================================
    # STORAGE
    # ============================================================

    transformed_data = {}


    # ============================================================
    # RUN SINGLE-DATASET TRANSFORMATION FOR EACH METHOD
    # ============================================================

    for method, data in normalized_datasets.items():

        (
            method_fig,
            raw_z,
            adjusted_z,
            _,
        ) = (
            PlotDiagnosticResults
            .plot_longitudinal_zscore_histograms(
                data=data,
                batch=batch_arr,
                subject=subject_arr,
                idp_names=idp_names,
                covariates=covariates,
                reml=True,
                bins=bins,
                xlim=xlim,
                figsize=(
                    8,
                    4,
                ),
                batch_order=batch_order,
                savepath=None,
                show=False,
                rep=None,
            )
        )


        # ------------------------------------------------------------
        # Close temporary single-dataset figure
        # ------------------------------------------------------------

        plt.close(
            method_fig
        )


        # ============================================================
        # NORMALISE RETURNED MATRICES
        # ============================================================

        raw_z_array = np.asarray(
            raw_z,
            dtype=float,
        )


        adjusted_z_array = np.asarray(
            adjusted_z,
            dtype=float,
        )


        expected_shape = (
            n_samples,
            n_idps,
        )


        if raw_z_array.shape != expected_shape:

            raise ValueError(
                f"{method}: raw robust z-score matrix has "
                f"shape {raw_z_array.shape}; "
                f"expected {expected_shape}."
            )


        if adjusted_z_array.shape != expected_shape:

            raise ValueError(
                f"{method}: adjusted robust z-score matrix has "
                f"shape {adjusted_z_array.shape}; "
                f"expected {expected_shape}."
            )


        raw_z_df = pd.DataFrame(
            raw_z_array,
            columns=idp_names,
        )


        adjusted_z_df = pd.DataFrame(
            adjusted_z_array,
            columns=idp_names,
        )


        # ============================================================
        # STORE TRANSFORMED MATRICES
        # ============================================================

        transformed_data[
            method
        ] = {
            "Original":
                raw_z_df,

            "Adjusted":
                adjusted_z_df,
        }


    # ============================================================
    # CHECK TRANSFORMED DATA
    # ============================================================

    if len(
        transformed_data
    ) == 0:

        return []


    # ============================================================
    # CREATE COMPARISON FIGURE
    #
    # Rows:
    #
    #     Original
    #     Adjusted
    #
    # Columns:
    #
    #     Methods
    # ============================================================

    n_methods = len(
        method_names
    )


    fig_width = max(
        10.0,
        figsize_per_method[0]
        * n_methods,
    )


    fig_height = max(
        8.0,
        figsize_per_method[1]
        * 2,
    )


    fig, axes = plt.subplots(
        2,
        n_methods,
        figsize=(
            fig_width,
            fig_height,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )


    # ============================================================
    # PANEL PLOTTING FUNCTION
    # ============================================================

    def _plot_panel(
        ax,
        z_df,
    ):

        for batch_name in batch_order:

            mask = (
                batch_arr
                == batch_name
            )


            batch_data = (
                z_df
                .loc[
                    mask
                ]
                .to_numpy(
                    dtype=float
                )
                .ravel()
            )


            batch_data = batch_data[
                np.isfinite(
                    batch_data
                )
            ]


            if batch_data.size == 0:

                continue


            # --------------------------------------------------------
            # Histogram
            # --------------------------------------------------------

            hist_result = ax.hist(
                batch_data,
                bins=bins,
                density=probability_distribution,
                alpha=0.45,
                label=str(
                    batch_name
                ),
            )


            # --------------------------------------------------------
            # Normal density overlay
            # --------------------------------------------------------

            if draw_pdf:

                mu = float(
                    np.mean(
                        batch_data
                    )
                )


                std = float(
                    np.std(
                        batch_data
                    )
                )


                if (
                    np.isfinite(
                        std
                    )
                    and
                    std > 0
                ):

                    from scipy.stats import norm


                    x = np.linspace(
                        xlim[0],
                        xlim[1],
                        250,
                    )


                    if probability_distribution:

                        y = norm.pdf(
                            x,
                            mu,
                            std,
                        )


                    else:

                        bin_edges = (
                            hist_result[
                                1
                            ]
                        )


                        if len(
                            bin_edges
                        ) > 1:

                            bin_width = (
                                bin_edges[1]
                                - bin_edges[0]
                            )


                        else:

                            bin_width = 1.0


                        y = (
                            norm.pdf(
                                x,
                                mu,
                                std,
                            )
                            * len(
                                batch_data
                            )
                            * bin_width
                        )


                    patches = (
                        hist_result[
                            2
                        ]
                    )


                    if len(
                        patches
                    ) > 0:

                        color = (
                            patches[
                                0
                            ]
                            .get_facecolor()
                        )


                        ax.plot(
                            x,
                            y,
                            linewidth=1.6,
                            color=color,
                        )


        # ------------------------------------------------------------
        # Panel formatting
        # ------------------------------------------------------------

        ax.set_xlim(
            xlim
        )


        ax.grid(
            axis="y",
            alpha=0.18,
        )


        ax.spines[
            "top"
        ].set_visible(
            False
        )


        ax.spines[
            "right"
        ].set_visible(
            False
        )


    # ============================================================
    # PLOT EACH METHOD
    # ============================================================

    for col, method in enumerate(
        method_names
    ):

        raw_z_df = (
            transformed_data[
                method
            ][
                "Original"
            ]
        )


        adjusted_z_df = (
            transformed_data[
                method
            ][
                "Adjusted"
            ]
        )


        # ------------------------------------------------------------
        # Original
        # ------------------------------------------------------------

        ax_raw = axes[
            0,
            col,
        ]


        _plot_panel(
            ax_raw,
            raw_z_df,
        )


        ax_raw.set_title(
            method,
            fontsize=13,
            fontweight="bold",
        )


        # ------------------------------------------------------------
        # Adjusted
        # ------------------------------------------------------------

        ax_adjusted = axes[
            1,
            col,
        ]


        _plot_panel(
            ax_adjusted,
            adjusted_z_df,
        )


        ax_adjusted.set_xlabel(
            "Robust z-score"
        )


    # ============================================================
    # ROW LABELS
    # ============================================================

    if probability_distribution:

        y_measure = (
            "Density"
        )


    else:

        y_measure = (
            "Frequency"
        )


    axes[
        0,
        0,
    ].set_ylabel(
        (
            "Original data\n\n"
            f"{y_measure}"
        ),
        fontsize=11,
        fontweight="bold",
    )


    if covariates:

        adjusted_row_label = (
            "Covariate- and\n"
            "subject-adjusted data"
        )


    else:

        adjusted_row_label = (
            "Subject-adjusted data"
        )


    axes[
        1,
        0,
    ].set_ylabel(
        (
            adjusted_row_label
            + "\n\n"
            + y_measure
        ),
        fontsize=11,
        fontweight="bold",
    )


    # ============================================================
    # SHARED LEGEND
    # ============================================================

    handles, labels = (
        axes[
            0,
            0,
        ]
        .get_legend_handles_labels()
    )


    if len(
        handles
    ) > 0:

        fig.legend(
            handles,
            labels,
            title="Batch",
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.955,
            ),
            ncol=min(
                len(
                    labels
                ),
                8,
            ),
            fontsize=8,
            frameon=False,
        )


    # ============================================================
    # OVERALL TITLE
    # ============================================================

    fig.suptitle(
        "Feature distributions across batches",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )


    # ============================================================
    # FIGURE NOTE
    # ============================================================

    if covariates:

        note = (
            "Adjusted distributions remove covariate effects "
            "and the subject random intercept while retaining "
            "batch effects."
        )


    else:

        note = (
            "Adjusted distributions remove the subject random "
            "intercept while retaining batch effects."
        )


    fig.text(
        0.5,
        0.015,
        note,
        ha="center",
        fontsize=8.5,
        alpha=0.70,
    )


    # ============================================================
    # LAYOUT
    # ============================================================

    fig.tight_layout(
        rect=[
            0.02,
            0.04,
            0.98,
            0.91,
        ]
    )


    # ============================================================
    # RETURN
    # ============================================================

    figs = [
        (
            (
                "Longitudinal comparison: "
                "robust feature distributions across batches"
            ),
            fig,
        )
    ]


    return figs

def plot_compare_longitudinal_scorecard(summary_df: pd.DataFrame):
    figs = []
    if summary_df is None or summary_df.empty:
        return figs
    if "method" not in summary_df.columns or "overall_score" not in summary_df.columns:
        return figs

    df = summary_df.copy()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.22, df.get("subject_stability_score", np.nan), width=0.22, label="Subject stability")
    ax.bar(x, df.get("batch_removal_score", np.nan), width=0.22, label="Batch removal")
    ax.bar(x + 0.22, df.get("biological_preservation_score", np.nan), width=0.22, label="Biological preservation")
    ax.plot(x, df["overall_score"], color="black", marker="o", linewidth=1.5, label="Overall")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Longitudinal method scorecard")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    figs.append(("Longitudinal comparison: method scorecard", fig))
    return figs
