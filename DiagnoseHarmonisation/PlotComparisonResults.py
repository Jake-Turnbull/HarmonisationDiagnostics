from __future__ import annotations

from typing import Any
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

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


def _batch_palette(batch: np.ndarray) -> tuple[np.ndarray, Any]:
    unique_batches = np.unique(batch)
    cmap = plt.get_cmap("tab10", len(unique_batches))
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


def _add_right_colorbar(fig, ax, mappable, label: str | None = None):
    cbar = fig.colorbar(mappable, ax=ax, location="right", fraction=0.04, pad=0.01, shrink=0.5)
    if label:
        cbar.set_label(label, fontsize=4)
    cbar.ax.tick_params(labelsize=3)
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


def plot_compare_zscore_distributions(results, batch, use_residual: bool = False):
    figs = []
    items = _method_items(results)
    if len(items) == 0:
        return figs

    batch_arr = np.asarray(batch)
    unique_batches, cmap = _batch_palette(batch_arr)

    fig, axes, _, _ = _make_method_grid(len(items))
    shared_bins = np.linspace(-5, 5, 60)
    n_valid = 0
    z_key = "zscore_residual" if use_residual else "zscore_raw"
    figure_caption = "Comparison: residual z-score distributions" if use_residual else "Comparison: raw z-score distributions"

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
            ax.hist(vals, bins=shared_bins, alpha=0.45, label=str(b), color=color)
            # Use a gaussian kernel density estimate to plot a smooth curve of the distribution
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(vals)
                x = np.linspace(-5, 5, 100)
                # Scale the kde to match the histogram height
                ax.plot(x, kde(x) * 0.9 * ax.get_ylim()[1], color=color, linewidth=1.5, alpha=0.7)
            except Exception:
                # KDE can fail for near-constant vectors; keep the histogram only.
                pass
            
        ax.set_xlim([-8, 8])
        ax.invert_xaxis()
        panel_suffix = "residual" if use_residual else "raw"
        ax.set_title(_title(f"{method} ({panel_suffix})"))
        ax.set_xlabel("Robust z-score")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, frameon=False, title="Batch", title_fontsize=8)
        n_valid += 1

    _hide_unused_axes(axes, len(items))

    if n_valid == 0:
        plt.close(fig)
        return figs

    fig.tight_layout()
    figs.append((figure_caption, fig))
    return figs


def plot_compare_cohens_d(results):
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
    ax.set_ylabel("Value")
    ax.set_title("Cohen's d summary")
    ax.legend()
    fig.tight_layout()
    figs.append(("Comparison: Cohen's d summary", fig))

    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        for ax2, (method, vec) in zip(axes, featurewise):
            vec = np.asarray(vec, dtype=float)
            x2 = np.arange(vec.size)
            ax2.plot(x2, vec, "b-", linewidth=1)
            ax2.plot(x2, vec, "r.", markersize=2)
            # Check the max values of x2, if x2 is too small, avoid plotting the horizontal lines to prevent clutter

            if max(abs(vec)) > 0.2:
                for thresh, color in [(0.2, "green"), (0.5, "orange"), (0.8, "red")]:
                    ax2.axhline(y=thresh, color=color, linestyle="--", linewidth=1)
                    #ax2.axhline(y=-thresh, color=color, linestyle="--", linewidth=1)
            elif max(abs(vec)) > 0.1:
                for thresh, color in [(0.1, "green"), (0.2, "orange"), (0.5, "red")]:
                    ax2.axhline(y=thresh, color=color, linestyle="--", linewidth=1)
                    #ax2.axhline(y=-thresh, color=color, linestyle="--", linewidth=1)
            elif max(abs(vec)) > 0.05:
                # Draw line at 1.1 times max and -1.1 max and label what the max observed value is
                max_val = max(abs(vec))
                ax2.axhline(y=1.1 * max_val, color="purple", linestyle="--", linewidth=1)
                #ax2.axhline(y=-1.1 * max_val, color="purple", linestyle="--", linewidth=1)
                ax2.text(0.5, 1.15 * max_val, f"Max observed |d|: {max_val:.3f}", color="purple", fontsize=6, ha="center")  
            
            y_max = np.nanmax(np.abs(vec)) if vec.size else 1.0
            y_max = max(1.0, y_max)

            ax2.set_ylim(-0.1, y_max)
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature")
            ax2.set_ylabel("Median |d| across comparisons")
            labels, rotation = _feature_labels(vec.size)
            ax2.set_xticks(np.arange(vec.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: Cohen's d feature-wise", fig2))

    return figs


def plot_compare_variance_ratios(results):
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
    ax.set_ylabel("Value")
    ax.set_title("Variance-ratio summary")
    ax.legend()
    fig.tight_layout()
    figs.append(("Comparison: variance ratios summary", fig))

    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        for ax2, (method, vec) in zip(axes, featurewise):
            vec = np.asarray(vec, dtype=float)
            x2 = np.arange(vec.size)
            ax2.plot(x2, vec, "b-", linewidth=1)
            ax2.plot(x2, vec, "r.", markersize=2)
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature")
            ax2.set_ylabel("Median log variance ratio")
            labels, rotation = _feature_labels(vec.size)
            ax2.set_xticks(np.arange(vec.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: variance ratios feature-wise", fig2))

    return figs


def plot_compare_lmm_icc(results):
    figs = []
    icc_items = []
    r2_items = []

    for method, res in _method_items(results):
        if res.lmm_results is None or res.lmm_results.empty:
            continue
        df = res.lmm_results.copy()
        icc = pd.to_numeric(df.get("ICC"), errors="coerce").to_numpy(dtype=float)
        r2m = pd.to_numeric(df.get("R2_marginal"), errors="coerce").to_numpy(dtype=float)
        r2c = pd.to_numeric(df.get("R2_conditional"), errors="coerce").to_numpy(dtype=float)
        if icc.size:
            icc_items.append((method, icc))
        if r2m.size and r2c.size:
            r2_items.append((method, r2m, r2c))

    if icc_items:
        fig_icc, axes, _, _ = _make_method_grid(len(icc_items))
        for ax, (method, icc) in zip(axes, icc_items):
            x = np.arange(len(icc))
            ax.bar(x, icc, alpha=0.8)
            ax.set_title(_title(method))
            ax.set_xlabel("Feature")
            ax.set_ylabel("ICC")
            ax.set_ylim(bottom=0)
            labels, rotation = _feature_labels(len(icc))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
        _hide_unused_axes(axes, len(icc_items))
        fig_icc.tight_layout()
        figs.append(("Comparison: LMM ICC per feature", fig_icc))

    if r2_items:
        fig_r2, axes, _, _ = _make_method_grid(len(r2_items))
        for ax, (method, r2m, r2c) in zip(axes, r2_items):
            x = np.arange(len(r2m))
            ax.plot(x, r2m, label="Marginal R2", linewidth=1.2)
            ax.plot(x, r2c, label="Conditional R2", linewidth=1.2)
            ax.set_title(_title(method))
            ax.set_xlabel("Feature")
            ax.set_ylabel("R2")
            ax.set_ylim(bottom=0)
            labels, rotation = _feature_labels(len(r2m))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax.legend(fontsize=7, frameon=False)
        _hide_unused_axes(axes, len(r2_items))
        fig_r2.tight_layout()
        figs.append(("Comparison: LMM marginal and conditional R2", fig_r2))

    return figs


def plot_compare_lmm_biological_effects(results):
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
            lmm_items.append((method, bio_df[["covariate", "mean_lmm_partial_r2"]].copy()))

        r2m = pd.to_numeric(res.lmm_results.get("R2_marginal"), errors="coerce").to_numpy(dtype=float)
        summary_rows.append(
            {
                "method": method,
                "median_r2_marginal": float(np.nanmedian(r2m)) if r2m.size else np.nan,
                "weighted_covariate_pc_correlation_top3": _weighted_covariate_pc_correlation(res.pca_results),
            }
        )

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - 0.18, df["median_r2_marginal"], width=0.36, label="Median marginal R2")
        ax.bar(
            x + 0.18,
            df["weighted_covariate_pc_correlation_top3"],
            width=0.36,
            label="Weighted covariate-PC correlation",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df["method"], rotation=20, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("Biological preservation summary")
        ax.legend(frameon=False)
        fig.tight_layout()
        figs.append(("Comparison: biological preservation summary", fig))

    for caption, items, value_col, title in [
        ("Comparison: covariate variance from OLS", ols_items, "mean_ols_partial_r2", "OLS partial R2"),
        (
            "Comparison: covariate variance with batch random effect",
            lmm_items,
            "mean_lmm_partial_r2",
            "Batch-adjusted partial R2",
        ),
    ]:
        if not items:
            continue
        fig_grid, axes, _, _ = _make_method_grid(len(items), square_size=5.2)
        for ax, (method, df_cov) in zip(axes, items):
            df_cov = df_cov.sort_values(value_col, ascending=True, na_position="last")
            ax.barh(df_cov["covariate"].astype(str), df_cov[value_col], color="C1" if "OLS" in caption else "C2", alpha=0.85)
            ax.set_title(_title(method))
            ax.set_xlabel(title)
            ax.grid(axis="x", alpha=0.2)
        _hide_unused_axes(axes, len(items))
        fig_grid.tight_layout()
        figs.append((caption, fig_grid))

    return figs


def plot_compare_pca_correlation_heatmaps(results, max_pcs: int = 3):
    figs = []
    items = []

    for method, res in _method_items(results):
        pca_results = res.pca_results or {}
        corr_dict = pca_results.get("pc_correlations", {}) or {}
        explained = np.asarray(pca_results.get("explained_variance", []), dtype=float)
        if not corr_dict or explained.size == 0:
            continue

        k = min(max_pcs, explained.shape[0])
        row_names = list(corr_dict.keys())
        matrix = []
        for row_name in row_names:
            corr = np.asarray((corr_dict[row_name] or {}).get("correlation", []), dtype=float)
            if corr.shape[0] < k:
                corr = np.pad(corr, (0, max(0, k - corr.shape[0])), constant_values=np.nan)
            matrix.append(corr[:k])
        items.append((method, np.asarray(matrix, dtype=float), row_names, explained[:k]))

    if not items:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items), square_size=5.6, max_cols=2)
    last_im = None
    for ax, (method, matrix, row_names, explained) in zip(axes, items):
        last_im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="jet", aspect="auto")
        ax.set_title(_title(method))
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([f"PC{i+1}\n({explained[i]:.1f}%)" for i in range(matrix.shape[1])], fontsize=7)
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels([str(name) for name in row_names], fontsize=7)
        ax.set_xlabel("Principal component")
        _add_right_colorbar(fig, ax, last_im, label="Correlation coefficient")

        if matrix.shape[0] * matrix.shape[1] <= 24:
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)

    _hide_unused_axes(axes, len(items))


    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.10, top=0.92, wspace=0.40, hspace=0.40)
    figs.append(("Comparison: PCA covariate correlation heatmaps", fig))
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


def plot_compare_ks(results):
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
    ax.set_ylabel("Proportion p < 0.05")
    ax.set_title("KS summary")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figs.append(("Comparison: KS summary", fig))

    if featurewise:
        fig2, axes, _, _ = _make_method_grid(len(featurewise))
        for ax2, (method, min_p) in zip(axes, featurewise):
            min_p = np.asarray(min_p, dtype=float)
            x2 = np.arange(min_p.size)
            y = -np.log10(np.clip(min_p, 1e-300, 1.0))
            ax2.plot(x2, y, "b-", linewidth=1)
            ax2.plot(x2, y, "r.", markersize=2)
            ax2.axhline(-np.log10(0.05), color="red", linestyle=":", linewidth=1)
            ax2.set_title(_title(method))
            ax2.set_xlabel("Feature")
            ax2.set_ylabel("-log10(min adjusted p)")
            labels, rotation = _feature_labels(min_p.size)
            ax2.set_xticks(np.arange(min_p.size))
            ax2.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center", fontsize=6)
            ax2.grid(True, alpha=0.2)

        _hide_unused_axes(axes, len(featurewise))
        fig2.tight_layout()
        figs.append(("Comparison: KS feature-wise", fig2))

    return figs


def plot_compare_covariance(results):
    figs = []
    mats = []
    names = []
    vmax = 0.0
    for method, res in _method_items(results):
        cov = res.covariance_results or {}
        mat = cov.get("pairwise_frobenius_normalized")
        if mat is None:
            continue
        arr = mat.to_numpy(dtype=float) if isinstance(mat, pd.DataFrame) else np.asarray(mat, dtype=float)
        if arr.ndim != 2:
            continue
        mats.append(arr)
        names.append(method)
        vmax = max(vmax, float(np.nanmax(arr)))

    if len(mats) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(mats))

    for ax, method, arr in zip(axes, names, mats):
        im = ax.imshow(arr, vmin=0.0, vmax=vmax if vmax > 0 else None, aspect="equal")
        ax.set_title(_title(method))
        ax.set_xlabel("Batch")
        ax.set_ylabel("Batch")
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


def plot_compare_batch_scree(results, batch):
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
        ax.set_title(_title(method))
        ax.set_xlabel("PC index")
        ax.set_ylabel("Fraction variance")
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
):
    figs = []
    items = list(embedding_by_method.items())
    if len(items) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items))

    for ax, (method, emb) in zip(axes, items):
        emb = np.asarray(emb, dtype=float)
        if emb.ndim != 2 or emb.shape[1] < 2:
            ax.set_title(_title(f"{method} (embedding unavailable)"))
            ax.axis("off")
            continue
        if is_categorical:
            cats = pd.Series(color_values).astype("category")
            codes = cats.cat.codes.to_numpy()
            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=codes,
                cmap=cm.get_cmap("tab10", max(2, len(cats.cat.categories))),
                s=8,
                alpha=0.7,
            )
        else:
            vals = pd.to_numeric(pd.Series(color_values), errors="coerce").to_numpy(dtype=float)
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=8, alpha=0.7)
        ax.set_title(_title(method))
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    _hide_unused_axes(axes, len(items))
    for ax in axes[:len(items)]:
        _add_right_colorbar(fig, ax, sc, label=color_label)
    fig.suptitle(_title(title, width=70))
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.90, wspace=0.30, hspace=0.35)
    figs.append((title, fig))
    return figs


def plot_compare_pca_embeddings(
    results,
    batch,
    covariates=None,
    plot_covariate_embeddings: bool = True,
    allow_many_covariates: bool = False,
):
    figs = []
    items = [(m, r) for m, r in _method_items(results) if isinstance(r.pca_results, dict) and "scores" in r.pca_results]
    if len(items) == 0:
        return figs

    fig, axes, _, _ = _make_method_grid(len(items), square_size=4.8)

    batch_arr = np.asarray(batch)
    unique_batches = np.unique(batch_arr)
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
            ax.scatter(score[idx, 0], score[idx, 1], s=8, alpha=0.6, label=str(b))
        ax.set_title(_title(method))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    _hide_unused_axes(axes, len(items))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
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
        figs.extend(
            _plot_embedding_grid(
                embeddings,
                color_values=col,
                title=f"Comparison: PCA embeddings coloured by covariate {i+1}",
                is_categorical=is_categorical,
                color_label=f"covariate_{i+1}",
            )
        )
    return figs


def Plot_compare_UMAP_embeddings(
    results,
    batch,
    covariates=None,
    n_neighbours=10,
    min_dist=0.1,
    random_state=None,
    plot_covariate_embeddings: bool = True,
    allow_many_covariates: bool = False,
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
    unique_batches = np.unique(batch_arr)
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
            ax.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.6, label=str(b))
        ax.set_title(_title(method))
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    _hide_unused_axes(axes, len(items))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
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
        figs.extend(
            _plot_embedding_grid(
                embeddings,
                color_values=col,
                title=f"Comparison: UMAP embeddings coloured by covariate {i+1}",
                is_categorical=is_categorical,
                color_label=f"covariate_{i+1}",
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
