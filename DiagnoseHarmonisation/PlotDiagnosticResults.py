from __future__ import annotations
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- TEST WRAPPER FUNCTION ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from scipy import stats

def LMM_Diagnostics_Plot(
    results_df,
    feature_order="original",
    max_labels=50,
    include_delta_r2=True,
    include_status_summary=True,
) -> list[tuple[str, mfig.Figure]]:
    """
    Plot LMM diagnostics from Run_LMM_cross_sectional output.

    Args:
        results_df (pd.DataFrame): Output dataframe from Run_LMM_cross_sectional.
        feature_order (str): 'original' to preserve input order, 'sorted_icc' to sort by ICC.
        max_labels (int): Maximum number of x-axis labels to show before thinning them.
        include_delta_r2 (bool): If True, add a delta_R2 plot.
        include_status_summary (bool): If True, add a status/notes summary plot.

    Returns:
        list[tuple[str, matplotlib.figure.Figure]]: Caption and figure pairs for
        the generated diagnostic plots.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---- Validation ----
    if not isinstance(results_df, pd.DataFrame):
        raise ValueError("results_df must be a pandas DataFrame.")

    required_cols = ["feature", "ICC", "R2_marginal", "R2_conditional"]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        raise ValueError(f"results_df is missing required columns: {missing}")

    df = results_df.copy()

    # Make sure there is a stable order column
    if "feature_index" not in df.columns:
        df["feature_index"] = np.arange(len(df))

    # Choose ordering
    if feature_order == "original":
        df_plot = df.sort_values("feature_index").reset_index(drop=True)
    elif feature_order == "sorted_icc":
        df_plot = df.sort_values("ICC", ascending=False, na_position="last").reset_index(drop=True)
    else:
        raise ValueError("feature_order must be either 'original' or 'sorted_icc'.")

    # Helper for x labels
    def _x_labels(frame):
        labels = frame["feature"].astype(str).tolist()
        if len(labels) <= max_labels:
            return labels, np.arange(len(labels))

        step = int(np.ceil(len(labels) / max_labels))
        shown = [lab if (i % step == 0) else "" for i, lab in enumerate(labels)]
        return shown, np.arange(len(labels))

    figs = []

    # -------------------------------------------------
    # 1) ICC per feature
    # -------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    x_labels, x = _x_labels(df_plot)

    icc_vals = pd.to_numeric(df_plot["ICC"], errors="coerce").to_numpy()
    ax1.bar(x, icc_vals, alpha=0.8)

    ax1.set_title("ICC per feature")
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("ICC")
    ax1.set_ylim(bottom=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=90 if len(df_plot) <= 100 else 0)

    figs.append(("ICC per feature", fig1))

    # -------------------------------------------------
    # 2) ICC sorted descending
    # -------------------------------------------------
    df_icc_sorted = df.sort_values("ICC", ascending=False, na_position="last").reset_index(drop=True)

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x_labels2, x2 = _x_labels(df_icc_sorted)
    icc_sorted_vals = pd.to_numeric(df_icc_sorted["ICC"], errors="coerce").to_numpy()

    ax2.bar(x2, icc_sorted_vals, alpha=0.8)
    ax2.set_title("ICC per feature (sorted descending)")
    ax2.set_xlabel("Feature")
    ax2.set_ylabel("ICC")
    ax2.set_ylim(bottom=0)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(x_labels2, rotation=90 if len(df_icc_sorted) <= 100 else 0)

    figs.append(("ICC per feature sorted", fig2))

    # -------------------------------------------------
    # 3) Marginal and conditional R²
    # -------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    x_r2 = np.arange(len(df_plot))

    r2_m = pd.to_numeric(df_plot["R2_marginal"], errors="coerce").to_numpy()
    r2_c = pd.to_numeric(df_plot["R2_conditional"], errors="coerce").to_numpy()

    ax3.plot(x_r2, r2_m, label="Marginal R²", linewidth=1.5)
    ax3.plot(x_r2, r2_c, label="Conditional R²", linewidth=1.5)

    ax3.set_title("Marginal and Conditional R² per feature")
    ax3.set_xlabel("Feature")
    ax3.set_ylabel("R²")
    ax3.set_ylim(bottom=0)

    ax3.set_xticks(x_r2)
    ax3.set_xticklabels(x_labels, rotation=90 if len(df_plot) <= 100 else 0)
    ax3.legend()

    figs.append(("Marginal and Conditional R² per feature", fig3))

    # -------------------------------------------------
    # 4) Delta R²
    # -------------------------------------------------
    if include_delta_r2 and "delta_R2" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(14, 5))
        df_delta = df_plot.copy()
        df_delta["delta_R2"] = pd.to_numeric(df_delta["delta_R2"], errors="coerce")

        delta_vals = df_delta["delta_R2"].to_numpy()
        ax4.bar(np.arange(len(df_delta)), delta_vals, alpha=0.8)

        ax4.set_title("Delta R² per feature (Conditional - Marginal)")
        ax4.set_xlabel("Feature")
        ax4.set_ylabel("Delta R²")

        max_labels = 100
        # Only show x labels if not too many features, otherwise, have no labels and just ticks:
        if len(df_delta) <= max_labels:
            ax4.set_xticks(np.arange(len(df_delta)))
            if len(df_delta) <= max_labels-20:
                ax4.set_xticklabels(
                    df_delta["feature"].astype(str).tolist(),
                    rotation=90
                )
            else:
                ax4.set_xticks(np.arange(len(df_delta)))
                ax4.set_xticklabels(
                    df_delta["feature"].astype(str).tolist(),
                    rotation=90 if len(df_delta) <= 100 else 0
                )
        else: 
            ax4.set_xticks(np.arange(len(df_delta)))
            ax4.set_xticklabels([""] * len(df_delta))

        figs.append(("Delta R² per feature", fig4))

    # -------------------------------------------------
    # 5) Status / notes summary
    # -------------------------------------------------
    if include_status_summary and "status" in df.columns:
        fig5, ax5 = plt.subplots(figsize=(10, 4))
        status_counts = df["status"].fillna("unknown").value_counts()

        ax5.bar(status_counts.index.astype(str), status_counts.values, alpha=0.85)
        ax5.set_title("LMM fit status counts")
        ax5.set_xlabel("Status")
        ax5.set_ylabel("Count")
        ax5.tick_params(axis="x", rotation=30)

        figs.append(("LMM fit status counts", fig5))

    return figs

def _is_figure(obj) -> bool:
    return isinstance(obj, mfig.Figure)

def _normalize_figs_from_result(result: Any) -> List[Tuple[Optional[str], mfig.Figure]]:
    """Normalize many possible return shapes into a list of (caption, Figure)."""
    if result is None:
        return []
    if _is_figure(result):
        return [(None, result)]
    if isinstance(result, tuple) and len(result) >= 1 and _is_figure(result[0]):
        return [(None, result[0])]
    if isinstance(result, (list, tuple)):
        out = []
        for item in result:
            if _is_figure(item):
                out.append((None, item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2 and _is_figure(item[1]):
                out.append((str(item[0]) if item[0] is not None else None, item[1]))
        return out
    if isinstance(result, dict):
        for k in ("fig", "figure", "figures"):
            if k in result:
                return _normalize_figs_from_result(result[k])
    return []

def rep_plot_wrapper(func: Callable) -> Callable:
    """
    Decorator that:
      - optionally forces show=False (if the wrapped function supports it),
      - intercepts and removes wrapper-only kwargs (rep, log_func, caption),
      - logs returned figure(s) into rep via rep.log_plot(fig, caption) if rep provided,
      - closes figures after logging to free memory.
    """
    @wraps(func)
    def _wrapper(*args, **kwargs):
        # Extract wrapper-only args and remove them from kwargs BEFORE calling func
        rep = kwargs.pop("rep", None)
        log_func = kwargs.pop("log_func", None)
        caption_kw = kwargs.pop("caption", None)

        # If function supports 'show', force show=False unless caller explicitly set it
        try:
            sig = inspect.signature(func)
            if "show" in sig.parameters and "show" not in kwargs:
                kwargs["show"] = False
        except Exception:
            pass

        # Call original function without rep/log_func/caption in kwargs
        result = func(*args, **kwargs)

        # If neither rep nor log_func provided, return the original result unchanged
        if rep is None and log_func is None:
            return result

        # Normalize any returned figures
        figs = _normalize_figs_from_result(result)
        if not figs:
            # nothing to log; return original result for backward compatibility
            return result

        # Log each figure (use caption from return value or fallback)
        for idx, (cap, fig) in enumerate(figs):
            used_caption = cap or caption_kw or f"{func.__name__} — plot {idx+1}"
            try:
                if rep is not None:
                    rep.log_plot(fig, used_caption)
                elif callable(log_func):
                    log_func(fig, used_caption)
            except Exception as e:
                # best-effort: if rep has log_text, write the error there
                try:
                    if rep is not None and hasattr(rep, "log_text"):
                        rep.log_text(f"Failed to log figure from {func.__name__}: {e}")
                except Exception:
                    pass
            finally:
                try:
                    plt.close(fig)
                except Exception:
                    pass

        # Return original result (keeps backward compatibility)
        return result

    return _wrapper

#%%

import matplotlib.pyplot as plt
from collections.abc import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence

"""
    Complementary plotting functions for the functions in DiagnosticFunctions.py

    Functions:
    - Z_Score_Plot: Plot histogram and heatmap of Z-scored data by batch.
    - Cohens_D_plot: Plot Cohen's d effect sizes with histograms.
    - variance_ratio_plot: Plot variance ratios between batches.
    - PC_corr_plot: Generate PCA diagnostic plots including scatter plots and correlation heatmaps.
    - PC_clustering_plot: K-means clustering and silhouette analysis of PCA results by batch.
    - Ks_Plot: Plot KS statistic between batches.
    - 


"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Optional
import pandas as pd
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Z-score results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def Z_Score_Plot(data, batch, probablity_distribution=False,draw_PDF=True):
    """
    Plots the median centered Z-score data as a heatmap and as a histogram of all scores.
    Re-order by batch for better visualisaion in the heatmap, also plot batch seperators on heatmap.
    Args:
        data (np.ndarray): 2D array of Z-scored data (samples x features).
    Returns:
        None: Displays plot of Z-scored data and a histogram of the values on different axes.
    """
    # Histogram of all Z-scores plotted by batch variable
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import stats

    # ---- Validation ----
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a NumPy array.")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (samples x features).")
    if not isinstance(batch, np.ndarray):
        if isinstance(batch, list):
            batch = np.array(batch)
        else:
            raise ValueError("batch must be a NumPy array or a list.")
    # Sort data by batch, batch can either be numeric or string labels here:
    sorted_indices = np.argsort(batch)
    sorted_data = data[sorted_indices, :]
    sorted_batch = batch[sorted_indices]
    unique_batches, batch_counts = np.unique(sorted_batch, return_counts=True)  
    # Create figure with gridspec
    fig = plt.figure(figsize=(14, 8))
    # Loop over unique batches and plot as histogram on same axis:
    ax1 = fig.add_subplot()
    import matplotlib
    # Define colours for each histogram based on number of unique batches:
    colors = matplotlib.pyplot.get_cmap('tab10', len(unique_batches))

    if probablity_distribution==True:
        plot_type = 'density'
    else:
        plot_type = 'frequency'

    for i in np.unique(batch):
        batch_data = data[batch == i, :].flatten()
        # Match colours of the histogram for each batch:
        color = colors(np.where(unique_batches == i)[0][0])
        ax1.hist(batch_data, bins=80, density=plot_type, alpha=0.5, label=str(i), color=color)
        # Draw an estimated normal distribution curve over histogram:
        if draw_PDF==True:
            mu, std = np.mean(batch_data), np.std(batch_data)
            xmin, xmax = ax1.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)            
            ax1.plot(x, p, color=color, linewidth=2)

    ax1.set_xlabel("Z-scores of all unique measures")
    # Set axis limits to -8 to 8 for better visualisation:
    ax1.set_xlim([-8, 8])
    ax1.invert_xaxis()
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.legend(title="Batch")

    figs = []
    figs.append(("Z-score histogram", fig))
    #figs.append(("Z-score heatmap", fig2))
    return figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Cohens D results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def Cohens_D_plot(
    cohens_d: np.ndarray,
    pair_labels: list,
    df: Optional[pd.DataFrame] = None,
    *,
    rep = None,            # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    
    """Plots Cohen's d effect sizes as a bar plot with histograms of the values.
    Args:
        cohens_d (np.ndarray): 2D array of Cohen's d values (num_pairs x num_features).
        pair_labels (list): List of labels for each pair of batches corresponding to rows in cohens_d.
        df (Optional[pd.DataFrame], optional): Optional DataFrame containing additional information. Defaults to None.
        rep (optional): Optional StatsReporter instance. Defaults to None.
        caption (Optional[str], optional): Optional caption for the plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
    Returns:
        plt.Figure: The generated plot figure."""
    # (validation code unchanged)...
    if not isinstance(cohens_d, np.ndarray):
        raise ValueError("cohens_d must be a NumPy array.")
    if cohens_d.ndim != 2:
        raise ValueError("cohens_d must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != cohens_d.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as cohens_d rows.")
    
    # Create one figure per pair and return a list or just create+log each inside loop:
    figs = []
    for i in range(cohens_d.shape[0]):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(cohens_d[i], bins=20, orientation='horizontal', color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(cohens_d.shape[1])
        ax2.bar(indices, cohens_d[i], color=[0.2, 0.4, 0.6])
        ax2.plot(indices, cohens_d[i], 'r.')
        # add effect size lines...
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Cohen's d")
        ax2.set_title(f"Effect Size (Cohen's d) for {pair_labels[i]}")
        #fig.tight_layout()
        ax2.grid(True)
        # Ensure equal y-limits for fair comparison
        # Draw horizontal lines for small/medium/large effect sizes, green small, orange medium, red large
        for thresh, color, label in [ (0.2, 'green', 'Small'), (0.5, 'orange', 'Medium'), (0.8, 'red', 'Large') ]:
            ax2.axhline(y=thresh, color=color, linestyle='--', linewidth=1)
            ax2.axhline(y=-thresh, color=color, linestyle='--', linewidth=1)
            ax2.text(cohens_d.shape[1]-1, thresh, f' {label}', color=color, va='bottom', ha='right', fontsize=8)
            ax2.text(cohens_d.shape[1]-1, -thresh, f' {label}', color=color, va='top', ha='right', fontsize=8)
        # Set limits to have equal negatice/positive range around zero
        ylims = ax2.get_ylim()
        max_abs = max(abs(ylims[0]), abs(ylims[1]))
        ax2.set_ylim(-max_abs, max_abs)
        ax1.set_ylim(-max_abs, max_abs)


        caption_i = caption or f"Cohen's d — {pair_labels[i]}"
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()
    # If rep used, figs list is empty; otherwise return list for caller
    return None if rep is not None else figs


@rep_plot_wrapper
def Levenes_Test_with_residuals(
    levene_results_raw: dict,
    levene_results_resid: dict | None = None,
    feature_names: list | None = None,
    *,
    alpha: float = 0.05,
    show: bool = False,
    rep=None,
) -> list[tuple[str, plt.Figure]]:
    """
    Plot raw and residualised Levene's test results side-by-side.

    Args:
        levene_results_raw: dict of raw Levene outputs.
        levene_results_resid: dict of Levene outputs after covariate residualisation.
        feature_names: optional list of feature names.
        alpha: significance threshold.
    """
    figs: list[tuple[str, plt.Figure]] = []
    if not isinstance(levene_results_raw, dict):
        raise ValueError("levene_results_raw must be a dict as returned by Levene_Test")

    first = None
    for k, v in levene_results_raw.items():
        first = v
        break
    if first is None:
        return []

    stat_arr = np.asarray(first.get("stat") if "stat" in first else first.get("statistic"))
    if stat_arr.ndim == 1:
        n_features = stat_arr.shape[0]
    else:
        raise ValueError("Expected 1D arrays of per-feature statistics in levene_results values")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    for comp, res_raw in levene_results_raw.items():
        stat_raw = np.asarray(res_raw.get("stat") if "stat" in res_raw else res_raw.get("statistic"))
        pval_raw = None
        for key in ("pvalue", "p_val", "pvalues", "p"):
            if key in res_raw:
                pval_raw = np.asarray(res_raw[key])
                break
        if pval_raw is None:
            pval_raw = np.full(stat_raw.shape, 1.0)

        stat_res = None
        pval_res = None
        if levene_results_resid is not None and comp in levene_results_resid:
            res_res = levene_results_resid[comp]
            stat_res = np.asarray(res_res.get("stat") if "stat" in res_res else res_res.get("statistic"))
            for key in ("pvalue", "p_val", "pvalues", "p"):
                if key in res_res:
                    pval_res = np.asarray(res_res[key])
                    break
            if pval_res is None:
                pval_res = np.full(stat_res.shape, 1.0)

        if stat_raw.shape[0] != n_features:
            continue

        ncols = 2 if stat_res is not None else 1
        fig, axes = plt.subplots(1, ncols, figsize=(max(6, n_features * 0.12) * ncols, 4), squeeze=False)
        x = np.arange(n_features)

        ax = axes[0, 0]
        bars = ax.bar(x, stat_raw, color="C0", alpha=0.8)
        sig_raw = pval_raw < alpha
        for xi, s in enumerate(sig_raw):
            if s:
                bars[xi].set_edgecolor("red")
                bars[xi].set_linewidth(1.5)
        ax.set_ylabel("Levene statistic")
        ax.set_title("Raw")
        # add significance markers above bars for p < alpha
        if sig_raw.any():
            ymax = np.nanmax(stat_raw)
            marker_y = ymax + 0.05 * max(1.0, abs(ymax))
            ax.plot(x[sig_raw], np.full(int(np.sum(sig_raw)), marker_y), marker="v", linestyle="", color="red", markersize=6)

        if stat_res is not None:
            ax2 = axes[0, 1]
            bars2 = ax2.bar(x, stat_res, color="C1", alpha=0.8)
            sig_res = pval_res < alpha
            for xi, s in enumerate(sig_res):
                if s:
                    bars2[xi].set_edgecolor("red")
                    bars2[xi].set_linewidth(1.5)
            ax2.set_title("Residual (covariates removed)")
            # add significance markers above bars for p < alpha
            if sig_res.any():
                ymax2 = np.nanmax(stat_res)
                marker_y2 = ymax2 + 0.05 * max(1.0, abs(ymax2))
                ax2.plot(x[sig_res], np.full(int(np.sum(sig_res)), marker_y2), marker="v", linestyle="", color="red", markersize=6)

        left_label, right_label = (comp[0], comp[1]) if isinstance(comp, (list, tuple)) and len(comp) >= 2 else (str(comp), "")
        fig.suptitle(f"Levene's test: {left_label} vs {right_label}")

        for ax_plot in axes[0, :ncols]:
            ax_plot.set_xticks(x)
            if n_features <= 50:
                ax_plot.set_xticklabels(feature_names, rotation=90)
            else:
                step = max(1, n_features // 40)
                labels = [feature_names[i] if (i % step == 0) else "" for i in range(n_features)]
                ax_plot.set_xticklabels(labels, rotation=90)

        ax.text(0.98, 0.95, f"n_significant_raw={int(np.sum(sig_raw))}", transform=ax.transAxes, ha="right", va="top")
        if stat_res is not None:
            ax2.text(0.98, 0.95, f"n_significant_resid={int(np.sum(sig_res))}", transform=ax2.transAxes, ha="right", va="top")

        figs.append((f"Levene: {left_label} vs {right_label}", fig))

    return None if rep is not None else figs


@rep_plot_wrapper
def Levenes_Test(
    levene_results: dict,
    feature_names: list | None = None,
    *,
    alpha: float = 0.05,
    show: bool = False,
    rep=None,
) -> list[tuple[str, plt.Figure]]:
    """
    Plot Levene's test results produced by DiagnosticFunctions.Levene_Test.

    Args:
        levene_results: dict keyed by comparison tuple (a,b) with values
            containing at least 'stat' and 'pvalue' arrays (per-feature).
        feature_names: optional list of feature names (length = n_features).
        alpha: significance threshold to highlight features.
        rep: optional report object used by the wrapper.

    Returns:
        list of (caption, Figure) tuples.
    """
    figs: list[tuple[str, plt.Figure]] = []

    # Basic validation
    if not isinstance(levene_results, dict):
        raise ValueError("levene_results must be a dict as returned by Levene_Test")

    # Determine number of features from first entry
    first = None
    for k, v in levene_results.items():
        first = v
        break
    if first is None:
        return []

    stat_arr = np.asarray(first.get("stat") if "stat" in first else first.get("statistic"))
    if stat_arr.ndim == 1:
        n_features = stat_arr.shape[0]
    else:
        raise ValueError("Expected 1D arrays of per-feature statistics in levene_results values")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # For each comparison create a bar plot of the test statistic and mark significant features
    for comp, res in levene_results.items():
        stat = np.asarray(res.get("stat") if "stat" in res else res.get("statistic"))
        # try a few common p-value keys
        pval = None
        for key in ("pvalue", "p_val", "pvalues", "p"):
            if key in res:
                pval = np.asarray(res[key])
                break
        if pval is None:
            # if absent, create non-significant mask
            pval = np.full(stat.shape, 1.0)

        if stat.shape[0] != n_features:
            continue

        fig, ax = plt.subplots(figsize=(max(6, n_features * 0.12), 4))
        x = np.arange(n_features)
        bars = ax.bar(x, stat, color="C0", alpha=0.8)

        # highlight significant
        sig = pval < alpha
        for xi, s in enumerate(sig):
            if s:
                bars[xi].set_edgecolor("red")
                bars[xi].set_linewidth(1.5)

        left_label, right_label = (comp[0], comp[1]) if isinstance(comp, (list, tuple)) and len(comp) >= 2 else (str(comp), "")
        ax.set_title(f"Levene's test: {left_label} vs {right_label}")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Levene statistic")
        ax.set_xticks(x)
        if n_features <= 50:
            ax.set_xticklabels(feature_names, rotation=90)
        else:
            step = max(1, n_features // 40)
            labels = [feature_names[i] if (i % step == 0) else "" for i in range(n_features)]
            ax.set_xticklabels(labels, rotation=90)

        # annotate number of significant features
        n_sig = int(np.sum(sig))
        ax.text(0.98, 0.95, f"n_significant={n_sig}", transform=ax.transAxes, ha="right", va="top")

        figs.append((f"Levene: {left_label} vs {right_label}", fig))

    return None if rep is not None else figs
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions ratio of variance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def variance_ratio_plot(variance_ratios:  np.ndarray, pair_labels: list,
                         df: None = None,rep = None,show: bool = False,caption: Optional[str] = None,) -> None:
    """
    Plots the explained variance ratio for each principal component as a bar plot.

    Args:
        variance_ratios (Sequence[float]): A sequence of explained variance ratios for each principal component.
        pair_labels (list): List of labels for each pair of batches corresponding to rows in variance_ratios.
        df (None, optional): Placeholder for potential future use. Defaults to None.
        rep (optional): Optional StatsReporter instance. Defaults to None.
        caption (Optional[str], optional): Optional caption for the plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
    Returns:

        None: Displays plot of vario per feature and a histogram of the values on different axes.
    Raises:
        ValueError: If variance_ratios is not a sequence of numbers.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import pandas as pd
    from matplotlib.figure import Figure

    # ---- Validation ----

    if not isinstance(variance_ratios, np.ndarray):
        raise ValueError("variance_ratios must be a NumPy array.")
    if variance_ratios.ndim != 2:
        raise ValueError("variance_ratios must be a 2D array (num_pairs x num_features).")
    if not isinstance(pair_labels, list) or len(pair_labels) != variance_ratios.shape[0]:
        raise ValueError("pair_labels must be a list with the same length as the number of rows in variance_ratios.")
    
    figs = []
    for i, label in enumerate(pair_labels):
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8], wspace=0.3)

        # Histogram (left)
        ax1 = fig.add_subplot(gs[0])
        ax1.hist(variance_ratios[i], bins=20, orientation="horizontal", color=[0.8, 0.2, 0.2])
        ax1.set_xlabel("Frequency")
        ax1.invert_xaxis()
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")

        # Bar plot (right)
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        indices = np.arange(variance_ratios.shape[1])
        ax2.plot(indices, variance_ratios[i], "b-")
        ax2.plot(indices, variance_ratios[i], "r.")

        # Labels and title
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Variance Ratio: $(\\sigma_1 / \\sigma_2)$")
        ax2.set_title(f"Feature wise ratio of variance between {label}")
        ax2.grid(True)

        caption_i = caption or f"Variance ratio — {pair_labels[i]}"

        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else:
            figs.append((caption_i, fig))
            if show:
                plt.show()

    return None if rep is not None else figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for dimensionality reduction ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
@rep_plot_wrapper
def PC_corr_plot(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    PC_correlations=False,
    *,
    show: bool = False,
    cluster_batches: bool = False
) -> list[tuple[str, plt.Figure]]:
    """
    Generate PCA diagnostic plots and return a list of (caption, fig).

    Args:
        PrincipleComponents (np.ndarray): 2D array of shape (n_samples, n_components) containing PCA scores.
        batch (np.ndarray or list): 1D array or list of batch labels corresponding to each sample.
        covariates (optional): Optional covariate data. Can be a DataFrame, structured array, or 2D array. Defaults to None.
        variable_names (optional): Optional list of variable names for
            covariates and batch. If supplied with covariates, the length should
            match the number of covariate columns. If the first element is
            `'batch'`, it is used as the batch column name.
        PC_correlations (optional): Optional output from `PC_Correlations` used
            to add correlation summary plots.
        show (bool, optional): Whether to display the figures interactively.
        cluster_batches (bool, optional): Whether to add batch-clustering
            overlays to the PCA plots.

    Returns:
        list[tuple[str, plt.Figure]]: Caption and figure pairs for the PCA
        diagnostic plots.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    figs = []

    # Basic validation
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray) or batch.ndim != 1:
        raise ValueError("batch must be a 1D numpy array.")
    if PrincipleComponents.shape[0] != len(batch):
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # Build DataFrame of PCs
    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names)

    # Decide batch column name (allow variable_names to include 'batch' as first element)
    batch_col_name = "batch"
    # If variable_names explicitly provided and starts with "batch", capture it as possible batch name
    if variable_names is not None and len(variable_names) > 0 and str(variable_names[0]).lower() == "batch":
        # use the exact provided first name (preserve case) as batch label
        batch_col_name = variable_names[0]
    

    df[batch_col_name] = batch
    # Change batch to numeric codes to prevent issues in plotting and calculating correlation:

    # --- Handle covariates robustly and determine covariate names ---
    cov_names = []
    cov_matrix = None  # numeric matrix (n_samples x n_covariates) used for correlations/plots

    if covariates is not None:
        # If DataFrame: use its column names
        if isinstance(covariates, pd.DataFrame):
            cov_matrix = covariates.values
            cov_names = list(map(str, covariates.columns))
        # Structured numpy array with named fields
        elif isinstance(covariates, np.ndarray) and covariates.dtype.names is not None:
            cov_names = [str(n) for n in covariates.dtype.names]
            # stack named columns into a 2D array
            cov_matrix = np.vstack([covariates[name] for name in cov_names]).T
        else:
            # array-like (convert to ndarray)
            cov_matrix = np.asarray(covariates)
            if cov_matrix.ndim != 2:
                raise ValueError("covariates must be 2D (samples x num_covariates).")
            if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
                raise ValueError("Number of rows in covariates must match number of samples.")

            # If variable_names provided: it may either be exactly covariate names,
            # or include 'batch' as first element followed by covariate names.
            if variable_names is not None:
                # If user included 'batch' as first element, strip it.
                if len(variable_names) == cov_matrix.shape[1] + 1 and str(variable_names[0]).lower() == "batch":
                    cov_names = [str(x) for x in variable_names[1:]]
                elif len(variable_names) == cov_matrix.shape[1]:
                    cov_names = [str(x) for x in variable_names]
                else:
                    # inconsistent lengths: raise helpful error
                    raise ValueError(
                        "variable_names length does not match number of covariates.\n"
                        f"covariates has {cov_matrix.shape[1]} columns, "
                        f"but variable_names has length {len(variable_names)}.\n"
                        "If you include 'batch' in variable_names, put it first (e.g. ['batch', 'Age', 'Sex'])."
                    )
            else:
                # No variable_names: create defaults
                cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        # Finally, assign covariate columns to df using cov_names
        # (if we reached here cov_matrix and cov_names should be set)
        if cov_matrix is None:
            raise ValueError("Unable to interpret covariates input; please supply a DataFrame, structured array, or 2D ndarray.")
        # Double-check shapes
        if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
            raise ValueError("Number of rows in covariates must match number of samples.")
        if cov_matrix.shape[1] != len(cov_names):
            # defensive: if Pandas columns count mismatch (shouldn't happen), regenerate names
            cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        for i, name in enumerate(cov_names):
            df[name] = cov_matrix[:, i]
    else:
        # No covariates present; ensure variable_names is either None or only contains 'batch'
        if variable_names is not None:
            if not (len(variable_names) == 1 and str(variable_names[0]).lower() == "batch"):
                raise ValueError("variable_names provided but covariates is None. Provide covariates or remove variable_names.")
        cov_names = []
    batch_numeric = pd.Categorical(batch).codes
    batch_col_code = f"{batch_col_name}_code"
    df[batch_col_code] = batch_numeric
    # --- 
    if PC_correlations:
        # create combined_data and combined_names in the same order used for corr matrix
        if cov_names:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1), df[cov_names].values))
            combined_names = PC_Names + [batch_col_code] + cov_names
        else:
            combined_data = np.column_stack((PrincipleComponents, df[batch_col_code].values.reshape(-1, 1)))
            combined_names = PC_Names + [batch_col_code]

        corr = np.corrcoef(combined_data.T)
        fig, ax = plt.subplots(figsize=(10, 8))
        # Check number of combined_names to adjust font size for readability:
        if len(combined_names) > 10:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=combined_names, yticklabels=combined_names, ax=ax)
        else: # Use just numbers if too many variables to avoid clutter:
            x_ticks = [f"{name}\n({i+1})" for i, name in enumerate(combined_names)]
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=x_ticks, yticklabels=x_ticks, ax=ax)       
        ax.set_title("Correlation Matrix of PCs, Batch, and Covariates")
        figs.append(("PCA correlation matrix", fig))
    
    # show only if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                # some backends may not support show on Figure objects; ignore safely
                pass

    return figs

@rep_plot_wrapper
def clustering_analysis_PCA(
    PrincipleComponents,
    batch,
    covariates=None,
    variable_names=None,
    PC_correlations=False,
    *,
    show: bool = False,
    cluster_batches: bool = False,
    UMAP_embedding=False,
    data = None):

    """
    Perform clustering analysis on PCA results and generate diagnostic plots.
    Args:
        PrincipleComponents (np.ndarray): 2D array of shape (n_samples, n_components) containing PCA scores.
        batch (np.ndarray or list): 1D array or list of batch labels corresponding to each sample.
        covariates (optional): Optional covariate data. Can be a DataFrame, structured array, or 2D array. Defaults to None.
        variable_names (optional): Optional list of variable names for covariates and batch. If covariates provided, should match number of covariate columns.
        If first element is 'batch', it will be used as batch column name. Defaults to None.
    Returns:
        List[Tuple[str, plt.Figure]]: A list of tuples containing captions and corresponding figures for the PCA diagnostic plots.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    figs = []

    # Basic validation
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray) or batch.ndim != 1:
        raise ValueError("batch must be a 1D numpy array.")
    if PrincipleComponents.shape[0] != len(batch):
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # Build DataFrame of PCs
    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names)

    # Decide batch column name (allow variable_names to include 'batch' as first element)
    batch_col_name = "batch"
    # If variable_names explicitly provided and starts with "batch", capture it as possible batch name
    if variable_names is not None and len(variable_names) > 0 and str(variable_names[0]).lower() == "batch":
        # use the exact provided first name (preserve case) as batch label
        batch_col_name = variable_names[0]
    

    df[batch_col_name] = batch
    # Change batch to numeric codes to prevent issues in plotting and calculating correlation:

    # --- Handle covariates robustly and determine covariate names ---
    cov_names = []
    cov_matrix = None  # numeric matrix (n_samples x n_covariates) used for correlations/plots

    if covariates is not None:
        # If DataFrame: use its column names
        if isinstance(covariates, pd.DataFrame):
            cov_matrix = covariates.values
            cov_names = list(map(str, covariates.columns))
        # Structured numpy array with named fields
        elif isinstance(covariates, np.ndarray) and covariates.dtype.names is not None:
            cov_names = [str(n) for n in covariates.dtype.names]
            # stack named columns into a 2D array
            cov_matrix = np.vstack([covariates[name] for name in cov_names]).T
        else:
            # array-like (convert to ndarray)
            cov_matrix = np.asarray(covariates)
            if cov_matrix.ndim != 2:
                raise ValueError("covariates must be 2D (samples x num_covariates).")
            if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
                raise ValueError("Number of rows in covariates must match number of samples.")

            # If variable_names provided: it may either be exactly covariate names,
            # or include 'batch' as first element followed by covariate names.
            if variable_names is not None:
                # If user included 'batch' as first element, strip it.
                if len(variable_names) == cov_matrix.shape[1] + 1 and str(variable_names[0]).lower() == "batch":
                    cov_names = [str(x) for x in variable_names[1:]]
                elif len(variable_names) == cov_matrix.shape[1]:
                    cov_names = [str(x) for x in variable_names]
                else:
                    # inconsistent lengths: raise helpful error
                    raise ValueError(
                        "variable_names length does not match number of covariates.\n"
                        f"covariates has {cov_matrix.shape[1]} columns, "
                        f"but variable_names has length {len(variable_names)}.\n"
                        "If you include 'batch' in variable_names, put it first (e.g. ['batch', 'Age', 'Sex'])."
                    )
            else:
                # No variable_names: create defaults
                cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        # Finally, assign covariate columns to df using cov_names
        # (if we reached here cov_matrix and cov_names should be set)
        if cov_matrix is None:
            raise ValueError("Unable to interpret covariates input; please supply a DataFrame, structured array, or 2D ndarray.")
        # Double-check shapes
        if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
            raise ValueError("Number of rows in covariates must match number of samples.")
        if cov_matrix.shape[1] != len(cov_names):
            # defensive: if Pandas columns count mismatch (shouldn't happen), regenerate names
            cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]

        for i, name in enumerate(cov_names):
            df[name] = cov_matrix[:, i]
    else:
        # No covariates present; ensure variable_names is either None or only contains 'batch'
        if variable_names is not None:
            if not (len(variable_names) == 1 and str(variable_names[0]).lower() == "batch"):
                raise ValueError("variable_names provided but covariates is None. Provide covariates or remove variable_names.")
        cov_names = []


    # --- 1) PCA scatter by batch ---
    figs = []
    fig1, ax = plt.subplots(figsize=(8, 6))
    for b in unique_batches:
        ax.scatter(df.loc[df[batch_col_name] == b, "PC1"], df.loc[df[batch_col_name] == b, "PC2"], label=f"{batch_col_name} {b}", alpha=0.7)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Scatter Plot by Batch")
    ax.legend()
    ax.grid(True)
    # put legend outside the plot area to avoid overlap with points, especially if many batches
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
    figs.append(("PCA scatter by batch", fig1))
    
    batch_numeric = pd.Categorical(batch).codes
    batch_col_code = f"{batch_col_name}_code"
    df[batch_col_code] = batch_numeric
    # Comment out for now as integrating t-SNE/UMAP as clustering here is better.
    # --- 2) PCA scatter by each covariate (if present) ---
    if cov_names:
        for name in cov_names:
            vals = df[name].values
            fig, ax = plt.subplots(figsize=(8, 6))
            # treat small-unique-count as categorical
            if len(np.unique(vals)) <= 10:
                for cat in np.unique(vals):
                    sel = df[name] == cat
                    ax.scatter(df.loc[sel, "PC1"], df.loc[sel, "PC2"], label=f"{name}={cat}", alpha=0.6)
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
            else:
                sc = ax.scatter(df["PC1"], df["PC2"], c=vals, cmap="viridis", alpha=0.7)
                plt.colorbar(sc, ax=ax, label=name)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"PCA Scatter Plot by {name}")
            # legend can be large; show only for categorical
            if len(np.unique(vals)) <= 20:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
            ax.grid(True)
            figs.append((f"PCA scatter by {name}", fig))
    
        # show only if requested
    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                # some backends may not support show on Figure objects; ignore safely
                pass
    return figs

@rep_plot_wrapper
def clustering_analysis_all(
    PrincipleComponents,
    data,
    batch,
    covariates=None,
    variable_names=None,
    show: bool = False,
    UMAP_embedding=True,
    UMAP_neighbors=15,
    UMAP_min_dist=0.1,
    UMAP_metric='euclidean',
    UMAP_tuning='auto', # Auto, batch or None: whether to automatically tune UMAP parameters based on data size, or allow user to specify tuning strategy
) -> list[tuple[str, plt.Figure]]:
    """
    Perform clustering diagnostics in PCA and optional UMAP space.

    Args:
        PrincipleComponents (np.ndarray): PCA score matrix with shape
            `(n_samples, n_components)`.
        data (np.ndarray): Original feature matrix used for the optional UMAP
            embedding.
        batch (np.ndarray or list): Batch labels for each sample.
        covariates (optional): Optional covariate data to color the plots.
        variable_names (optional): Optional names for the covariates and batch
            variables.
        show (bool, optional): Whether to display the figures interactively.
        UMAP_embedding (bool, optional): Whether to compute and plot a UMAP
            embedding of the raw data.
        UMAP_neighbors (int, optional): Number of neighbors for UMAP.
        UMAP_min_dist (float, optional): Minimum distance parameter for UMAP.
        UMAP_metric (str, optional): Distance metric passed to UMAP.

    Returns:
        list[tuple[str, plt.Figure]]: Caption and figure pairs for the
        generated PCA and optional UMAP plots.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap


    figs = []

    # Basic validation
    if not isinstance(PrincipleComponents, np.ndarray) or PrincipleComponents.ndim != 2:
        raise ValueError("PrincipleComponents must be a 2D numpy array (samples x components).")
    if not isinstance(batch, np.ndarray) or batch.ndim != 1:
        raise ValueError("batch must be a 1D numpy array.")
    if PrincipleComponents.shape[0] != len(batch):
        raise ValueError("Number of samples in PrincipleComponents and batch must match.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # Build DataFrame of PCs
    PC_Names = [f"PC{i+1}" for i in range(PrincipleComponents.shape[1])]
    df = pd.DataFrame(PrincipleComponents, columns=PC_Names)

    # Decide batch column name (allow variable_names to include 'batch' as first element)
    batch_col_name = "batch"
    # If variable_names explicitly provided and starts with "batch", capture it as possible batch name
    if variable_names is not None and len(variable_names) > 0 and str(variable_names[0]).lower() == "batch":
        # use the exact provided first name (preserve case) as batch label
        batch_col_name = variable_names[0]
    
    df[batch_col_name] = batch
    # Change batch to numeric codes to prevent issues in plotting and calculating correlation:

    # --- Handle covariates robustly and determine covariate names ---
    cov_names = []
    cov_matrix = None  # numeric matrix (n_samples x n_covariates) used for correlations/plots
    # If covariates given as matrix, dataframe or structured array, handle accordingly to extract covariate names and matrix
    if covariates is not None:
        # If DataFrame: use its column names
        if isinstance(covariates, pd.DataFrame):
            cov_matrix = covariates.values
            cov_names = list(map(str, covariates.columns))
        # Structured numpy array with named fields
        elif isinstance(covariates, np.ndarray) and covariates.dtype.names is not None:
            cov_names = [str(n) for n in covariates.dtype.names]
            # stack named columns into a 2D array
            cov_matrix = np.vstack([covariates[name] for name in cov_names]).T
        else:
            # array-like (convert to ndarray)
            cov_matrix = np.asarray(covariates)
            if cov_matrix.shape[0] != PrincipleComponents.shape[0]:
                raise ValueError("Number of rows in covariates must match number of samples.")

            # If variable_names provided: it may either be exactly covariate names,
            # or include 'batch' as first element followed by covariate names.
            if variable_names is not None:
                # If user included 'batch' as first element, strip it.
                if len(variable_names) == cov_matrix.shape[1] + 1 and str(variable_names[0]).lower() == "batch":
                    cov_names = [str(x) for x in variable_names[1:]]
                elif len(variable_names) == cov_matrix.shape[1]:
                    cov_names = [str(x) for x in variable_names]
                else:
                    # inconsistent lengths: raise helpful error
                    raise ValueError(
                        "variable_names length does not match number of covariates.\n"
                        f"covariates has {cov_matrix.shape[1]} columns, "
                        f"but variable_names has length {len(variable_names)}.\n"
                        "If you include 'batch' in variable_names, put it first (e.g. ['batch', 'cov1', 'cov2']).\n")
    # Overall, returned covariate dataframe will have columns named according to cov_names, and batch column named according to batch_col_name, regardless of input format.
            else:
                # No variable_names: create defaults
                cov_names = [f"Covariate{i+1}" for i in range(cov_matrix.shape[1])]
                cov_df = pd.DataFrame(cov_matrix, columns=cov_names)
                df = pd.concat([df, cov_df], axis=1)

    # Plot PCA scatter by batch and covariates as in previous function, create subplots for each covariate and batch
    # Here we will then display the low dimensional UMAP embedding of the data coloured by batch and covariates next to PCA for comparisson
    if data is not None and len(data) > 10000:
        print("Data has more than 10,000 samples; This may make UMAP very slow and memory intensive.")

    # Perform UMAP embedding if data is provided and not too large (to avoid long runtime and memory issues)

    if data is not None:
        # If UMAP_tuning is set to 'auto', we can adjust UMAP parameters based on data size (e.g. reduce n_neighbors for larger datasets)
        n_samples = data.shape[0]

        if UMAP_tuning == "auto":
            if n_samples < 1000:
                UMAP_neighbors = 15
                UMAP_min_dist = 0.1
            elif n_samples < 5000:
                UMAP_neighbors = 30
                UMAP_min_dist = 0.15
            elif n_samples < 10000:
                UMAP_neighbors = 50
                UMAP_min_dist = 0.15
            else:
                UMAP_neighbors = round(max(50, n_samples // 250))  # heuristic: 0.4% of samples, but at least 50
                UMAP_min_dist = 0.2

            UMAP_neighbors = min(UMAP_neighbors, max(2, n_samples - 1))

        if UMAP_tuning == "batch":
            # Define set of neighbours to try, test silhouette score for each, and pick best. This is more computationally intensive but can yield better results.
            from sklearn.metrics import silhouette_score
            best_score = -1
            param_grid = [
            (n, d)
            for n in [10, 15, 30, 50]
            for d in [0.01, 0.1, 0.3]
            ]
            best_score = -1
            best_params = None
            best_embedding = None
            # If dataset large, subsample for tuning to speed up:
            if n_samples > 5000:
                idx = np.random.choice(n_samples, size=5000, replace=False)
                data_sub = data[idx]
                batch_sub = batch[idx]
            else:
                data_sub = data
                batch_sub = batch

            for n, d in param_grid:
                reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=42)
                emb = reducer.fit_transform(data_sub)

                score = silhouette_score(emb, batch_sub)

                if score > best_score:
                    best_score = score
                    best_params = (n, d)
                    best_embedding = emb

            UMAP_neighbors, UMAP_min_dist = best_params

        
        reducer = umap.UMAP(n_neighbors=UMAP_neighbors, min_dist=UMAP_min_dist, metric=UMAP_metric, random_state=42)
        embedding = reducer.fit_transform(data)

        df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
        df_umap[batch_col_name] = batch

        if cov_names:
            for i, name in enumerate(cov_names):
                df_umap[name] = cov_matrix[:, i]
                df[name] = cov_matrix[:, i]
        # Plot UMAP on right, PCA on left for batch and covariates:

        fig_umap, ax = plt.subplots(1, 2, figsize=(16, 6))
        sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue=batch_col_name, palette="tab10", alpha=0.7, ax=ax[0])
        ax[0].set_title("UMAP Embedding Colored by Batch")
        ax[0].legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
        figs.append(("UMAP embedding colored by batch", fig_umap))
        # Plot PCA on left for batch
        for b in unique_batches:
            ax[1].scatter(df.loc[df[batch_col_name] == b, "PC1"], df.loc[df[batch_col_name] == b, "PC2"], label=f"{batch_col_name} {b}", alpha=0.7)
        
        ax[1].set_xlabel("Principal Component 1")
        ax[1].set_ylabel("Principal Component 2")
        ax[1].set_title("PCA Scatter Plot by Batch")
        ax[1].legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
        ax[1].grid(True)
        figs.append(("UMAP and PCA embedding by batch", fig_umap))

        # Plot UMAP colored by covariates if present
        if cov_names:
            for name in cov_names:
                fig_cov, ax_cov = plt.subplots(1, 2, figsize=(16, 6))
                sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue=name, palette="viridis", alpha=0.7, ax=ax_cov[0])
                ax_cov[0].set_title(f"UMAP Embedding Colored by {name}")
                if len(np.unique(df_umap[name])) <= 20:
                    ax_cov[0].legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
                else:
                    ax_cov[0].legend().remove()
                    plt.colorbar(ax_cov[0].collections[0], ax=ax_cov[0], label=name)
                # Plot PCA on left for covariate
                vals = df[name].values
                if len(np.unique(vals)) <= 10:
                    for cat in np.unique(vals):
                        sel = df[name] == cat
                        ax_cov[1].scatter(df.loc[sel, "PC1"], df.loc[sel, "PC2"], label=f"{name}={cat}", alpha=0.6)
                    ax_cov[1].legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
                else:
                    sc = ax_cov[1].scatter(df["PC1"], df["PC2"], c=vals, cmap="viridis", alpha=0.7)
                    plt.colorbar(sc, ax=ax_cov[1], label=name)
                ax_cov[1].set_xlabel("Principal Component 1")
                ax_cov[1].set_ylabel("Principal Component 2")
                ax_cov[1].set_title(f"PCA Scatter Plot by {name}")
                if len(np.unique(vals)) <= 20:
                    ax_cov[1].legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
                ax_cov[1].grid(True)
                # Add text to x_axis to indicate UMAP parameters used for clarity
                ax_cov[0].set_xlabel(f"UMAP1 (n_neighbors={UMAP_neighbors}, min_dist={UMAP_min_dist})")
                ax_cov[0].set_ylabel("UMAP2")

                figs.append((f"UMAP and PCA embedding by {name}", fig_cov))
                # Add the n_neighbors and min_dist parameters to the caption for clarity on what was used for UMAP                
    # Create a string summary UMAP parametrs and have the function return it along with the figures, so it can be logged to the report text as well.
    umap_param_summary = f"UMAP parameters: n_neighbors={UMAP_neighbors}, min_dist={UMAP_min_dist}, metric={UMAP_metric}, tuning={UMAP_tuning}"

    if show:
        for _, f in figs:
            try:
                f.show()
            except Exception:
                # some backends may not support show on Figure objects; ignore safely
                pass
    return figs

@rep_plot_wrapper
def clustering_analysis_UMAP(data, batch,
                             covariates=None,
                             variable_names=None,
                             rep=None):
    """Perform UMAP dimensionality reduction and plot the embedding colored by batch and covariates.
    
    Args:
        data: 2D array-like (n_samples x n_features) input data for U
        batch: 1D array-like (n_samples,) batch labels for each sample.
        covariates: Optional 2D array-like (n_samples x n_covariates
        variable_names: Optional list of covariate names (if covariates provided).
        rep: Optional report object with rep.log_plot(plt, caption=...) method for logging"""
    
    # Check length of data and batch
    if len(data) != len(batch):
        raise ValueError("Length of data and batch must match.")
    if covariates is not None and len(data) != len(covariates):
        raise ValueError("Length of data and covariates must match.")
    
    # Check data size, if too large, log a warning and skip UMAP to avoid long runtime
    if len(data) > 100000:
        if rep is not None:
            rep.log_text(f"Data has {len(data)} samples, which may lead to long UMAP runtime. Skipping UMAP embedding.")
        return []
    else:
        import umap
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(data)

        df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
        df["batch"] = batch
        if covariates is not None:
            for i in range(covariates.shape[1]):
                df[f"Covariate{i+1}"] = covariates[:, i]
        figs = []

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="batch", palette="tab10", alpha=0.7, ax=ax)
        ax.set_title("UMAP Embedding Colored by Batch")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
        figs.append(("UMAP embedding colored by batch", fig))
        if rep is not None:
            rep.log_plot(fig, "UMAP embedding colored by batch")
            plt.close(fig)
        
        # Check covarariates and plot if present:
        if covariates is not None:
            for i in range(covariates.shape[1]):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue=f"Covariate{i+1}", palette="viridis", alpha=0.7, ax=ax)
                ax.set_title(f"UMAP Embedding Colored by Covariate {i+1}")
                if len(np.unique(covariates[:, i])) <= 20:
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", frameon=True)
                else:
                    ax.legend().remove()
                    plt.colorbar(ax.collections[0], ax=ax, label=f"Covariate {i+1}")
                figs.append((f"UMAP embedding colored by covariate {i+1}", fig))
                if rep is not None:
                    rep.log_plot(fig, f"UMAP embedding colored by covariate {i+1}")
                    plt.close(fig)

    # Check if figs is empty (e.g. if data too large and UMAP skipped), and log a message if so
    if not figs and rep is not None:
        rep.log_text("No UMAP plots generated (data may have been too large).")
    else:  
        return figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""Plotting per batch variance across first N PCs"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def plot_eigen_spectra_and_cumulative(
    score: np.ndarray,
    batch: np.ndarray,
    rep,
    max_components: int = 50,
    caption_prefix: str = "PC spectrum",
) -> dict[str, Any]:
    """
    Compute per-batch variance along PCs (scree / cumulative) and log plots to the report.

    Args:
        score: (n_samples, n_pcs) PCA score matrix returned by your PCA routine.
        batch: (n_samples,) batch labels (numeric or strings).
        rep: report object that has rep.log_plot(plt, caption=...) and rep.log_text(...)
        max_components: maximum number of PCs to visualise (keeps plots cheap).
        caption_prefix: prefix for plot captions.

    Returns:
        dict[str, Any]: A dictionary containing per-batch variance curves,
        per-batch fraction-of-variance curves, and the number of principal
        components used.
    """
    # Basic checks
    if score.ndim != 2:
        raise ValueError("score must be a 2D array (n_samples x n_pcs)")

    n, n_pcs = score.shape
    k = min(n_pcs, max_components)
    if k < 2:
        rep.log_text("Not enough PCs to produce spectrum plots (k < 2).")
        return {}

    unique_batches = np.unique(batch)
    per_batch_variance = {}
    per_batch_frac_var = {}

    # Compute per-batch variance along each PC (diagonal / axis variances)
    for b in unique_batches:
        idx = np.where(batch == b)[0]
        if len(idx) < 2:
            # Variance undefined or uninformative
            rep.log_text(f"Batch {b}: too few samples ({len(idx)}) to estimate per-PC variance reliably.")
            per_batch_variance[b] = np.full(k, np.nan)
            per_batch_frac_var[b] = np.full(k, np.nan)
            continue
        scores_b = score[idx, :k]
        # ddof=1 to mirror sample covariance convention
        var_b = np.nanvar(scores_b, axis=0, ddof=1)
        total_var_b = np.nansum(var_b)
        if total_var_b == 0 or np.isnan(total_var_b):
            frac = np.full_like(var_b, np.nan)
        else:
            frac = var_b / total_var_b
        per_batch_variance[b] = var_b
        per_batch_frac_var[b] = frac

    results = {
        "pcs_used": k,
        "per_batch_variance": per_batch_variance,
        "per_batch_frac_var": per_batch_frac_var,
    }

        # --- One figure with two horizontally aligned subplots ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4.5),
        sharex=True
    )

    # --- Scree plot (fraction of variance per PC) ---
    for b in unique_batches:
        frac = per_batch_frac_var[b]
        if np.all(np.isnan(frac)):
            continue
        ax1.plot(
            np.arange(1, k + 1),
            frac,
            marker="o",
            label=f"Batch {b}",
            alpha=0.8
        )

    ax1.set_xlabel("PC index")
    ax1.set_ylabel("Fraction variance (per-batch)")
    ax1.set_title("Per-batch scree plot")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, fontsize="small")

    # --- Cumulative variance plot (per-batch) ---
    for b in unique_batches:
        frac = per_batch_frac_var[b]
        if np.all(np.isnan(frac)):
            continue
        cum = np.nancumsum(frac)
        ax2.plot(
            np.arange(1, k + 1),
            cum,
            marker="o",
            label=f"Batch {b}",
            alpha=0.8
        )

    ax2.set_xlabel("PC index")
    ax2.set_ylabel("Cumulative fraction variance")
    ax2.set_title("Per-batch cumulative variance explained")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    rep.log_plot(
        plt,
        caption=f"{caption_prefix}: Per-batch scree + cumulative variance explained"
    )
    plt.close()

    # short textual summary (helpful for users)
    rep.log_text(
        f"{caption_prefix}: Used first {k} PCs. "
        "Scree and cumulative plots show how variance is distributed across PCs per batch."
    )

    return results
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting covariance Frobenius norm results ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""

def plot_covariance_frobenius(
    score: np.ndarray,
    batch: np.ndarray,
    rep,
    max_components: int = 50,
    normalize: bool = True,
    caption_prefix: str = "Covariance comparison (PC space)",
) -> dict[str, Any]:
    """
    Compute pairwise Frobenius norms of covariance differences between batches

    Args:
        score: (n_samples, n_pcs) whole data matrix in real space
        batch: (n_samples,) batch labels.
        rep: report object (must support rep.log_plot and rep.log_text).
        normalize: if True, divide pairwise norms by Frobenius norm of pooled covariance.
        caption_prefix: prefix for plot captions.

    Returns:
        dict[str, Any]: A dictionary containing the batch covariance matrices,
        pairwise Frobenius distances, optional normalized distances, and the
        number of principal components used.
    """
    import pandas as pd  

    if score.ndim != 2:
        raise ValueError("score must be a 2D array (n_samples x n_pcs)")

    n, n_pcs = score.shape
    k = min(n_pcs, max_components)
    if k < 1:
        rep.log_text("Not enough PCs to compute covariance diagnostics.")
        return {}

    unique_batches = np.unique(batch)
    G = len(unique_batches)

    cov_matrices = {}
    valid_batches = []
    for b in unique_batches:
        idx = np.where(batch == b)[0]
        if len(idx) < 2:
            rep.log_text(f"Batch {b}: too few samples ({len(idx)}) to estimate covariance reliably.")
            # store NaN matrix to preserve indexing
            cov_matrices[b] = np.full((k, k), np.nan)
            continue
        scores_b = score[idx, :k]
        cov_b = np.cov(scores_b, rowvar=False, ddof=1)  # k x k
        cov_matrices[b] = cov_b
        valid_batches.append(b)

    # pooled covariance (using all samples)
    pooled_scores = score[:, :k]
    pooled_cov = np.cov(pooled_scores, rowvar=False, ddof=1)
    pooled_frob = np.linalg.norm(pooled_cov, ord='fro')

    # pairwise frobenius norms
    pairwise = np.full((G, G), np.nan)
    batch_list = list(unique_batches)
    for i, bi in enumerate(batch_list):
        for j, bj in enumerate(batch_list):
            Ci = cov_matrices[bi]
            Cj = cov_matrices[bj]
            if np.isnan(Ci).all() or np.isnan(Cj).all():
                pairwise[i, j] = np.nan
            else:
                diff = Ci - Cj
                pairwise[i, j] = np.linalg.norm(diff, ord='fro')

    # normalized version
    pairwise_norm = pairwise.copy()
    if normalize and pooled_frob > 0 and not np.isnan(pooled_frob):
        pairwise_norm = pairwise / pooled_frob

    # Turn results into a pandas DataFrame for nicer human-readable output if desired
    try:
        df_pairwise = pd.DataFrame(pairwise, index=batch_list, columns=batch_list)
        df_pairwise_norm = pd.DataFrame(pairwise_norm, index=batch_list, columns=batch_list)
    except Exception:
        df_pairwise = pairwise
        df_pairwise_norm = pairwise_norm

    results = {
        "Number of features": k,
        "cov_matrices": cov_matrices,
        "pairwise_frobenius": df_pairwise,
        "pairwise_frobenius_normalized": df_pairwise_norm,
        "pooled_frobenius": pooled_frob,
    }
        # --- One figure with two horizontally aligned subplots ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 2]}
    )

    # --- Heatmap of normalized pairwise differences ---
    im = ax1.imshow(pairwise_norm, interpolation="nearest", aspect="auto")

    # Add value to each cell
    for i in range(G):
        for j in range(G):
            val = pairwise_norm[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(np.arange(G))
    ax1.set_xticklabels(batch_list, rotation=45, ha="right")
    ax1.set_yticks(np.arange(G))
    ax1.set_yticklabels(batch_list)
    ax1.set_title("Normalized Frobenius norm of covariance differences (original feature space)")

    # --- Bar plot: max per batch ---
    max_per_batch = np.nanmax(pairwise_norm, axis=1)
    ax2.bar(range(G), max_per_batch)
    ax2.set_xticks(range(G))
    ax2.set_xticklabels(batch_list, rotation=45, ha="right")
    ax2.set_ylabel("Max normalized Frobenius distance")
    ax2.set_title("Max covariance difference per batch (normalized)")

    plt.tight_layout()
    if rep is not None:
        rep.log_plot(plt, caption=f"{caption_prefix}: Pairwise normalized Frobenius norms (heatmap)")
        plt.close()
    else:
        plt.show()
        

    # --- Bar plot showing max difference per batch (summary) ---
    # compute max distance from each batch to others
    if rep is not None:
        rep.log_text(
            f"{caption_prefix}: used first {k} features. Pooled Frobenius norm = {pooled_frob:.4g}.\n "
            "Pairwise normalized Frobenius matrix and per-batch summaries added to report."
    )
    return results

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mahalanobis distance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def mahalanobis_distance_plot(results: dict,
                               rep=None,
                                 annotate: bool = True,
                                   figsize=(14,5),
                                     cmap="viridis",
                                       show: bool = False):

    """
    Plot Mahalanobis distances from (...) all on ONE figure:
      - Heatmap of pairwise RAW distances
      - Heatmap of pairwise RESIDUAL distances (if available)
      - Bar chart of centroid-to-global distances (raw vs residual)

    Args:
        results (dict): Output from MahalanobisDistance(...)
        annotate (bool): Write numeric values inside heatmap cells/bars.
        figsize (tuple): Matplotlib figure size.
        cmap (str): Colormap for heatmaps.
        show (bool): If True, plt.show(); otherwise just return (fig, axes).

    Returns:
        (fig, axes): The matplotlib Figure and dict of axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ---- Validation ----
    if not isinstance(results, dict):
        raise ValueError("results must be a dict produced by MahalanobisDistance(...)")

    req = ["pairwise_raw", "centroid_raw", "batches"]
    for k in req:
        if k not in results:
            raise ValueError(f"Missing required key '{k}' in results.")
    # Optional
    pairwise_resid = results.get("pairwise_resid", None)
    centroid_resid = results.get("centroid_resid", None)

    pairwise_raw = results["pairwise_raw"]
    centroid_raw = results["centroid_raw"]
    batches = results["batches"]
    if isinstance(batches, np.ndarray):
        batches = batches.tolist()
    n = len(batches)
    if n < 2:
        raise ValueError("Need at least two batches to plot distances.")

    # ---- Helpers ----
    def build_matrix(pw: dict) -> np.ndarray:
        M = np.full((n, n), np.nan, dtype=float)
        # Fill symmetric entries from pairwise dict keys (b1, b2)
        # Diagonal defined as 0 (distance of a batch to itself)
        for i in range(n):
            M[i, i] = 0.0
        if pw is None:
            return M
        for (b1, b2), d in pw.items():
            i = batches.index(b1)
            j = batches.index(b2)
            M[i, j] = d
            M[j, i] = d
        return M

    def centroid_array(cent: dict) -> np.ndarray:
        if cent is None:
            return None
        # keys like (b, 'global')
        return np.array([float(cent[(b, "global")]) for b in batches], dtype=float)

    def annotate_heatmap(ax, M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    # ---- Data prep ----
    M_raw = build_matrix(pairwise_raw)
    M_resid = build_matrix(pairwise_resid) if pairwise_resid is not None else None

    # Use a shared color scale across heatmaps for fair comparison
    vmax_candidates = [np.nanmax(M_raw)]
    if M_resid is not None:
        vmax_candidates.append(np.nanmax(M_resid))
    vmax = np.nanmax(vmax_candidates)
    vmin = 0.0

    c_raw = centroid_array(centroid_raw)
    c_res = centroid_array(centroid_resid) if centroid_resid is not None else None

    # ---- Figure layout ----
    # If residuals exist: 3 panels (raw, resid, bars)
    # Else: 2 panels (raw, bars)
    has_resid = (pairwise_resid is not None) and (centroid_resid is not None)
    num_cols = 3 if has_resid else 2

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, num_cols, figure=fig, width_ratios=[1, 1, 0.9] if has_resid else [1, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    im_raw = ax_raw.imshow(M_raw, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_raw.set_title("Pairwise Mahalanobis (Raw)")
    ax_raw.set_xticks(range(n))
    ax_raw.set_yticks(range(n))
    ax_raw.set_xticklabels(batches, rotation=45, ha="right")
    ax_raw.set_yticklabels(batches)
    ax_raw.set_xlabel("Batch")
    ax_raw.set_ylabel("Batch")
    if annotate:
        annotate_heatmap(ax_raw,M_raw)

    if has_resid:
        ax_resid = fig.add_subplot(gs[0, 1])
        im_resid = ax_resid.imshow(M_resid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_resid.set_title("Pairwise Mahalanobis (Residual)")
        ax_resid.set_xticks(range(n))
        ax_resid.set_yticks(range(n))
        ax_resid.set_xticklabels(batches, rotation=45, ha="right")
        ax_resid.set_yticklabels(batches)
        ax_resid.set_xlabel("Batch")
        ax_resid.set_ylabel("Batch")
        if annotate:
            annotate_heatmap(ax_resid,M_resid)

        # One colorbar shared by both heatmaps
        cbar = fig.colorbar(im_resid, ax=ax_raw, fraction=0.046, pad=0.2,orientation="horizontal",location="top")
        cbar = fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.2,orientation="horizontal",location="top")

        cbar.set_label("Mahalanobis distance")
    else:
        # Single colorbar for the single heatmap
        cbar = fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
        cbar.set_label("Mahalanobis distance")

    # ---- Bar chart of centroid-to-global ----
    ax_bar = fig.add_subplot(gs[0, -1])
    x = np.arange(n)
    if c_res is None:
        # Only raw bars
        width = 0.6
        bars = ax_bar.bar(x, c_raw, width, label="Raw")
        ax_bar.set_title("Centroid → Global")
        if annotate:
            for b in bars:
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()
    else:
        width = 0.38
        bars_raw = ax_bar.bar(x - width/2, c_raw, width, label="Raw")
        bars_res = ax_bar.bar(x + width/2, c_res, width, label="Residual")
        ax_bar.set_title("Centroid → Global (Raw vs Residual)")
        if annotate:
            for b in list(bars_raw) + list(bars_res):
                ax_bar.text(b.get_x() + b.get_width()/2., b.get_height(),
                            f"{b.get_height():.2f}",
                            ha='center', va='bottom', fontsize=8)
        ax_bar.legend()

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(batches, rotation=45, ha="right")
    ax_bar.set_ylabel("Mahalanobis distance")
    ax_bar.set_xlabel("Batch")

    axes = {"heatmap_raw": ax_raw, "bars": ax_bar}
    if has_resid:
        axes["heatmap_resid"] = ax_resid
    #fig.tight_layout()
    if rep is not None:
        rep.log_plot(fig, "Mahalanobis distances (raw vs residual)")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig, axes
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Mixed effects model ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
# @rep_plot_wrapper To be added at a later date (currently plotted in the report directly as we haven't decided on a standard plot format)
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for Two-sample Kolmogorov-Smirnov test ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""


from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def KS_plot(
    ks_results: dict,
    feature_names: list = None,
    rep=None,                      # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False
) -> Any:
    """
    Plot detailed KS-test results for each comparison.

    Args:
        ks_results (dict): Output from `KS_Test`, keyed by tuples such as
            `(batch, "overall")` or `(batch1, batch2)`.
        feature_names (list, optional): Optional feature names for plot labels.
        rep (optional): Report object used to log plots instead of returning
            them.
        caption (str, optional): Optional caption prefix.
        show (bool, optional): Whether to display the generated figures.

    Returns:
        Any: The report object if `rep` is provided; otherwise a list of
        `(caption, figure)` tuples.
    """



    def _bh_fdr(pvals):
        """Benjamini-Hochberg adjustment with NaNs preserved."""
        p = np.asarray(pvals, dtype=float)
        out = np.full_like(p, np.nan, dtype=float)
        mask = ~np.isnan(p)
        p_nonan = p[mask]
        m = p_nonan.size
        if m == 0:
            return out

        order = np.argsort(p_nonan)
        ranked = p_nonan[order]

        adj = np.empty(m, dtype=float)
        cummin = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            val = ranked[i] * m / rank
            cummin = min(cummin, val)
            adj[i] = cummin

        adj_back = np.empty(m, dtype=float)
        adj_back[order] = np.minimum(adj, 1.0)
        out[mask] = adj_back
        return out

    def _ks_critical_d(n1, n2, alpha=0.05):
        """
        Approximate two-sample KS critical value for rejecting H0.
        D_crit = c(alpha) * sqrt((n1+n2)/(n1*n2))
        where c(alpha) = sqrt(-0.5 * ln(alpha/2))
        """
        n1 = np.asarray(n1, dtype=float)
        n2 = np.asarray(n2, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_alpha = np.sqrt(-0.5 * np.log(alpha / 2.0))
            denom = np.sqrt((n1 * n2) / (n1 + n2))
            dcrit = c_alpha / denom
        dcrit[(n1 <= 0) | (n2 <= 0)] = np.nan
        return dcrit

    def _prepare_feature_names(n_features):
        if feature_names is None:
            return np.array([f"feature_{i + 1}" for i in range(n_features)], dtype=object)
        if len(feature_names) != n_features:
            raise ValueError("feature_names length must match number of features in ks_results.")
        return np.asarray(feature_names, dtype=object)

    def _comparison_label(key):
        if key[1] == "overall":
            return f"Batch {key[0]} vs overall"
        return f"Batch {key[0]} vs batch {key[1]}"

    def _plot_pvals_ordered(result, feat_names, title):
        p = np.asarray(result["p_value"], dtype=float)
        fdr = result.get("p_value_fdr", None)
        if fdr is not None:
            fdr = np.asarray(fdr, dtype=float)

        valid = ~np.isnan(p)
        if not np.any(valid):
            return None

        p_valid = p[valid]
        fdr_valid = fdr[valid] if fdr is not None else None
        feat_valid = feat_names[valid]

        order = np.argsort(p_valid)
        p_sorted = p_valid[order]
        feat_sorted = feat_valid[order]
        fdr_sorted = fdr_valid[order] if fdr_valid is not None else None

        x = np.arange(p_sorted.size)

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(x, -np.log10(np.clip(p_sorted, 1e-300, 1.0)),
                marker="o", linestyle="-", linewidth=1.2, markersize=4,
                label="Raw p-value")

        if fdr_sorted is not None:
            ax.plot(x, -np.log10(np.clip(fdr_sorted, 1e-300, 1.0)),
                    marker="s", linestyle="--", linewidth=1.2, markersize=4,
                    label="BH-adjusted p-value")

        ax.axhline(-np.log10(0.05), color="red", linestyle=":", linewidth=1.2, label="0.05 threshold")
        ax.set_xlabel("Features ordered by raw p-value")
        ax.set_ylabel(r"$-\log_{10}(p)$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Keep labels readable without overcrowding.
        if p_sorted.size <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(feat_sorted, rotation=90, fontsize=8)
        else:
            step = max(1, p_sorted.size // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(feat_sorted[::step], rotation=90, fontsize=8)

        fig.tight_layout()
        return fig

    def _plot_pvals_feature_order(result, feat_names, title):
        p = np.asarray(result["p_value"], dtype=float)
        fdr = result.get("p_value_fdr", None)
        if fdr is not None:
            fdr = np.asarray(fdr, dtype=float)

        valid = ~np.isnan(p)
        if not np.any(valid):
            return None

        x = np.arange(p.size)
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(x[valid], -np.log10(np.clip(p[valid], 1e-300, 1.0)),
                marker="o", linestyle="", markersize=5, label="Raw p-value")

        if fdr is not None:
            fdr_valid = ~np.isnan(fdr)
            ax.plot(x[fdr_valid], -np.log10(np.clip(fdr[fdr_valid], 1e-300, 1.0)),
                    marker="s", linestyle="", markersize=5, label="BH-adjusted p-value")

        ax.axhline(-np.log10(0.05), color="red", linestyle=":", linewidth=1.2, label="0.05 threshold")
        ax.set_xlabel("Feature index (original order)")
        ax.set_ylabel(r"$-\log_{10}(p)$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if p.size <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(feat_names, rotation=90, fontsize=8)
        else:
            step = max(1, p.size // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(feat_names[::step], rotation=90, fontsize=8)

        fig.tight_layout()
        return fig

    def _plot_distance(result, feat_names, title):
        d = np.asarray(result["statistic"], dtype=float)
        n1 = np.asarray(result["n_group1"], dtype=float)
        n2 = np.asarray(result["n_group2"], dtype=float)
        dcrit = _ks_critical_d(n1, n2, alpha=0.05)

        valid = ~np.isnan(d)
        if not np.any(valid):
            return None

        x = np.arange(d.size)
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(x[valid], d[valid],
                marker="o", linestyle="", markersize=5, label="Observed KS D")
        ax.plot(x[valid], dcrit[valid],
                marker="s", linestyle="", markersize=5, label="Critical D to reject H0")

        ax.set_xlabel("Feature index (original order)")
        ax.set_ylabel("KS D statistic")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if d.size <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(feat_names, rotation=90, fontsize=8)
        else:
            step = max(1, d.size // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(feat_names[::step], rotation=90, fontsize=8)

        fig.tight_layout()
        return fig

    # -------------------------
    # Main plotting logic
    # -------------------------
    figs = []

    tuple_keys = [
        k for k in ks_results.keys()
        if isinstance(k, tuple) and len(k) == 2
    ]

    if not tuple_keys:
        raise ValueError("ks_results does not contain any comparison keys of the form (batch, 'overall') or (batch1, batch2).")

    for key in tuple_keys:
        res = ks_results[key]
        pvals = np.asarray(res["p_value"], dtype=float)
        n_features = pvals.size
        feat_names = _prepare_feature_names(n_features)

        label = _comparison_label(key)

        fig1 = _plot_pvals_ordered(
            res,
            feat_names,
            title=f"KS test p-values ordered by raw p-value: {label}"
        )
        if fig1 is not None:
            figs.append((caption or f"KS ordered p-values: {label}", fig1))

        fig2 = _plot_pvals_feature_order(
            res,
            feat_names,
            title=f"KS test p-values in feature order: {label}"
        )
        if fig2 is not None:
            figs.append((caption or f"KS feature-order p-values: {label}", fig2))

        fig3 = _plot_distance(
            res,
            feat_names,
            title=f"KS distance and rejection threshold: {label}"
        )
        if fig3 is not None:
            figs.append((caption or f"KS distance plot: {label}", fig3))

    # Keep the package-style behavior: log figures to rep when provided.
    if rep is not None:
        for cap, fig in figs:
            rep.log_plot(fig, cap)
            plt.close(fig)
        return rep

    if show:
        for _, fig in figs:
            fig.show()

    return figs

"""
plotting_harmonisation.py
=========================
Plotting functions for longitudinal neuroimaging harmonisation metrics.

Visual style is controlled centrally via STYLE and apply_plot_theme().
All public functions follow the same call convention:
  - rep    : optional report object; if provided the figure is logged and closed
  - show   : bool, call plt.show() before returning
  - return : Figure (or None when rep is not None)
"""

from typing import Any

import math
import random
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# CENTRAL STYLE CONSTANTS
# ============================================================================

class STYLE:
    # Palette (three-colour, colour-blind friendly)
    PRIMARY   = "#4C78A8"
    SECONDARY = "#F58518"
    ACCENT    = "#54A24B"
    NEUTRAL   = "#BDBDBD"

    # ICC threshold colours
    ICC_EXCELLENT = "#54A24B"   # >=0.90
    ICC_GOOD      = "#F2CF5B"   # 0.75–0.90
    ICC_LOWER     = "#E45756"   # <0.75

    # Typography — one consistent hierarchy
    SUPTITLE_SIZE  = 13   # figure-level suptitle
    TITLE_SIZE     = 12   # axes title
    AXIS_SIZE      = 10   # axis labels (x/y)
    TICK_SIZE      = 9    # tick labels
    ANNOT_SIZE     = 8    # in-cell heatmap annotations
    LABEL_SIZE     = 9    # bar / scatter value labels

    # Layout
    GRID_ALPHA     = 0.18
    BAR_ALPHA      = 0.88
    DPI            = 150
    TIGHT_RECT     = (0.0, 0.0, 0.92, 0.95)   # common tight_layout rect

    # Heatmap
    HEATMAP_CMAP        = "viridis"
    HEATMAP_LINEWIDTHS  = 0.25
    HEATMAP_LINECOLOR   = "white"
    ANNOT_KWS           = dict(fontsize=ANNOT_SIZE, fontweight="bold")

    # Bar labels
    BAR_LABEL_PAD = 0.01   # fraction of axis range used as right-pad for h-bars


def apply_plot_theme() -> None:
    """Apply the shared visual theme to all subsequent Matplotlib/Seaborn plots."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        # background
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        # fonts
        "font.size":         STYLE.TICK_SIZE,
        "axes.titlesize":    STYLE.TITLE_SIZE,
        "axes.titleweight":  "bold",
        "axes.labelsize":    STYLE.AXIS_SIZE,
        "xtick.labelsize":   STYLE.TICK_SIZE,
        "ytick.labelsize":   STYLE.TICK_SIZE,
        "legend.fontsize":   STYLE.TICK_SIZE,
        # spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # grid
        "grid.alpha":        STYLE.GRID_ALPHA,
        # spacing
        "axes.titlepad":     10,
        "axes.labelpad":     6,
        "xtick.major.pad":   3,
        "ytick.major.pad":   3,
        "legend.frameon":    False,
        # output
        "savefig.dpi":       STYLE.DPI,
    })

# ============================================================================
# SHARED HELPERS
# ============================================================================

def _bar_label_offset(ax: plt.Axes, axis: str = "x") -> float:
    """Return a consistent absolute offset for bar-end labels."""
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    return (hi - lo) * STYLE.BAR_LABEL_PAD


def _annotate_hbar(ax: plt.Axes, bars, fmt: str = "{:.1f}") -> None:
    """Add right-aligned value labels to horizontal bars (uniform style)."""
    offset = _bar_label_offset(ax, "x")
    for bar in bars:
        v = bar.get_width()
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(v),
            va="center", ha="left",
            fontsize=STYLE.LABEL_SIZE, fontweight="bold",
        )


def _annotate_vbar(ax: plt.Axes, bars, fmt_int: str = "{:d}", fmt_float: str = "{:.1f}") -> None:
    """Add top-aligned value labels to vertical bars (uniform style)."""
    for bar in bars:
        v = bar.get_height()
        if np.isnan(v):
            continue
        label = fmt_int.format(int(round(v))) if float(v).is_integer() else fmt_float.format(v)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            label,
            ha="center", va="bottom",
            fontsize=STYLE.LABEL_SIZE, fontweight="bold",
        )


def _strip_spines(ax: plt.Axes) -> None:
    """Remove top and right spines explicitly (belt-and-braces over rcParams)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_hbar_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    yticklabels: list[str] | None = None,
) -> None:
    ax.set_title(title, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=STYLE.AXIS_SIZE)
    ax.set_ylabel("", fontsize=STYLE.AXIS_SIZE)
    ax.tick_params(axis="x", labelsize=STYLE.TICK_SIZE, pad=4)
    ax.tick_params(axis="y", labelsize=STYLE.TICK_SIZE, pad=4)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels, fontsize=STYLE.TICK_SIZE)
    ax.grid(axis="x", linestyle="--", alpha=STYLE.GRID_ALPHA)
    _strip_spines(ax)


def _finalise(fig: plt.Figure, suptitle: str | None = None) -> None:
    """Apply suptitle (if any) and a standard tight_layout rect."""
    if suptitle:
        fig.suptitle(suptitle, fontsize=STYLE.SUPTITLE_SIZE, fontweight="bold")
    fig.tight_layout(rect=list(STYLE.TIGHT_RECT))


def _handle_rep_show(fig: plt.Figure, rep, label: str, show: bool):
    """Log to report or optionally show; return None when rep is set."""
    if rep is not None:
        rep.log_plot(fig, label)
        plt.close(fig)
        return None, None
    if show:
        plt.show()
    return fig


def _set_ticklabels(
    ax: plt.Axes,
    axis: str,
    labels,
    rotation: float | int = 0,
    ha: str = "center",
    fontsize: int = STYLE.TICK_SIZE,
) -> None:
    """Apply tick labels in one place so size/rotation stay consistent."""
    if axis == "x":
        ax.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=fontsize)
    elif axis == "y":
        ax.set_yticklabels(labels, rotation=rotation, ha=ha, fontsize=fontsize)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def _adaptive_rotation(labels, *, threshold: int = 8, steep: int = 35, mild: int = 20) -> int:
    """Choose a conservative tick rotation based on label count and length."""
    labels = [str(x) for x in labels]
    if not labels:
        return 0
    longest = max(len(x) for x in labels)
    if len(labels) > threshold or longest > 12:
        return steep
    if longest > 8:
        return mild
    return 0


def _icc_legend_handles() -> list:
    """Legend handles explaining ICC bar colours and threshold bands."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    return [
        Patch(facecolor=STYLE.ICC_EXCELLENT, edgecolor="none", label="ICC ≥ 0.90"),
        Patch(facecolor=STYLE.ICC_GOOD, edgecolor="none", label="0.75 ≤ ICC < 0.90"),
        Patch(facecolor=STYLE.ICC_LOWER, edgecolor="none", label="ICC < 0.75"),
        Patch(facecolor=STYLE.NEUTRAL, edgecolor="none", label="Missing / NaN"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.9, alpha=0.5, label="Threshold lines"),
    ]


def _batch_difference_legend_handles() -> list:
    """Legend handles for the multivariate batch difference bar colours."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=STYLE.SECONDARY, edgecolor="none", label="Average batch"),
        Patch(facecolor=STYLE.PRIMARY, edgecolor="none", label="Other batches"),
    ]


def _configure_hbar_panel(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    yticklabels: list[str],
    *,
    xlim_pad: float = 0.0,
    show_legend=None,
    legend_loc: str = "lower right",
    legend_title: str | None = None,
) -> None:
    """Standardise horizontal bar panels so titles/ticks look uniform."""
    ax.set_title(title, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=STYLE.AXIS_SIZE)
    ax.set_ylabel("", fontsize=STYLE.AXIS_SIZE)
    ax.tick_params(axis="both", labelsize=STYLE.TICK_SIZE, pad=4)
    ax.set_yticklabels(yticklabels, fontsize=STYLE.TICK_SIZE)
    ax.grid(axis="x", linestyle="--", alpha=STYLE.GRID_ALPHA)
    _strip_spines(ax)

    if xlim_pad:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + xlim_pad)

    if show_legend is not None:
        if legend_loc == "center left":
            ax.legend(
                handles=show_legend,
                title=legend_title,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=STYLE.TICK_SIZE,
                title_fontsize=STYLE.TICK_SIZE,
                handlelength=1.5,
                borderaxespad=0.25,
            )
        else:
            ax.legend(
                handles=show_legend,
                loc=legend_loc,
                frameon=False,
                fontsize=STYLE.TICK_SIZE,
                title=legend_title,
                title_fontsize=STYLE.TICK_SIZE,
                handlelength=1.5,
                borderaxespad=0.25,
            )


# ============================================================================
# STYLE REGISTRY FOR SUBJECTS / IDPs
# ============================================================================

def build_style_registry(
    subjects,
    idps,
    subject_palette: str = "tab10",
    idp_palette: str = "Set2",
    subject_markers: list | None = None,
    idp_markers: list | None = None,
) -> tuple[dict[Any, tuple[Any, str]], dict[Any, tuple[Any, str]]]:
    """
    Build colour and marker registries for subjects and IDPs.

    Returns
    -------
    subject_style, idp_style : dicts mapping each identifier to (color, marker).
    """
    if subject_markers is None:
        subject_markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*", "h"]
    if idp_markers is None:
        idp_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

    subj_colors = sns.color_palette(subject_palette, len(subjects))
    idp_colors  = sns.color_palette(idp_palette,     len(idps))

    subject_style = {
        s: (subj_colors[i % len(subj_colors)], subject_markers[i % len(subject_markers)])
        for i, s in enumerate(subjects)
    }
    idp_style = {
        n: (idp_colors[i % len(idp_colors)], idp_markers[i % len(idp_markers)])
        for i, n in enumerate(idps)
    }
    return subject_style, idp_style


def _build_default_idp_style(idps, palette: str = "Set2", markers: list | None = None) -> dict:
    if markers is None:
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H"]
    colors = sns.color_palette(palette, max(2, len(idps)))
    return {idp: (colors[i % len(colors)], markers[i % len(markers)]) for i, idp in enumerate(idps)}


# ============================================================================
# P-VALUE COMBINATION / CORRECTION UTILITIES
# ============================================================================

try:
    from scipy.stats import combine_pvalues, norm as _scipy_norm
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    flat = p.flatten()
    nanmask = np.isnan(flat)
    idx = np.where(~nanmask)[0]
    if idx.size == 0:
        return p
    pv = flat[~nanmask]
    order = np.argsort(pv)
    ranked = np.empty_like(order)
    ranked[order] = np.arange(pv.size) + 1
    m = pv.size
    adj_vals = pv * m / ranked
    adj_vals_sorted = np.empty_like(adj_vals)
    adj_vals_sorted[order] = adj_vals
    cummin = np.minimum.accumulate(adj_vals_sorted[::-1])[::-1]
    adj = np.empty_like(pv)
    adj[order] = np.minimum(cummin, 1.0)
    flat_adj = flat.copy()
    flat_adj[~nanmask] = adj
    return flat_adj.reshape(p.shape)


def _bonferroni_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    flat = p.flatten()
    nanmask = np.isnan(flat)
    pv = flat[~nanmask]
    flat_adj = flat.copy()
    flat_adj[~nanmask] = np.minimum(pv * pv.size, 1.0)
    return flat_adj.reshape(p.shape)

import json
import textwrap
from collections import Counter
from matplotlib.colors import ListedColormap, BoundaryNorm


def _build_pairwise_pair_matrix(
    df: pd.DataFrame,
    idp_col: str = "IDP",
    records_col: str = "pairwise_site_tests",
    sig_key: str = "sig_bonf",
    top_n: int = 20,
    max_full_pairs: int = 50,
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    """
    Build a binary 0/1 matrix of significant pairwise site differences.

    Returns
    -------
    mat : pd.DataFrame
        Rows = IDPs, columns = pair codes like '1-2'
    site_code_map : dict
        Maps code -> original site name, e.g. '1' -> 'siteA'
    use_compact : bool
        True when top-N compact mode is used.
    """
    rows = []
    pair_freq = Counter()
    all_sites = []

    for _, r in df.iterrows():
        recs = r.get(records_col, [])
        if isinstance(recs, str):
            try:
                recs = json.loads(recs)
            except Exception:
                recs = []

        row_map = {}
        for rec in recs:
            a = str(rec.get("siteA", ""))
            b = str(rec.get("siteB", ""))
            if not a or not b:
                continue

            all_sites.extend([a, b])
            pair_name = tuple(sorted((a, b)))
            is_sig = bool(rec.get(sig_key, False))

            pair_freq[pair_name] += int(is_sig)
            row_map[pair_name] = 1.0 if is_sig else 0.0

        rows.append((str(r[idp_col]), row_map))

    unique_sites = list(dict.fromkeys(all_sites))
    site_code_map = {str(i + 1): s for i, s in enumerate(unique_sites)}
    site_to_code = {v: k for k, v in site_code_map.items()}

    all_pairs = sorted(pair_freq.keys())
    use_compact = len(all_pairs) > max_full_pairs

    if use_compact:
        selected_pairs = [p for p, _ in pair_freq.most_common(top_n)]
    else:
        selected_pairs = all_pairs

    pair_codes = [f"{site_to_code[a]}-{site_to_code[b]}" for a, b in selected_pairs]

    mat = pd.DataFrame(index=[x[0] for x in rows], columns=pair_codes, dtype=float)

    for idp, row_map in rows:
        for (a, b), code in zip(selected_pairs, pair_codes):
            mat.at[idp, code] = row_map.get((a, b), 0.0)

    if len(mat) > 0 and len(mat.columns) > 0:
        pair_rank = mat.sum(axis=0).sort_values(ascending=False)
        idp_rank = mat.sum(axis=1).sort_values(ascending=False)
        mat = mat.loc[idp_rank.index, pair_rank.index]

    return mat, site_code_map, use_compact


def _apply_correction(p_matrix: np.ndarray, method: str | None) -> np.ndarray:
    if method is None:
        return p_matrix
    if method == "fdr_bh":
        return _bh_adjust(p_matrix)
    if method == "bonferroni":
        return _bonferroni_adjust(p_matrix)
    raise ValueError("p_correction must be 'fdr_bh', 'bonferroni', or None")
# ============================================================================
# 1.  RAW IDP DISTRIBUTIONS ACROSS SITES
# ============================================================================

def plot_RawIDPBoxplotsAcrossSites(
    df,
    batch_col: str = "batch",
    subject_col: str | None = "subject",
    idp_cols: list | None = None,
    site_order: list | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple = (6.0, 4.5),
    show_points: bool = True,
    point_size: float = 2.2,
    point_alpha: float = 0.18,
    point_jitter: float = 0.22,
    savepath: str | None = None,
    rep=None,
    show: bool = False,
    site_threshold_for_horizontal: int = 12,
    feature_display_limit: int = 10,
    add_pca_summary: bool = True,
):
    """
    Raw IDP distributions across sites/batches.

    Behaviour
    ----------
    - If the number of sites is small, boxplots are shown vertically.
    - If the number of sites is large, boxplots are shown horizontally.
    - If there are many features, only the top features by site dispersion are shown.
    - Optionally adds a PCA summary panel for the full feature set when truncated.

    Notes
    -----
    The batch/site order is deterministic:
    - if site_order is provided, it is used as-is
    - otherwise sites are ordered alphabetically
    This keeps colors and tick-label order consistent across raw and harmonised runs.
    """
    apply_plot_theme()

    if batch_col not in df.columns:
        raise ValueError(f"'{batch_col}' not found in dataframe columns.")

    if idp_cols is None:
        idp_cols = [c for c in df.columns if c != batch_col and c != subject_col]

    idp_cols = list(idp_cols)
    if len(idp_cols) == 0:
        raise ValueError("No IDP columns found to plot.")

    plot_df = df.copy()
    plot_df[batch_col] = plot_df[batch_col].astype(str)

    if subject_col is not None and subject_col in plot_df.columns:
        plot_df[subject_col] = plot_df[subject_col].astype(str)
    else:
        subject_col = None

    # ------------------------------------------------------------------
    # Site ordering (deterministic)
    # ------------------------------------------------------------------
    if site_order is None:
        site_order = sorted(plot_df[batch_col].dropna().astype(str).unique().tolist())
    else:
        site_order = [str(s) for s in site_order]

    n_sites = len(site_order)
    use_horizontal = n_sites > site_threshold_for_horizontal

    # Batch-size summary for user awareness
    batch_counts = plot_df.groupby(batch_col).size().reindex(site_order).fillna(0).astype(int)
    min_batch = int(batch_counts.min()) if len(batch_counts) else 0
    med_batch = int(batch_counts.median()) if len(batch_counts) else 0
    max_batch = int(batch_counts.max()) if len(batch_counts) else 0
    n_lt5 = int((batch_counts < 5).sum())
    n_lt10 = int((batch_counts < 10).sum())
    n_lt20 = int((batch_counts < 20).sum())

    summary_text = (
        f"Batch size check: min={min_batch}, median={med_batch}, max={max_batch}   |   "
        f"Batches <5: {n_lt5}   <10: {n_lt10}   <20: {n_lt20}"
    )

    # ------------------------------------------------------------------
    # Feature ranking if there are many IDPs
    # ------------------------------------------------------------------
    shown_idps = idp_cols
    truncated = False

    if len(idp_cols) > feature_display_limit:
        med = plot_df.groupby(batch_col)[idp_cols].median(numeric_only=True)
        if isinstance(med, pd.DataFrame) and not med.empty:
            disp = med.max(axis=0) - med.min(axis=0)
            shown_idps = disp.sort_values(ascending=False).head(feature_display_limit).index.tolist()
        else:
            shown_idps = idp_cols[:feature_display_limit]
        truncated = True

    n_idps = len(shown_idps)

    # ------------------------------------------------------------------
    # Optional PCA summary for all features
    # ------------------------------------------------------------------
    pca_summary = None
    if truncated and add_pca_summary:
        X = plot_df[idp_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        row_mask = ~np.isnan(X).any(axis=1)
        Xc = X[row_mask]

        if Xc.shape[0] >= 2 and Xc.shape[1] >= 2:
            Xc = Xc - Xc.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            scores = U[:, :2] * S[:2]

            var = (S ** 2) / max(Xc.shape[0] - 1, 1)
            total_var = var.sum() if var.size else 1.0
            explained = (var[:2] / total_var * 100.0) if total_var > 0 else np.array([0.0, 0.0])

            pca_summary = {
                "scores": scores,
                "explained": explained,
                "sites": plot_df.loc[row_mask, batch_col].astype(str).to_numpy(),
            }

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    n_plot_panels = n_idps + (1 if (truncated and add_pca_summary and pca_summary is not None) else 0)
    max_site_label_len = max((len(s) for s in site_order), default=0)
    left_margin = min(0.42, max(0.22, 0.10 + 0.008 * max_site_label_len))

    if use_horizontal:
        fig_w = max(8.0, figsize_per_panel[0] * 1.25)
        fig_h = max(4.0, figsize_per_panel[1] * n_plot_panels)
        fig, axes = plt.subplots(n_plot_panels, 1, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.flatten()
    else:
        ncols = max(1, min(int(ncols), n_idps))
        nrows = int(math.ceil(n_idps / ncols))
        extra_rows = 1 if (truncated and add_pca_summary and pca_summary is not None) else 0

        fig_w = figsize_per_panel[0] * ncols
        fig_h = figsize_per_panel[1] * (nrows + extra_rows)
        fig, axes_grid = plt.subplots(nrows + extra_rows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes_grid.flatten()

    # Make room for the batch-size note
    fig.subplots_adjust(top=0.93 if use_horizontal else 0.94)

    fig.text(
        0.5, 0.995,
        summary_text,
        ha="center",
        va="top",
        fontsize=max(7, STYLE.TICK_SIZE - 1),
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.85"),
    )

    # Palette by site (stable mapping because site_order is stable)
    palette = sns.color_palette("Set2", n_colors=max(3, len(site_order)))
    site_palette = {site: palette[i % len(palette)] for i, site in enumerate(site_order)}

    # ------------------------------------------------------------------
    # Plot each IDP
    # ------------------------------------------------------------------
    for i, idp in enumerate(shown_idps):
        ax = axes[i]

        if use_horizontal:
            sns.boxplot(
                data=plot_df,
                y=batch_col,
                x=idp,
                order=site_order,
                ax=ax,
                palette=site_palette,
                width=0.65,
                fliersize=0,
                linewidth=1.0,
                orient="h",
            )

            if show_points:
                sns.stripplot(
                    data=plot_df,
                    y=batch_col,
                    x=idp,
                    order=site_order,
                    ax=ax,
                    color="black",
                    size=point_size,
                    alpha=point_alpha,
                    jitter=point_jitter,
                    orient="h",
                )

            ax.set_title(idp, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=12)
            ax.set_xlabel("IDP value", fontsize=STYLE.AXIS_SIZE)
            ax.set_ylabel("Site", fontsize=STYLE.AXIS_SIZE)
            _set_ticklabels(
                ax,
                "y",
                site_order,
                fontsize=max(7, STYLE.TICK_SIZE - 1),
                ha="right",
            )
            ax.tick_params(axis="y", pad=8, labelsize=max(7, STYLE.TICK_SIZE - 1))
            ax.tick_params(axis="x", labelsize=STYLE.TICK_SIZE)
            ax.grid(axis="x", linestyle="--", alpha=STYLE.GRID_ALPHA)
        else:
            sns.boxplot(
                data=plot_df,
                x=batch_col,
                y=idp,
                order=site_order,
                ax=ax,
                palette=site_palette,
                width=0.65,
                fliersize=0,
                linewidth=1.0,
            )

            if show_points:
                sns.stripplot(
                    data=plot_df,
                    x=batch_col,
                    y=idp,
                    order=site_order,
                    ax=ax,
                    color="black",
                    size=point_size,
                    alpha=point_alpha,
                    jitter=point_jitter,
                )

            ax.set_title(idp, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=10)
            ax.set_xlabel("Site", fontsize=STYLE.AXIS_SIZE)
            ax.set_ylabel("Raw value", fontsize=STYLE.AXIS_SIZE)
            rot = _adaptive_rotation(site_order, threshold=7, steep=45, mild=30)
            _set_ticklabels(ax, "x", site_order, rotation=rot, ha="right" if rot else "center")
            ax.tick_params(axis="y", labelsize=STYLE.TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=STYLE.GRID_ALPHA)

        _strip_spines(ax)

    # ------------------------------------------------------------------
    # PCA summary panel if truncated
    # ------------------------------------------------------------------
    if truncated and add_pca_summary:
        ax = axes[n_idps]

        if pca_summary is None:
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                "PCA summary unavailable\n(insufficient complete data)",
                ha="center", va="center",
                fontsize=STYLE.AXIS_SIZE,
                transform=ax.transAxes,
            )
        else:
            scores = pca_summary["scores"]
            sites = pca_summary["sites"]
            explained = pca_summary["explained"]
            color_map = {site: site_palette.get(site, STYLE.NEUTRAL) for site in site_order}

            for site in site_order:
                idx = sites == site
                if np.any(idx):
                    ax.scatter(
                        scores[idx, 0],
                        scores[idx, 1],
                        s=18,
                        alpha=0.6,
                        color=color_map[site],
                        label=site,
                    )

            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
            ax.set_title(
                f"PCA summary of all IDPs\n"
                f"PC1: {explained[0]:.1f}%  |  PC2: {explained[1]:.1f}%",
                fontsize=STYLE.TITLE_SIZE,
                fontweight="bold",
                pad=10,
            )
            ax.set_xlabel("PC1 score", fontsize=STYLE.AXIS_SIZE)
            ax.set_ylabel("PC2 score", fontsize=STYLE.AXIS_SIZE)
            ax.tick_params(axis="both", labelsize=STYLE.TICK_SIZE)
            ax.legend(
                title="Site",
                fontsize=STYLE.TICK_SIZE,
                title_fontsize=STYLE.TICK_SIZE,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )
            _strip_spines(ax)

        for j in range(n_idps + 1, len(axes)):
            axes[j].axis("off")
    else:
        for j in range(n_idps, len(axes)):
            axes[j].axis("off")

    # ------------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------------
    if use_horizontal:
        fig.subplots_adjust(left=left_margin, right=0.98, top=0.93, bottom=0.06, hspace=0.35)
    else:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06, hspace=0.38)

    _finalise(fig, "Raw IDP distributions across sites")

    if savepath:
        plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

    return _handle_rep_show(fig, rep, "", show)


# ============================================================================
# 1.  SUBJECT ORDER CONSISTENCY
# ============================================================================

def plot_SubjectOrder(
    df,
    idp_col:          str   = "IDP",
    time_a_col:       str   = "TimeA",
    time_b_col:       str   = "TimeB",
    rho_col:          str   = "SpearmanRho",
    p_col:            str   = "pValue",
    times_order             = None,
    significance:     float = 0.05,
    ncols:            int   = 2,
    figsize_per_plot: tuple = (5, 5),
    cmap:             str   = STYLE.HEATMAP_CMAP,
    fmt:              str   = ".1f",
    center:           float = 0,
    limit_idps:       int | None = None,
    sample_method:    str   = "first",
    random_state            = None,
    rep                     = None,
    show:             bool  = False,
    combine_method:   str   = "stouffer",
    p_correction:     str   = "fdr_bh",
):
    """
    Subject order consistency heatmaps with combined p-values.

    Per-feature heatmaps show raw permutation p-values.
    Summary heatmaps show combined p-values corrected by p_correction.
    """
    apply_plot_theme()

    if not _SCIPY_OK:
        raise ImportError("scipy is required. Run: pip install scipy")

    _TINY = 1e-300

    all_idps = sorted(df[idp_col].unique())
    if not all_idps:
        raise ValueError("No IDPs found in dataframe.")

    if times_order is None:
        times = sorted(set(df[time_a_col].unique()) | set(df[time_b_col].unique()))
    else:
        times = list(times_order)
    n_times = len(times)
    if not n_times:
        raise ValueError("No time points found.")

    rho_mats_all: dict[str, pd.DataFrame] = {}
    p_mats_all: dict[str, pd.DataFrame] = {}
    all_rho_vals: list[float] = []

    for idp in all_idps:
        sub = df[df[idp_col] == idp].copy()
        rho = sub.pivot(index=time_a_col, columns=time_b_col, values=rho_col)
        pmat = sub.pivot(index=time_a_col, columns=time_b_col, values=p_col)
        rho = rho.reindex(index=times, columns=times).combine_first(rho.T.reindex(index=times, columns=times))
        pmat = pmat.reindex(index=times, columns=times).combine_first(pmat.T.reindex(index=times, columns=times))
        np.fill_diagonal(rho.values, np.nan)
        np.fill_diagonal(pmat.values, np.nan)
        rho_mats_all[idp] = rho.astype(float)
        p_mats_all[idp] = pmat.astype(float)
        vals = rho.values.flatten()
        all_rho_vals.extend(vals[~np.isnan(vals)].tolist())

    if not all_rho_vals:
        raise ValueError("No numeric SpearmanRho values found.")

    if limit_idps is None:
        if len(all_idps) > 10:
            ranked = sorted(
                all_idps,
                key=lambda idp: np.nanmean(rho_mats_all[idp].values),
                reverse=True,
            )
            idps_to_plot = ranked[:10]
        else:
            idps_to_plot = list(all_idps)
    else:
        if not (isinstance(limit_idps, int) and limit_idps >= 1):
            raise ValueError("limit_idps must be None or a positive int.")
        limit = min(limit_idps, len(all_idps))
        if sample_method == "first":
            idps_to_plot = list(all_idps[:limit])
        elif sample_method == "random":
            idps_to_plot = random.Random(random_state).sample(all_idps, limit)
        else:
            raise ValueError("sample_method must be 'first' or 'random'.")

    n_idp_plot = len(idps_to_plot)

    stacked = np.stack([rho_mats_all[idp].values for idp in all_idps], axis=0)
    stacked_p = np.stack([p_mats_all[idp].values for idp in all_idps], axis=0)
    sp = stacked_p.copy()
    sp[sp == 0] = _TINY

    combined_p = np.full((n_times, n_times), np.nan)
    mean_rho_mx = np.nanmean(stacked, axis=0)

    for i in range(n_times):
        for j in range(n_times):
            pv = sp[:, i, j]
            pv = pv[~np.isnan(pv)]
            if pv.size == 0:
                continue
            if combine_method.lower() == "fisher":
                _, combined_p[i, j] = combine_pvalues(pv, method="fisher")
            elif combine_method.lower() == "stouffer":
                rhos_ij = stacked[:, i, j]
                mean_r = np.nanmean(rhos_ij[~np.isnan(rhos_ij)]) if np.any(~np.isnan(rhos_ij)) else 0.0
                sign = np.sign(mean_r) if not np.isnan(mean_r) and mean_r != 0 else 1.0
                p_clip = np.clip(pv, _TINY, 1 - 1e-16)
                zs = _scipy_norm.ppf(1.0 - p_clip / 2.0)
                z_comb = np.sum(sign * zs) / math.sqrt(zs.size)
                combined_p[i, j] = float(np.clip(2.0 * (1.0 - _scipy_norm.cdf(abs(z_comb))), _TINY, 1.0))
            else:
                raise ValueError("combine_method must be 'stouffer' or 'fisher'")

    combined_p_adj = _apply_correction(combined_p, p_correction)
    np.fill_diagonal(mean_rho_mx, np.nan)

    idp_combined_ps: list[float] = []
    idp_mean_rhos: list[float] = []
    offdiag = ~np.eye(n_times, dtype=bool)

    for idp in all_idps:
        pm = p_mats_all[idp].values[offdiag]
        rho = rho_mats_all[idp].values[offdiag]
        pm = pm[~np.isnan(pm)]
        rho = rho[~np.isnan(rho)]
        idp_mean_rhos.append(float(np.nanmean(rho)) if rho.size else np.nan)
        if pm.size == 0:
            idp_combined_ps.append(np.nan)
            continue
        if combine_method.lower() == "fisher":
            _, p_c = combine_pvalues(np.clip(pm, _TINY, 1.0), method="fisher")
        else:
            mean_r = float(np.nanmean(rho)) if rho.size else 0.0
            sign = np.sign(mean_r) if not np.isnan(mean_r) and mean_r != 0 else 1.0
            zs = _scipy_norm.ppf(1.0 - np.clip(pm, _TINY, 1 - 1e-16) / 2.0)
            z_c = np.sum(sign * zs) / math.sqrt(zs.size)
            p_c = float(np.clip(2.0 * (1.0 - _scipy_norm.cdf(abs(z_c))), _TINY, 1.0))
        idp_combined_ps.append(p_c)

    idp_combined_ps_arr = np.asarray(idp_combined_ps, dtype=float)
    idp_combined_ps_adj = _apply_correction(idp_combined_ps_arr.reshape(-1, 1), p_correction).reshape(-1)

    max_idp_len = max((len(str(x)) for x in all_idps), default=0)
    extra_gap = 1 if max_idp_len > 18 else 0

    nrows = math.ceil(n_idp_plot / ncols) if n_idp_plot else 0
    fig_w = figsize_per_plot[0] * ncols
    fig_h = max(1, nrows) * figsize_per_plot[1] + 1.8 * figsize_per_plot[1]

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        max(1, nrows) + 2 + extra_gap,
        ncols,
        height_ratios=[1] * max(1, nrows) + ([0.18] if extra_gap else []) + [0.9, 0.9],
        hspace=0.55,
        wspace=0.35,
    )

    _hm_kws = dict(
        fmt="",
        cmap=cmap,
        center=center,
        vmin=0,
        vmax=1,
        linewidths=STYLE.HEATMAP_LINEWIDTHS,
        linecolor=STYLE.HEATMAP_LINECOLOR,
        annot_kws=STYLE.ANNOT_KWS,
        cbar=False,
        square=False,
    )

    def _make_annot(rho_df: pd.DataFrame, pmat_df: pd.DataFrame) -> np.ndarray:
        a = np.full(rho_df.shape, "", dtype=object)
        for i in range(n_times):
            for j in range(n_times):
                v = rho_df.iat[i, j]
                pv = pmat_df.iat[i, j]
                if not np.isnan(v):
                    star = "*" if (not pd.isna(pv) and pv < significance) else ""
                    a[i, j] = f"{v:{fmt}}{star}"
        return a

    idp_tick_size = max(7, STYLE.TICK_SIZE - 1) if max_idp_len > 18 else STYLE.TICK_SIZE

    for idx, idp in enumerate(idps_to_plot):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        rho = rho_mats_all[idp]
        pmat = p_mats_all[idp]
        sns.heatmap(rho, ax=ax, annot=_make_annot(rho, pmat), **_hm_kws)
        ax.set_title(idp, fontsize=STYLE.TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(times, rotation=45, ha="right", fontsize=idp_tick_size)
        ax.set_yticklabels(times, rotation=0, fontsize=STYLE.TICK_SIZE)
        ax.tick_params(axis="x", pad=3)

    for k in range(n_idp_plot, max(1, nrows) * ncols):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        ax.axis("off")

    summary_row = max(0, nrows) + extra_gap

    ax_tt = fig.add_subplot(gs[summary_row, :])
    annot_mean = np.full(mean_rho_mx.shape, "", dtype=object)
    for i in range(n_times):
        for j in range(n_times):
            v = mean_rho_mx[i, j]
            pv = combined_p_adj[i, j]
            if not np.isnan(v):
                star = "*" if (not np.isnan(pv) and pv < significance) else ""
                annot_mean[i, j] = f"{v:{fmt}}{star}"

    sns.heatmap(
        mean_rho_mx, ax=ax_tt, annot=annot_mean,
        linewidths=0.35, linecolor="gray",
        **{k: v for k, v in _hm_kws.items() if k not in ("linewidths", "linecolor")},
    )
    ax_tt.set_title(
        "Mean subject order consistency across timepoints and IDPs",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
    )
    ax_tt.set_xlabel("")
    ax_tt.set_ylabel("")
    ax_tt.set_xticklabels(times, rotation=45, ha="right", fontsize=STYLE.TICK_SIZE)
    ax_tt.set_yticklabels(times, rotation=0, fontsize=STYLE.TICK_SIZE)

    ax_idp = fig.add_subplot(gs[summary_row + 1, :])
    idp_mean_mat = np.array(idp_mean_rhos, dtype=float).reshape(-1, 1)
    annot_idp = np.array([
        f"{v:{fmt}}{'*' if (not np.isnan(p) and p < significance) else ''}"
        for v, p in zip(idp_mean_rhos, idp_combined_ps_adj)
    ]).reshape(-1, 1)

    sns.heatmap(
        idp_mean_mat, ax=ax_idp, annot=annot_idp,
        yticklabels=all_idps, xticklabels=["Mean across time-pairs"],
        **_hm_kws,
    )
    ax_idp.set_title(
        "Per-IDP mean subject order consistency",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
    )
    ax_idp.set_xlabel("")
    ax_idp.set_ylabel("")
    ax_idp.set_xticklabels([""], rotation=0)
    ax_idp.tick_params(axis="y", labelsize=STYLE.TICK_SIZE)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.018, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Spearman ρ", fontsize=STYLE.AXIS_SIZE)
    cbar.ax.tick_params(labelsize=STYLE.TICK_SIZE)

    corr_name = {
        "fdr_bh": "FDR (Benjamini-Hochberg)",
        "bonferroni": "Bonferroni",
        None: "none",
    }.get(p_correction, str(p_correction))

    note = (
        f"Per-feature heatmaps: raw permutation p-values (α = {significance:g})\n"
        f"Summary heatmaps: combined p-values corrected by {corr_name}"
    )
    fig.text(
        0.985, 0.985, note,
        ha="right", va="top",
        fontsize=max(7, STYLE.TICK_SIZE - 2),
        fontstyle="italic",
    )

    plt.suptitle(
        f"Subject order consistency  ({len(all_idps)} features analysed)",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.94])

    return _handle_rep_show(fig, rep, "", show)


# ============================================================================
# 2.  WITHIN-SUBJECT VARIABILITY
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


def plot_WithinSubjVar(
    df,
    subject_col:           str        = "subject",
    idp_cols:              list | None = None,
    subject_style:         dict | None = None,
    idp_style:             dict | None = None,
    limit_subjects:        int        = 30,
    limit_idps_for_legend: int        = 30,
    figsize:               tuple      = (16, 13),
    savepath:              str | None = None,
    rep                               = None,
    show:                  bool       = False,
    debug:                 bool       = False,
):
    """
    Plot within-subject variability summary across IDPs.

    Panels:
        A — distribution across subjects for each IDP
            * few subjects: colored subject markers + subject legend
            * many subjects: black dots only, no subject legend

        B — mean variability per IDP
            * few IDPs: colored IDP markers + IDP legend
            * many IDPs: black dots only, no IDP legend

        C — top 10 subjects with highest average variability
    """
    apply_plot_theme()

    metadata_cols = {subject_col, "n_obs", "metric_type", "MetricType", "N"}

    if idp_cols is None:
        idp_cols = [
            c for c in df.columns
            if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    if len(idp_cols) == 0:
        raise ValueError("No IDP columns found.")

    subjects_all = pd.unique(df[subject_col]).tolist()
    idps_all = list(idp_cols)

    if subject_style is None or idp_style is None:
        subject_style, idp_style = build_style_registry(subjects_all, idps_all)

    def _get_subject_style(subj):
        sty = subject_style.get(subj, None)
        if sty is None:
            return STYLE.NEUTRAL, "o"
        if isinstance(sty, (tuple, list)) and len(sty) >= 2:
            return sty[0], sty[1]
        if isinstance(sty, dict):
            return sty.get("color", STYLE.NEUTRAL), sty.get("marker", "o")
        return STYLE.NEUTRAL, "o"

    def _get_idp_style(idp):
        sty = idp_style.get(idp, None)
        if sty is None:
            return STYLE.NEUTRAL, "o"
        if isinstance(sty, (tuple, list)) and len(sty) >= 2:
            return sty[0], sty[1]
        if isinstance(sty, dict):
            return sty.get("color", STYLE.NEUTRAL), sty.get("marker", "o")
        return STYLE.NEUTRAL, "o"

    long = df.melt(
        id_vars=subject_col,
        value_vars=idp_cols,
        var_name="IDP",
        value_name="value",
    ).dropna(subset=["value"])

    idp_to_x = {idp: i for i, idp in enumerate(idp_cols)}
    rng = np.random.default_rng(12345)  # stable jitter

    n_subjects = df[subject_col].nunique()
    n_idps = len(idp_cols)

    show_subject_style = n_subjects <= limit_subjects
    show_idp_style = n_idps <= limit_idps_for_legend

    if debug:
        print("=== WithinSubjVar DEBUG ===")
        print(f"rows in df: {len(df)}")
        print(f"subject_col: {subject_col}")
        print(f"n_subjects: {n_subjects}")
        print(f"limit_subjects: {limit_subjects}")
        print(f"show_subject_style: {show_subject_style} (type={type(show_subject_style)})")
        print(f"n_idps: {n_idps}")
        print(f"limit_idps_for_legend: {limit_idps_for_legend}")
        print(f"show_idp_style: {show_idp_style} (type={type(show_idp_style)})")
        print(f"subjects_all[:10]: {subjects_all[:10]}")
        print(f"idps_all[:10]: {idps_all[:10]}")
        print("==========================")

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.45, 0.85, 1.55],
        hspace=0.70
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[2, 0])

    fig.subplots_adjust(right=0.78)

    # ---------------- Panel A: per-IDP distribution ----------------
    sns.boxplot(
        x="IDP", y="value", data=long, ax=axA,
        order=idp_cols,
        color=STYLE.PRIMARY, width=0.55,
        showfliers=False,
        boxprops=dict(alpha=0.35),
        whiskerprops=dict(linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
    )

    if debug:
        print("\n--- Panel A ---")
        print(f"show_subject_style = {show_subject_style}")

    if show_subject_style:
        if debug:
            print("Panel A branch: SUBJECT-SPECIFIC COLORS/MARKERS + LEGEND")

        handles = []
        labels = []

        for subj in subjects_all:
            sub = long[long[subject_col] == subj]
            color, marker = _get_subject_style(subj)

            x = sub["IDP"].map(idp_to_x).to_numpy(dtype=float)
            x = x + rng.normal(0, 0.08, size=len(x))  # jitter
            y = sub["value"].to_numpy(dtype=float)

            axA.scatter(
                x, y,
                s=38,
                c=[color],
                marker=marker,
                alpha=0.90,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

            handles.append(
                mlines.Line2D(
                    [], [], linestyle="None",
                    marker=marker, markersize=7,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    label=str(subj),
                )
            )
            labels.append(str(subj))

        axA.legend(
            handles, labels,
            title="Subject",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor="0.85",
            fontsize=STYLE.TICK_SIZE,
            title_fontsize=STYLE.TICK_SIZE,
            ncol=1,
            borderpad=0.6,
            labelspacing=0.5,
            handletextpad=0.4,
        )
    else:
        if debug:
            print("Panel A branch: BLACK DOTS ONLY, NO LEGEND")

        x = long["IDP"].map(idp_to_x).to_numpy(dtype=float)
        x = x + rng.normal(0, 0.08, size=len(x))
        y = long["value"].to_numpy(dtype=float)

        axA.scatter(
            x, y,
            s=28,
            c="black",
            marker="o",
            alpha=0.28,
            edgecolors="none",
            zorder=3,
        )

        leg = axA.get_legend()
        if debug:
            print(f"Panel A legend exists before remove: {leg is not None}")
        if leg is not None:
            leg.remove()

    axA.set_xticks(range(len(idp_cols)))
    axA.set_xticklabels(idp_cols)
    axA.set_title(
        "Within-subject variability across IDPs",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    axA.set_xlabel("")
    axA.set_ylabel("Pairwise RPD (%)", fontsize=STYLE.AXIS_SIZE)
    rotA = _adaptive_rotation(idp_cols, threshold=8, steep=40, mild=25)
    axA.tick_params(axis="x", labelrotation=rotA, labelsize=STYLE.TICK_SIZE)
    axA.tick_params(axis="y", labelsize=STYLE.TICK_SIZE)
    _strip_spines(axA)
    axA.margins(x=0.02, y=0.06)

    # ---------------- Panel B: mean across subjects per IDP ----------------
    idp_means = df[idp_cols].mean(axis=0, skipna=True)

    if debug:
        print("\n--- Panel B ---")
        print(f"show_idp_style = {show_idp_style}")

    sns.boxplot(
        x=idp_means.values, ax=axB, orient="h",
        color=STYLE.PRIMARY, width=0.50,
        showfliers=False,
        boxprops=dict(alpha=0.35),
        whiskerprops=dict(linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
    )

    axB.set_yticks([])
    axB.set_title(
        "Mean pairwise RPD per feature",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    axB.set_xlabel("Pairwise RPD (%)", fontsize=STYLE.AXIS_SIZE)
    axB.set_ylabel("")
    axB.tick_params(axis="both", labelsize=STYLE.TICK_SIZE)
    _strip_spines(axB)
    axB.margins(x=0.03, y=0.10)

    if show_idp_style:
        if debug:
            print("Panel B branch: IDP-SPECIFIC COLORS/MARKERS + LEGEND")

        handles = []
        labels = []

        for idp, mean_val in idp_means.items():
            color, marker = _get_idp_style(idp)

            axB.scatter(
                mean_val, 0,
                s=70,
                marker=marker,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

            handles.append(
                mlines.Line2D(
                    [], [], linestyle="None",
                    marker=marker, markersize=7,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    label=idp,
                )
            )
            labels.append(idp)

        axB.legend(
            handles, labels,
            title="IDP",
            loc="center left",
            bbox_to_anchor=(1.01, 0.50),
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            edgecolor="0.85",
            fontsize=STYLE.TICK_SIZE,
            title_fontsize=STYLE.TICK_SIZE,
        )
    else:
        if debug:
            print("Panel B branch: BLACK DOTS ONLY, NO LEGEND")

        xvals = idp_means.values
        yvals = np.zeros(len(idp_means))

        axB.scatter(
            xvals, yvals,
            s=58,
            marker="o",
            color="black",
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        leg = axB.get_legend()
        if debug:
            print(f"Panel B legend exists before remove: {leg is not None}")
        if leg is not None:
            leg.remove()

    # ---------------- Panel C: top 10 subjects ----------------
    subj_means = df.set_index(subject_col)[idp_cols].mean(axis=1, skipna=True)
    top_subjects = subj_means.sort_values(ascending=False).head(10)

    if debug:
        print("\n--- Panel C ---")
        print("Top subjects:", list(top_subjects.index))

    if "n_obs" in df.columns:
        n_map = df.set_index(subject_col)["n_obs"].to_dict()
        labels = [
            f"{str(subj)} (n={int(n_map.get(subj, 0))})"
            for subj in top_subjects.index
        ]
    else:
        labels = [str(subj) for subj in top_subjects.index]

    bars = axC.barh(
        labels,
        top_subjects.values,
        color=STYLE.PRIMARY,
        alpha=STYLE.BAR_ALPHA,
        height=0.82,
        edgecolor="none",
    )
    axC.invert_yaxis()
    axC.set_title(
        "Top 10 subjects with the highest average pairwise RPD",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )
    axC.set_xlabel("Mean pairwise RPD (%)", fontsize=STYLE.AXIS_SIZE)
    axC.set_ylabel("")
    axC.tick_params(axis="both", labelsize=STYLE.TICK_SIZE)
    _strip_spines(axC)
    axC.margins(y=0.08)
    _annotate_hbar(axC, bars, fmt="{:.1f}")

    fig.tight_layout(rect=[0, 0, 0.78, 0.97])

    if savepath:
        plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

    return _handle_rep_show(fig, rep, "", show)

# ============================================================================
# 3.  MULTIVARIATE BATCH DIFFERENCE (MAHALANOBIS)
# ============================================================================

def plot_MultivariateBatchDifference(
    df,
    batch_col:    str        = "batch",
    value_col:    str        = "mdval",
    avg_label:    str        = "average_batch",
    figsize:      tuple      = (8, 6),
    sort_by_value:bool       = True,
    sort_rest_desc:bool      = True,
    value_format: str        = "{:.1f}",
    savepath:     str | None = None,
    rep                      = None,
    show:         bool       = False,
):
    """
    Horizontal bar chart of Mahalanobis distances per batch.
    The average batch is pinned to the top; remaining batches are optionally sorted.
    """
    apply_plot_theme()

    if batch_col not in df.columns or value_col not in df.columns:
        raise ValueError("DataFrame must contain the specified batch and value columns.")

    avg_df  = df[df[batch_col] == avg_label][[batch_col, value_col]]
    rest_df = df[df[batch_col] != avg_label][[batch_col, value_col]].copy()

    if sort_by_value:
        rest_df = rest_df.sort_values(value_col, ascending=not sort_rest_desc)

    plot_df = pd.concat([avg_df, rest_df], ignore_index=True)
    batches = plot_df[batch_col].astype(str).tolist()
    values  = plot_df[value_col].astype(float).tolist()
    y_pos   = np.arange(len(batches))

    fig, ax = plt.subplots(figsize=figsize)
    colors  = [STYLE.SECONDARY if b == avg_label else STYLE.PRIMARY for b in batches]
    bars    = ax.barh(
        y_pos, values, color=colors,
        edgecolor="none", alpha=STYLE.BAR_ALPHA, height=0.78,
    )

    ax.set_yticks(y_pos)
    ax.invert_yaxis()
    _configure_hbar_panel(
        ax,
        title="Multivariate batch differences",
        xlabel="Mahalanobis distance",
        yticklabels=batches,
        show_legend=_batch_difference_legend_handles(),
        legend_loc="lower right",
        legend_title="Colour key",
    )
    _annotate_hbar(ax, bars, fmt=value_format)
    ax.text(
        0.01, 0.98,
        "Mahalanobis distance per batch",
        transform=ax.transAxes,
        fontsize=STYLE.TICK_SIZE,
        fontstyle="italic",
        va="bottom",
        ha="left",
    )

    _finalise(fig)
    if savepath:
        plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

    return _handle_rep_show(fig, rep, "", show)

# ============================================================================
# 4.  MIXED EFFECTS — PART 1  (ICC / n_sig / WCV)
# ============================================================================

def plot_MixedEffectsPart1(
    df,
    idp_col:          str        = "IDP",
    metrics:          list | None = None,
    plot_type:        str        = "bar",
    idp_style:        dict | None = None,
    limit_idps:       int         = 10,
    figsize:          tuple       = (4, 6),
    seed:             int | None  = None,
    savepath:         str | None  = None,
    rep                          = None,
    show:             bool       = False,
    display:          str        = "subplots",
    metric:           str | None = None,
    show_pairwise_heatmap: bool = True,
):
    """
    Plot ICC, residual batch significance counts, and WCV per IDP.

    display : 'subplots' (all metrics in one figure),
              'separate'  (one figure per metric),
              'single'    (one figure for `metric`).
    """
    apply_plot_theme()

    if display not in ("subplots", "separate", "single"):
        raise ValueError("display must be 'subplots', 'separate', or 'single'.")

    if metrics is None:
        metrics = ["n_is_batchSig", "ICC", "WCV"]

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metrics in df: {missing}")

    idps = list(df[idp_col].astype(str).values)
    n_idps = len(idps)

    if idp_style is None:
        idp_style = _build_default_idp_style(idps)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    def _draw(ax: plt.Axes, metric_name: str) -> None:
        vals = pd.to_numeric(df[metric_name], errors="coerce").values
        y_pos = np.arange(n_idps)

        if plot_type == "bar":
            # ------------------------------------------------------------
            # Pairwise batch/site-count plot
            # ------------------------------------------------------------
            if metric_name == "n_is_batchSig":
                bars = ax.barh(
                    y_pos, vals, height=0.85,
                    color=STYLE.PRIMARY, edgecolor="none", alpha=STYLE.BAR_ALPHA
                )
                ax.set_yticks(y_pos)
                ax.invert_yaxis()

                _configure_hbar_panel(
                    ax,
                    title="Site-pairwise batch effects",
                    xlabel="Number of significant pairwise site differences\n(Bonferroni-adjusted within each feature)",
                    yticklabels=idps,
                )
                ax.margins(y=0.12)

                offset = _bar_label_offset(ax, "x")
                for bar, v in zip(bars, vals):
                    if np.isnan(v):
                        continue
                    ax.text(
                        v + offset,
                        bar.get_y() + bar.get_height() / 2,
                        f"{int(v)}",
                        va="center", ha="left",
                        fontsize=STYLE.LABEL_SIZE, fontweight="bold",
                    )
                return

            # ------------------------------------------------------------
            # ICC plot
            # ------------------------------------------------------------
            if metric_name.upper() == "ICC":
                colors = [
                    (STYLE.NEUTRAL if np.isnan(v)
                     else STYLE.ICC_EXCELLENT if v >= 0.90
                     else STYLE.ICC_GOOD if v >= 0.75
                     else STYLE.ICC_LOWER)
                    for v in vals
                ]
                bars = ax.barh(
                    y_pos, vals, height=0.85,
                    color=colors, edgecolor="none", alpha=STYLE.BAR_ALPHA
                )
                for thr in (0.50, 0.75, 0.90):
                    ax.axvline(thr, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)

                ax.set_yticks(y_pos)
                ax.set_xlim(0, 1.05)
                ax.invert_yaxis()

                _configure_hbar_panel(
                    ax,
                    title="Between-subject reliability",
                    xlabel="Intraclass correlation (ICC)",
                    yticklabels=idps,
                    show_legend=_icc_legend_handles(),
                    legend_loc="center left",
                    legend_title="ICC color key",
                )
                ax.margins(y=0.12)

                offset = _bar_label_offset(ax, "x")
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(
                            v + offset,
                            bar.get_y() + bar.get_height() / 2,
                            f"{v:.2f}",
                            va="center", ha="left",
                            fontsize=STYLE.LABEL_SIZE, fontweight="bold",
                        )
                return

            # ------------------------------------------------------------
            # Generic bar plot
            # ------------------------------------------------------------
            colors_g = [idp_style[idp][0] for idp in idps]
            bars = ax.bar(
                y_pos, vals,
                color=colors_g, edgecolor="none", alpha=STYLE.BAR_ALPHA
            )
            ax.set_xticks(y_pos)
            ax.set_xticklabels(idps, rotation=45, ha="right", fontsize=STYLE.TICK_SIZE)
            ax.set_ylabel(metric_name, fontsize=STYLE.AXIS_SIZE)
            ax.set_title(metric_name, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=10)
            ax.tick_params(axis="y", labelsize=STYLE.TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=STYLE.GRID_ALPHA)
            _strip_spines(ax)
            _annotate_vbar(ax, bars)

        elif plot_type == "box":
            clean = vals[~np.isnan(vals)]
            if clean.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                return

            ax.boxplot(clean, vert=True, widths=0.6, patch_artist=True)
            ax.set_xticks([])
            ax.set_ylabel(metric_name, fontsize=STYLE.AXIS_SIZE)
            ax.set_title(metric_name, fontsize=STYLE.TITLE_SIZE, fontweight="bold", pad=10)
            ax.tick_params(axis="both", labelsize=STYLE.TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=STYLE.GRID_ALPHA)
            _strip_spines(ax)

            series = pd.Series(vals, index=idps)
            show_legend = n_idps <= limit_idps
            handles, labels_l = [], []

            for idp in idps:
                v = series.loc[idp]
                if np.isnan(v):
                    continue
                jx = rng.uniform(-0.06, 0.06) + 1.0
                if show_legend:
                    color, marker = idp_style[idp]
                    sc = ax.scatter(
                        jx, v, marker=marker, color=color,
                        edgecolor="k", s=70, linewidths=0.6, zorder=5
                    )
                    if idp not in labels_l:
                        handles.append(sc)
                        labels_l.append(idp)
                else:
                    ax.scatter(
                        jx, v, marker="o", color=STYLE.NEUTRAL,
                        edgecolor="k", s=40, linewidths=0.6, zorder=4
                    )

            vmin, vmax_ = np.nanmin(clean), np.nanmax(clean)
            rng_ = vmax_ - vmin if vmax_ != vmin else max(abs(vmax_), 1.0)
            ax.set_ylim(vmin - 0.06 * rng_, vmax_ + 0.12 * rng_)

    # ------------------------------------------------------------------
    # SUBPLOTS MODE
    # ------------------------------------------------------------------
    if display == "subplots":
        n_met = len(metrics)
        fig_h = max(figsize[1], 4.8 * n_met)
        fig_w = max(figsize[0], 8.0)

        fig, axes = plt.subplots(
            n_met, 1,
            figsize=(fig_w, fig_h),
            squeeze=False
        )
        axes = axes.flatten()

        for i, m in enumerate(metrics):
            _draw(axes[i], m)

        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

        if rep is not None:
            rep.log_plot(fig, "")
            plt.close(fig)

            # optional pairwise heatmap
            if show_pairwise_heatmap and "pairwise_site_tests" in df.columns and "n_is_batchSig" in metrics:
                heatmap_fig = plot_PairwiseSiteDifferencesHeatmap(
                    df,
                    idp_col=idp_col,
                    records_col="pairwise_site_tests",
                    sig_key="sig_bonf",
                    rep=None,
                    show=False,
                )
                if heatmap_fig is not None:
                    rep.log_plot(heatmap_fig, "")
                    plt.close(heatmap_fig)

            return None, None

        if show:
            plt.show()
        return fig, axes[:len(metrics)].tolist()

    # ------------------------------------------------------------------
    # SINGLE MODE
    # ------------------------------------------------------------------
    elif display == "single":
        if metric is None or metric not in metrics:
            raise ValueError(f"Provide a valid metric name via `metric=`; got {metric!r}.")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _draw(ax, metric)
        _finalise(fig)

        if savepath:
            import os
            base, ext = os.path.splitext(savepath)
            plt.savefig(f"{base}_{metric}{ext or '.png'}", dpi=STYLE.DPI, bbox_inches="tight")

        if rep is not None:
            rep.log_plot(fig, f"Mixed effects metric {metric}")
            plt.close(fig)

            if show_pairwise_heatmap and metric == "n_is_batchSig" and "pairwise_site_tests" in df.columns:
                heatmap_fig = plot_PairwiseSiteDifferencesHeatmap(
                    df,
                    idp_col=idp_col,
                    records_col="pairwise_site_tests",
                    sig_key="sig_bonf",
                    rep=None,
                    show=False,
                )
                if heatmap_fig is not None:
                    rep.log_plot(heatmap_fig, "Pairwise site-difference heatmap")
                    plt.close(heatmap_fig)

            return None, None

        if show:
            plt.show()
        return fig, [ax]

    # ------------------------------------------------------------------
    # SEPARATE MODE
    # ------------------------------------------------------------------
    else:
        figs: dict[str, plt.Figure] = {}
        for m in metrics:
            fig_i, ax_i = plt.subplots(1, 1, figsize=figsize)
            _draw(ax_i, m)
            _finalise(fig_i)
            figs[m] = fig_i

            if savepath:
                import os
                base, ext = os.path.splitext(savepath)
                fig_i.savefig(f"{base}_{m}{ext or '.png'}", dpi=STYLE.DPI, bbox_inches="tight")

            if rep is not None:
                rep.log_plot(fig_i, f"Mixed effects metric {m}")
                plt.close(fig_i)
            elif show:
                fig_i.show()

        if rep is not None and show_pairwise_heatmap and "pairwise_site_tests" in df.columns and "n_is_batchSig" in metrics:
            heatmap_fig = plot_PairwiseSiteDifferencesHeatmap(
                df,
                idp_col=idp_col,
                records_col="pairwise_site_tests",
                sig_key="sig_bonf",
                rep=None,
                show=False,
            )
            if heatmap_fig is not None:
                rep.log_plot(heatmap_fig, "Pairwise site-difference heatmap")
                plt.close(heatmap_fig)

        if rep is not None:
            return None, None
        return figs

def plot_PairwiseSiteDifferencesHeatmap(
    df: pd.DataFrame,
    idp_col: str = "IDP",
    records_col: str = "pairwise_site_tests",
    sig_key: str = "sig_bonf",
    title: str = "Pairwise site differences by feature",
    p_thr: float = 0.05,
    figsize: tuple = (10, 8),
    top_n: int = 20,
    max_full_pairs: int = 50,
    rep=None,
    show: bool = False,
):
    """
    Discrete binary heatmap:
    grey = not significant
    red  = significant after Bonferroni correction
    """
    apply_plot_theme()

    mat, site_code_map, use_compact = _build_pairwise_pair_matrix(
        df,
        idp_col=idp_col,
        records_col=records_col,
        sig_key=sig_key,
        top_n=top_n,
        max_full_pairs=max_full_pairs,
    )

    if mat.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, "No pairwise site-test records found.", ha="center", va="center")
        return _handle_rep_show(fig, rep, "", show)

    fig_h = max(figsize[1], 0.35 * len(mat.index) + 2.5)
    fig_w = max(figsize[0], 0.35 * len(mat.columns) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    annot = mat.replace({0.0: "", 1.0: "•"})

    cmap = ListedColormap(["#f2f2f2", "#b2182b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        norm=norm,
        linewidths=0.25,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar=False,
        annot_kws=dict(fontsize=max(7, STYLE.ANNOT_SIZE), fontweight="bold"),
    )

    mode_text = (
        f"Compact view: top {top_n} most frequent pairs"
        if use_compact else
        "Full view: all pairwise site comparisons"
    )

    ax.set_title(
        f"{title}\n"
        f"({mode_text}; • = significant after Bonferroni correction, grey = not significant)",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Pair code", fontsize=STYLE.AXIS_SIZE)
    ax.set_ylabel("Feature / IDP", fontsize=STYLE.AXIS_SIZE)
    ax.tick_params(axis="x", labelsize=STYLE.TICK_SIZE, rotation=45)
    ax.tick_params(axis="y", labelsize=STYLE.TICK_SIZE)
    _strip_spines(ax)

    key_lines = [f"{code} = {site}" for code, site in site_code_map.items()]
    key_text = "Site code key: " + ", ".join(key_lines)

    fig.text(
        0.99, -0.04,
        textwrap.fill(key_text, width=110),
        ha="right", va="bottom",
        fontsize=max(7, STYLE.TICK_SIZE - 2),
        style="italic",
    )
    fig.subplots_adjust(bottom=0.16)

    _finalise(fig)
    return _handle_rep_show(fig, rep, "", show)

# ============================================================================
# 5.  MIXED EFFECTS — PART 2  (fixed-effect estimates + CIs)
# ============================================================================

def plot_MixedEffectsPart2(
    df,
    idp_col:         str        = "IDP",
    fix_eff:         tuple      = ("age", "sex"),
    p_thr:           float      = 0.05,
    effect_style:    dict | None = None,
    idp_order:       list | None = None,
    marker_size:     int         = 160,
    figsize:         tuple       = (9.0, 3.2),
    cap_width:       float       = 0.03,
    linewidth:       float       = 2.4,
    xtick_rotation:  int         = 25,
    highlight_color: str         = "red",
    savepath:        str | None  = None,
    rep                          = None,
    show:            bool        = False,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Fixed-effect estimates and 95% confidence intervals per IDP.

    Significant estimates (p < p_thr) are filled red; non-significant are open.
    One covariate per row, compact layout.
    """
    apply_plot_theme()

    from matplotlib.lines import Line2D

    effs = list(fix_eff)

    if idp_order is None:
        idps = list(df[idp_col].astype(str).values)
    else:
        bad = [i for i in idp_order if i not in df[idp_col].values]
        if bad:
            raise ValueError(f"Unknown IDPs in idp_order: {bad}")
        idps = list(idp_order)

    n_idps = len(idps)
    if not n_idps:
        raise ValueError("No IDPs provided.")

    if effect_style is None:
        palette = sns.color_palette("tab10", max(3, len(effs)))
        markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]
        effect_style = {
            e: (palette[i % len(palette)], markers[i % len(markers)])
            for i, e in enumerate(effs)
        }

    nrows = len(effs)
    ncols = 1

    fig_w = max(7.5, figsize[0])
    fig_h = max(3.2 * nrows, 3.0)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    axes = axes_grid.flatten()

    for idx, eff in enumerate(effs):
        ax = axes[idx]
        est_col = f"{eff}_est"
        p_col   = f"{eff}_pval"
        ciL_col = f"{eff}_ciL"
        ciU_col = f"{eff}_ciU"

        missing = [c for c in (est_col, p_col, ciL_col, ciU_col) if c not in df.columns]
        if missing:
            warnings.warn(f"Skipping effect '{eff}' — missing columns: {missing}")
            ax.set_visible(False)
            continue

        def _get(idp: str, col: str) -> float:
            row = df[df[idp_col] == idp]
            return pd.to_numeric(row.iloc[0].get(col, np.nan), errors="coerce") if row.shape[0] else np.nan

        ests  = np.array([_get(i, est_col) for i in idps], dtype=float)
        pvals = np.array([_get(i, p_col)   for i in idps], dtype=float)
        ciLs  = np.array([_get(i, ciL_col) for i in idps], dtype=float)
        ciUs  = np.array([_get(i, ciU_col) for i in idps], dtype=float)
        x     = np.arange(n_idps)

        for xi, lo, hi in zip(x, ciLs, ciUs):
            if np.isnan(lo) and np.isnan(hi):
                continue
            ax.plot([xi, xi], [lo, hi], color="black", linewidth=linewidth, zorder=1)
            ax.plot([xi - cap_width, xi + cap_width], [lo, lo], color="black", linewidth=linewidth, zorder=1)
            ax.plot([xi - cap_width, xi + cap_width], [hi, hi], color="black", linewidth=linewidth, zorder=1)

        for xi, est, pv in zip(x, ests, pvals):
            if np.isnan(est):
                continue
            sig = not np.isnan(pv) and pv < p_thr
            fc = highlight_color if sig else "white"
            lw_m = 1.4 if sig else 1.8
            ax.scatter(
                xi, est,
                marker="o",
                s=marker_size,
                facecolors=fc,
                edgecolors="black",
                linewidths=lw_m,
                zorder=5,
            )

        ax.set_xticks(x)
        rot = xtick_rotation if xtick_rotation is not None else _adaptive_rotation(idps, threshold=7, steep=45, mild=25)
        _set_ticklabels(ax, "x", idps, rotation=rot, ha="right" if rot else "center")
        ax.set_xlim(-0.6, n_idps - 1 + 0.6)
        ax.axhline(0, color="gray", linewidth=0.7, alpha=0.6)

        ax.set_title(
            eff.replace("_", " ").title(),
            fontsize=STYLE.TITLE_SIZE,
            fontweight="bold",
            pad=6,
        )
        ax.set_ylabel("Effect size (β)", fontsize=STYLE.AXIS_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=STYLE.GRID_ALPHA)
        _strip_spines(ax)

        all_vals = np.concatenate([v[~np.isnan(v)] for v in (ciLs, ciUs, ests)])
        if all_vals.size:
            vmin, vmax_ = np.nanmin(all_vals), np.nanmax(all_vals)
            vr = vmax_ - vmin if vmax_ != vmin else max(abs(vmax_), 1.0)
            ax.set_ylim(vmin - 0.10 * vr, vmax_ + 0.10 * vr)

    for j in range(len(effs), len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="black", markerfacecolor=highlight_color,
               markersize=8, linestyle="None", label=f"p < {p_thr}"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
               markersize=8, linestyle="None", label=f"p ≥ {p_thr}"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=STYLE.TICK_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath:
        plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

    if rep is not None:
        rep.log_plot(fig, "")
        plt.close(fig)
        return None, None

    if show:
        plt.show()

    return fig, axes[:len(effs)].tolist()


# ============================================================================
# 6.  ADDITIVE / MULTIPLICATIVE EFFECTS  (p-value matrix)
# ============================================================================
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_AddMultEffects(
    dfs,
    feature_col:    str        = "Feature",
    p_col:          str        = "p-value",
    labels:         list | None = None,
    p_thr:          float       = 0.05,
    cmap:           str         = "viridis",
    annot_fmt:      str         = "{:.1g}",
    vmax_logp:      float       = 10,
    figsize:        tuple       = (10, 8),
    show_colorbar:  bool        = True,
    savepath:       str | None  = None,
    value_scale:    str         = "p",
    rep                         = None,
    show:           bool        = False,
    annot_fontsize: float       = STYLE.ANNOT_SIZE,
    tick_fontsize:  float       = STYLE.TICK_SIZE,
    cbar_shrink:    float       = 0.2,
    linewidths:     float       = STYLE.HEATMAP_LINEWIDTHS,
    square:         bool        = False,
):
    """
    Feature significance heatmap (one or multiple DataFrames).

    value_scale : 'p'    — raw p-values in [0, 1]
                  'logp' — -log10(p) clipped at vmax_logp
    """
    apply_plot_theme()

    if value_scale not in ("p", "logp"):
        raise ValueError("value_scale must be 'p' or 'logp'.")

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    if labels is None:
        labels = [f"col{i+1}" for i in range(len(dfs))]

    if len(labels) != len(dfs):
        raise ValueError("labels must have the same length as dfs.")

    # Preserve first-seen feature order; do not sort by significance
    all_features = list(dict.fromkeys(
        str(f) for df in dfs for f in df[feature_col]
    ))

    _TINY = 1e-300

    # ---- single DF: horizontal bar plot ----------------------------------
    if len(dfs) == 1:
        s = dfs[0].set_index(feature_col)[p_col]

        plot_df = (
            pd.DataFrame({
                "Feature": all_features,
                "p": [s.get(f, np.nan) for f in all_features],
            })
            .assign(logp=lambda d: -np.log10(np.maximum(d["p"], _TINY)))
        )

        sig_color = "#E45756"   # red
        ns_color  = "#54A24B"   # green
        colors = np.where(plot_df["p"] < p_thr, sig_color, ns_color)

        fig, ax = plt.subplots(figsize=(6, max(3, 0.42 * len(plot_df))))

        bars = ax.barh(
            plot_df["Feature"], plot_df["logp"],
            color=colors, alpha=STYLE.BAR_ALPHA, height=0.85
        )
        ax.axvline(
            -np.log10(p_thr),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"p = {p_thr}"
        )

        _apply_hbar_style(
            ax,
            title=f"Batch effect significance per feature (p < {p_thr})",
            xlabel=r"$-\log_{10}(p-value)$",
        )

        legend_handles = [
            Patch(facecolor=sig_color, edgecolor="none", label=f"p < {p_thr}"),
            Patch(facecolor=ns_color, edgecolor="none", label=f"p ≥ {p_thr}"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.2,
                   label=f"threshold = {p_thr}"),
        ]
        ax.legend(handles=legend_handles, frameon=False, fontsize=STYLE.TICK_SIZE)

        # annotate, using ">" for clipped values
        offset = _bar_label_offset(ax, "x")
        for bar in bars:
            v = bar.get_width()
            label = ">300" if v >= 299.5 else f"{v:.2f}"
            ax.text(
                v + offset,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center", ha="left",
                fontsize=STYLE.LABEL_SIZE, fontweight="bold",
            )

        if savepath:
            plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")
        return _handle_rep_show(fig, rep, "", show)

    # ---- multiple DFs: heatmap -------------------------------------------
    pv_mat = pd.DataFrame(index=all_features, columns=labels, dtype=float)

    for df, lab in zip(dfs, labels):
        tmp = (
            df[[feature_col, p_col]]
            .drop_duplicates(subset=feature_col)
            .set_index(feature_col)[p_col]
            .astype(float)
        )
        for feat in all_features:
            pv_mat.at[feat, lab] = tmp.get(feat, np.nan)

    pv_safe = pv_mat.copy().astype(float)
    pv_safe[np.isclose(pv_safe, 0.0)] = _TINY
    logp_mat = (-np.log10(pv_safe)).clip(upper=vmax_logp)

    def _star(p_orig: float) -> str:
        return "*" if (not pd.isna(p_orig) and p_orig < p_thr) else ""

    if value_scale == "p":
        plot_mat = pv_mat.copy().fillna(1.0).clip(lower=0.0, upper=1.0)
        annot = pv_mat.copy().astype(object)
        for r in all_features:
            for c in labels:
                v = pv_mat.at[r, c]
                annot.at[r, c] = "" if pd.isna(v) else f"{annot_fmt.format(float(v))}{_star(v)}"
        cbar_label = "p-value"
    else:
        plot_mat = logp_mat.copy().fillna(0.0)
        annot = logp_mat.copy().astype(object)
        for r in all_features:
            for c in labels:
                v = logp_mat.at[r, c]
                p_orig = pv_mat.at[r, c]
                annot.at[r, c] = "" if pd.isna(v) else f"{v:.3f}{_star(p_orig)}"
        cbar_label = "-log\u2081\u2080(p-value)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_mat, ax=ax, annot=annot, fmt="", cmap=cmap,
        cbar=show_colorbar, linewidths=linewidths, linecolor="white",
        xticklabels=labels, yticklabels=all_features,
        square=square, annot_kws=dict(fontsize=annot_fontsize),
    )

    if show_colorbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=STYLE.AXIS_SIZE)

        max_val = float(np.nanmax(plot_mat.values))
        if value_scale == "p":
            ticks = np.linspace(0.0, min(1.0, max_val), num=5)
            if not np.isclose(ticks, p_thr).any():
                ticks = np.sort(np.unique(np.append(ticks, p_thr)))
            tick_labels = [
                f"p_thr={p_thr:g}" if np.isclose(t, p_thr) else f"{t:.2g}"
                for t in ticks
            ]
        else:
            ticks = np.linspace(0.0, min(vmax_logp, max_val), num=5)
            thr_log = -np.log10(p_thr) if p_thr and p_thr > 0 else None
            if thr_log is not None and not np.isclose(ticks, thr_log).any():
                ticks = np.sort(np.unique(np.append(ticks, thr_log)))
            tick_labels = []
            for t in ticks:
                if thr_log is not None and np.isclose(t, thr_log):
                    tick_labels.append(f"p={p_thr:g}\n(−log₁₀={t:.2f})")
                else:
                    tick_labels.append(f"{t:.2f}")

        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=tick_fontsize)

        try:
            pos = cbar.ax.get_position()
            cbar.ax.set_position([pos.x0, pos.y0, pos.width * cbar_shrink, pos.height])
        except Exception:
            pass

    ax.set_xlabel("", fontsize=STYLE.AXIS_SIZE)
    ax.set_ylabel("", fontsize=STYLE.AXIS_SIZE)
    ax.set_title(
        f"Feature significance (p < {p_thr})",
        fontsize=STYLE.TITLE_SIZE,
        fontweight="bold",
        pad=10,
    )

    rot = _adaptive_rotation(labels, threshold=6, steep=45, mild=25)
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=rot)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    _finalise(fig)

    if savepath:
        plt.savefig(savepath, dpi=STYLE.DPI, bbox_inches="tight")

    return _handle_rep_show(fig, rep, "", show)





