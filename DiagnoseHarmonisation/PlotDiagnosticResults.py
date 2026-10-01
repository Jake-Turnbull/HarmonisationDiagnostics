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


FEATURE_LABEL_HIDE_THRESHOLD = 20


def _feature_tick_policy(labels: list[str]) -> tuple[list[str], int]:
    """For many features keep ticks but hide labels; for few features use diagonal labels."""
    if len(labels) > FEATURE_LABEL_HIDE_THRESHOLD:
        return ["" for _ in labels], 0
    return labels, 45


def summarise_lmm_biological_effects(results_df) -> pd.DataFrame:
    """summarise per-covariate beta magnitude and variance contribution columns."""
    import pandas as pd

    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return pd.DataFrame(columns=["covariate", "mean_abs_beta", "mean_ols_partial_r2", "mean_batch_adjusted_partial_r2"])

    prefixes = {
        "mean_abs_beta": "ols_abs_beta_",
        "mean_ols_partial_r2": "ols_partial_r2_",
    }

    covariates = set()
    for prefix in prefixes.values():
        covariates.update(col[len(prefix):] for col in results_df.columns if col.startswith(prefix))

    rows = []
    for covariate in sorted(covariates):
        row = {"covariate": covariate}
        has_value = False
        for output_name, prefix in prefixes.items():
            col = f"{prefix}{covariate}"
            if col in results_df.columns:
                values = pd.to_numeric(results_df[col], errors="coerce").to_numpy(dtype=float)
                row[output_name] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                has_value = has_value or np.isfinite(values).any()
            else:
                row[output_name] = np.nan
        if has_value:
            rows.append(row)

    # Add a dedicated batch row so all new biology summary plots explicitly
    # show residual batch effects alongside covariates.
    batch_row = {
        "covariate": "batch",
        "mean_abs_beta": np.nan,
        "mean_ols_partial_r2": np.nan,
        "mean_batch_adjusted_partial_r2": np.nan,
    }

    batch_beta_cols = [c for c in results_df.columns if c.startswith("batch_fixed_beta_")]
    if batch_beta_cols:
        # Being passed as a numpy array in some called, write check only apply if pan
        batch_beta_values = pd.to_numeric(
            np.asarray(results_df[batch_beta_cols], dtype=float).ravel(),
            errors="coerce",
        )
        if np.isfinite(batch_beta_values).any():
            batch_row["mean_abs_beta"] = float(np.nanmean(np.abs(batch_beta_values)))

    if "batch_fixed_partial_r2" in results_df.columns:
        batch_partial = pd.to_numeric(results_df["batch_fixed_partial_r2"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(batch_partial).any():
            batch_row["mean_ols_partial_r2"] = float(np.nanmean(batch_partial))

    if "delta_R2_batch" in results_df.columns:
        delta = pd.to_numeric(results_df["delta_R2_batch"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(delta).any():
            batch_row["mean_batch_adjusted_partial_r2"] = float(np.nanmean(delta))

    if any(np.isfinite(batch_row[key]) for key in ("mean_abs_beta", "mean_ols_partial_r2", "mean_batch_adjusted_partial_r2")):
        rows.append(batch_row)

    return pd.DataFrame(rows)


def plot_lmm_biological_effects(
    results_df,
    title_prefix: str = "LMM covariate effects",
) -> list[tuple[str, mfig.Figure]]:
    """Plot covariate-level beta magnitude and explained variance summaries."""
    summary_df = summarise_lmm_biological_effects(results_df)
    if summary_df.empty:
        return []

    plot_df = summary_df.sort_values(
        ["mean_batch_adjusted_partial_r2", "mean_ols_partial_r2", "mean_abs_beta"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4.5, 0.6 * len(plot_df) + 1.5)), squeeze=False)
    ax_beta, ax_ols, ax_lmm = axes.ravel()

    ax_beta.barh(y, plot_df["mean_abs_beta"], color="C0", alpha=0.85)
    ax_beta.set_title("Mean |beta|")
    ax_beta.set_xlabel("Coefficient magnitude")
    ax_beta.set_yticks(y)
    ax_beta.set_yticklabels(plot_df["covariate"].astype(str).tolist())
    ax_beta.invert_yaxis()

    ax_ols.barh(y, plot_df["mean_ols_partial_r2"], color="C1", alpha=0.85)
    ax_ols.set_title("OLS partial R2")
    ax_ols.set_xlabel("Mean explained variance")
    ax_ols.set_yticks(y)
    ax_ols.set_yticklabels([])
    ax_ols.invert_yaxis()

    ax_lmm.barh(y, plot_df["mean_batch_adjusted_partial_r2"], color="C2", alpha=0.85)
    ax_lmm.set_title("Batch incremental partial R2")
    ax_lmm.set_xlabel("Mean explained variance")
    ax_lmm.set_yticks(y)
    ax_lmm.set_yticklabels([])
    ax_lmm.invert_yaxis()

    for ax in (ax_beta, ax_ols, ax_lmm):
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    return [(f"{title_prefix}: covariate beta and variance summary", fig)]

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
        include_delta_r2 (bool): If True, add a delta_R2_batch plot.
        include_status_summary (bool): Retained for backward compatibility.

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

    required_cols = ["feature", "ICC", "R2_covariates", "R2_covariates_batch"]
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
    # Get Pvalues for ICC and convert to numeric, coercing errors to NaN
    # If significant, plot in red otherwise in blue
    # Prefer BH-FDR corrected p-values (multiple comparisons across features).
    icc_pvalues = pd.to_numeric(df.get("pval_LRT_random_mixture_fdr", pd.Series([np.nan] * len(df_plot))), errors="coerce").to_numpy()
    if not np.isfinite(icc_pvalues).any():
        icc_pvalues = pd.to_numeric(df.get("pval_LRT_random_mixture", pd.Series([np.nan] * len(df_plot))), errors="coerce").to_numpy()

    icc_vals = pd.to_numeric(df_plot["ICC"], errors="coerce").to_numpy()
    # Handle NaN p-values by coloring them gray
    colors = np.where(np.isnan(icc_pvalues), "gray", np.where(icc_pvalues < 0.05, "red", "blue"))
    ax1.bar(x, icc_vals, color=colors, alpha=0.8)

    # Add color legend for significance
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Significant ICC (p < 0.05)'),
        Line2D([0], [0], color='blue', lw=4, label='Non-significant ICC (p >= 0.05)'),
        Line2D([0], [0], color='gray', lw=4, label='P-value unavailable')
    ]
    ax1.legend(handles=legend_elements, frameon=False)
    ax1.set_title("ICC per feature")
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("ICC")
    ax1.set_ylim(bottom=0)
    # Add a legend explaining colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Significant ICC (p < 0.05)'),
        Line2D([0], [0], color='blue', lw=4, label='Non-significant ICC (p >= 0.05)')
    ]

    ax1.set_xticks(x)
    shown_labels_1, rot_1 = _feature_tick_policy(list(x_labels))
    ax1.set_xticklabels(shown_labels_1, rotation=rot_1, ha="right" if rot_1 else "center")

    figs.append(("ICC per feature", fig1))

    """ # Remove the ordered plot as unnecessary; the unsorted plot is more informative and can be used to identify features with high ICC.
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
    shown_labels_2, rot_2 = _feature_tick_policy(list(x_labels2))
    ax2.set_xticklabels(shown_labels_2, rotation=rot_2, ha="right" if rot_2 else "center")

    figs.append(("ICC per feature sorted", fig2)) """

    # -------------------------------------------------
    # 3) Covariates and covariates+batch R²
    # -------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    x_r2 = np.arange(len(df_plot))

    r2_cov = pd.to_numeric(df_plot["R2_covariates"], errors="coerce").to_numpy()
    r2_cov_batch = pd.to_numeric(df_plot["R2_covariates_batch"], errors="coerce").to_numpy()

    ax3.plot(x_r2, r2_cov, label="Covariates R²", linewidth=1.5)
    ax3.plot(x_r2, r2_cov_batch, label="Covariates + Batch R²", linewidth=1.5)

    ax3.set_title("Covariates and Covariates + Batch R² per feature")
    ax3.set_xlabel("Feature")
    ax3.set_ylabel("R²")
    ax3.set_ylim(bottom=0)

    ax3.set_xticks(x_r2)
    shown_labels_3, rot_3 = _feature_tick_policy(list(x_labels))
    ax3.set_xticklabels(shown_labels_3, rotation=rot_3, ha="right" if rot_3 else "center")
    ax3.legend()

    figs.append(("Covariates and Covariates + Batch R² per feature", fig3))

    # -------------------------------------------------
    # 4) Delta R²
    # -------------------------------------------------
    if include_delta_r2 and "delta_R2_batch" in df.columns:
        fig4, ax4 = plt.subplots(figsize=(14, 5))
        df_delta = df_plot.copy()
        df_delta["delta_R2_batch"] = pd.to_numeric(df_delta["delta_R2_batch"], errors="coerce")

        delta_vals = df_delta["delta_R2_batch"].to_numpy()
        ax4.bar(np.arange(len(df_delta)), delta_vals, alpha=0.8)

        ax4.set_title("Delta R² per feature (Covariates + Batch - Covariates)")
        ax4.set_xlabel("Feature")
        ax4.set_ylabel("Delta R²")

        ax4.set_xticks(np.arange(len(df_delta)))
        delta_labels = df_delta["feature"].astype(str).tolist()
        shown_delta_labels, rot_delta = _feature_tick_policy(delta_labels)
        ax4.set_xticklabels(shown_delta_labels, rotation=rot_delta, ha="right" if rot_delta else "center")

        figs.append(("Delta R² per feature", fig4))

    figs.extend(plot_lmm_biological_effects(df, title_prefix="LMM covariate effects"))

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
    # Create figure with a single axis for batch-wise histogram overlays.
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot()
    _plot_zscore_hist_on_axis(
        ax=ax1,
        data=data,
        batch=batch,
        probablity_distribution=probablity_distribution,
        draw_PDF=draw_PDF,
    )

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


def _plot_zscore_hist_on_axis(ax, data, batch, probablity_distribution=False, draw_PDF=True):
    import matplotlib
    from scipy import stats

    unique_batches = np.unique(batch)
    colors = matplotlib.pyplot.get_cmap("tab10", len(unique_batches))
    density = bool(probablity_distribution)

    for i in unique_batches:
        batch_data = data[batch == i, :].flatten()
        color = colors(np.where(unique_batches == i)[0][0])
        ax.hist(batch_data, bins=80, density=density, alpha=0.5, label=str(i), color=color)
        if draw_PDF:
            mu, std = np.mean(batch_data), np.std(batch_data)
            if np.isfinite(std) and std > 0:
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                ax.plot(x, p, color=color, linewidth=2)


@rep_plot_wrapper
def Z_Score_Raw_Residual_Plot(raw_data, residual_data, batch, probablity_distribution=True, draw_PDF=True):
    """Plot side-by-side histograms for raw and covariate-residual robust z-scores."""
    if not isinstance(raw_data, np.ndarray):
        raise ValueError("raw_data must be a NumPy array.")
    if raw_data.ndim != 2:
        raise ValueError("raw_data must be a 2D array (samples x features).")
    if not isinstance(batch, np.ndarray):
        if isinstance(batch, list):
            batch = np.array(batch)
        else:
            raise ValueError("batch must be a NumPy array or a list.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    ax_raw, ax_resid = axes.ravel()

    _plot_zscore_hist_on_axis(
        ax=ax_raw,
        data=raw_data,
        batch=batch,
        probablity_distribution=probablity_distribution,
        draw_PDF=draw_PDF,
    )
    ax_raw.set_title("Raw data")
    ax_raw.set_xlabel("Robust z-score")
    ax_raw.set_ylabel("Frequency" if not probablity_distribution else "Density")
    ax_raw.set_xlim([-8, 8])
    ax_raw.invert_xaxis()
    ax_raw.legend(title="Batch")

    if residual_data is None:
        ax_resid.set_title("Covariate-residual data (unavailable)")
        ax_resid.set_axis_off()
    else:
        residual_data = np.asarray(residual_data)
        if residual_data.ndim != 2:
            raise ValueError("residual_data must be a 2D array (samples x features).")
        _plot_zscore_hist_on_axis(
            ax=ax_resid,
            data=residual_data,
            batch=batch,
            probablity_distribution=probablity_distribution,
            draw_PDF=draw_PDF,
        )
        ax_resid.set_title("Covariate-residual data")
        ax_resid.set_xlabel("Robust z-score")
        ax_resid.set_ylabel("Frequency" if not probablity_distribution else "Density")
        ax_resid.set_xlim([-8, 8])
        ax_resid.invert_xaxis()
        ax_resid.legend(title="Batch")

    fig.tight_layout()
    return [("Z-score histograms (raw vs residual)", fig)]
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
    FK_mode: bool = True,
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
        for key in ("p_value_fdr", "p_value", "pvalue", "p_val", "pvalues", "p"):
            if key in res_raw and res_raw[key] is not None:
                pval_raw = np.asarray(res_raw[key])
                break
        if pval_raw is None:
            pval_raw = np.full(stat_raw.shape, 1.0)

        stat_res = None
        pval_res = None
        if levene_results_resid is not None and comp in levene_results_resid:
            res_res = levene_results_resid[comp]
            stat_res = np.asarray(res_res.get("stat") if "stat" in res_res else res_res.get("statistic"))
            for key in ("p_value_fdr", "p_value", "pvalue", "p_val", "pvalues", "p"):
                if key in res_res and res_res[key] is not None:
                    pval_res = np.asarray(res_res[key])
                    break
            if pval_res is None:
                pval_res = np.full(stat_res.shape, 1.0)

        if stat_raw.shape[0] != n_features:
            continue

        ncols = 2 if stat_res is not None else 1
        square_side = max(6.0, min(12.0, n_features * 0.2))
        fig, axes = plt.subplots(1, ncols, figsize=(square_side * ncols, square_side), squeeze=False)
        x = np.arange(n_features)

        ax = axes[0, 0]
        bars = ax.bar(x, stat_raw, color="C0", alpha=0.8)
        sig_raw = pval_raw < alpha
        for xi, s in enumerate(sig_raw):
            if s:
                bars[xi].set_edgecolor("red")
                bars[xi].set_linewidth(1.5)
        if FK_mode:
            ax.set_ylabel("Fligner-Killeen statistic")
        else:
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
        if FK_mode:
            fig.suptitle(f"Fligner-Killeen test: {left_label} vs {right_label}")
        else:
            fig.suptitle(f"Levene's test: {left_label} vs {right_label}")

        for ax_plot in axes[0, :ncols]:
            if hasattr(ax_plot, "set_box_aspect"):
                ax_plot.set_box_aspect(1)
            ax_plot.set_xticks(x)
            shown_labels, rotation = _feature_tick_policy(list(feature_names))
            ax_plot.set_xticklabels(shown_labels, rotation=rotation, ha="right" if rotation else "center")

        ax.text(0.98, 0.95, f"n_significant_raw={int(np.sum(sig_raw))}", transform=ax.transAxes, ha="right", va="top")
        if stat_res is not None:
            ax2.text(0.98, 0.95, f"n_significant_resid={int(np.sum(sig_res))}", transform=ax2.transAxes, ha="right", va="top")

        figs.append((f"{'Fligner-Killeen' if FK_mode else 'Levene'}: {left_label} vs {right_label}", fig))

    return None if rep is not None else figs


@rep_plot_wrapper
def Levenes_Test(
    levene_results: dict,
    feature_names: list | None = None,
    *,
    alpha: float = 0.05,
    show: bool = False,
    rep=None,
    FK_mode: bool = True,
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
        # try a few common p-value keys, preferring BH-FDR corrected values
        pval = None
        for key in ("p_value_fdr", "p_value", "pvalue", "p_val", "pvalues", "p"):
            if key in res and res[key] is not None:
                pval = np.asarray(res[key])
                break
        if pval is None:
            # if absent, create non-significant mask
            pval = np.full(stat.shape, 1.0)

        if stat.shape[0] != n_features:
            continue

        square_side = max(6.0, min(12.0, n_features * 0.2))
        fig, ax = plt.subplots(figsize=(square_side, square_side))
        x = np.arange(n_features)
        bars = ax.bar(x, stat, color="C0", alpha=0.8)

        # highlight significant
        sig = pval < alpha
        for xi, s in enumerate(sig):
            if s:
                bars[xi].set_edgecolor("red")
                bars[xi].set_linewidth(1.5)

        left_label, right_label = (comp[0], comp[1]) if isinstance(comp, (list, tuple)) and len(comp) >= 2 else (str(comp), "")

        ax.set_title(f"{'Fligner-Killeen' if FK_mode else 'Levene'} test: {left_label} vs {right_label}")
        ax.set_xlabel("Feature")
        ax.set_ylabel(f"{'Fligner-Killeen' if FK_mode else 'Levene'} statistic")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)
        ax.set_xticks(x)
        shown_labels, rotation = _feature_tick_policy(list(feature_names))
        ax.set_xticklabels(shown_labels, rotation=rotation, ha="right" if rotation else "center")

        # annotate number of significant features
        n_sig = int(np.sum(sig))
        ax.text(0.98, 0.95, f"n_significant={n_sig}", transform=ax.transAxes, ha="right", va="top")

        figs.append((f"{'Fligner-Killeen' if FK_mode else 'Levene'}: {left_label} vs {right_label}", fig))

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
    # Note: PC-vs-metadata association is reported separately via
    # calculate_pc_associations()/plot_pc_r2_heatmap (see DiagnosticFunctions.py),
    # which avoids a Pearson correlation matrix that is sensitive to arbitrary
    # batch/categorical label encoding.

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
def plot_pc_r2_heatmap(r2_matrix, explained_variance=None, max_pcs: int = 5, *, show: bool = False) -> list[tuple[str, plt.Figure]]:
    """
    Heatmap of omnibus R^2 between each metadata variable (rows) and each
    principal component (columns), as produced by
    `DiagnosticFunctions.calculate_pc_associations`.

    Args:
        r2_matrix: DataFrame (or array-like) with variables as rows and PCs as
            columns, values in [0, 1].
        explained_variance: optional array-like of per-PC explained variance
            percentages (same order/length as `r2_matrix` columns), shown in
            the PC axis labels.
        max_pcs (int, optional): Hard limit on the number of PCs plotted
            (regardless of how many were computed), to keep the heatmap
            readable when many low-variance PCs are retained. Defaults to 5.
        show (bool, optional): Whether to display the figure interactively.

    Returns:
        list[tuple[str, plt.Figure]]: Single (caption, figure) pair.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(r2_matrix, pd.DataFrame):
        r2_matrix = pd.DataFrame(r2_matrix)
    r2_matrix = r2_matrix.astype(float)

    k = min(max(int(max_pcs), 1), 5, r2_matrix.shape[1])
    r2_matrix = r2_matrix.iloc[:, :k]

    if explained_variance is not None:
        explained_arr = np.asarray(explained_variance, dtype=float)
        col_labels = [
            f"{col}\n({explained_arr[i]:.1f}%)" if i < explained_arr.shape[0] else col
            for i, col in enumerate(r2_matrix.columns)
        ]
    else:
        col_labels = list(r2_matrix.columns)

    fig_width = max(6.0, 0.6 * r2_matrix.shape[1] + 3.0)
    fig_height = max(4.0, 0.5 * r2_matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        r2_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Omnibus R2"},
        xticklabels=col_labels,
        ax=ax
    )
    ax.set_title("Variance associated with metadata (R2)")
    ax.set_xlabel("Principal component (% variance explained)")
    ax.set_ylabel("Metadata variable")

    if show:
        plt.show()

    return [("PC-metadata association (Omnibus R2) heatmap", fig)]


@rep_plot_wrapper
def plot_batch_covariate_confounding(covariates, batch, confounding, *, show: bool = False, fontsize: float = 10) -> list[tuple[str, plt.Figure]]:
    """
    Batch-wise distribution plots per covariate, annotated with the batch-covariate
    confounding statistic from `DiagnosticFunctions.calculate_batch_covariate_confounding`
    (omnibus R2 for continuous covariates, Cramer's V for binary/categorical).
    Larger values indicate greater covariate imbalance across batches and
    therefore greater potential for confounding.

    Args:
        covariates: DataFrame (or 2D array-like) of shape (n_samples, n_covariates).
        batch: 1D array-like of batch labels.
        confounding: dict returned by `calculate_batch_covariate_confounding`
            (with "tidy" and "variable_types" keys).
        show (bool, optional): Whether to display the figure interactively.

    Returns:
        list[tuple[str, plt.Figure]]: Single (caption, figure) pair.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(covariates, pd.DataFrame):
        cov_df = covariates.reset_index(drop=True)
    else:
        cov_arr = np.asarray(covariates)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr[:, None]
        cov_df = pd.DataFrame(cov_arr, columns=[f"covariate_{i+1}" for i in range(cov_arr.shape[1])])

    if cov_df.shape[1] == 0:
        return []

    batch_series = pd.Series(np.asarray(batch)).astype(str).reset_index(drop=True)

    tidy = confounding.get("tidy") if isinstance(confounding, dict) else None
    var_types = confounding.get("variable_types", {}) if isinstance(confounding, dict) else {}
    stat_lookup: dict[str, tuple[str, float]] = {}
    if isinstance(tidy, pd.DataFrame):
        for _, row in tidy.iterrows():
            stat_lookup[str(row["Variable"])] = (str(row["Statistic"]), float(row["Value"]))

    n_cov = cov_df.shape[1]
    ncols = min(3, n_cov)
    nrows = int(np.ceil(n_cov / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(cov_df.columns):
        ax = axes_flat[i]
        vtype = var_types.get(col, "continuous")
        stat_name, stat_value = stat_lookup.get(col, (None, np.nan))
        plot_df = pd.DataFrame({"batch": batch_series, str(col): cov_df[col].to_numpy()}).dropna()

        if vtype == "continuous":
            sns.boxplot(data=plot_df, x="batch", y=str(col), ax=ax)
        else:
            counts = plot_df.groupby(["batch", str(col)]).size().unstack(fill_value=0)
            proportions = counts.div(counts.sum(axis=1), axis=0)
            proportions.plot(kind="bar", stacked=True, ax=ax, legend=(i == 0))
            ax.set_ylabel("Proportion", fontsize=fontsize)

        stat_label = f"{stat_name}={stat_value:.2f}" if stat_name is not None and np.isfinite(stat_value) else "n/a"
        ax.set_title(f"{col} ({vtype})\nbatch association: {stat_label}", fontsize=fontsize)
        ax.set_xlabel("Batch", fontsize=fontsize)
        ax.tick_params(axis="x", rotation=45, labelsize=fontsize)

    for j in range(n_cov, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Batch-covariate confounding: distributions by batch (larger values = greater imbalance)")
    fig.tight_layout()

    if show:
        plt.show()

    return [("Batch-covariate confounding: distributions by batch", fig)]


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
    explained_variance=None,
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
        explained_variance (optional): Optional array-like of per-PC explained
            variance percentages, used to annotate the PC1/PC2 axis labels.
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

    def _pc_axis_label(pc_index: int) -> str:
        if explained_variance is not None and pc_index < len(explained_variance):
            return f"Principal Component {pc_index+1} ({explained_variance[pc_index]:.1f}% variance)"
        return f"Principal Component {pc_index+1}"



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
        
        if UMAP_tuning == "cluster":
            # Test random distances and neighbourhood sizes, pick the one that forms the most compact clusters (e.g. using silhouette score or Davies-Bouldin index)
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_params = None
            best_embedding = None
            param_grid = [
                (n, d)
                for n in [10, 15, 30, 50]
                for d in [0.01, 0.1, 0.3]
            ]
            
            # Take several random subsamples of the data to choose optimal parameters for UMAP, to avoid overfitting to a single subsample
            n_samples = data.shape[0]
            if n_samples > 5000:
                idx = np.random.choice(n_samples, size=int(np.round(n_samples/10)), replace=False)
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
        
        ax[1].set_xlabel(_pc_axis_label(0))
        ax[1].set_ylabel(_pc_axis_label(1))
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
                ax_cov[1].set_xlabel(_pc_axis_label(0))
                ax_cov[1].set_ylabel(_pc_axis_label(1))
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
    data: np.ndarray,
    batch: np.ndarray,
    rep,
    max_components: int = 50,
    normalize: bool = True,
    caption_prefix: str = "Covariance comparison (PC space)",
) -> dict[str, Any]:
    """
    Compute pairwise Frobenius norms of covariance differences between batches

    Args:
        data: (n_samples, n_pcs) whole data matrix in real space
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

    if data.ndim != 2:
        raise ValueError("data must be a 2D array (n_samples x n_features)")

    n, n_features = data.shape
    k = min(n_features, max_components)
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
        data_b = data[idx, :k]
        cov_b = np.cov(data_b, rowvar=False, ddof=1)  # k x k
        cov_matrices[b] = cov_b
        valid_batches.append(b)

    # pooled covariance (using all samples)
    pooled_data = data[:, :k]
    pooled_cov = np.cov(pooled_data, rowvar=False, ddof=1)
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
    im = ax1.imshow(pairwise_norm, interpolation="nearest", aspect="auto", cmap="viridis")

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
    ax_raw.set_yticklabels(batches, rotation=45, ha="right")
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
        ax_resid.set_yticklabels(batches, rotation=45, ha="right")
        ax_resid.set_xlabel("Batch")
        if annotate:
            annotate_heatmap(ax_resid,M_resid)

        # One colorbar shared by both heatmaps
        cbar = fig.colorbar(im_resid, ax=ax_raw, fraction=0.046, pad=0.2,orientation="horizontal",location="top")
        cbar.set_label("Mahalanobis distance")
        cbar = fig.colorbar(im_resid, ax=ax_resid, fraction=0.046, pad=0.2,orientation="horizontal",location="top")

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
    if has_resid:
        ax_bar.set_ylabel("Mahalanobis distance")
        # Else we leave blank to avoid cluttering the y-axis with a single bar chart
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

def n_dim_umap_visualisation(data, batch, n_dimensions=2, covariates=None, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, show=False,marker_size=30):
    """
    Perform UMAP dimensionality reduction on the provided data and visualize the results in 2D or 3D scatter plots, colored by batch and optionally by covariates.

    This visualisation is standalone and not suitable for inclusion in report workflows due to interactive elements. For report workflows, use the `cluster_analysis_all` or 'cluster_analysis_umap' function instead.
    Parameters
    ----------
    data : array-like
        The input data for UMAP.
    batch : array-like
        Batch labels for coloring the points.
    n_dimensions : int, optional
        Number of dimensions for UMAP (default is 2).
    covariates : array-like, optional
        Additional covariates for coloring the points.
    n_neighbors : int, optional
        Number of neighbors for UMAP (default is 15).
    min_dist : float, optional
        Minimum distance for UMAP (default is 0.1).
    metric : str, optional
        Distance metric for UMAP (default is 'euclidean').
    random_state : int, optional
        Random state for reproducibility (default is 42).
    show : bool, optional
        Whether to show the plot (default is False).
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap

    def _is_numeric(values):
        arr = np.asarray(values)
        return np.issubdtype(arr.dtype, np.number)

    active_cbar = [None]
    active_cbar_ax = [None]

    def _ensure_cbar_ax():
        if active_cbar_ax[0] is None:
            active_cbar_ax[0] = fig.add_axes([0.82, 0.20, 0.16, 0.04])
        return active_cbar_ax[0]

    def _clear_active_cbar():
        active_cbar[0] = None
        if active_cbar_ax[0] is not None:
            active_cbar_ax[0].cla()
            active_cbar_ax[0].set_visible(False)

    def _plot_colored_scatter(ax, emb, values, label, is_3d=False):
        """Plot continuous values with a colorbar, categorical values with a legend."""
        vals = np.asarray(values).reshape(-1)
        if vals.shape[0] != emb.shape[0]:
            raise ValueError(
                f"Color values length ({vals.shape[0]}) does not match embedding rows ({emb.shape[0]})."
            )

        _clear_active_cbar()

        if _is_numeric(vals):
            vals = vals.astype(float)
            if is_3d:
                scatter_obj = ax.scatter(
                    emb[:, 0], emb[:, 1], emb[:, 2],
                    c=vals, cmap='Spectral', s=marker_size
                )
                # Reduce marker size for 3D splots to avoid clutter, use total sample size to adjust size dynamically
                total_samples = emb.shape[0]

            else:
                scatter_obj = ax.scatter(
                    emb[:, 0], emb[:, 1],
                    c=vals, cmap='Spectral', s=marker_size
                )
                # Adjust marker size for 2D plots based on total sample size
                total_samples = emb.shape[0]


            cax = _ensure_cbar_ax()
            cax.clear()
            cax.set_visible(True)
            cbar = fig.colorbar(scatter_obj, cax=cax, orientation='horizontal')
            cbar.set_label(str(label))
            active_cbar[0] = cbar
            return

        categories = pd.Categorical(vals)
        cats = categories.categories
        cmap = plt.get_cmap('tab10', max(len(cats), 1))
        for i, cat in enumerate(cats):
            mask = categories == cat
            if is_3d:
                ax.scatter(
                    emb[mask, 0], emb[mask, 1], emb[mask, 2],
                    color=cmap(i), s=50, label=str(cat)
                )
            else:
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    color=cmap(i), s=50, label=str(cat)
                )
        ax.legend(title=str(label), loc='upper right')

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_dimensions, metric=metric, random_state=random_state)
    embedding = reducer.fit_transform(data)
    returned_embedding = embedding  # Keep the full embedding for return

    # Warn if n_dimensions is not 2 or 3
    if n_dimensions not in [2, 3]:
        warnings.warn("n_dimensions should be 2 or 3 for visualization. Plotting only the first 2 dimensions in this case, but returning full embedding to user")
        n_dimensions = 2
        # Take first two dimensions of embedding for plotting
        embedding = embedding[:, :2]

    is_3d = (n_dimensions == 3)

    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        _plot_colored_scatter(ax, embedding, batch, label='Batch', is_3d=True)
        ax.set_title('UMAP projection (3D)', fontsize=15)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_colored_scatter(ax, embedding, batch, label='Batch', is_3d=False)
        ax.set_title('UMAP projection (2D)', fontsize=15)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    if covariates is not None and show:
        if isinstance(covariates, pd.DataFrame):
            covariate_keys = list(covariates.columns)
            get_values = lambda key: covariates[key].to_numpy()
        elif isinstance(covariates, dict):
            covariate_keys = list(covariates.keys())
            get_values = lambda key: np.asarray(covariates[key])
        else:
            covariate_keys = []
            get_values = None

        if len(covariate_keys) > 0:
            from matplotlib.widgets import RadioButtons

            fig.subplots_adjust(right=0.80)
            selector_ax = fig.add_axes([0.82, 0.35, 0.16, 0.30])
            selector_ax.set_title('Color by', fontsize=10)
            options = ['batch'] + covariate_keys
            selector = RadioButtons(selector_ax, options, active=0)

            def _redraw(color_by):
                try:
                    ax.clear()
                    values = batch if color_by == 'batch' else get_values(color_by)
                    _plot_colored_scatter(ax, embedding, values, label=color_by, is_3d=is_3d)
                    if is_3d:
                        ax.set_title(f'UMAP projection (3D) colored by {color_by}', fontsize=15)
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                        ax.set_zlabel('UMAP 3')
                    else:
                        ax.set_title(f'UMAP projection (2D) colored by {color_by}', fontsize=15)
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                    fig.canvas.draw_idle()
                except Exception as exc:
                    warnings.warn(f"Failed to redraw UMAP colors for '{color_by}': {exc}. Reverting to batch coloring.")
                    ax.clear()
                    _plot_colored_scatter(ax, embedding, batch, label='batch', is_3d=is_3d)
                    if is_3d:
                        ax.set_title('UMAP projection (3D) colored by batch', fontsize=15)
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                        ax.set_zlabel('UMAP 3')
                    else:
                        ax.set_title('UMAP projection (2D) colored by batch', fontsize=15)
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                    fig.canvas.draw_idle()

            selector.on_clicked(_redraw)
            # Keep a reference so widgets stay alive while the figure is open.
            fig._umap_color_selector = selector
        else:
            warnings.warn("Covariates provided without named columns/keys; color selector not shown.")

    if show:
        plt.show()

    # Return the full embedding to the user, even if only 2D or 3D is plotted, this code is not suitable for rep workflow:
    # Only return the data
    return returned_embedding
    
    


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
# 1B.  RESIDUALISED IDP DISTRIBUTIONS ACROSS SITES
# ============================================================================

def _adjust_for_covariates_preserve_batch(
    df,
    idp_cols,
    batch_col,
    covariates=None,
):
    """
    Remove covariate effects from each feature while preserving batch effects.

    Fits:
        feature ~ covariates + batch

    and returns:
        feature - fitted covariate contribution

    Batch and intercept effects are retained.
    """
    if not covariates:
        return df.copy()

    adjusted = df.copy()

    for idp in idp_cols:
        model_df = pd.DataFrame(
            {
                "_y": pd.to_numeric(df[idp], errors="coerce"),
                "_batch": df[batch_col].astype("category"),
            },
            index=df.index,
        )

        continuous = []
        categorical = []

        # Add covariates and retain categorical information where possible
        for name, values in covariates.items():
            values = pd.Series(values).reset_index(drop=True)
            values.index = df.index

            model_df[name] = values

            if (
                isinstance(values.dtype, pd.CategoricalDtype)
                or not pd.api.types.is_numeric_dtype(values)
            ):
                categorical.append(name)
            else:
                continuous.append(name)

        model_df = model_df.dropna()

        if model_df.empty:
            adjusted[idp] = np.nan
            continue

        # Continuous covariates
        X_cov = model_df[continuous].astype(float).copy()

        # Categorical covariates
        if categorical:
            X_cat = pd.get_dummies(
                model_df[categorical],
                drop_first=True,
                dtype=float,
            )
            X_cov = pd.concat([X_cov, X_cat], axis=1)

        # Batch is included in the model but will NOT be removed
        X_batch = pd.get_dummies(
            model_df["_batch"],
            prefix="batch",
            drop_first=True,
            dtype=float,
        )

        intercept = pd.DataFrame(
            {"intercept": 1.0},
            index=model_df.index,
        )

        X = pd.concat(
            [intercept, X_cov, X_batch],
            axis=1,
        ).astype(float)

        y = model_df["_y"].to_numpy(dtype=float)

        beta, *_ = np.linalg.lstsq(
            X.to_numpy(),
            y,
            rcond=None,
        )

        beta = pd.Series(beta, index=X.columns)

        # Remove ONLY covariate effects
        if X_cov.shape[1]:
            cov_effect = (
                X_cov.to_numpy()
                @ beta[X_cov.columns].to_numpy()
            )
        else:
            cov_effect = 0.0

        # Missing model rows should remain missing rather than raw
        adjusted[idp] = np.nan
        adjusted.loc[model_df.index, idp] = y - cov_effect

    return adjusted

def plot_CovariateAdjustedBoxplotsAcrossSites(
    df,
    batch_col: str = "batch",
    subject_col: str | None = "subject",
    covariates: dict | None = None,
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
    feature_display_limit: int = 5,
):
    """
    Plot covariate-adjusted IDP distributions across sites/batches.

    Covariate effects are removed using:

        IDP ~ covariates + batch

    Only the covariate contribution is removed. Batch/site effects
    remain in the adjusted values so that residual between-batch
    differences can be visualised.

    Feature display is deliberately scalable:

        <= feature_display_limit IDPs:
            show all IDPs

        > feature_display_limit IDPs:
            show the FIRST feature_display_limit IDPs in the exact
            order supplied in idp_cols.

    No feature ranking or PCA is performed.

    This makes the figure stable for datasets ranging from a few IDPs
    to thousands of IDPs, and ensures raw/harmonised figures can show
    the same IDPs when the same idp_cols ordering is supplied.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing batch/site information and IDPs.

    batch_col : str, default="batch"
        Column containing batch/site labels.

    subject_col : str or None, default="subject"
        Subject identifier column. It is excluded from IDP detection.

    covariates : dict or None
        Dictionary of covariates. Each value must have one value per
        observation.

        Example:

            covariates = {
                "age": age_array,
                "sex": sex_array,
            }

        Categorical variables such as M/F are automatically handled
        by the adjustment helper.

    idp_cols : list or None
        IDP columns to consider.

        If None, all columns except batch_col and subject_col are used.

    site_order : list or None
        Explicit batch/site ordering.

        If None, sites are ordered alphabetically.

    ncols : int, default=2
        Number of IDP panels shown side-by-side when the number of
        sites is small.

    figsize_per_panel : tuple, default=(6.0, 4.5)
        Size of each IDP panel.

        Only applies to the displayed IDP panels.

    show_points : bool, default=True
        Overlay individual observations.

    point_size : float, default=2.2
        Size of individual observation points.

    point_alpha : float, default=0.18
        Transparency of individual observation points.

    point_jitter : float, default=0.22
        Jitter applied to individual observations.

    savepath : str or None
        Output path for the figure.

    rep : optional
        Existing report object.

    show : bool, default=False
        Display the figure.

    site_threshold_for_horizontal : int, default=12
        If there are more than this number of sites, use horizontal
        boxplots to keep site labels readable.

    feature_display_limit : int, default=5
        Maximum number of IDPs displayed.

        IMPORTANT:
        IDPs are selected by supplied order, not by site dispersion.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.

    axes : list of matplotlib.axes.Axes
        Axes corresponding to displayed IDPs.
    """

    # ============================================================
    # IMPORTS
    # ============================================================

    import math
    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ============================================================
    # THEME
    # ============================================================

    apply_plot_theme()

    # ============================================================
    # VALIDATION
    # ============================================================

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if batch_col not in df.columns:
        raise ValueError(
            f"'{batch_col}' not found in dataframe columns."
        )

    if (
        feature_display_limit is None
        or int(feature_display_limit) < 1
    ):
        raise ValueError(
            "feature_display_limit must be an integer >= 1."
        )

    feature_display_limit = int(
        feature_display_limit
    )

    # ============================================================
    # DETERMINE IDP COLUMNS
    # ============================================================

    if idp_cols is None:

        excluded = {
            batch_col
        }

        if (
            subject_col is not None
            and subject_col in df.columns
        ):
            excluded.add(
                subject_col
            )

        idp_cols = [
            c
            for c in df.columns
            if c not in excluded
        ]

    else:

        idp_cols = list(
            idp_cols
        )

    if len(idp_cols) == 0:

        raise ValueError(
            "No IDP columns found."
        )

    missing_idps = [
        c
        for c in idp_cols
        if c not in df.columns
    ]

    if missing_idps:

        raise ValueError(
            "The following IDP columns were not found in df: "
            + ", ".join(
                map(
                    str,
                    missing_idps,
                )
            )
        )

    # ============================================================
    # COPY / NORMALISE DATA
    # ============================================================

    plot_df = df.copy()

    plot_df[
        batch_col
    ] = (
        plot_df[
            batch_col
        ]
        .astype(str)
    )

    if (
        subject_col is not None
        and subject_col in plot_df.columns
    ):

        plot_df[
            subject_col
        ] = (
            plot_df[
                subject_col
            ]
            .astype(str)
        )

    else:

        subject_col = None

    # ============================================================
    # SITE ORDER
    # ============================================================

    if site_order is None:

        site_order = sorted(
            plot_df[
                batch_col
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    else:

        site_order = [
            str(s)
            for s in site_order
        ]

    if len(site_order) == 0:

        raise ValueError(
            "No batch/site labels were found."
        )

    # Keep only sites actually represented in data.
    observed_sites = set(
        plot_df[
            batch_col
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    missing_sites = [
        site
        for site in site_order
        if site not in observed_sites
    ]

    if missing_sites:

        warnings.warn(
            "The following entries in site_order do not occur "
            "in the data and will be shown with empty boxes: "
            + ", ".join(
                missing_sites
            )
        )

    n_sites = len(
        site_order
    )

    # ============================================================
    # BATCH SAMPLE SIZE SUMMARY
    # ============================================================

    batch_counts = (
        plot_df
        .groupby(
            batch_col
        )
        .size()
        .reindex(
            site_order
        )
        .fillna(0)
        .astype(int)
    )

    min_batch = (
        int(
            batch_counts.min()
        )
        if len(batch_counts)
        else 0
    )

    med_batch = (
        int(
            batch_counts.median()
        )
        if len(batch_counts)
        else 0
    )

    max_batch = (
        int(
            batch_counts.max()
        )
        if len(batch_counts)
        else 0
    )

    n_lt5 = int(
        (
            batch_counts < 5
        ).sum()
    )

    n_lt10 = int(
        (
            batch_counts < 10
        ).sum()
    )

    n_lt20 = int(
        (
            batch_counts < 20
        ).sum()
    )

    summary_text = (
        f"Batch size: min={min_batch}, "
        f"median={med_batch}, max={max_batch}   |   "
        f"Batches <5: {n_lt5}   "
        f"<10: {n_lt10}   "
        f"<20: {n_lt20}"
    )

    # ============================================================
    # COVARIATE ADJUSTMENT
    # ============================================================

    # This retains batch effects and removes only the covariate
    # contribution.
    #
    # The helper handles numeric and categorical covariates.
    # ============================================================

    adjusted_df = _adjust_for_covariates_preserve_batch(
        df=plot_df,
        idp_cols=idp_cols,
        batch_col=batch_col,
        covariates=covariates,
    )

    adjusted_df[
        batch_col
    ] = plot_df[
        batch_col
    ]

    if (
        subject_col is not None
        and subject_col in plot_df.columns
    ):

        adjusted_df[
            subject_col
        ] = plot_df[
            subject_col
        ]

    # ============================================================
    # SELECT DISPLAYED IDPs
    # ============================================================

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT rank by dispersion.
    # Do NOT perform PCA.
    #
    # Simply take the first N IDPs in supplied order.
    # ------------------------------------------------------------

    shown_idps = (
        idp_cols[
            :feature_display_limit
        ]
    )

    n_total_idps = len(
        idp_cols
    )

    n_displayed_idps = len(
        shown_idps
    )

    truncated = (
        n_total_idps
        > n_displayed_idps
    )

    # ============================================================
    # ORIENTATION
    # ============================================================

    use_horizontal = (
        n_sites
        > site_threshold_for_horizontal
    )

    # ============================================================
    # PALETTE
    # ============================================================

    palette = sns.color_palette(
        "Set2",
        n_colors=max(
            3,
            n_sites,
        ),
    )

    site_palette = {
        site: palette[
            i % len(palette)
        ]
        for i, site in enumerate(
            site_order
        )
    }

    # ============================================================
    # FIGURE LAYOUT
    # ============================================================

    if use_horizontal:

        fig_w = max(
            8.0,
            figsize_per_panel[0] * 1.30,
        )

        fig_h = max(
            5.2,
            figsize_per_panel[1] * 1.10,
        )

        fig, axes_grid = plt.subplots(
            nrows=n_displayed_idps,
            ncols=1,
            figsize=(
                fig_w,
                fig_h * n_displayed_idps,
            ),
            squeeze=False,
        )

        axes = (
            axes_grid.flatten()
        )

    else:

        ncols_use = max(
            1,
            min(
                int(ncols),
                n_displayed_idps,
            ),
        )

        nrows = int(
            math.ceil(
                n_displayed_idps
                / ncols_use
            )
        )

        fig_w = (
            figsize_per_panel[0]
            * ncols_use
        )

        fig_h = (
            figsize_per_panel[1]
            * nrows
        )

        fig, axes_grid = plt.subplots(
            nrows=nrows,
            ncols=ncols_use,
            figsize=(
                fig_w,
                fig_h,
            ),
            squeeze=False,
        )

        axes = (
            axes_grid.flatten()
        )

    # ============================================================
    # TITLE / TOP NOTE
    # ============================================================

    if truncated:

        top_note = (
            f"Showing first {n_displayed_idps} of "
            f"{n_total_idps} IDPs in supplied order"
        )

    else:

        top_note = (
            f"Showing all {n_total_idps} IDPs"
        )

    fig.suptitle(
        "Covariate-adjusted IDP distributions across sites",
        fontsize=14.5,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.972,
        top_note,
        ha="center",
        va="top",
        fontsize=max(
            7,
            STYLE.TICK_SIZE - 1,
        ),
        fontstyle="italic",
        alpha=0.72,
    )

    fig.text(
        0.5,
        0.950,
        summary_text,
        ha="center",
        va="top",
        fontsize=max(
            7,
            STYLE.TICK_SIZE - 1,
        ),
        fontstyle="italic",
        alpha=0.68,
    )

    # ============================================================
    # PLOT EACH IDP
    # ============================================================

    for i, idp in enumerate(
        shown_idps
    ):

        ax = axes[i]

        # --------------------------------------------------------
        # Explicit numeric IDP values
        # --------------------------------------------------------

        adjusted_df[
            idp
        ] = pd.to_numeric(
            adjusted_df[
                idp
            ],
            errors="coerce",
        )

        # --------------------------------------------------------
        # Long-format data for seaborn.
        # --------------------------------------------------------

        temp = adjusted_df[
            [
                batch_col,
                idp,
            ]
        ].copy()

        temp = temp.rename(
            columns={
                idp: "_value"
            }
        )

        # --------------------------------------------------------
        # Horizontal orientation for many sites.
        # --------------------------------------------------------

        if use_horizontal:

            sns.boxplot(
                data=temp,
                y=batch_col,
                x="_value",
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
                    data=temp,
                    y=batch_col,
                    x="_value",
                    order=site_order,
                    ax=ax,
                    color="black",
                    size=point_size,
                    alpha=point_alpha,
                    jitter=point_jitter,
                    orient="h",
                )

            ax.set_xlabel(
                "Covariate-adjusted value",
                fontsize=STYLE.AXIS_SIZE,
            )

            ax.set_ylabel(
                "Site",
                fontsize=STYLE.AXIS_SIZE,
            )

            # Explicitly set labels rather than relying on automatic
            # tick-label handling.
            ax.set_yticks(
                np.arange(
                    len(
                        site_order
                    )
                )
            )

            ax.set_yticklabels(
                site_order,
                fontsize=max(
                    7,
                    STYLE.TICK_SIZE - 1,
                ),
            )

            ax.tick_params(
                axis="y",
                pad=7,
            )

            ax.tick_params(
                axis="x",
                labelsize=STYLE.TICK_SIZE,
            )

            ax.grid(
                axis="x",
                linestyle="--",
                alpha=STYLE.GRID_ALPHA,
            )

        else:

            sns.boxplot(
                data=temp,
                x=batch_col,
                y="_value",
                order=site_order,
                ax=ax,
                palette=site_palette,
                width=0.65,
                fliersize=0,
                linewidth=1.0,
            )

            if show_points:

                sns.stripplot(
                    data=temp,
                    x=batch_col,
                    y="_value",
                    order=site_order,
                    ax=ax,
                    color="black",
                    size=point_size,
                    alpha=point_alpha,
                    jitter=point_jitter,
                )

            ax.set_xlabel(
                "Site",
                fontsize=STYLE.AXIS_SIZE,
            )

            ax.set_ylabel(
                "Covariate-adjusted value",
                fontsize=STYLE.AXIS_SIZE,
            )

            # Use existing helper if available; otherwise use the
            # explicit labels.
            try:

                rot = _adaptive_rotation(
                    site_order,
                    threshold=7,
                    steep=45,
                    mild=30,
                )

            except Exception:

                rot = (
                    45
                    if len(site_order) > 7
                    else 0
                )

            ax.set_xticks(
                np.arange(
                    len(
                        site_order
                    )
                )
            )

            ax.set_xticklabels(
                site_order,
                rotation=rot,
                ha=(
                    "right"
                    if rot
                    else "center"
                ),
                fontsize=max(
                    7,
                    STYLE.TICK_SIZE - 1,
                ),
            )

            ax.tick_params(
                axis="y",
                labelsize=STYLE.TICK_SIZE,
            )

            ax.tick_params(
                axis="x",
                pad=5,
            )

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=STYLE.GRID_ALPHA,
            )

        # --------------------------------------------------------
        # IDP title
        # --------------------------------------------------------

        ax.set_title(
            str(idp),
            fontsize=STYLE.TITLE_SIZE,
            fontweight="bold",
            pad=10,
        )

        # --------------------------------------------------------
        # Clean spines
        # --------------------------------------------------------

        _strip_spines(
            ax
        )

    # ============================================================
    # HIDE UNUSED AXES
    # ============================================================

    for j in range(
        n_displayed_idps,
        len(axes),
    ):

        axes[j].axis(
            "off"
        )

    # ============================================================
    # FOOTNOTE
    # ============================================================

    covariate_names = []

    if covariates:

        covariate_names = [
            str(name)
            for name in covariates.keys()
        ]

    if covariate_names:

        covariate_note = (
            "Covariate effects removed: "
            + ", ".join(
                covariate_names
            )
            + ". Batch/site effects retained."
        )

    else:

        covariate_note = (
            "No covariates supplied. "
            "Original IDP values are plotted."
        )

    fig.text(
        0.5,
        0.005,
        covariate_note,
        ha="center",
        va="bottom",
        fontsize=max(
            7,
            STYLE.TICK_SIZE - 1,
        ),
        alpha=0.68,
    )

    # ============================================================
    # FINAL LAYOUT
    # ============================================================

    if use_horizontal:

        fig.subplots_adjust(
            left=0.25,
            right=0.98,
            top=0.925,
            bottom=0.065,
            hspace=0.60,
        )

    else:

        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.91,
            bottom=0.075,
            hspace=0.50,
            wspace=0.28,
        )

    # ============================================================
    # SAVE
    # ============================================================

    if savepath:

        plt.savefig(
            savepath,
            dpi=STYLE.DPI,
            bbox_inches="tight",
            facecolor="white",
        )

    # ============================================================
    # REPORT
    # ============================================================

    if rep is not None:

        try:

            rep.log_plot(
                fig,
                "Covariate-adjusted IDP distributions across sites",
            )

        except Exception as exc:

            warnings.warn(
                "Could not log figure to report: "
                f"{exc}"
            )

    # ============================================================
    # SHOW
    # ============================================================

    if show:

        plt.show()

    # ============================================================
    # RETURN
    # ============================================================

    return (
        fig,
        axes[
            :n_displayed_idps
        ],
    )

# ============================================================================
# 1.  SUBJECT ORDER CONSISTENCY
# ============================================================================
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_SubjectOrder(
    df,
    idp_col: str = "IDP",
    time_a_col: str = "TimeA",
    time_b_col: str = "TimeB",
    rho_col: str = "SpearmanRho",
    p_col: str = "pValue",
    summary_stat: str = "median",
    significance: float = 0.05,
    savepath: str | None = None,
    rep=None,
    show: bool = False,
):
    """
    Publication-style diagnostic for subject-order consistency.
    Panels
    ------
    A
        Mean/median subject-order consistency across acquisitions.
        Significance is based on feature-level acquisition-pair p-values
        corrected together using Benjamini-Hochberg FDR across the full
        family of valid Feature x unique-acquisition-pair tests.
        A star indicates that at least one feature-level test for that
        acquisition pair survives the global FDR correction.
    B
        Per-Feature mean/median subject-order consistency.
        Feature index is shown on the x-axis and Spearman rho on the y-axis.
    C
        Summary statistics across Features.
        For median:
            overall Feature-level median rho [IQR]
        For mean:
            overall Feature-level mean rho +/- SD
    Notes
    -----
    * Feature order is preserved.
    * Acquisition labels are encoded A1, A2, ...
    * Actual acquisition labels are shown in the acquisition mapping.
    """
    # ============================================================
    # VALIDATION
    # ============================================================
    if summary_stat not in {"median", "mean"}:
        raise ValueError("summary_stat must be either 'median' or 'mean'.")
    if not (0 < significance < 1):
        raise ValueError("significance must be between 0 and 1.")
    required_cols = [idp_col, time_a_col, time_b_col, rho_col, p_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError("Missing required columns:\n" + "\n".join(missing_cols))
    # ============================================================
    # CLEAN DATA
    # ============================================================
    data = df[required_cols].copy()
    data[idp_col] = data[idp_col].astype(str).str.strip()
    data[rho_col] = pd.to_numeric(data[rho_col], errors="coerce")
    data[p_col] = pd.to_numeric(data[p_col], errors="coerce")
    data = data.dropna(subset=[idp_col, time_a_col, time_b_col])
    if data.empty:
        raise ValueError("No usable rows remain after cleaning.")
    # ============================================================
    # FEATURE ORDER
    # ============================================================
    all_idps = list(pd.unique(data[idp_col]))
    n_idps = len(all_idps)
    if n_idps == 0:
        raise ValueError("No Features found.")
    # ============================================================
    # ACQUISITION ORDER
    # ============================================================
    acquisitions = list(
        pd.unique(pd.concat([data[time_a_col], data[time_b_col]], ignore_index=True))
    )
    n_acquisitions = len(acquisitions)
    if n_acquisitions < 2:
        raise ValueError("At least two acquisition/timepoint/batch labels " "are required.")
    # ============================================================
    # COMPACT ACQUISITION CODES
    # ============================================================
    acquisition_codes = {original: f"B{i + 1}" for i, original in enumerate(acquisitions)}
    acquisition_display = [acquisition_codes[x] for x in acquisitions]
    acquisition_mapping = [
        {"index": i + 1, "code": acquisition_codes[original], "label": str(original)}
        for i, original in enumerate(acquisitions)
    ]
    # ============================================================
    # GLOBAL PLOTTING SETTINGS
    # ============================================================
    annotate_heatmaps = n_acquisitions <= 12
    if n_acquisitions <= 8:
        heatmap_tick_font = 7.5
        heatmap_annotation_font = 8.5
    elif n_acquisitions <= 12:
        heatmap_tick_font = 6.5
        heatmap_annotation_font = 7.0
    elif n_acquisitions <= 16:
        heatmap_tick_font = 5.5
        heatmap_annotation_font = 6.0
    else:
        heatmap_tick_font = 4.8
        heatmap_annotation_font = 5.5
    heatmap_tick_positions = np.arange(n_acquisitions, dtype=float) + 0.5
    cmap = "RdYlBu_r"
    # ============================================================
    # BUILD MATRICES
    # ============================================================
    rho_mats = {}
    p_mats = {}
    for idp in all_idps:
        subset = data[data[idp_col] == idp]
        rho = subset.pivot_table(
            index=time_a_col, columns=time_b_col, values=rho_col, aggfunc="first"
        )
        pmat = subset.pivot_table(
            index=time_a_col, columns=time_b_col, values=p_col, aggfunc="first"
        )
        rho = rho.reindex(index=acquisitions, columns=acquisitions)
        pmat = pmat.reindex(index=acquisitions, columns=acquisitions)
        rho = rho.combine_first(rho.T)
        pmat = pmat.combine_first(pmat.T)
        rho = rho.astype(float)
        pmat = pmat.astype(float)
        np.fill_diagonal(rho.values, np.nan)
        np.fill_diagonal(pmat.values, np.nan)
        rho_mats[idp] = rho
        p_mats[idp] = pmat
    # ============================================================
    # UNIQUE ACQUISITION PAIRS
    # ============================================================
    pair_i, pair_j = np.triu_indices(n_acquisitions, k=1)
    n_unique_pairs = len(pair_i)
    if n_unique_pairs == 0:
        raise ValueError("No unique acquisition pairs were found.")
    # ============================================================
    # PER-FEATURE SUMMARY
    # ============================================================
    idp_pairwise_values = {}
    idp_mean = {}
    idp_median = {}
    idp_q10 = {}
    idp_q25 = {}
    idp_q75 = {}
    idp_q90 = {}
    for idp in all_idps:
        rho_values = rho_mats[idp].values[pair_i, pair_j]
        valid_rho = rho_values[np.isfinite(rho_values)]
        idp_pairwise_values[idp] = valid_rho
        if len(valid_rho) == 0:
            idp_mean[idp] = np.nan
            idp_median[idp] = np.nan
            idp_q10[idp] = np.nan
            idp_q25[idp] = np.nan
            idp_q75[idp] = np.nan
            idp_q90[idp] = np.nan
            continue
        idp_mean[idp] = float(np.mean(valid_rho))
        idp_median[idp] = float(np.median(valid_rho))
        idp_q10[idp] = float(np.percentile(valid_rho, 10))
        idp_q25[idp] = float(np.percentile(valid_rho, 25))
        idp_q75[idp] = float(np.percentile(valid_rho, 75))
        idp_q90[idp] = float(np.percentile(valid_rho, 90))
    # ============================================================
    # FEATURE SUMMARY DATAFRAME
    # ============================================================
    idp_summary = pd.DataFrame(
        {
            "central": [
                (idp_median[idp] if summary_stat == "median" else idp_mean[idp]) for idp in all_idps
            ],
            "mean": [idp_mean[idp] for idp in all_idps],
            "median": [idp_median[idp] for idp in all_idps],
            "q10": [idp_q10[idp] for idp in all_idps],
            "q25": [idp_q25[idp] for idp in all_idps],
            "q75": [idp_q75[idp] for idp in all_idps],
            "q90": [idp_q90[idp] for idp in all_idps],
        },
        index=all_idps,
    )
    # ============================================================
    # STACK MATRICES
    # ============================================================
    rho_stack = np.stack([rho_mats[idp].values for idp in all_idps], axis=0)
    p_stack = np.stack([p_mats[idp].values for idp in all_idps], axis=0)
    # ============================================================
    # SUMMARY RHO MATRIX
    # ============================================================
    summary_rho = np.full((n_acquisitions, n_acquisitions), np.nan, dtype=float)
    for i in range(n_acquisitions):
        for j in range(n_acquisitions):
            if i == j:
                continue
            values = rho_stack[:, i, j]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            if summary_stat == "median":
                summary_rho[i, j] = np.median(values)
            else:
                summary_rho[i, j] = np.mean(values)
    # ============================================================
    # GLOBAL FDR CORRECTION
    #
    # All valid Feature-level p-values across ALL unique
    # acquisition pairs are corrected together.
    #
    # The multiple-testing family therefore contains up to:
    #
    #     N Features x N unique acquisition pairs
    #
    # tests.
    # ============================================================
    raw_global_p = []
    global_locations = []
    for idp_idx, idp in enumerate(all_idps):
        p_values = p_mats[idp].values[pair_i, pair_j]
        for pair_idx, p_value in enumerate(p_values):
            if np.isfinite(p_value):
                raw_global_p.append(float(np.clip(p_value, 0.0, 1.0)))
                global_locations.append((idp_idx, pair_idx))
    raw_global_p = np.asarray(raw_global_p, dtype=float)
    n_global_tests = len(raw_global_p)
    global_p_adj = np.full(raw_global_p.shape, np.nan, dtype=float)
    if n_global_tests > 0:
        order = np.argsort(raw_global_p)
        sorted_p = raw_global_p[order]
        adjusted_sorted = sorted_p * n_global_tests / np.arange(1, n_global_tests + 1)
        adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
        global_p_adj[order] = adjusted_sorted
    # ============================================================
    # ADJUSTED P MATRIX FOR EACH FEATURE
    # ============================================================
    p_adj_mats = {
        idp: pd.DataFrame(
            np.full((n_acquisitions, n_acquisitions), np.nan, dtype=float),
            index=acquisitions,
            columns=acquisitions,
        )
        for idp in all_idps
    }
    for adjusted_p, (idp_idx, pair_idx) in zip(global_p_adj, global_locations):
        idp = all_idps[idp_idx]
        i = pair_i[pair_idx]
        j = pair_j[pair_idx]
        p_adj_mats[idp].iat[i, j] = adjusted_p
        p_adj_mats[idp].iat[j, i] = adjusted_p
    # ============================================================
    # SUMMARY SIGNIFICANCE MATRIX
    #
    # A summary acquisition pair receives a star if at least one
    # Feature-level test for that acquisition pair survives the
    # GLOBAL Feature x acquisition-pair FDR correction.
    #
    # The displayed significance value is the minimum globally
    # adjusted Feature-level p-value for that acquisition pair.
    # ============================================================
    summary_p_adj = np.full((n_acquisitions, n_acquisitions), np.nan, dtype=float)
    for pair_idx, (i, j) in enumerate(zip(pair_i, pair_j)):
        adjusted_pair_p = []
        for idp in all_idps:
            p_value = p_adj_mats[idp].iat[i, j]
            if np.isfinite(p_value):
                adjusted_pair_p.append(p_value)
        if len(adjusted_pair_p) == 0:
            continue
        pair_min_adjusted_p = float(np.min(adjusted_pair_p))
        summary_p_adj[i, j] = pair_min_adjusted_p
        summary_p_adj[j, i] = pair_min_adjusted_p

    # ============================================================
    # HEATMAP ANNOTATION HELPER
    # ============================================================
    def make_heatmap_annotation(rho_df, p_df):
        if not annotate_heatmaps:
            return False
        annotation = np.full(rho_df.shape, "", dtype=object)
        for i in range(n_acquisitions):
            for j in range(n_acquisitions):
                value = rho_df.iat[i, j]
                p_value = p_df.iat[i, j]
                if not np.isfinite(value):
                    continue
                if np.isfinite(p_value) and p_value < significance:
                    star = "*"
                else:
                    star = ""
                annotation[i, j] = f"{value:.2f}" f"{star}"
        return annotation

    # ============================================================
    # OVERALL FEATURE VALUE AND DISPERSION
    #
    # median -> median [Q1, Q3]
    # mean   -> mean +/- SD
    # ============================================================
    valid_central = idp_summary["central"].dropna().to_numpy(dtype=float)
    if len(valid_central) == 0:
        overall_central = np.nan
        overall_q25 = np.nan
        overall_q75 = np.nan
        overall_sd = np.nan
    elif summary_stat == "median":
        overall_central = float(np.median(valid_central))
        overall_q25 = float(np.percentile(valid_central, 25))
        overall_q75 = float(np.percentile(valid_central, 75))
        overall_sd = np.nan
    else:
        overall_central = float(np.mean(valid_central))
        if len(valid_central) > 1:
            overall_sd = float(np.std(valid_central, ddof=1))
        else:
            overall_sd = np.nan
        overall_q25 = np.nan
        overall_q75 = np.nan
    # ============================================================
    # FORMATTED OVERALL SUMMARY
    # ============================================================
    if summary_stat == "median":
        overall_label = "Overall Feature-level median ρ [IQR]"
        if np.isfinite(overall_central) and np.isfinite(overall_q25) and np.isfinite(overall_q75):
            overall_text = f"{overall_central:.2f} " f"[{overall_q25:.2f}, " f"{overall_q75:.2f}]"
        else:
            overall_text = "NA"
    else:
        overall_label = "Overall Feature-level mean ρ ± SD"
        if np.isfinite(overall_central):
            if np.isfinite(overall_sd):
                overall_text = f"{overall_central:.2f} " f"± {overall_sd:.2f}"
            else:
                overall_text = f"{overall_central:.2f} " "± NA"
        else:
            overall_text = "NA"

    # ============================================================
    # COMMON AXIS STYLE
    # ============================================================
    def clean_axis(ax, grid_axis="y"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if grid_axis is not None:
            ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)
        ax.tick_params(labelsize=9)

    # ============================================================
    # PANEL A DRAWING
    #
    # MEAN/MEDIAN SUBJECT-ORDER CONSISTENCY ACROSS ACQUISITIONS
    # ============================================================
    def draw_panel_A(ax, standalone=False):
        annotation = make_heatmap_annotation(pd.DataFrame(summary_rho), pd.DataFrame(summary_p_adj))
        sns.heatmap(
            summary_rho,
            ax=ax,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            annot=annotation,
            fmt="",
            linewidths=0.45,
            linecolor="white",
            cbar=True,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.80},
            square=False,
            mask=np.isnan(summary_rho),
            annot_kws={"fontsize": heatmap_annotation_font},
        )
        title = f"{summary_stat.capitalize()} " "subject-order consistency across acquisitions"
        ax.set_title(
            title, loc="left", fontsize=14 if standalone else 11.5, fontweight="bold", pad=7
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(heatmap_tick_positions)
        ax.set_yticks(heatmap_tick_positions)
        ax.set_xticklabels(acquisition_display, rotation=45, ha="right", fontsize=heatmap_tick_font)
        ax.set_yticklabels(acquisition_display, rotation=0, fontsize=heatmap_tick_font)

    # ============================================================
    # PANEL B DRAWING
    #
    # PER-FEATURE MEAN/MEDIAN SUBJECT-ORDER CONSISTENCY
    #
    # FEATURE INDEX = X AXIS
    # SPEARMAN RHO = Y AXIS
    # ============================================================
    def draw_panel_B(ax, standalone=False):
        positions = np.arange(1, n_idps + 1)
        if n_idps <= 100:
            dot_size = 22
        elif n_idps <= 200:
            dot_size = 17
        elif n_idps <= 400:
            dot_size = 13
        elif n_idps <= 800:
            dot_size = 9
        else:
            dot_size = 7
        if n_idps <= 25:
            idp_step = 1
        elif n_idps <= 60:
            idp_step = 3
        elif n_idps <= 120:
            idp_step = 8
        elif n_idps <= 250:
            idp_step = 20
        elif n_idps <= 500:
            idp_step = 40
        else:
            idp_step = max(1, int(np.ceil(n_idps / 12)))
        ax.scatter(
            positions,
            idp_summary["central"].to_numpy(dtype=float),
            s=dot_size,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )
        if np.isfinite(overall_central):
            ax.axhline(overall_central, linestyle="--", linewidth=1.05, alpha=0.60, zorder=0)
        x_ticks = positions[::idp_step]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x)) for x in x_ticks])
        title = f"Per-Feature {summary_stat} " "subject-order consistency"
        ax.set_title(
            title, loc="left", fontsize=14 if standalone else 11.5, fontweight="bold", pad=7
        )
        ax.text(
            0.0,
            0.975,
            ("Each point represents one Feature; " "Features are shown in the order supplied."),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            alpha=0.68,
            clip_on=False,
        )
        ax.set_xlabel("Feature index", fontsize=10)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlim(0.5, n_idps + 0.5)
        clean_axis(ax, grid_axis="y")

    # ============================================================
    # SUMMARY TABLE DATA
    # ============================================================
    summary_rows = [
        ["N Features", f"{n_idps:,}"],
        ["N acquisitions", f"{n_acquisitions:,}"],
        ["N unique acquisition pairs", f"{n_unique_pairs:,}"],
        ["N Feature × pair tests", f"{n_global_tests:,}"],
        [overall_label, overall_text],
    ]
    # ============================================================
    # CONTENT-AWARE FIGURE SIZE
    # ============================================================
    if n_acquisitions <= 8:
        heatmap_height = 4.8
    elif n_acquisitions <= 12:
        heatmap_height = 5.2
    elif n_acquisitions <= 16:
        heatmap_height = 5.6
    elif n_acquisitions <= 24:
        heatmap_height = 6.0
    else:
        heatmap_height = 6.4
    combined_height = heatmap_height + 5.0
    # ============================================================
    # CREATE COMBINED FIGURE
    # ============================================================
    fig = plt.figure(figsize=(15, combined_height), facecolor="white")
    outer = fig.add_gridspec(
        nrows=2,
        ncols=16,
        height_ratios=[heatmap_height, 4.0],
        hspace=0.42,
        wspace=0.0,
        left=0.055,
        right=0.965,
        top=0.91,
        bottom=0.09,
    )
    # ============================================================
    # PANEL A
    # ============================================================
    axA = fig.add_subplot(outer[0, 0:7])
    draw_panel_A(axA, standalone=False)
    # ============================================================
    # PANEL B
    # ============================================================
    axB = fig.add_subplot(outer[0, 9:16])
    draw_panel_B(axB, standalone=False)
    # ============================================================
    # PANEL C
    #
    # SUMMARY ACROSS FEATURES
    # ============================================================
    axC = fig.add_subplot(outer[1, 0:7])
    axC.axis("off")
    axC.set_title("Summary across Features", loc="left", fontsize=11.5, fontweight="bold", pad=5)
    table_C = axC.table(
        cellText=summary_rows,
        colLabels=["Measure", "Value"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.20, 1.0, 0.68],
        colWidths=[0.72, 0.28],
    )
    table_C.auto_set_font_size(False)
    table_C.set_fontsize(7.4)
    for (row_idx, col_idx), cell in table_C.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.40)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx == 1:
            cell.set_text_props(ha="right")
    # ============================================================
    # ACQUISITION MAPPING
    # ============================================================
    axD = fig.add_subplot(outer[1, 9:16])
    axD.axis("off")
    axD.set_title("Batch/Site mapping", loc="left", fontsize=11.5, fontweight="bold", pad=5)
    mapping_data = [[item["code"], item["label"]] for item in acquisition_mapping]
    table_D = axD.table(
        cellText=mapping_data,
        colLabels=["Code used in plots", "Actual label"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.02, 1.0, 0.86],
        colWidths=[0.28, 0.72],
    )
    table_D.auto_set_font_size(False)
    table_D.set_fontsize(7.0)
    for (row_idx, col_idx), cell in table_D.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.40)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
    # ============================================================
    # FIGURE TITLE
    # ============================================================
    fig.suptitle(
        ("Subject order consistency" f"  ({n_idps:,} Features analysed)"),
        fontsize=14.5,
        fontweight="bold",
        y=0.975,
    )
    # ============================================================
    # FOOTNOTE
    # ============================================================
    fig.text(
        0.055,
        0.018,
        (
            f"N Features = {n_idps:,}   |   "
            f"N acquisitions = {n_acquisitions:,}   |   "
            f"N unique pairs = {n_unique_pairs:,}   |   "
            f"N Feature × pair tests = {n_global_tests:,}   |   "
            f"Primary statistic = {summary_stat}   |   "
            f"* = global BH-FDR-adjusted p < {significance:g}"
        ),
        fontsize=7.3,
        alpha=0.65,
    )
    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    if savepath is not None:
        base, ext = os.path.splitext(savepath)
        if ext == "":
            ext = ".png"
        # ========================================================
        # COMBINED
        # ========================================================
        combined_path = f"{base}{ext}"
        fig.savefig(combined_path, dpi=300, bbox_inches="tight", facecolor="white")
        # ========================================================
        # STANDALONE A
        # ========================================================
        a_width = min(10.5, max(7.0, 0.60 * n_acquisitions + 3.0))
        a_height = min(9.0, max(5.2, 0.55 * n_acquisitions + 2.5))
        fig_A = plt.figure(figsize=(a_width, a_height), facecolor="white")
        ax_A = fig_A.add_axes([0.14, 0.17, 0.70, 0.68])
        draw_panel_A(ax_A, standalone=True)
        fig_A.text(
            0.14,
            0.045,
            (
                "* = BH-FDR-adjusted p < "
                f"{significance:g}; correction includes all "
                f"{n_global_tests:,} valid Feature × "
                "acquisition-pair tests."
            ),
            fontsize=8,
            alpha=0.65,
        )
        fig_A.savefig(f"{base}_A{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_A)
        # ========================================================
        # STANDALONE B
        # ========================================================
        fig_B = plt.figure(figsize=(10.5, 6.5), facecolor="white")
        ax_B = fig_B.add_axes([0.10, 0.14, 0.84, 0.72])
        draw_panel_B(ax_B, standalone=True)
        if summary_stat == "median":
            standalone_summary_text = (
                f"Overall Feature-level median ρ = "
                f"{overall_central:.2f} "
                f"[IQR {overall_q25:.2f}, "
                f"{overall_q75:.2f}]."
            )
        else:
            if np.isfinite(overall_sd):
                standalone_summary_text = (
                    f"Overall Feature-level mean ρ = "
                    f"{overall_central:.2f} "
                    f"± {overall_sd:.2f} SD."
                )
            else:
                standalone_summary_text = (
                    f"Overall Feature-level mean ρ = " f"{overall_central:.2f} " "± NA SD."
                )
        fig_B.text(0.10, 0.035, standalone_summary_text, fontsize=8, alpha=0.65)
        fig_B.savefig(f"{base}_B{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_B)
        # ========================================================
        # STANDALONE C
        # ========================================================
        fig_C = plt.figure(figsize=(7.0, 5.0), facecolor="white")
        ax_C = fig_C.add_axes([0.10, 0.12, 0.80, 0.74])
        ax_C.axis("off")
        ax_C.set_title(
            "Summary across Features", loc="left", fontsize=14, fontweight="bold", pad=12
        )
        table_C_standalone = ax_C.table(
            cellText=summary_rows,
            colLabels=["Measure", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.18, 1.0, 0.64],
            colWidths=[0.70, 0.30],
        )
        table_C_standalone.auto_set_font_size(False)
        table_C_standalone.set_fontsize(9)
        for (row_idx, col_idx), cell in table_C_standalone.get_celld().items():
            cell.set_edgecolor("0.85")
            cell.set_linewidth(0.40)
            if row_idx == 0:
                cell.set_text_props(weight="bold")
            if col_idx == 1:
                cell.set_text_props(ha="right")
        fig_C.savefig(f"{base}_C{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_C)
        # ========================================================
        # STANDALONE ACQUISITION MAPPING
        # ========================================================
        standalone_mapping_height = max(4.0, min(10.0, 2.0 + n_acquisitions * 0.42))
        fig_D = plt.figure(figsize=(8.5, standalone_mapping_height), facecolor="white")
        ax_D = fig_D.add_axes([0.08, 0.12, 0.84, 0.72])
        ax_D.axis("off")
        ax_D.set_title("Acquisition mapping", loc="left", fontsize=14, fontweight="bold", pad=12)
        table_D_standalone = ax_D.table(
            cellText=mapping_data,
            colLabels=["Code used in plots", "Actual label"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.08, 1.0, 0.74],
            colWidths=[0.28, 0.72],
        )
        table_D_standalone.auto_set_font_size(False)
        table_D_standalone.set_fontsize(9)
        fig_D.savefig(f"{base}_D{ext}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig_D)
        # ========================================================
        # REPORT OUTPUT
        # ========================================================
        print("\nSaved subject-order consistency figures:")
        print(f"  Combined : {combined_path}")
        print(f"  Panel A  : {base}_A{ext}")
        print(f"  Panel B  : {base}_B{ext}")
        print(f"  Panel C  : {base}_C{ext}")
        print(f"  Mapping  : {base}_D{ext}")
        print("\nMultiple-comparison correction:")
        print(f"  Features                    : {n_idps:,}")
        print(f"  Unique acquisition pairs    : {n_unique_pairs:,}")
        print(f"  Valid Feature x pair tests  : {n_global_tests:,}")
        print("  Correction                  : Benjamini-Hochberg FDR")
        print(f"  Significance threshold      : {significance:g}")
        print("\nAcquisition mapping:")
        for item in acquisition_mapping:
            print(f"  {item['code']} = " f"{item['label']}")
    # ============================================================
    # SHOW
    # ============================================================
    if rep is not None:
        rep.log_plot(fig, "Subject order consistency")
    if show:
        plt.show()
    return fig



# ============================================================================
# 2.  WITHIN SUBJECT VARIABILITY
# ============================================================================
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_WithinSubjVar(
    df,
    subject_col: str = "subject",
    idp_cols: list | None = None,
    summary_stat: str = "median",
    figsize: tuple | None = None,
    savepath: str | None = None,
    rep=None,
    show: bool = False,
    debug: bool = False,
):
    """
    Publication-style visualization of within-subject variability
    across Features.
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing one row per subject/observation and
        numeric Feature columns containing pairwise RPD values.
    subject_col : str, default="subject"
        Subject identifier column.
    idp_cols : list or None, default=None
        Feature columns to plot.
        If supplied, the exact order is preserved in Panel A.
        This is important for comparing raw and harmonised results.
        If None, numeric Feature columns are detected automatically.
    summary_stat : {"median", "mean"}, default="median"
        Primary summary statistic used throughout the figure.
        The selected statistic determines:
            - Panel A central value
            - Panel B distribution
            - Panel C subject ranking
            - Panels D/E Feature ranking
            - Panel F primary summary
        Summary reporting:
            median -> median [Q1, Q3]
            mean   -> mean ± SD
    figsize : tuple or None, default=None
        Figure size for the combined figure.
        Standalone A/B/C figures use their own automatically chosen
        dimensions unless otherwise handled internally.
    savepath : str or None, default=None
        Path for saving the combined figure.
        If supplied, standalone panels are also saved automatically:_A.png,_B.png,_C.png
    rep : optional
        Retained for compatibility with the existing framework.
    show : bool, default=False
        Display the combined figure.
    debug : bool, default=False
        Print diagnostic information.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Combined A-F figure.
    Notes
    -----
    Panel A
        <= 40 Features:
            compact vertical boxplots.
        > 40 Features:
            one dot per Feature.
        Feature order is always preserved.
    Panel B
        Histogram of the selected Feature-level summary statistic.
    Panel C
        Ten subjects with highest selected variability.
        Subject labels have a dedicated axis so long subject IDs
        cannot overlap Panel B.
    Panel D/E
        Ten most and least variable Features.
    Panel F
        Summary statistics.
    In addition to the complete figure, Panels A, B and C are
    independently saved for manuscript use.
    """
    # ============================================================
    # VALIDATION
    # ============================================================
    if summary_stat not in {"median", "mean"}:
        raise ValueError("summary_stat must be either 'median' or 'mean'.")
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' was not found.")
    # ============================================================
    # IDENTIFY FEATURE COLUMNS
    # ============================================================
    metadata_cols = {
        subject_col,
        "n_obs",
        "metric_type",
        "MetricType",
        "N",
        "timepoint",
        "scan_session",
        "age",
        "sex",
    }
    if idp_cols is None:
        idp_cols = [
            col
            for col in df.columns
            if (col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col]))
        ]
    # Preserve exact supplied order
    idp_cols = list(idp_cols)
    if len(idp_cols) == 0:
        raise ValueError("No Feature columns were found.")
    # ============================================================
    # CHECK FEATURE COLUMNS
    # ============================================================
    missing_idps = [col for col in idp_cols if col not in df.columns]
    if missing_idps:
        raise ValueError("The following Feature columns are missing:\n" + "\n".join(missing_idps))
    # ============================================================
    # WORKING DATAFRAME
    # ============================================================
    work = df[[subject_col] + idp_cols].copy()
    work[subject_col] = work[subject_col].astype(str).str.strip()
    work[idp_cols] = work[idp_cols].apply(pd.to_numeric, errors="coerce")
    # ============================================================
    # BASIC COUNTS
    # ============================================================
    n_subjects = work[subject_col].nunique()
    requested_idp_count = len(idp_cols)
    # ============================================================
    # REMOVE ONLY COMPLETELY EMPTY FEATURES
    #
    # Preserve original order.
    # ============================================================
    valid_idps = [col for col in idp_cols if work[col].notna().any()]
    removed_idps = [col for col in idp_cols if col not in valid_idps]
    idp_cols = valid_idps
    n_idps = len(idp_cols)
    if n_idps == 0:
        raise ValueError("No Features contain usable numeric data.")
    # ============================================================
    # DEBUG
    # ============================================================
    if debug:
        print("\n" + "=" * 70)
        print("WITHIN-SUBJECT VARIABILITY")
        print("=" * 70)
        print(f"Subjects               : {n_subjects:,}")
        print(f"Requested Features     : {requested_idp_count:,}")
        print(f"Valid Features         : {n_idps:,}")
        print(f"Removed empty Features : {len(removed_idps):,}")
        print(f"Primary statistic      : {summary_stat}")
    # ============================================================
    # PRIMARY STATISTIC
    # ============================================================
    if summary_stat == "median":
        central_values = work[idp_cols].median(axis=0, skipna=True)
        subject_values = work.set_index(subject_col)[idp_cols].median(axis=1, skipna=True)
        central_label = "Median"
    else:
        central_values = work[idp_cols].mean(axis=0, skipna=True)
        subject_values = work.set_index(subject_col)[idp_cols].mean(axis=1, skipna=True)
        central_label = "Mean"
    # ============================================================
    # FEATURE SUMMARY
    #
    # NEVER SORT.
    #
    # This preserves the exact Feature order in Panel A.
    # ============================================================
    idp_summary = pd.DataFrame(
        {
            "central": central_values,
            "mean": (work[idp_cols].mean(axis=0, skipna=True)),
            "median": (work[idp_cols].median(axis=0, skipna=True)),
            "q25": (work[idp_cols].quantile(0.25, axis=0)),
            "q75": (work[idp_cols].quantile(0.75, axis=0)),
            "n": (work[idp_cols].count(axis=0)),
        },
        index=idp_cols,
    )
    idp_summary = idp_summary.dropna(subset=["central"])
    # ============================================================
    # SORTED COPIES FOR D/E
    # ============================================================
    top_idps = idp_summary.sort_values("central", ascending=False).head(10)
    bottom_idps = idp_summary.sort_values("central", ascending=True).head(10)
    # ============================================================
    # SUBJECT SUMMARY
    # ============================================================
    subject_values = subject_values.dropna().sort_values(ascending=False)
    top_subjects = subject_values.head(10)
    # ============================================================
    # DISTRIBUTION
    # ============================================================
    distribution = idp_summary["central"].dropna().to_numpy(dtype=float)
    if len(distribution) == 0:
        raise ValueError("No valid Feature-level summary values available.")
    # ============================================================
    # OVERALL SUMMARY
    #
    # median -> median [Q1, Q3]
    # mean   -> mean ± sample SD
    # ============================================================
    if summary_stat == "median":
        overall_value = float(np.median(distribution))
        overall_q25 = float(np.percentile(distribution, 25))
        overall_q75 = float(np.percentile(distribution, 75))
        overall_sd = np.nan
        overall_summary_label = "Feature-level median [IQR]"
        overall_summary_text = (
            f"{overall_value:.2f}% " f"[{overall_q25:.2f}%, " f"{overall_q75:.2f}%]"
        )
    else:
        overall_value = float(np.mean(distribution))
        overall_q25 = np.nan
        overall_q75 = np.nan
        if len(distribution) > 1:
            overall_sd = float(np.std(distribution, ddof=1))
        else:
            overall_sd = np.nan
        overall_summary_label = "Feature-level mean ± SD"
        if np.isfinite(overall_sd):
            overall_summary_text = f"{overall_value:.2f}% " f"± {overall_sd:.2f}%"
        else:
            overall_summary_text = f"{overall_value:.2f}% " "± NA"
    # ============================================================
    # PANEL A MODE
    # ============================================================
    use_boxplots = n_idps <= 40
    total_observations = n_subjects * n_idps
    show_fliers = total_observations <= 1500 and n_subjects <= 50 and n_idps <= 30
    # ============================================================
    # DYNAMIC PANEL A SIZING
    # ============================================================
    if n_idps <= 40:
        marker_size = 30
        axis_font = 8.5
        box_width = 0.60
    elif n_idps <= 100:
        marker_size = 23
        axis_font = 7.5
        box_width = 0.55
    elif n_idps <= 200:
        marker_size = 19
        axis_font = 7.0
        box_width = 0.50
    elif n_idps <= 400:
        marker_size = 15
        axis_font = 6.5
        box_width = 0.47
    elif n_idps <= 800:
        marker_size = 11
        axis_font = 6.0
        box_width = 0.44
    else:
        marker_size = 8
        axis_font = 5.5
        box_width = 0.40
    # ============================================================
    # COMBINED FIGURE SIZE
    # ============================================================
    if figsize is None:
        if n_idps <= 50:
            figure_height = 11.0
        elif n_idps <= 100:
            figure_height = 12.0
        elif n_idps <= 200:
            figure_height = 13.0
        elif n_idps <= 400:
            figure_height = 14.5
        elif n_idps <= 800:
            figure_height = 16.0
        else:
            figure_height = 18.0
        figsize = (15, figure_height)

    # ============================================================
    # COMMON STYLE
    # ============================================================
    def clean_axis(ax, grid_axis="x"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)
        if grid_axis == "x":
            ax.grid(axis="y", visible=False)
        else:
            ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=9)

    # ============================================================
    # HELPER: PANEL A CONTENT
    #
    # FEATURE INDEX = X AXIS
    # PAIRWISE RPD = Y AXIS
    # ============================================================
    def draw_panel_A(ax, standalone=False):
        positions = np.arange(1, len(idp_summary) + 1)
        # --------------------------------------------------------
        # Small dataset -> vertical boxplots
        # --------------------------------------------------------
        if use_boxplots:
            box_data = [work[idp].dropna().to_numpy(dtype=float) for idp in idp_summary.index]
            ax.boxplot(
                box_data,
                vert=True,
                positions=positions,
                widths=box_width,
                whis=1.5,
                showfliers=show_fliers,
                patch_artist=True,
                boxprops=dict(alpha=0.32, linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.15),
                whiskerprops=dict(linewidth=0.85),
                capprops=dict(linewidth=0.85),
                flierprops=dict(
                    marker="o",
                    markersize=3.0,
                    markerfacecolor="0.35",
                    markeredgecolor="none",
                    alpha=0.30,
                ),
            )
            # Mean dot only when mean is requested
            if summary_stat == "mean":
                ax.scatter(
                    positions,
                    idp_summary["central"].to_numpy(dtype=float),
                    s=marker_size,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.5,
                )
        # --------------------------------------------------------
        # Large dataset -> dot-only
        # --------------------------------------------------------
        else:
            ax.scatter(
                positions,
                idp_summary["central"].to_numpy(dtype=float),
                s=marker_size,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )
        # --------------------------------------------------------
        # Overall selected statistic
        # --------------------------------------------------------
        ax.axhline(overall_value, linestyle="--", linewidth=1.05, alpha=0.55, zorder=0)
        # --------------------------------------------------------
        # Feature index spacing
        # --------------------------------------------------------
        if n_idps <= 25:
            index_step = 1
        elif n_idps <= 60:
            index_step = 3
        elif n_idps <= 120:
            index_step = 8
        elif n_idps <= 250:
            index_step = 20
        elif n_idps <= 500:
            index_step = 40
        else:
            index_step = max(1, int(np.ceil(n_idps / 12)))
        tick_positions = positions[::index_step]
        tick_labels = [str(int(x)) for x in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------
        ax.set_title(
            ("Within-subject variability across Features"),
            loc="left",
            fontsize=15 if not standalone else 15,
            fontweight="bold",
            pad=24 if not standalone else 12,
        )
        # --------------------------------------------------------
        # Subtitle
        # --------------------------------------------------------
        if use_boxplots:
            subtitle = (
                f"Features are shown in the order supplied. "
                f"The {summary_stat} is shown for each Feature; "
                "boxes summarise values across subjects."
            )
        else:
            subtitle = (
                f"Each dot shows the {summary_stat} "
                "pairwise RPD for one Feature across subjects. "
                "Features are shown in the order supplied."
            )
        ax.text(
            0.0,
            1.012 if standalone else 1.008,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9 if standalone else 8.2,
            alpha=0.68,
            clip_on=False,
        )
        ax.set_xlabel("Feature index", fontsize=10)
        ax.set_ylabel("Pairwise RPD (%)", fontsize=11)
        clean_axis(ax, grid_axis="y")
        ax.tick_params(axis="x", labelsize=axis_font)
        # --------------------------------------------------------
        # Minimal legend
        # --------------------------------------------------------
        if use_boxplots:
            if summary_stat == "median":
                handles = [
                    Line2D([0], [0], color="black", linewidth=1.15, label="Median"),
                    Line2D([0], [0], color="0.35", linewidth=5, alpha=0.32, label="25–75% (IQR)"),
                    Line2D(
                        [0], [0], linestyle="--", linewidth=1.05, alpha=0.55, label="Overall median"
                    ),
                ]
            else:
                handles = [
                    Line2D([0], [0], marker="o", linestyle="None", markersize=5.5, label="Mean"),
                    Line2D([0], [0], color="black", linewidth=1.15, label="Median"),
                    Line2D([0], [0], color="0.35", linewidth=5, alpha=0.32, label="25–75% (IQR)"),
                    Line2D(
                        [0], [0], linestyle="--", linewidth=1.05, alpha=0.55, label="Overall mean"
                    ),
                ]
        else:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markersize=5.5,
                    label=f"Feature {summary_stat}",
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    linewidth=1.05,
                    alpha=0.55,
                    label=f"Overall {summary_stat}",
                ),
            ]
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.015),
            frameon=False,
            fontsize=8.0,
            ncol=len(handles),
        )

    # ============================================================
    # HELPER: PANEL B CONTENT
    # ============================================================
    def draw_panel_B(ax, standalone=False):
        x = distribution
        # --------------------------------------------------------
        # Automatic binning
        # --------------------------------------------------------
        if len(x) >= 2:
            q25_b = np.percentile(x, 25)
            q75_b = np.percentile(x, 75)
            iqr_b = q75_b - q25_b
            data_range_b = np.max(x) - np.min(x)
            if iqr_b > 0 and data_range_b > 0:
                bin_width_b = 2 * iqr_b / (len(x) ** (1 / 3))
                if bin_width_b > 0:
                    n_bins_b = int(np.ceil(data_range_b / bin_width_b))
                else:
                    n_bins_b = 12
            else:
                n_bins_b = 12
        else:
            n_bins_b = 8
        n_bins_b = int(np.clip(n_bins_b, 8, 25))
        counts, bins, patches = ax.hist(
            x, bins=n_bins_b, alpha=0.70, edgecolor="white", linewidth=0.6
        )
        # --------------------------------------------------------
        # Contained x-axis
        # --------------------------------------------------------
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_range = x_max - x_min
        if x_range > 0:
            x_padding = 0.06 * x_range
        else:
            x_padding = max(0.5, abs(x_max) * 0.06)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        # --------------------------------------------------------
        # Reference line
        # --------------------------------------------------------
        ax.axvline(overall_value, linestyle="--", linewidth=1.05, alpha=0.60, zorder=3)
        # --------------------------------------------------------
        # Summary label
        # --------------------------------------------------------
        if summary_stat == "median":
            panel_b_summary = (
                f"Median: {overall_value:.2f}% " f"[IQR {overall_q25:.2f}%, " f"{overall_q75:.2f}%]"
            )
        else:
            if np.isfinite(overall_sd):
                panel_b_summary = f"Mean: {overall_value:.2f}% " f"± {overall_sd:.2f}% SD"
            else:
                panel_b_summary = f"Mean: {overall_value:.2f}% " "± NA SD"
        ax.text(
            0.50,
            0.92,
            panel_b_summary,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            clip_on=True,
        )
        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------
        ax.set_title(
            (f"Distribution of Feature-level " f"{summary_stat} pairwise RPD"),
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel(f"{central_label} pairwise RPD (%)", fontsize=10)
        ax.set_ylabel("Number of Features", fontsize=10)
        clean_axis(ax)

    # ============================================================
    # HELPER: PANEL C CONTENT
    # ============================================================
    def draw_panel_C(ax_labels, ax_plot, standalone=False):
        subject_positions = np.arange(len(top_subjects))
        subject_values_top = top_subjects.to_numpy(dtype=float)
        # --------------------------------------------------------
        # Bars
        # --------------------------------------------------------
        bars = ax_plot.barh(
            subject_positions, subject_values_top, height=0.62, alpha=0.75, edgecolor="none"
        )
        ax_plot.invert_yaxis()
        # No y-axis labels on bar panel
        ax_plot.set_yticks([])
        ax_plot.tick_params(axis="y", left=False, labelleft=False)
        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------
        ax_labels.text(
            0.0,
            1.10 if not standalone else 1.06,
            (f"Subjects with highest " f"{summary_stat} within-subject variability"),
            transform=ax_labels.transAxes,
            ha="left",
            va="bottom",
            fontsize=13 if not standalone else 15,
            fontweight="bold",
            clip_on=False,
        )
        # --------------------------------------------------------
        # Dedicated label axis
        # --------------------------------------------------------
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(ax_plot.get_ylim())
        ax_labels.axis("off")
        # --------------------------------------------------------
        # n per subject
        # --------------------------------------------------------
        if "n_obs" in df.columns:
            n_map = df.groupby(subject_col)["n_obs"].first().to_dict()
        else:
            n_map = {}
        # --------------------------------------------------------
        # Dynamic wrapping
        # --------------------------------------------------------
        if figsize is not None:
            label_width_fraction = 3 / 16
            approximate_label_inches = figsize[0] * label_width_fraction
        else:
            approximate_label_inches = 2.5
        if approximate_label_inches >= 3.0:
            wrap_width = 30
        elif approximate_label_inches >= 2.5:
            wrap_width = 25
        elif approximate_label_inches >= 2.0:
            wrap_width = 20
        else:
            wrap_width = 16
        # --------------------------------------------------------
        # Draw labels
        # --------------------------------------------------------
        for i, subject in enumerate(top_subjects.index):
            subject = str(subject)
            if subject in n_map:
                try:
                    label = f"{subject} " f"(n={int(n_map[subject])})"
                except (ValueError, TypeError):
                    label = subject
            else:
                label = subject
            label = textwrap.fill(
                label, width=wrap_width, break_long_words=True, break_on_hyphens=False
            )
            ax_labels.text(
                0.98,
                i,
                label,
                transform=ax_labels.transData,
                ha="right",
                va="center",
                fontsize=8.5 if standalone else 8,
                clip_on=True,
            )
        # --------------------------------------------------------
        # x-axis
        # --------------------------------------------------------
        ax_plot.set_xlabel((f"{central_label} pairwise RPD " "across Features (%)"), fontsize=10)
        clean_axis(ax_plot)
        ax_plot.tick_params(axis="y", left=False, labelleft=False)
        # --------------------------------------------------------
        # Bar values
        # --------------------------------------------------------
        if len(subject_values_top) > 0:
            max_subject = np.max(subject_values_top)
            for bar, value in zip(bars, subject_values_top):
                ax_plot.text(
                    bar.get_width() + max_subject * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}",
                    va="center",
                    fontsize=8.5,
                    clip_on=True,
                )
            ax_plot.set_xlim(0, max_subject * 1.10)

    # ============================================================
    # HELPER: FEATURE TABLE
    # ============================================================
    def add_idp_table(ax, summary, title):
        ax.axis("off")
        table_data = []
        for rank, (idp, row) in enumerate(summary.iterrows(), start=1):
            table_data.append([rank, str(idp), f"{row['central']:.2f}"])
        column_labels = ["#", "Feature", f"{central_label} RPD (%)"]
        if not table_data:
            ax.text(0.02, 0.5, "No data available", fontsize=9, va="center")
            return
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.08, 0.67, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.20)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("0.85")
            cell.set_linewidth(0.45)
            if row_idx == 0:
                cell.set_text_props(weight="bold")
            if col_idx == 2:
                cell.set_text_props(ha="right")
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=7)

    # ============================================================
    # COMBINED FIGURE
    # ============================================================
    fig = plt.figure(figsize=figsize, facecolor="white")
    # ------------------------------------------------------------
    # Combined figure grid
    # ------------------------------------------------------------
    gs = fig.add_gridspec(
        nrows=3,
        ncols=16,
        height_ratios=[4.8, 2.25, 2.65],
        wspace=0.25,
        hspace=0.82,
        left=0.055,
        right=0.985,
        top=0.915,
        bottom=0.085,
    )
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0:7])
    axC_labels = fig.add_subplot(gs[1, 7:10])
    axC = fig.add_subplot(gs[1, 10:16])
    axD = fig.add_subplot(gs[2, 0:5])
    axE = fig.add_subplot(gs[2, 5:10])
    axF = fig.add_subplot(gs[2, 10:16])
    # ------------------------------------------------------------
    # Draw combined panels
    # ------------------------------------------------------------
    draw_panel_A(axA, standalone=False)
    draw_panel_B(axB, standalone=False)
    draw_panel_C(axC_labels, axC, standalone=False)
    # ------------------------------------------------------------
    # D
    # ------------------------------------------------------------
    add_idp_table(axD, top_idps, (f"10 most variable Features " f"(by {summary_stat})"))
    # ------------------------------------------------------------
    # E
    # ------------------------------------------------------------
    add_idp_table(axE, bottom_idps, (f"10 least variable Features " f"(by {summary_stat})"))
    # ------------------------------------------------------------
    # F
    # ------------------------------------------------------------
    axF.axis("off")
    axF.set_title("Summary statistics", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    summary_rows = [
        ["N subjects", f"{n_subjects:,}"],
        ["N Features", f"{n_idps:,}"],
        [overall_summary_label, overall_summary_text],
    ]
    summary_table = axF.table(
        cellText=summary_rows,
        colLabels=["Measure", "Value"],
        colWidths=[0.68, 0.32],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(8.0)
    summary_table.scale(1, 1.32)
    for (row_idx, col_idx), cell in summary_table.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
        if col_idx == 1:
            cell.set_text_props(ha="right")
    # ============================================================
    # FOOTNOTE
    # ============================================================
    fig.text(
        0.055,
        0.025,
        (
            f"N subjects = {n_subjects:,}   |   "
            f"N Features = {n_idps:,}   |   "
            f"Within-subject metric = pairwise RPD (%)   |   "
            f"Primary statistic = {summary_stat}"
        ),
        fontsize=8,
        alpha=0.65,
    )
    # ============================================================
    # SAVE COMBINED FIGURE
    # ============================================================
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
    # ============================================================
    # SAVE STANDALONE PANELS A, B, C
    # ============================================================
    if savepath is not None:
        # --------------------------------------------------------
        # Derive filenames
        # --------------------------------------------------------
        base, ext = os.path.splitext(savepath)
        if ext == "":
            ext = ".png"
        panel_a_path = f"{base}_A{ext}"
        panel_b_path = f"{base}_B{ext}"
        panel_c_path = f"{base}_C{ext}"
        # ========================================================
        # STANDALONE PANEL A
        # ========================================================
        # Reversed orientation means Feature count controls width
        # rather than height.
        if n_idps <= 40:
            standalone_a_width = max(8.5, min(16, 6.5 + n_idps * 0.18))
        else:
            standalone_a_width = max(10.5, min(18, 9.5 + n_idps * 0.008))
        figA = plt.figure(figsize=(standalone_a_width, 6.5), facecolor="white")
        axA_standalone = figA.add_axes([0.10, 0.15, 0.85, 0.70])
        draw_panel_A(axA_standalone, standalone=True)
        figA.savefig(panel_a_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figA)
        # ========================================================
        # STANDALONE PANEL B
        # ========================================================
        figB = plt.figure(figsize=(8.5, 5.5), facecolor="white")
        axB_standalone = figB.add_axes([0.12, 0.15, 0.82, 0.70])
        draw_panel_B(axB_standalone, standalone=True)
        figB.savefig(panel_b_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figB)
        # ========================================================
        # STANDALONE PANEL C
        # ========================================================
        figC = plt.figure(figsize=(10.5, 5.5), facecolor="white")
        gsC = figC.add_gridspec(
            nrows=1, ncols=10, wspace=0.05, left=0.05, right=0.98, top=0.82, bottom=0.17
        )
        axC_labels_standalone = figC.add_subplot(gsC[0, 0:4])
        axC_plot_standalone = figC.add_subplot(gsC[0, 4:10])
        draw_panel_C(axC_labels_standalone, axC_plot_standalone, standalone=True)
        figC.savefig(panel_c_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figC)
        # --------------------------------------------------------
        # Debug output
        # --------------------------------------------------------
        if debug:
            print("\nSaved figures:")
            print(f"  Combined : {savepath}")
            print(f"  Panel A  : {panel_a_path}")
            print(f"  Panel B  : {panel_b_path}")
            print(f"  Panel C  : {panel_c_path}")
    # ============================================================
    # SHOW
    # ============================================================
    if rep is not None:
        rep.log_plot(fig, "Within Subject variability")
    if show:
        plt.show()
    return fig



# ============================================================================
# 3.  MULTIVARIATE BATCH DIFFERENCE (MAHALANOBIS)
# ============================================================================
def plot_MultivariateBatchDifference(
    results: dict,
    rep=None,
    info: dict | None = None,
    annotate: bool = True,
    figsize=(14, 5),
    cmap="viridis",
    show: bool = False,
):
    """
    Plot Mahalanobis distances from the longitudinal Mahalanobis function.
    Produces one figure containing:
      - Heatmap of pairwise RAW Mahalanobis distances
      - Heatmap of pairwise RESIDUAL Mahalanobis distances (if available)
      - Bar chart of centroid-to-global distances (raw vs residual)
    Parameters
    ----------
    results : dict
        Output from MultiVariateBatchDifference_long(...).
    rep : optional
        Optional DiagnosticReport/report object. If supplied, the figure
        is logged using rep.log_plot(...).
    info : dict, optional
        Optional diagnostics dictionary returned by
        MultiVariateBatchDifference_long(..., return_info=True).
        Used to display the detected longitudinal analysis mode.
    annotate : bool
        Annotate heatmap cells and bars with numeric values.
    figsize : tuple
        Matplotlib figure size.
    cmap : str
        Matplotlib colormap for heatmaps.
    show : bool
        If True, display the figure using plt.show().
    Returns
    -------
    fig, axes
        Matplotlib Figure and dictionary of Axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    # Validation
    if not isinstance(results, dict):
        raise ValueError(
            "results must be a dictionary produced by "
            "MultiVariateBatchDifference_long(...)."
        )
    required = ["pairwise_raw", "centroid_raw", "batches"]
    for key in required:
        if key not in results:
            raise ValueError(f"Missing required key '{key}' in results.")
    pairwise_raw = results["pairwise_raw"]
    pairwise_resid = results.get("pairwise_resid", None)
    centroid_raw = results["centroid_raw"]
    centroid_resid = results.get("centroid_resid", None)
    batches = results["batches"]
    if isinstance(batches, np.ndarray):
        batches = batches.tolist()
    batches = list(batches)
    n = len(batches)
    if n < 2:
        raise ValueError("Need at least two batches to plot distances.")
    # Helpers
    def build_matrix(pairwise_dict):
        M = np.full((n, n), np.nan, dtype=float)
        # Distance of batch to itself = 0
        np.fill_diagonal(M, 0.0)
        if pairwise_dict is None:
            return M
        for (b1, b2), distance in pairwise_dict.items():
            if b1 not in batches:
                continue
            if b2 not in batches:
                continue
            i = batches.index(b1)
            j = batches.index(b2)
            M[i, j] = float(distance)
            M[j, i] = float(distance)
        return M
    def centroid_array(centroid_dict):
        if centroid_dict is None:
            return None
        values = []
        for batch_name in batches:
            key = (
                batch_name,
                "global",
            )
            if key in centroid_dict:
                values.append(float(centroid_dict[key]))
            else:
                values.append(np.nan)
        return np.asarray(values, dtype=float)
    def annotate_heatmap(ax, M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                value = M[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    def format_batch_labels(ax):
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(
            batches,
            rotation=45,
            ha="right",
        )
        ax.set_yticklabels(
            batches,
            rotation=45,
            ha="right",
        )
        ax.set_xlabel("Batch")
        ax.set_ylabel("Batch")
    # Prepare matrices
    M_raw = build_matrix(pairwise_raw)
    M_resid = None
    if pairwise_resid is not None:
        M_resid = build_matrix(pairwise_resid)
    c_raw = centroid_array(centroid_raw)
    c_resid = centroid_array(centroid_resid)
    has_residual = M_resid is not None and c_resid is not None
    # Shared heatmap scale
    max_values = []
    finite_raw = M_raw[np.isfinite(M_raw)]
    if finite_raw.size > 0:
        max_values.append(np.max(finite_raw))
    if M_resid is not None:
        finite_resid = M_resid[np.isfinite(M_resid)]
        if finite_resid.size > 0:
            max_values.append(np.max(finite_resid))
    if len(max_values) == 0:
        vmax = 1.0
    else:
        vmax = max(max_values)
    # Avoid an invalid colour scale if all values are zero.
    if vmax <= 0:
        vmax = 1.0
    vmin = 0.0
    # Figure title
    if info is not None:
        mode = info.get(
            "mode",
            None,
        )
        longitudinal_case = info.get(
            "longitudinal_case",
            None,
        )
        if longitudinal_case == ("batch_constant_within_subject"):
            subtitle = "Longitudinal: batch constant within subject"
        elif longitudinal_case == ("batch_varies_within_subject"):
            subtitle = "Longitudinal: batch varies within subject"
        elif mode == "regular":
            subtitle = "Regular"
        else:
            subtitle = ""
    else:
        subtitle = ""
    # Figure layout
    if has_residual:
        num_cols = 3
        width_ratios = [
            1.0,
            1.0,
            0.95,
        ]
    else:
        num_cols = 2
        width_ratios = [
            1.0,
            0.95,
        ]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1,
        num_cols,
        figure=fig,
        width_ratios=width_ratios,
    )
    # Raw heatmap
    ax_raw = fig.add_subplot(gs[0, 0])
    im_raw = ax_raw.imshow(
        M_raw,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax_raw.set_title("Pairwise Mahalanobis (Raw)")
    format_batch_labels(ax_raw)
    if annotate:
        annotate_heatmap(
            ax_raw,
            M_raw,
        )
    # Residual heatmap
    if has_residual:
        ax_resid = fig.add_subplot(gs[0, 1])
        im_resid = ax_resid.imshow(
            M_resid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax_resid.set_title("Pairwise Mahalanobis (Residual)")
        format_batch_labels(ax_resid)
        if annotate:
            annotate_heatmap(
                ax_resid,
                M_resid,
            )
        # Shared colourbar
        cbar = fig.colorbar(
            im_resid,
            ax=[
                ax_raw,
                ax_resid,
            ],
            fraction=0.046,
            pad=0.15,
            orientation="horizontal",
            location="top",
        )
        cbar.set_label("Mahalanobis distance")
    else:
        ax_resid = None
        cbar = fig.colorbar(
            im_raw,
            ax=ax_raw,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label("Mahalanobis distance")
    # Centroid bar plot
    if has_residual:
        ax_bar = fig.add_subplot(gs[0, 2])
    else:
        ax_bar = fig.add_subplot(gs[0, 1])
    x = np.arange(n)
    if c_resid is None:
        width = 0.6
        bars = ax_bar.bar(
            x,
            c_raw,
            width,
            label="Raw",
        )
        if annotate:
            for bar in bars:
                height = bar.get_height()
                if np.isfinite(height):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax_bar.set_title("Centroid → Global")
        ax_bar.legend()
    else:
        width = 0.38
        bars_raw = ax_bar.bar(
            x - width / 2,
            c_raw,
            width,
            label="Raw",
        )
        bars_res = ax_bar.bar(
            x + width / 2,
            c_resid,
            width,
            label="Residual",
        )
        if annotate:
            for bar in list(bars_raw) + list(bars_res):
                height = bar.get_height()
                if np.isfinite(height):
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
        ax_bar.set_title("Centroid → Global (Raw vs Residual)")
        ax_bar.legend()
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        batches,
        rotation=45,
        ha="right",
    )
    ax_bar.set_ylabel("Mahalanobis distance")
    ax_bar.set_xlabel("Batch")
    # Overall figure title
    if subtitle:
        fig.suptitle(
            f"Mahalanobis Batch Differences\n{subtitle}",
            fontsize=13,
        )
        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.92,
            ]
        )
    else:
        fig.suptitle(
            "Mahalanobis Batch Differences",
            fontsize=13,
        )
        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.94,
            ]
        )
    # Report integration
    axes = {
        "heatmap_raw": ax_raw,
        "bars": ax_bar,
    }
    if ax_resid is not None:
        axes["heatmap_resid"] = ax_resid
    if rep is not None:
        rep.log_plot(
            fig,
            "Mahalanobis distances (raw vs residual)",
        )
        plt.close(fig)
        return None, None
    if show:
        plt.show()
    return fig, axes
    
# ============================================================================
# 4.  MIXED EFFECTS — PART 1  (ICC / n_sig / WCV)
# ============================================================================

def plot_MixedEffectsPart1(
    df,
    idp_col: str = "IDP",
    version: str = "icc",
    figsize: tuple = (11, 7),
    savepath: str | None = None,
    rep=None,
    show: bool = False,
):
    """
    Scalable mixed-effects diagnostic.
    Two independent report versions are available.
    version="icc"
        Displays:
            A. ICC dot plot across IDPs
            B. ICC summary statistics
    version="batch"
        Displays:
            A. Distribution of significant pairwise batch/site
               differences per IDP
            B. Pairwise batch/site significance heatmap
            C. Batch/site code mapping
            D. Summary statistics for pairwise differences
    The two versions are intentionally independent because they
    answer different diagnostic questions.
    Important
    ---------
    IDPs are NEVER ranked in the ICC or batch-distribution plots.
    IDP index follows the exact row order of df.
    The pairwise heatmap uses the `sig_bonf` field from
    `pairwise_site_tests`.
    Pairwise significance therefore refers to the Bonferroni-adjusted
    significance stored in that field.
    Parameters
    ----------
    df : pandas.DataFrame
        Mixed-effects result dataframe.
    idp_col : str, default="IDP"
        Column containing IDP names.
    version : {"icc", "batch"}, default="icc"
        Which diagnostic to generate.
        "icc":
            ICC plot + ICC summary.
        "batch":
            significant-pair histogram + pairwise heatmap +
            batch/site mapping + pairwise summary.
    figsize : tuple, default=(11, 7)
        Figure size. It does not scale with the number of IDPs.
    savepath : str or None
        Output path.
        For example:
            savepath="mixed_effects_icc.png"
        or:
            savepath="mixed_effects_batch.png"
    rep : optional
        Existing report object supporting rep.log_plot().
    show : bool, default=False
        Display the figure.
    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    axes : dict
        Axes for the selected diagnostic.
    Examples
    --------
    ICC version:
        fig, axes = plot_MixedEffectsPart1(
            df,
            version="icc",
            savepath="mixed_effects_icc.png",
            show=True,
        )
    Batch version:
        fig, axes = plot_MixedEffectsPart1(
            df,
            version="batch",
            savepath="mixed_effects_batch.png",
            show=True,
        )
    """
    # IMPORTS
    import os
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    # THEME
    apply_plot_theme()
    # STYLE
    TITLE_SIZE = 14
    SECTION_TITLE_SIZE = 12
    AXIS_SIZE = 10
    TICK_SIZE = 8
    NOTE_SIZE = 8
    # VALIDATION
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if idp_col not in df.columns:
        raise ValueError(f"IDP column {idp_col!r} was not found in df.")
    version = str(version).lower().strip()
    if version not in ("icc", "batch"):
        raise ValueError("version must be either 'icc' or 'batch'.")
    # PRESERVE IDP ORDER
    idps = df[idp_col].astype(str).str.strip().tolist()
    n_idps = len(idps)
    if n_idps == 0:
        raise ValueError("No IDPs were found.")
    # No sorting / ranking.
    idp_index = np.arange(1, n_idps + 1, dtype=float)

    # COMMON IDP TICKS
    def _adaptive_idp_ticks():
        if n_idps <= 12:
            step = 1
        elif n_idps <= 50:
            step = 5
        elif n_idps <= 100:
            step = 10
        elif n_idps <= 250:
            step = 25
        elif n_idps <= 500:
            step = 50
        elif n_idps <= 1000:
            step = 100
        else:
            step = max(1, int(np.ceil(n_idps / 10)))
        ticks = np.arange(1, n_idps + 1, step)
        if len(ticks) == 0 or ticks[-1] != n_idps:
            ticks = np.append(ticks, n_idps)
        return np.unique(ticks)

    y_ticks = _adaptive_idp_ticks()

    # COMMON AXIS STYLE
    def _style_axis(ax, grid_axis="x"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
        ax.grid(axis=grid_axis, alpha=0.18, linewidth=0.8)

    # VERSION 1: ICC
    if version == "icc":
        # Validation
        if "ICC" not in df.columns:
            raise ValueError("'ICC' column was not found in df.")
        # Extract ICC
        icc = pd.to_numeric(df["ICC"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(icc)
        if not np.any(valid):
            raise ValueError("No valid ICC values were found.")
        # Reversed plot orientation
        #
        # x = IDP index
        # y = ICC
        x = idp_index[valid]
        y = icc[valid]
        # Figure
        fig = plt.figure(
            figsize=(max(8.0, float(figsize[0])), max(6.0, float(figsize[1]))),
            facecolor="white",
        )
        # Layout
        #
        # Top: ICC plot
        # Bottom: summary
        outer = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[2.0, 0.75],
            hspace=0.55,
            left=0.10,
            right=0.96,
            top=0.88,
            bottom=0.11,
        )
        # ICC axis
        axA = fig.add_subplot(outer[0, 0])
        axA.scatter(
            x,
            y,
            s=30 if n_idps <= 500 else 10,
            alpha=0.82,
            color=STYLE.PRIMARY,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        # ICC reference lines
        for threshold in (0.50, 0.75, 0.90):
            axA.axhline(
                threshold,
                linestyle="--",
                linewidth=0.85,
                color="gray",
                alpha=0.38,
                zorder=1,
            )
            axA.text(
                0.985,
                threshold,
                f"{threshold:.2f}",
                transform=(axA.get_yaxis_transform()),
                ha="right",
                va="bottom",
                fontsize=7,
                alpha=0.55,
            )
        # ICC limits
        ymin = min(-0.02, float(np.nanmin(y)) - 0.05)
        ymax = max(1.02, float(np.nanmax(y)) + 0.05)
        ymin = max(-1.05, ymin)
        ymax = min(1.05, ymax)
        axA.set_ylim(ymin, ymax)
        # IDP index x-axis
        axA.set_xlim(0.5, n_idps + 0.5)
        axA.set_xticks(y_ticks)
        axA.set_xticklabels([str(int(v)) for v in y_ticks], fontsize=TICK_SIZE)
        axA.set_xlabel("Feature index", fontsize=AXIS_SIZE)
        # ICC y-axis
        axA.set_ylabel("Intraclass correlation (ICC)", fontsize=AXIS_SIZE)
        axA.set_title(
            "Between-subject reliability",
            loc="left",
            fontsize=SECTION_TITLE_SIZE,
            fontweight="bold",
            pad=8,
        )
        _style_axis(axA, grid_axis="y")
        # Summary
        axB = fig.add_subplot(outer[1, 0])
        axB.axis("off")
        # Summary statistics
        mean_icc = float(np.mean(y))
        sd_icc = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
        median_icc = float(np.median(y))
        q25_icc = float(np.percentile(y, 25))
        q75_icc = float(np.percentile(y, 75))
        min_icc = float(np.min(y))
        max_icc = float(np.max(y))
        # Vertical summary table
        summary_rows = [
            [
                "Mean ± SD",
                (
                    f"{mean_icc:.2f} ± {sd_icc:.2f}"
                    if np.isfinite(sd_icc)
                    else f"{mean_icc:.2f} ± NA"
                ),
            ],
            ["Median [IQR]", (f"{median_icc:.2f} " f"[{q25_icc:.2f}, {q75_icc:.2f}]")],
            ["Minimum", f"{min_icc:.2f}"],
            ["Maximum", f"{max_icc:.2f}"],
        ]
        table = axB.table(
            cellText=summary_rows,
            colLabels=["Measure", "ICC"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, 0.02, 0.55, 0.78],
            colWidths=[0.55, 0.45],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row_i, col_i), cell in table.get_celld().items():
            cell.set_edgecolor("0.85")
            cell.set_linewidth(0.4)
            if row_i == 0:
                cell.set_text_props(weight="bold")
            if col_i > 0:
                cell.set_text_props(ha="right")
        axB.set_title(
            "Summary across IDPs",
            loc="left",
            fontsize=SECTION_TITLE_SIZE,
            fontweight="bold",
            pad=8,
        )
        # Global title
        fig.suptitle(
            ("Mixed-effects: between-subject reliability" f"  ({n_idps:,} IDPs)"),
            fontsize=14.5,
            fontweight="bold",
            y=0.95,
        )
        fig.text(
            0.10,
            0.035,
            ("IDPs are shown in the order supplied; " "no IDP ranking is applied."),
            fontsize=NOTE_SIZE,
            alpha=0.65,
        )
        # Save
        if savepath is not None:
            fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
            print("\nSaved mixed-effects ICC diagnostic:")
            print(f"  {savepath}")
        # Report
        if rep is not None:
            try:
                rep.log_plot(fig, "Mixed-effects ICC diagnostic")
            except Exception as exc:
                warnings.warn("Could not log ICC diagnostic: " f"{exc}")
        if show:
            plt.show()
        return (fig, {"ICC": axA, "summary": axB})
    # VERSION 2: BATCH
    # Validation
    required_batch_columns = ["n_is_batchSig", "pairwise_site_tests"]
    missing = [c for c in required_batch_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Batch diagnostic requires the following columns: " + ", ".join(missing)
        )
    # PANEL A: HISTOGRAM
    n_sig_values = pd.to_numeric(df["n_is_batchSig"], errors="coerce").to_numpy(
        dtype=float
    )
    n_sig_values = n_sig_values[np.isfinite(n_sig_values)]
    if len(n_sig_values) == 0:
        raise ValueError("No valid n_is_batchSig values were found.")
    n_sig_values = np.maximum(n_sig_values, 0)
    n_sig_int = np.rint(n_sig_values).astype(int)
    max_count = int(np.max(n_sig_int))

    # PAIRWISE DATA
    def _get_dict_value(d, keys, default=None):
        if not isinstance(d, dict):
            return default
        for key in keys:
            if key in d:
                return d[key]
        return default

    site_a_keys = [
        "site_a",
        "siteA",
        "batch_a",
        "batchA",
        "group_a",
        "groupA",
        "site1",
        "site_1",
        "batch1",
        "batch_1",
        "group1",
        "group_1",
        "A",
    ]
    site_b_keys = [
        "site_b",
        "siteB",
        "batch_b",
        "batchB",
        "group_b",
        "groupB",
        "site2",
        "site_2",
        "batch2",
        "batch_2",
        "group2",
        "group_2",
        "B",
    ]
    sig_keys = [
        "sig_bonf",
        "significant_bonf",
        "bonferroni_significant",
        "sig",
        "significant",
    ]

    def _extract_sig(value):
        if isinstance(value, dict):
            return bool(_get_dict_value(value, sig_keys, default=False))
        return bool(value)

    def _parse_pair(key, value):
        # Tuple/list dictionary key:
        #
        #   {(site1, site2): {...}}
        if isinstance(key, (tuple, list)) and len(key) == 2:
            return (str(key[0]), str(key[1]), _extract_sig(value))
        # Dictionary containing explicit fields
        if isinstance(value, dict):
            a = _get_dict_value(value, site_a_keys, default=None)
            b = _get_dict_value(value, site_b_keys, default=None)
            if a is not None and b is not None:
                return (str(a), str(b), _extract_sig(value))
        return None

    def _walk_pairwise(obj, output):
        if obj is None:
            return
        # Dictionary
        if isinstance(obj, dict):
            found_direct = False
            for key, value in obj.items():
                parsed = _parse_pair(key, value)
                if parsed is not None:
                    output.append(parsed)
                    found_direct = True
            # Explicit pair record
            if not found_direct:
                parsed = _parse_pair(None, obj)
                if parsed is not None:
                    output.append(parsed)
                    return
                # Nested structures
                for value in obj.values():
                    _walk_pairwise(value, output)
            return
        # List / tuple
        if isinstance(obj, (list, tuple)):
            for item in obj:
                parsed = _parse_pair(None, item)
                if parsed is not None:
                    output.append(parsed)
                else:
                    _walk_pairwise(item, output)

    def _build_pairwise_data():
        records = df["pairwise_site_tests"].tolist()
        ordered_sites = []
        row_pairs = []
        # Discover pairs/sites
        for record in records:
            parsed = []
            _walk_pairwise(record, parsed)
            row_seen = set()
            clean = []
            for a, b, sig in parsed:
                a = str(a)
                b = str(b)
                if a == b:
                    continue
                canonical = tuple(sorted((a, b)))
                if canonical in row_seen:
                    continue
                row_seen.add(canonical)
                if a not in ordered_sites:
                    ordered_sites.append(a)
                if b not in ordered_sites:
                    ordered_sites.append(b)
                clean.append((a, b, sig))
            row_pairs.append(clean)
        if len(ordered_sites) < 2:
            return None
        # Generate every unique site pair in the discovered site
        # order.
        pair_list = []
        for i in range(len(ordered_sites) - 1):
            for j in range(i + 1, len(ordered_sites)):
                pair_list.append((ordered_sites[i], ordered_sites[j]))
        pair_to_col = {pair: i for i, pair in enumerate(pair_list)}
        matrix = np.zeros((n_idps, len(pair_list)), dtype=float)
        # Fill the IDP x pair matrix
        for row_i, clean in enumerate(row_pairs):
            for a, b, sig in clean:
                canonical = tuple(sorted((a, b)))
                if canonical not in pair_to_col:
                    continue
                col_i = pair_to_col[canonical]
                matrix[row_i, col_i] = 1.0 if sig else 0.0
        # Compact codes
        code_map = {site: f"B{i + 1}" for i, site in enumerate(ordered_sites)}
        pair_labels = [(f"{code_map[a]}-{code_map[b]}") for a, b in pair_list]
        return {
            "matrix": matrix,
            "pair_labels": pair_labels,
            "pair_list": pair_list,
            "ordered_sites": ordered_sites,
            "code_map": code_map,
        }

    pairwise_data = _build_pairwise_data()
    # FIGURE
    # Figure height remains stable regardless of IDP count.
    fig = plt.figure(
        figsize=(max(10.5, float(figsize[0])), max(9.5, float(figsize[1]))),
        facecolor="white",
    )
    # BATCH VERSION LAYOUT
    #
    # A: histogram
    # B: heatmap
    # C: mapping
    # D: summary
    outer = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[1.05, 2.15, 0.90, 0.22],
        width_ratios=[1, 1],
        hspace=0.95,
        wspace=0.30,
        left=0.08,
        right=0.96,
        top=0.88,
        bottom=0.09,
    )
    # PANEL A
    # HISTOGRAM
    axA = fig.add_subplot(outer[0, 0])
    bins = np.arange(-0.5, max_count + 1.5, 1)
    axA.hist(
        n_sig_int,
        bins=bins,
        rwidth=0.84,
        color=STYLE.PRIMARY,
        alpha=STYLE.BAR_ALPHA,
        edgecolor="white",
        linewidth=0.4,
    )
    mean_count = float(np.mean(n_sig_int))
    median_count = float(np.median(n_sig_int))
    axA.axvline(mean_count, linestyle="--", linewidth=1.0, color="gray", alpha=0.72)
    axA.axvline(median_count, linestyle=":", linewidth=1.2, color="gray", alpha=0.72)
    ymax = axA.get_ylim()[1]
    if np.isclose(mean_count, median_count):
        axA.text(
            mean_count,
            ymax * 0.90,
            f"Mean/median = {mean_count:.2f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            alpha=0.75,
        )
    else:
        axA.text(
            mean_count,
            ymax * 0.90,
            f"Mean = {mean_count:.2f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            alpha=0.75,
        )
        axA.text(
            median_count,
            ymax * 0.90,
            f"Median = {median_count:.2f}",
            rotation=90,
            ha="left",
            va="top",
            fontsize=7,
            alpha=0.75,
        )
    if max_count <= 20:
        axA.set_xticks(np.arange(0, max_count + 1, 1))
    else:
        step = max(1, int(np.ceil(max_count / 10)))
        axA.set_xticks(np.arange(0, max_count + 1, step))
    axA.set_xlabel(
        "Significant pairwise batch/site differences per feature", fontsize=AXIS_SIZE
    )
    axA.set_ylabel("Number of IDPs", fontsize=AXIS_SIZE)
    axA.set_title(
        "Distribution of significant pairwise batch/site differences",
        loc="left",
        fontsize=SECTION_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    _style_axis(axA)
    # PANEL B
    # PAIRWISE HEATMAP
    axB = fig.add_subplot(outer[0, 1])
    # Keep panel B dedicated to the histogram, while the heatmap
    # gets the large row below.
    axB.axis("off")
    """ axB.text(
        0.0,
        0.95,
        (
            "Pairwise significance is shown below for "
            "each IDP and batch/site pair."
        ),
        transform=axB.transAxes,
        ha="left",
        va="top",
        fontsize=NOTE_SIZE,
        alpha=0.68,
    ) """
    # ACTUAL HEATMAP AXIS
    ax_heat = fig.add_subplot(outer[1, :])
    if pairwise_data is None:
        ax_heat.axis("off")
        ax_heat.text(
            0.5,
            0.5,
            (
                "Pairwise batch/site information "
                "is unavailable or could not be parsed."
            ),
            transform=ax_heat.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            alpha=0.70,
        )
    else:
        matrix = pairwise_data["matrix"]
        pair_labels = pairwise_data["pair_labels"]
        n_pairs = len(pair_labels)
        cmap = ListedColormap(["white", STYLE.PRIMARY])
        image = ax_heat.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        # X-axis pair labels
        if n_pairs <= 28:
            x_positions = np.arange(n_pairs)
        elif n_pairs <= 60:
            x_positions = np.arange(0, n_pairs, 2)
        else:
            step = max(1, int(np.ceil(n_pairs / 18)))
            x_positions = np.arange(0, n_pairs, step)
        ax_heat.set_xticks(x_positions)
        ax_heat.set_xticklabels(
            [pair_labels[i] for i in x_positions],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=7,
        )
        ax_heat.tick_params(axis="x", pad=5)
        ax_heat.set_xlabel("Batch/site pair", fontsize=AXIS_SIZE, labelpad=7)
        # Y-axis IDP indices
        ax_heat.set_yticks(y_ticks - 1)
        ax_heat.set_yticklabels([str(int(v)) for v in y_ticks], fontsize=TICK_SIZE)
        ax_heat.set_ylabel("IDP index", fontsize=AXIS_SIZE)
        ax_heat.set_title(
            "Pairwise batch/site significance",
            loc="left",
            fontsize=SECTION_TITLE_SIZE,
            fontweight="bold",
            pad=9,
        )
        # Simple significance legend
        legend_handle = Patch(
            facecolor=STYLE.PRIMARY, edgecolor="none", label="Bonferroni-significant"
        )
        ax_heat.legend(
            handles=[legend_handle],
            frameon=False,
            fontsize=7.5,
            loc="upper right",
            borderaxespad=0.2,
            handlelength=1.0,
            handletextpad=0.5,
        )
        # Explanation below heatmap
        """ ax_heat.text(
            0.0,
            -0.18,
            (
                "Filled cells indicate Bonferroni-adjusted "
                "significant pairwise batch/site differences "
                "within each IDP. See Panel C for the batch/site "
                "code mapping."
            ),
            transform=ax_heat.transAxes,
            ha="left",
            va="top",
            fontsize=NOTE_SIZE,
            alpha=0.68,
        ) """
        ax_heat.spines["top"].set_visible(False)
        ax_heat.spines["right"].set_visible(False)
    # PANEL C
    # BATCH/SITE MAPPING
    axC = fig.add_subplot(outer[2, 0])
    axC.axis("off")
    axC.set_title(
        "Batch/site mapping",
        loc="left",
        fontsize=SECTION_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    if pairwise_data is None:
        axC.text(
            0.5,
            0.5,
            "Mapping unavailable.",
            transform=axC.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        code_map = pairwise_data["code_map"]
        ordered_sites = pairwise_data["ordered_sites"]
        mapping_rows = [[code_map[site], site] for site in ordered_sites]
        table_C = axC.table(
            cellText=mapping_rows,
            colLabels=["Code", "Actual batch/site"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.0, -0.02, 1.0, 0.95],
            colWidths=[0.28, 0.72],
        )
        table_C.auto_set_font_size(False)
        if len(mapping_rows) <= 10:
            font_size = 8
        elif len(mapping_rows) <= 20:
            font_size = 7
        else:
            font_size = 6
        table_C.set_fontsize(font_size)
        for (row_i, col_i), cell in table_C.get_celld().items():
            cell.set_edgecolor("0.85")
            cell.set_linewidth(0.4)
            if row_i == 0:
                cell.set_text_props(weight="bold")
    # PANEL D
    # SUMMARY
    axD = fig.add_subplot(outer[2, 1])
    axD.axis("off")
    axD.set_title(
        "Summary across IDPs",
        loc="left",
        fontsize=SECTION_TITLE_SIZE,
        fontweight="bold",
        pad=8,
    )
    summary_rows = [
        [
            "Significant batch/site pairs per feature",
            f"{np.mean(n_sig_int):.2f}",
            f"{np.median(n_sig_int):.2f}",
            f"{np.min(n_sig_int):.0f}",
            f"{np.max(n_sig_int):.0f}",
        ]
    ]
    table_D = axD.table(
        cellText=summary_rows,
        colLabels=["Measure", "Mean", "Median", "Minimum", "Maximum"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.12, 1.0, 0.62],
        colWidths=[0.48, 0.13, 0.13, 0.13, 0.13],
    )
    table_D.auto_set_font_size(False)
    table_D.set_fontsize(8)
    for (row_i, col_i), cell in table_D.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.4)
        if row_i == 0:
            cell.set_text_props(weight="bold")
        if col_i > 0:
            cell.set_text_props(ha="right")
    # EMPTY FOOTER ROW
    ax_empty_left = fig.add_subplot(outer[3, 0])
    ax_empty_left.axis("off")
    ax_empty_right = fig.add_subplot(outer[3, 1])
    ax_empty_right.axis("off")
    # MAIN TITLE
    fig.suptitle(
        f"Mixed-effects diagnostics across features ({n_idps:,} features analysed)",
        fontsize=14.5,
        fontweight="bold",
        y=0.95,
    )
    # FOOTNOTE
    fig.text(
        0.08,
        0.035,
        (
            f"N Features = {n_idps:,}   |   "
            "Features are shown in the order supplied   |   "
            "Pairwise significance = Bonferroni-adjusted within each feature"
        ),
        fontsize=7.5,
        alpha=0.65,
    )
    # SAVE
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight", facecolor="white")
        print("\nSaved mixed-effects batch diagnostic:")
        print(f"  {savepath}")
    # REPORT
    if rep is not None:
        try:
            rep.log_plot(fig, "Mixed-effects batch/site diagnostic")
        except Exception as exc:
            warnings.warn("Could not log batch diagnostic: " f"{exc}")
    # SHOW
    if show:
        plt.show()
    # RETURN
    return (
        fig,
        {"histogram": axA, "pairwise_heatmap": ax_heat, "mapping": axC, "summary": axD},
    )

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
def plot_long_biological_effects(
    df: pd.DataFrame,
    covariates,
    feature_col: str = "IDP",
    significance: float = 0.05,
    plot_fdr: bool = False,
    figsize_per_panel: tuple[float, float] = (7.0, 5.0),
    jitter_width: float = 0.18,
    random_seed: int = 42,
    top_n: int = 5,
    savepath: str | None = None,
    rep=None,
    show: bool = False,
):
    """
    Plot biological fixed-effect results for one dataset.
    Two combined figures are produced.
    Figure 1
    --------
    Fixed-effect distributions across Features. Covariates are arranged in a two-column layout. Each point represents one Feature.
    Significant Features are filled. Non-significant Features are open.

    Figure 2
    --------
    Left:
        Horizontal bar plot showing the proportion of significant Features for each covariate.
    Right:
        Table containing the top Features ranked separately within each covariate by absolute effect size.
        Columns: Covariate, Feature, |beta|

    Expected columns
    ----------------
    For each covariate:
        <covariate>_est
        <covariate>_pval

    Parameters
    ----------
    df : pd.DataFrame, Mixed-effects result dataframe.
    covariates : sequence of str, Biological covariates.
    feature_col : str
        Column containing Feature names.

    significance : float
        Significance threshold.

    plot_fdr : bool
        If True, BH-FDR correction is applied separately
        for each covariate across Features.

    figsize_per_panel : tuple
        Approximate size allocated to each effect panel.

    jitter_width : float
        Maximum vertical jitter for individual Feature points.

    random_seed : int
        Seed for reproducible jitter.

    top_n : int
        Number of Features retained per covariate in the
        top-effect table.

    savepath : str, optional
        Base path for saving figures.

    rep : optional
        Report object supporting rep.log_plot().

        If None, figures are not added to a report.

    show : bool
        Display figures.

    Returns
    -------
    figs : list
        List of (caption, figure) tuples.

    summary_df : pd.DataFrame
        Covariate-level summary statistics.

    top_idps_df : pd.DataFrame
        Columns:

            Covariate
            Feature
            |beta|
    """

    # Imports
    import os
    import re

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from statsmodels.stats.multitest import multipletests

    # Validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df is empty.")

    if feature_col not in df.columns:
        raise ValueError(f"Feature column '{feature_col}' was not found.")

    if not covariates:
        raise ValueError("covariates must contain at least one covariate.")

    if not (0 < significance < 1):
        raise ValueError("significance must be between 0 and 1.")

    if jitter_width <= 0:
        raise ValueError("jitter_width must be greater than 0.")

    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be an integer >= 1.")

    covariates = list(covariates)

    # Feature names
    feature_names = df[feature_col].astype(str).str.strip().to_numpy()
    n_idps = len(feature_names)

    # Storage
    figs = []
    summary_rows = []
    top_idp_rows = []
    covariate_results = {}

    # Shared jitter
    rng = np.random.default_rng(random_seed)
    feature_jitter = rng.uniform(-jitter_width, jitter_width, size=n_idps)

    # Safe filename
    def _safe_filename(value):
        value = str(value).strip()
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
        return value.strip("_")

    # Prepare results for each covariate
    for covariate in covariates:
        est_col = f"{covariate}_est"
        pval_col = f"{covariate}_pval"

        # Required columns
        if est_col not in df.columns or pval_col not in df.columns:
            continue

        # Effect estimates
        estimates = pd.to_numeric(df[est_col], errors="coerce").to_numpy(dtype=float)

        # P-values
        pvals = pd.to_numeric(df[pval_col], errors="coerce").to_numpy(dtype=float)

        valid_est = np.isfinite(estimates)

        # Significance
        if plot_fdr:
            valid_p = np.isfinite(pvals)
            qvals = np.full(len(pvals), np.nan, dtype=float)

            if np.any(valid_p):
                _, qvals_valid, _, _ = multipletests(
                    pvals[valid_p],
                    alpha=significance,
                    method="fdr_bh",
                )
                qvals[valid_p] = qvals_valid

            significant_mask = (
                valid_est & np.isfinite(qvals) & (qvals < significance)
            )

            significance_label = rf"$p_{{\mathrm{{FDR}}}} < {significance:g}$"
            nonsignificance_label = (
                rf"$p_{{\mathrm{{FDR}}}} \geq {significance:g}$"
            )
            correction_name = "BH-FDR"

        else:
            qvals = np.full(len(pvals), np.nan, dtype=float)

            significant_mask = (
                valid_est & np.isfinite(pvals) & (pvals < significance)
            )

            significance_label = f"p < {significance:g}"
            nonsignificance_label = f"p >= {significance:g}"
            correction_name = "None"

        nonsignificant_mask = valid_est & ~significant_mask

        valid_values = estimates[valid_est]

        if valid_values.size == 0:
            continue

        # Summary statistics
        n_valid = int(np.sum(valid_est))
        n_significant = int(np.sum(significant_mask))

        prop_significant = float(n_significant / n_valid) if n_valid > 0 else np.nan

        median_estimate = float(np.median(valid_values))
        q1_estimate = float(np.percentile(valid_values, 25))
        q3_estimate = float(np.percentile(valid_values, 75))
        mean_estimate = float(np.mean(valid_values))

        sd_estimate = (
            float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else np.nan
        )

        summary_rows.append(
            {
                "Covariate": covariate,
                "N Features": int(len(df)),
                "N valid estimates": n_valid,
                "Median estimate": median_estimate,
                "Q1 estimate": q1_estimate,
                "Q3 estimate": q3_estimate,
                "IQR estimate": q3_estimate - q1_estimate,
                "Mean estimate": mean_estimate,
                "SD estimate": sd_estimate,
                "N significant Features": n_significant,
                "Proportion significant": prop_significant,
                "Correction": correction_name,
            }
        )

        # Store covariate results
        covariate_results[covariate] = {
            "estimates": estimates,
            "pvalues": pvals,
            "qvalues": qvals,
            "valid": valid_est,
            "significant": significant_mask,
            "nonsignificant": nonsignificant_mask,
            "n_valid": n_valid,
            "n_significant": n_significant,
            "prop_significant": prop_significant,
            "significance_label": significance_label,
            "nonsignificance_label": nonsignificance_label,
            "correction": correction_name,
        }

        # Top Features: rank separately within each covariate by |beta|
        top_df = pd.DataFrame(
            {
                "Covariate": np.repeat(covariate, len(feature_names)),
                "Feature": feature_names,
                "|beta|": np.abs(estimates),
            }
        )

        top_df = (
            top_df[np.isfinite(top_df["|beta|"])]
            .sort_values("|beta|", ascending=False, kind="stable")
            .head(top_n)
            .reset_index(drop=True)
        )

        top_idp_rows.extend(top_df.to_dict(orient="records"))

    # Check usable covariates
    if len(covariate_results) == 0:
        return (
            [],
            pd.DataFrame(),
            pd.DataFrame(columns=["Covariate", "Feature", "|beta|"]),
        )

    valid_covariates = list(covariate_results.keys())

    # Figure 1: biological effect distributions (two-column layout)
    n_covariates = len(valid_covariates)
    n_cols = min(2, n_covariates)
    n_rows = int(np.ceil(n_covariates / n_cols))

    fig_effects, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
        facecolor="white",
    )

    axes = axes.flatten()

    # Effect-distribution panels
    for ax, covariate in zip(axes, valid_covariates):
        result = covariate_results[covariate]

        estimates = result["estimates"]
        valid_est = result["valid"]
        significant_mask = result["significant"]
        nonsignificant_mask = result["nonsignificant"]
        valid_values = estimates[valid_est]
        n_valid = result["n_valid"]
        n_significant = result["n_significant"]
        prop_significant = result["prop_significant"]

        # Boxplot
        ax.boxplot(
            valid_values,
            vert=False,
            positions=[0],
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(alpha=0.25),
            medianprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="", markersize=0),
        )

        # Non-significant Features
        if np.any(nonsignificant_mask):
            ax.scatter(
                estimates[nonsignificant_mask],
                feature_jitter[nonsignificant_mask],
                s=22,
                facecolors="white",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.85,
                zorder=3,
            )

        # Significant Features
        if np.any(significant_mask):
            ax.scatter(
                estimates[significant_mask],
                feature_jitter[significant_mask],
                s=24,
                facecolors="C3",
                edgecolors="black",
                linewidths=0.5,
                alpha=0.90,
                zorder=4,
            )

        # Zero-effect line
        ax.axvline(0.0, linestyle="--", linewidth=1.0)

        # X-axis limits
        xmin = min(float(np.min(valid_values)), 0.0)
        xmax = max(float(np.max(valid_values)), 0.0)

        span = xmax - xmin

        if span == 0:
            span = 1.0

        padding = 0.08 * span

        ax.set_xlim(xmin - padding, xmax + padding)

        # Significance summary
        ax.text(
            0.02,
            0.94,
            (
                f"Significant: "
                f"{n_significant}/{n_valid} "
                f"("
                f"{100 * prop_significant:.1f}%"
                f")"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        # Axes
        ax.set_xlabel(rf"{covariate} effect estimate ($\beta$)")
        ax.set_ylabel("Features")
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([0])
        ax.set_yticklabels([])
        ax.grid(axis="x", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Title
        ax.set_title(covariate.capitalize(), fontsize=13, fontweight="bold", pad=10)

        # Legend
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="C3",
                markeredgecolor="black",
                markersize=7,
                label=result["significance_label"],
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=7,
                label=result["nonsignificance_label"],
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=False,
            fontsize=8,
        )

    # Hide unused effect panels
    for ax in axes[n_covariates:]:
        ax.axis("off")

    # Effect figure title
    if plot_fdr:
        effects_title = (
            "Biological fixed-effect estimates across Features (FDR-corrected)"
        )
    else:
        effects_title = "Biological fixed-effect estimates across Features"

    fig_effects.suptitle(effects_title, fontsize=16, fontweight="bold", y=0.99)

    fig_effects.text(
        0.5,
        0.015,
        (
            "Each point represents one Feature. "
            "The dashed line represents "
            r"$\beta = 0$."
        ),
        ha="center",
        fontsize=8.5,
        alpha=0.65,
    )

    fig_effects.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    # Save effect figure
    if savepath is not None:
        base, ext = os.path.splitext(savepath)

        if ext == "":
            ext = ".png"

        fig_effects.savefig(
            f"{base}_biological_effects{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    # Report effect figure
    effects_caption = "Biological fixed-effect estimates across Features"

    if rep is not None:
        rep.log_plot(fig_effects, caption=effects_caption)

    figs.append((effects_caption, fig_effects))

    if show:
        plt.show()

    # Top Feature dataframe
    top_idps_df = pd.DataFrame(
        top_idp_rows,
        columns=["Covariate", "Feature", "|beta|"],
    )

    # Preserve covariate order and descending |beta|
    if not top_idps_df.empty:
        top_idps_df["Covariate"] = pd.Categorical(
            top_idps_df["Covariate"],
            categories=valid_covariates,
            ordered=True,
        )

        top_idps_df = top_idps_df.sort_values(
            ["Covariate", "|beta|"],
            ascending=[True, False],
            kind="stable",
        ).reset_index(drop=True)

        top_idps_df["Covariate"] = top_idps_df["Covariate"].astype(str)

    # Figure 2
    # Left: proportion of significant Features
    # Right: top-Feature table
    fig_summary, (ax_sig, ax_table) = plt.subplots(
        1,
        2,
        figsize=(16, max(5.5, 0.42 * len(top_idps_df) + 2.5)),
        gridspec_kw={"width_ratios": [0.85, 1.55]},
        facecolor="white",
    )

    # Left panel: horizontal proportion plot
    sig_covariates = []
    sig_proportions = []
    sig_counts = []
    sig_totals = []

    for covariate in valid_covariates:
        result = covariate_results[covariate]

        sig_covariates.append(covariate)
        sig_proportions.append(result["prop_significant"])
        sig_counts.append(result["n_significant"])
        sig_totals.append(result["n_valid"])

    # Reverse so first covariate appears at top
    plot_covariates = sig_covariates[::-1]
    plot_proportions = np.asarray(sig_proportions[::-1], dtype=float)
    plot_counts = sig_counts[::-1]
    plot_totals = sig_totals[::-1]

    y_positions = np.arange(len(plot_covariates))

    bars = ax_sig.barh(y_positions, plot_proportions, height=0.58, alpha=0.75)

    # Bar labels
    for bar, n_sig, n_total, proportion in zip(
        bars,
        plot_counts,
        plot_totals,
        plot_proportions,
    ):
        ax_sig.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{n_sig}/{n_total} ({100 * proportion:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Axes
    ax_sig.set_yticks(y_positions)
    ax_sig.set_yticklabels([str(c) for c in plot_covariates], fontsize=10)
    ax_sig.set_xlabel("Proportion of significant Features")
    ax_sig.set_ylabel("Biological covariate")
    ax_sig.set_xlim(0, 1.18)
    ax_sig.grid(axis="x", alpha=0.20)
    ax_sig.spines["top"].set_visible(False)
    ax_sig.spines["right"].set_visible(False)

    if plot_fdr:
        sig_panel_title = "Significant Features\n(FDR-corrected)"
    else:
        sig_panel_title = "Significant Features"

    ax_sig.set_title(sig_panel_title, fontsize=13, fontweight="bold", pad=12)

    # Right panel: top-Feature table
    ax_table.axis("off")

    if not top_idps_df.empty:
        table_display = top_idps_df.copy()

        table_display["|beta|"] = table_display["|beta|"].map(
            lambda value: f"{value:.4g}"
        )

        table = ax_table.table(
            cellText=table_display[["Covariate", "Feature", "|beta|"]].values,
            colLabels=["Covariate", "Feature", r"$|\beta|$"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=[0.18, 0.64, 0.18],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.30)

        # Header formatting
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_linewidth(0.4)

            if row_index == 0:
                cell.set_text_props(weight="bold")

        ax_table.set_title(
            f"Top {top_n} Features by " r"$|\beta|$",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

    else:
        ax_table.text(
            0.5,
            0.5,
            "No valid biological effect estimates",
            ha="center",
            va="center",
        )

    # Summary figure title
    fig_summary.suptitle(
        "Biological association summary",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    # fig_summary.text(
    #     0.5,
    #     0.015,
    #     (
    #         f"Top Features are ranked separately within each "
    #         f"covariate by absolute effect size."
    #     ),
    #     ha="center",
    #     fontsize=8.5,
    #     alpha=0.65,
    # )

    fig_summary.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])

    # Save summary figure
    if savepath is not None:
        base, ext = os.path.splitext(savepath)

        if ext == "":
            ext = ".png"

        fig_summary.savefig(
            f"{base}_biological_summary{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    # Report summary figure
    summary_caption = (
        "Biological association summary: significant Features and strongest effects"
    )

    if rep is not None:
        rep.log_plot(fig_summary, caption=summary_caption)

    figs.append((summary_caption, fig_summary))

    if show:
        plt.show()

    # Summary dataframe
    summary_df = pd.DataFrame(summary_rows)

    # Formatted summary columns
    # Median -> Median [Q1, Q3]; Mean -> Mean ± SD
    if not summary_df.empty:
        summary_df["Median estimate [IQR]"] = summary_df.apply(
            lambda row: (
                f"{row['Median estimate']:.3f} "
                f"[{row['Q1 estimate']:.3f}, {row['Q3 estimate']:.3f}]"
                if (
                    pd.notna(row["Median estimate"])
                    and pd.notna(row["Q1 estimate"])
                    and pd.notna(row["Q3 estimate"])
                )
                else "NA"
            ),
            axis=1,
        )

        summary_df["Mean estimate ± SD"] = summary_df.apply(
            lambda row: (
                f"{row['Mean estimate']:.3f} ± {row['SD estimate']:.3f}"
                if (pd.notna(row["Mean estimate"]) and pd.notna(row["SD estimate"]))
                else "NA"
            ),
            axis=1,
        )

        summary_df["Significant Features"] = summary_df.apply(
            lambda row: (
                f"{int(row['N significant Features'])}"
                f"/"
                f"{int(row['N valid estimates'])}"
                f" ({100 * row['Proportion significant']:.1f}%)"
                if row["N valid estimates"] > 0
                else "NA"
            ),
            axis=1,
        )

    # Print summary
    if not summary_df.empty:
        display_summary = summary_df[
            [
                "Covariate",
                "Median estimate [IQR]",
                "Mean estimate ± SD",
                "Significant Features",
                "N Features",
                "Correction",
            ]
        ].copy()

        print("\n" + "=" * 70)
        print("BIOLOGICAL FIXED-EFFECT SUMMARY")
        print("=" * 70)
        print(display_summary.to_string(index=False))

    # Print top Features
    if not top_idps_df.empty:
        print("\n" + "=" * 70)
        print(f"TOP {top_n} FEATURES BY ABSOLUTE EFFECT SIZE")
        print("=" * 70)
        print(top_idps_df.to_string(index=False))

    return (
        figs,
        summary_df,
        top_idps_df,
    )

# ============================================================================
# 6.  ADDITIVE / MULTIPLICATIVE EFFECTS  (p-value matrix)
# ============================================================================
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_AddMultEffects(
    dfs,
    feature_col: str = "Feature",
    p_col: str = "p-value",
    labels: list | None = None,
    p_thr: float = 0.05,
    figsize: tuple = (12, 8),
    savepath: str | None = None,
    rep=None,
    show: bool = False,
):
    """
    Plot feature-level additive/multiplicative effect significance.

    Main panel
    ----------
    One point per Feature.

    x-axis:
        Feature index in the order supplied to the function.

    y-axis:
        -log10(p-value)

    A horizontal dashed line marks the significance threshold.

    Extremely small p-values are capped for display so that they do
    not stretch the y-axis and make the threshold visually useless.
    Capped values are indicated with a triangle.

    For multiple DataFrames, each dataset is shown using a small
    horizontal offset so raw/harmonised results can be compared at the
    same Feature position.

    Bottom panel
    ------------
    Summary statistics across all Features.

    Parameters
    ----------
    dfs : pandas.DataFrame or sequence of pandas.DataFrame
        One or more result DataFrames.

    feature_col : str, default="Feature"
        Feature name column.

    p_col : str, default="p-value"
        P-value column.

    labels : list[str] or None
        Dataset labels. Example:

            ["Raw", "Harmonised"]

    p_thr : float, default=0.05
        Nominal significance threshold.

    figsize : tuple, default=(12, 8)
        Size of the combined figure.

    savepath : str or None
        Output filename for the combined figure.

        Example:

            "additive_effects.png"

        Also creates:

            "additive_effects_A.png"
            "additive_effects_B.png"

    rep : retained for compatibility.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        Combined figure.
    """

    # Constants
    tiny = 1e-300

    # Maximum -log10(p) shown explicitly.
    # Larger values are displayed at this limit using triangles.
    display_logp_max = 10.0

    # Fixed visual sizes.
    title_size = 13
    axis_size = 10
    tick_size = 8

    # Validation
    if not (0 < p_thr < 1):
        raise ValueError("p_thr must be between 0 and 1.")

    # Normalise input
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    else:
        try:
            dfs = list(dfs)
        except TypeError:
            raise TypeError(
                "dfs must be a pandas DataFrame or a sequence of pandas DataFrames."
            )

    if len(dfs) == 0:
        raise ValueError("At least one DataFrame must be supplied.")

    n_datasets = len(dfs)

    # Labels
    if labels is None:
        if n_datasets == 1:
            labels = ["Dataset"]
        else:
            labels = [f"Dataset {i + 1}" for i in range(n_datasets)]
    else:
        labels = list(labels)

    if len(labels) != n_datasets:
        raise ValueError("labels must have the same length as dfs.")

    # Clean data
    clean_dfs = []

    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"dfs[{i}] must be a pandas DataFrame.")

        required = [feature_col, p_col]

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                f"DataFrame {i + 1} is missing columns: " + ", ".join(missing)
            )

        tmp = df[[feature_col, p_col]].copy()

        tmp[feature_col] = tmp[feature_col].astype(str).str.strip()

        tmp[p_col] = pd.to_numeric(tmp[p_col], errors="coerce")

        # Keep first feature occurrence.
        tmp = tmp.drop_duplicates(subset=feature_col, keep="first")

        clean_dfs.append(tmp)

    # Preserve feature order
    all_features = list(
        dict.fromkeys(
            feature for df in clean_dfs for feature in df[feature_col].tolist()
        )
    )

    n_features = len(all_features)

    if n_features == 0:
        raise ValueError("No Features were found.")

    # P-value matrix
    pmat = pd.DataFrame(
        np.nan,
        index=all_features,
        columns=labels,
        dtype=float,
    )

    for df, label in zip(clean_dfs, labels):
        series = df.set_index(feature_col)[p_col].astype(float)

        pmat.loc[series.index, label] = series

    # Summary statistics
    # Only retained for display: N Features, significant Features,
    # proportion significant.
    summaries = {}

    for label in labels:
        values = pmat[label].dropna().to_numpy(dtype=float)

        values = values[np.isfinite(values)]

        values = np.clip(values, tiny, 1.0)

        if len(values) == 0:
            summaries[label] = {
                "n": 0,
                "n_sig": 0,
                "pct_sig": np.nan,
            }

        else:
            summaries[label] = {
                "n": int(len(values)),
                "n_sig": int(np.sum(values < p_thr)),
                "pct_sig": float(100 * np.mean(values < p_thr)),
            }

    # Common axis style
    def style_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(axis="y", alpha=0.18, linewidth=0.8)

        ax.tick_params(axis="both", labelsize=tick_size)

    # Panel A
    # Reversed orientation: x = feature index, y = -log10(p)
    def draw_panel_A(ax, standalone=False):
        x = np.arange(1, n_features + 1, dtype=float)

        # Horizontal offsets for multiple datasets.
        if n_datasets == 1:
            offsets = [0.0]
        elif n_datasets == 2:
            offsets = [-0.16, 0.16]
        elif n_datasets == 3:
            offsets = [-0.22, 0.0, 0.22]
        else:
            offsets = np.linspace(-0.30, 0.30, n_datasets)

        # Plot each dataset.
        for label, offset in zip(labels, offsets):
            raw_p = pmat[label].to_numpy(dtype=float)

            valid = np.isfinite(raw_p)

            if not np.any(valid):
                continue

            safe_p = np.clip(raw_p[valid], tiny, 1.0)

            logp = -np.log10(safe_p)

            # Capped values.
            capped = logp > display_logp_max

            displayed = np.minimum(logp, display_logp_max)

            x_valid = x[valid] + offset

            # Normal points.
            normal = ~capped

            if np.any(normal):
                ax.scatter(
                    x_valid[normal],
                    displayed[normal],
                    s=(24 if n_features < 500 else 10),
                    alpha=0.82,
                    edgecolors="white",
                    linewidths=0.40,
                    label=label,
                    zorder=3,
                )

            # Capped extreme values.
            if np.any(capped):
                ax.scatter(
                    x_valid[capped],
                    displayed[capped],
                    s=(36 if n_features < 500 else 15),
                    marker="^",
                    alpha=0.90,
                    edgecolors="white",
                    linewidths=0.45,
                    label=(f"{label} (>10)" if n_datasets == 1 else None),
                    zorder=4,
                )

        # Significance threshold
        threshold_logp = -np.log10(p_thr)

        ax.axhline(
            threshold_logp,
            linestyle="--",
            linewidth=1.25,
            alpha=0.75,
            zorder=1,
        )

        ax.text(
            0.985,
            threshold_logp,
            f"p = {p_thr:g}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.80,
        )

        # X axis: feature index
        ax.set_xlim(0.5, n_features + 0.5)

        if n_features <= 20:
            x_step = 1
        elif n_features <= 50:
            x_step = 5
        elif n_features <= 100:
            x_step = 10
        elif n_features <= 250:
            x_step = 25
        elif n_features <= 500:
            x_step = 50
        elif n_features <= 1000:
            x_step = 100
        else:
            x_step = max(1, int(np.ceil(n_features / 10)))

        x_ticks = np.arange(1, n_features + 1, x_step)

        if len(x_ticks) == 0 or x_ticks[-1] != n_features:
            x_ticks = np.append(x_ticks, n_features)

        x_ticks = np.unique(x_ticks)

        ax.set_xticks(x_ticks)

        ax.set_xticklabels([str(int(v)) for v in x_ticks], fontsize=tick_size)

        ax.set_xlabel("Feature index", fontsize=axis_size)

        # Y axis: -log10(p)
        ax.set_ylim(0, display_logp_max)

        y_ticks = np.array([0, 1, 2, 3, 5, 7, 10], dtype=float)

        ax.set_yticks(y_ticks)

        ax.set_yticklabels([f"{y:g}" for y in y_ticks], fontsize=tick_size)

        ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=axis_size)

        # Title
        if standalone:
            title = "Batch-effect significance across Features"
        else:
            title = "Batch-effect significance across Features"

        ax.set_title(
            title,
            loc="left",
            fontsize=(14 if standalone else title_size),
            fontweight="bold",
            pad=8,
        )

        # Explanatory note
        ax.text(
            0.0,
            -0.20,
            (
                "Features are displayed in the order supplied to the function. "
                "Triangles indicate "
                r"$-\log_{10}(p)>10$."
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            alpha=0.68,
        )

        # Legend
        if n_datasets > 1:
            ax.legend(frameon=False, fontsize=tick_size, loc="upper right")

        style_axis(ax)

    # Panel B: summary table
    def draw_panel_B(ax, standalone=False):
        ax.axis("off")

        if standalone:
            title = "Summary across Features"
        else:
            title = "Summary across Features"

        ax.set_title(
            title,
            loc="left",
            fontsize=(14 if standalone else title_size),
            fontweight="bold",
            pad=8,
        )

        # Single dataset
        if n_datasets == 1:
            label = labels[0]

            s = summaries[label]

            rows = [
                ["N Features", f"{s['n']:,}"],
                [f"Significant Features (p < {p_thr:g})", f"{s['n_sig']:,}"],
                [
                    "Proportion significant",
                    (f"{s['pct_sig']:.1f}%" if np.isfinite(s["pct_sig"]) else "NA"),
                ],
            ]

            table = ax.table(
                cellText=rows,
                colLabels=["Measure", label],
                cellLoc="left",
                colLoc="left",
                bbox=[0.0, 0.20, 1.0, 0.62],
                colWidths=[0.72, 0.28],
            )

        # Multiple datasets
        else:
            measures = [
                "N Features",
                f"Significant Features (p < {p_thr:g})",
                "Proportion significant",
            ]

            rows = []

            for measure in measures:
                row = [measure]

                for label in labels:
                    s = summaries[label]

                    if measure == "N Features":
                        value = f"{s['n']:,}"

                    elif measure.startswith("Significant"):
                        value = f"{s['n_sig']:,}"

                    else:
                        value = (
                            f"{s['pct_sig']:.1f}%" if np.isfinite(s["pct_sig"]) else "NA"
                        )

                    row.append(value)

                rows.append(row)

            table = ax.table(
                cellText=rows,
                colLabels=["Measure", *labels],
                cellLoc="left",
                colLoc="left",
                bbox=[0.0, 0.20, 1.0, 0.62],
            )

        # Table styling
        table.auto_set_font_size(False)

        table.set_fontsize(8)

        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("0.85")

            cell.set_linewidth(0.40)

            if row_idx == 0:
                cell.set_text_props(weight="bold")

            if col_idx > 0:
                cell.set_text_props(ha="right")

    # Figure size (does NOT scale with n_features)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), facecolor="white")

    # Figure layout
    outer = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3.0, 1.15],
        hspace=0.28,
        left=0.08,
        right=0.96,
        top=0.89,
        bottom=0.11,
    )

    # Panel A
    axA = fig.add_subplot(outer[0, 0])

    draw_panel_A(axA, standalone=False)

    # Panel B
    axB = fig.add_subplot(outer[1, 0])

    draw_panel_B(axB, standalone=False)

    # Main title
    fig.suptitle(
        f"Feature significance diagnostic  ({n_features:,} Features analysed)",
        fontsize=14.5,
        fontweight="bold",
        y=0.965,
    )

    # Footnote
    fig.text(
        0.08,
        0.035,
        (
            f"N Features = {n_features:,}   |   "
            f"Significance threshold = p < {p_thr:g}   |   "
            "Feature order preserved from supplied results"
        ),
        fontsize=7.5,
        alpha=0.65,
    )

    # Save
    if savepath is not None:
        base, ext = os.path.splitext(savepath)

        if ext == "":
            ext = ".png"

        # Combined
        combined_path = f"{base}{ext}"

        fig.savefig(
            combined_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        # Standalone A
        fig_A = plt.figure(figsize=(11, 7), facecolor="white")

        ax_A = fig_A.add_axes([0.11, 0.17, 0.82, 0.70])

        draw_panel_A(ax_A, standalone=True)

        fig_A.text(
            0.11,
            0.045,
            (
                f"N Features = {n_features:,}. "
                "Features are displayed in the order supplied."
            ),
            fontsize=8,
            alpha=0.65,
        )

        fig_A.savefig(
            f"{base}_A{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig_A)

        # Standalone B
        fig_B = plt.figure(figsize=(7.5, 5.5), facecolor="white")

        ax_B = fig_B.add_axes([0.10, 0.13, 0.80, 0.75])

        draw_panel_B(ax_B, standalone=True)

        fig_B.savefig(
            f"{base}_B{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig_B)

        print("\nSaved additive/multiplicative effect figures:")
        print(f"  Combined : {combined_path}")
        print(f"  Panel A  : {base}_A{ext}")
        print(f"  Panel B  : {base}_B{ext}")

    # Show
    if rep is not None:
        rep.log_plot(fig, "Batch effects")

    if show:
        plt.show()

    return fig

def plot_age_percentile_chart(
    df,
    age_col="age",
    value_col="y",
    batch_col="batch",
    percentiles=(5, 25, 50, 75, 95),
    n_bins=25,
    smooth_frac=0.25,
    show_points=True,
    show_batch_coloring=True,
    figsize=(10, 6),
    title=None,
):
    """
    Plot smoothed percentile curves of a value against age.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    age_col : str
        Age variable to use on the x-axis.
    value_col : str
        Outcome / feature to summarise on the y-axis.
    batch_col : str or None
        Optional batch column for point coloring.
    percentiles : tuple[int]
        Percentiles to plot.
    n_bins : int
        Number of age bins used to estimate local percentile curves.
    smooth_frac : float
        LOWESS smoothing fraction applied to the binned percentile estimates.
    show_points : bool
        If True, scatter the raw data faintly.
    show_batch_coloring : bool
        If True, use batch colors for raw points.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title. If None, a default title is used.

    Returns
    -------
    fig, ax
    """
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as e:
        raise ImportError(
            "This function requires statsmodels. Install it with `pip install statsmodels`."
        ) from e

    if age_col not in df.columns:
        raise ValueError(f"'{age_col}' was not found in the dataframe.")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' was not found in the dataframe.")
    if batch_col is not None and batch_col not in df.columns:
        batch_col = None

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if not (0 < smooth_frac <= 1):
        raise ValueError("smooth_frac must be in (0, 1].")

    try:
        percentiles = sorted({float(p) for p in percentiles})
    except Exception as e:
        raise ValueError("percentiles must be an iterable of numeric values.") from e
    if not percentiles:
        raise ValueError("percentiles must not be empty.")
    if any((p < 0 or p > 100) for p in percentiles):
        raise ValueError("All percentiles must be between 0 and 100.")

    d = df[[age_col, value_col] + ([batch_col] if batch_col else [])].copy()
    d = d.dropna(subset=[age_col, value_col])

    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[age_col, value_col])

    if len(d) < 10:
        raise ValueError("Not enough data to plot percentile curves.")

    # Build age bins
    try:
        bins = pd.qcut(d[age_col], q=min(n_bins, d.shape[0]), duplicates="drop")
    except ValueError:
        bins = pd.cut(d[age_col], bins=min(n_bins, max(3, d.shape[0] // 2)))

    d = d.assign(_age_bin=bins)

    # Compute binned percentile estimates
    rows = []
    for bin_id, g in d.groupby("_age_bin", observed=True):
        if g.shape[0] < 3:
            continue
        row = {
            "age_center": float(g[age_col].median()),
            "n": int(g.shape[0]),
        }
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(g[value_col], p))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("age_center")
    if summary.empty:
        raise ValueError(
            "Could not compute percentile bins. Try fewer bins or more data."
        )

    x_grid = np.linspace(d[age_col].min(), d[age_col].max(), 400)

    def smooth_curve(x, y, frac=smooth_frac):
        """
        LOWESS smoothing on binned percentile estimates, then interpolate to the grid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if len(x) < 4:
            return np.interp(x_grid, x, y)

        sm = lowess(y, x, frac=frac, return_sorted=True)
        xs, ys = sm[:, 0], sm[:, 1]

        # Ensure unique x values for interpolation
        xs_unique, idx = np.unique(xs, return_index=True)
        ys_unique = ys[idx]

        return np.interp(x_grid, xs_unique, ys_unique)

    curves = {}
    for p in percentiles:
        curves[p] = smooth_curve(summary["age_center"], summary[f"p{p}"])

    fig, ax = plt.subplots(figsize=figsize)

    # Raw points, if requested
    if show_points:
        if show_batch_coloring and batch_col is not None:
            batches = list(pd.unique(d[batch_col]))
            cmap = plt.get_cmap("tab10")
            for i, b in enumerate(batches):
                g = d[d[batch_col] == b]
                ax.scatter(
                    g[age_col],
                    g[value_col],
                    s=12,
                    alpha=0.18,
                    label=str(b),
                    color=cmap(i % 10),
                    edgecolors="none",
                )
        else:
            ax.scatter(
                d[age_col],
                d[value_col],
                s=12,
                alpha=0.15,
                color="0.5",
                edgecolors="none",
            )

    lower = [p for p in percentiles if p < 50]
    upper = [p for p in percentiles if p > 50]

    # Shaded percentile bands (adaptive to supplied percentiles)
    if lower and upper:
        if 25.0 in curves and 75.0 in curves:
            p_lo_inner, p_hi_inner = 25.0, 75.0
        else:
            p_lo_inner = max(lower)
            p_hi_inner = min(upper)

        ax.fill_between(
            x_grid,
            curves[p_lo_inner],
            curves[p_hi_inner],
            alpha=0.18,
            label=f"{p_lo_inner:g}th-{p_hi_inner:g}th",
        )

        if 5.0 in curves and 95.0 in curves:
            p_lo_outer, p_hi_outer = 5.0, 95.0
        else:
            p_lo_outer = min(lower)
            p_hi_outer = max(upper)

        if (p_lo_outer, p_hi_outer) != (p_lo_inner, p_hi_inner):
            ax.fill_between(
                x_grid,
                curves[p_lo_outer],
                curves[p_hi_outer],
                alpha=0.08,
                label=f"{p_lo_outer:g}th-{p_hi_outer:g}th",
            )

    # Main percentile curves
    for p in percentiles:
        linewidth = 2.6 if np.isclose(p, 50.0) else 1.6
        ax.plot(x_grid, curves[p], linewidth=linewidth, label=f"{p:g}th percentile")

    ax.set_xlabel(age_col.capitalize())
    ax.set_ylabel(value_col.capitalize() if value_col else "Value")
    ax.set_title(title or f"Smoothed percentile chart of {value_col} by age")

    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep the legend readable without clutter
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=2, loc="best")
    plt.show()

    fig.tight_layout()
    return fig, ax

# ============================================================
# PLOT IDP DISTRIBUTION
# ============================================================
from pathlib import Path
from typing import Optional, Sequence, Dict, Union

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import norm


# ============================================================
# ROBUST Z-SCORE
# ============================================================

def _robust_zscore(values: np.ndarray) -> np.ndarray:
    """
    Robust standardisation using median and MAD.

    z = (x - median) / (1.4826 * MAD)
    """

    values = np.asarray(values, dtype=float)

    out = np.full(values.shape, np.nan, dtype=float)

    valid = np.isfinite(values)

    if not valid.any():
        return out

    median = np.median(values[valid])

    mad = np.median(np.abs(values[valid] - median))

    mad_scaled = 1.4826 * mad

    if not np.isfinite(mad_scaled) or mad_scaled == 0:
        out[valid] = 0.0
        return out

    out[valid] = (values[valid] - median) / mad_scaled

    return out


# Prepare input
def _prepare_longitudinal_input(
    data: Union[pd.DataFrame, np.ndarray],
    batch: Sequence,
    subject: Sequence,
    idp_names: Optional[Sequence[str]] = None,
):
    """
    Convert DataFrame or NumPy input to a common internal format.
    """

    if isinstance(data, pd.DataFrame):
        df = data.copy()

        n_samples = len(df)

        if batch is None:
            if "batch" not in df.columns:
                raise ValueError(
                    "batch must be supplied when the DataFrame "
                    "does not contain a 'batch' column."
                )

            batch_arr = df["batch"].astype(str).to_numpy()

        else:
            batch_arr = np.asarray(batch).astype(str)

        if subject is None:
            if "subject" not in df.columns:
                raise ValueError(
                    "subject must be supplied when the DataFrame "
                    "does not contain a 'subject' column."
                )

            subject_arr = df["subject"].astype(str).to_numpy()

        else:
            subject_arr = np.asarray(subject).astype(str)

        if idp_names is None:
            excluded = {"batch", "subject"}

            idp_names = [c for c in df.columns if c not in excluded]

        else:
            idp_names = list(idp_names)

        missing = [c for c in idp_names if c not in df.columns]

        if missing:
            raise ValueError("IDP columns not found: " + ", ".join(map(str, missing)))

        idp_matrix = (
            df[idp_names]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("data must be a 2D NumPy array.")

        idp_matrix = np.asarray(data, dtype=float)

        n_samples, n_features = idp_matrix.shape

        batch_arr = np.asarray(batch).astype(str)

        subject_arr = np.asarray(subject).astype(str)

        if idp_names is None:
            idp_names = [f"idp_{i+1}" for i in range(n_features)]

        else:
            idp_names = list(idp_names)

            if len(idp_names) != n_features:
                raise ValueError(
                    "idp_names must contain one name per column of data."
                )

    else:
        raise TypeError("data must be either a pandas DataFrame or a NumPy array.")

    if len(batch_arr) != idp_matrix.shape[0]:
        raise ValueError("Length of batch does not match number of observations.")

    if len(subject_arr) != idp_matrix.shape[0]:
        raise ValueError("Length of subject does not match number of observations.")

    return (
        idp_matrix,
        np.asarray(batch_arr).astype(str),
        np.asarray(subject_arr).astype(str),
        list(idp_names),
    )


# Prepare covariates
def _prepare_covariates(
    covariates: Optional[Dict[str, Sequence]],
    n_samples: int,
):
    """
    Prepare covariates for model fitting.

    Numeric covariates remain numeric.

    Categorical covariates are converted to strings and represented
    in the formula using C(Q('name')).
    """

    covariate_df = pd.DataFrame(index=np.arange(n_samples))

    if not covariates:
        return covariate_df, []

    formula_terms = []

    for name, values in covariates.items():
        values = np.asarray(values)

        if len(values) != n_samples:
            raise ValueError(
                f"Covariate '{name}' has "
                f"{len(values)} values but "
                f"{n_samples} observations are present."
            )

        safe_name = str(name)

        series = pd.Series(values, index=covariate_df.index)

        if series.isna().any():
            raise ValueError(f"Covariate '{name}' contains missing values.")

        if pd.api.types.is_numeric_dtype(series):
            covariate_df[safe_name] = pd.to_numeric(
                series,
                errors="raise",
            ).astype(float)

            formula_terms.append(safe_name)

        else:
            covariate_df[safe_name] = series.astype(str)

            formula_terms.append(f"C(Q('{safe_name}'))")

    return covariate_df, formula_terms


# Fit one mixed model
def _fit_and_adjust_feature(
    df: pd.DataFrame,
    feature: str,
    batch_col: str,
    subject_col: str,
    covariate_df: pd.DataFrame,
    covariate_terms: list,
    reml: bool = True,
):
    """
    Fit:

        IDP ~ batch + covariates + (1 | subject)

    Then remove:
        - covariate fixed effects
        - subject random intercept

    Batch effects are retained.

    Returns:
        adjusted values
        fitted model
    """

    work = pd.DataFrame(index=df.index)

    work["_y"] = pd.to_numeric(df[feature], errors="coerce")

    work["_batch"] = df[batch_col].astype(str)

    work["_subject"] = df[subject_col].astype(str)

    for col in covariate_df.columns:
        work[col] = covariate_df[col]

    # Complete cases
    valid = np.isfinite(work["_y"].to_numpy(dtype=float))

    for col in covariate_df.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            valid &= np.isfinite(
                pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
            )

        else:
            valid &= work[col].notna().to_numpy()

    work = work.loc[valid].copy()

    if len(work) < 3:
        return np.full(len(df), np.nan, dtype=float), None

    # Formula
    formula_terms = ["C(_batch)"]

    formula_terms.extend(covariate_terms)

    formula = "_y ~ " + " + ".join(formula_terms)

    try:
        model = smf.mixedlm(
            formula=formula,
            data=work,
            groups=work["_subject"],
            re_formula="1",
        )

        fitted = model.fit(reml=reml, method="lbfgs", disp=False)

    except Exception as exc:
        warnings.warn(f"Mixed model failed for {feature}: {exc}")

        return np.full(len(df), np.nan, dtype=float), None

    # Identify fixed effects
    params = fitted.fe_params

    exog = fitted.model.exog
    exog_names = fitted.model.exog_names

    # Covariate contribution
    # Do NOT remove: Intercept, Batch
    covariate_contribution = np.zeros(len(work), dtype=float)

    for j, name in enumerate(exog_names):
        if name == "Intercept":
            continue

        if "_batch" in name:
            continue

        covariate_contribution += exog[:, j] * params[name]

    # Random intercept
    random_effects = fitted.random_effects

    subject_random_intercept = np.zeros(len(work), dtype=float)

    for pos, subject_id in enumerate(work["_subject"]):
        effect = random_effects.get(subject_id)

        if effect is None:
            continue

        try:
            subject_random_intercept[pos] = float(np.asarray(effect).reshape(-1)[0])

        except Exception:
            subject_random_intercept[pos] = 0.0

    # Adjusted values
    adjusted_values = (
        work["_y"].to_numpy(dtype=float)
        - covariate_contribution
        - subject_random_intercept
    )

    output = np.full(len(df), np.nan, dtype=float)

    output[work.index.to_numpy()] = adjusted_values

    return output, fitted


# Main function
def plot_longitudinal_zscore_histograms(
    data: Union[pd.DataFrame, np.ndarray],
    batch: Optional[Sequence] = None,
    subject: Optional[Sequence] = None,
    idp_names: Optional[Sequence[str]] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
    reml: bool = True,
    bins: int = 80,
    xlim: tuple = (-8, 8),
    figsize: tuple = (16, 6),
    batch_order: Optional[Sequence[str]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    rep=None,
):
    """
    Plot robust MAD-standardised IDP distributions by batch for
    repeated-measures data.

    Model used for each IDP:

        IDP ~ Batch + covariates + (1 | Subject)

    The visualisation uses two representations:

        1. Raw robust z-scores.
        2. Covariate- and subject-adjusted robust z-scores.

    The adjustment removes:
        - covariate fixed-effect contributions
        - subject-specific random intercepts

    The batch effect is deliberately retained.

    Parameters
    ----------
    data
        Either:

        pandas.DataFrame
            containing IDPs and optionally "batch" and "subject".

        NumPy array
            shape (n_samples, n_features).

    batch
        Batch/site/scanner labels.

        If data is a DataFrame and this is None, a "batch"
        column is used.

    subject
        Subject identifiers.

        If data is a DataFrame and this is None, a "subject"
        column is used.

    idp_names
        IDP names when data is a NumPy array.

    covariates
        Dictionary of covariates.

        Example:

            {
                "age": age,
                "sex": sex,
            }

    reml
        Whether to fit the mixed models using REML.

    bins
        Number of histogram bins.

    xlim
        x-axis limits.

    figsize
        Figure size.

    batch_order
        Explicit batch order.

    savepath
        Optional PDF/PNG path.

    show
        Whether to display the figure.

    Returns
    -------
    fig
        Matplotlib figure.

    raw_z
        DataFrame of raw robust z-scores.

    adjusted_z
        DataFrame of covariate/subject-adjusted robust z-scores.

    models
        Dictionary of fitted mixed models by IDP.
    """

    # Prepare data
    (
        idp_matrix,
        batch_arr,
        subject_arr,
        idp_names,
    ) = _prepare_longitudinal_input(
        data=data,
        batch=batch,
        subject=subject,
        idp_names=idp_names,
    )

    n_samples = idp_matrix.shape[0]
    work_df = pd.DataFrame(idp_matrix, columns=idp_names)
    work_df["_batch"] = batch_arr
    work_df["_subject"] = subject_arr

    # Batch order
    if batch_order is None:
        batch_order = pd.Series(batch_arr).dropna().unique().tolist()
    else:
        batch_order = [str(x) for x in batch_order]

    # Raw robust z-scores
    raw_z = np.full((n_samples, len(idp_names)), np.nan, dtype=float)
    for j, feature in enumerate(idp_names):
        raw_z[:, j] = _robust_zscore(idp_matrix[:, j])
    raw_z = pd.DataFrame(raw_z, columns=idp_names)

    # Covariates
    covariate_df, covariate_terms = _prepare_covariates(
        covariates=covariates,
        n_samples=n_samples,
    )

    # Fit mixed models
    adjusted_values = np.full((n_samples, len(idp_names)), np.nan, dtype=float)
    models = {}
    for j, feature in enumerate(idp_names):
        adjusted, fitted = _fit_and_adjust_feature(
            df=work_df,
            feature=feature,
            batch_col="_batch",
            subject_col="_subject",
            covariate_df=covariate_df,
            covariate_terms=covariate_terms,
            reml=reml,
        )
        adjusted_values[:, j] = adjusted
        if fitted is not None:
            models[feature] = fitted

    # Adjusted robust z-scores
    adjusted_z = np.full(adjusted_values.shape, np.nan, dtype=float)

    for j in range(len(idp_names)):
        adjusted_z[:, j] = _robust_zscore(adjusted_values[:, j])

    adjusted_z = pd.DataFrame(adjusted_z, columns=idp_names)

    # Print model information
    covariate_formula = " + ".join(covariate_terms) if covariate_terms else "none"

    print("\n" + "=" * 72)
    print("LONGITUDINAL DISTRIBUTION MODEL")
    print("=" * 72)
    print("General model:")

    if covariate_terms:
        print("    IDP ~ Batch + " + covariate_formula + " + (1 | Subject)")
    else:
        print("    IDP ~ Batch + (1 | Subject)")

    print(f"REML: {reml}")
    print("\nAdjustment applied:")
    print("    - covariate fixed effects removed")
    print("    - subject random intercept removed")
    print("    - batch effects retained")
    print("\nFixed effects are estimated separately for each IDP.")
    print("Random effects: subject-specific random intercepts.")

    # Print an example full model
    if models:
        first_feature = next(iter(models))
        model = models[first_feature]
        print("\n" + "-" * 72)
        print(f"Example fitted model: {first_feature}")
        print("-" * 72)
        print(model.summary())

        # Random-effect variance
        try:
            random_var = float(np.asarray(model.cov_re)[0, 0])
            random_sd = np.sqrt(max(random_var, 0.0))
            print("\nSubject random-intercept variance:")
            print(f"    Variance = {random_var:.6g}")
            print(f"    SD       = {random_sd:.6g}")
        except Exception:
            pass

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _plot_panel(ax, score_df, title):
        for batch_label in batch_order:
            mask = batch_arr == batch_label
            values = score_df.loc[mask].to_numpy(dtype=float).ravel()
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.45,
                label=str(batch_label),
            )

            # PDF in corresponding histogram colour
            patches = ax.patches

            if len(patches) > 0:
                color = patches[-1].get_facecolor()
                mu = float(np.mean(values))
                sd = float(np.std(values))

                if np.isfinite(sd) and sd > 0:
                    x = np.linspace(xlim[0], xlim[1], 300)
                    y = norm.pdf(x, mu, sd)
                    ax.plot(x, y, color=color, linewidth=2)

        ax.set_xlim(xlim)
        ax.set_xlabel("Robust z-score")
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")
        ax.legend(title="Batch", fontsize=8)
        ax.grid(axis="y", alpha=0.2)

    # Raw
    _plot_panel(axes[0], raw_z, "Raw data")

    # Adjusted
    if covariates:
        adjusted_title = "Covariate- and subject-adjusted data"

    else:
        adjusted_title = "Subject-adjusted data"

    _plot_panel(axes[1], adjusted_z, adjusted_title)

    fig.suptitle(
        "Robust z-score distributions across batches",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()

    # Save
    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )

    # Show
    if show:
        plt.show()

    if rep is not None:
        rep.log_plot(fig, "Batch effects")

    return (
        fig,
        raw_z,
        adjusted_z,
        models,
    )