#%%


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
"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions ratio of variance ----------------------------------"""
"""----------------------------------------------------------------------------------------------------------------------------"""
def variance_ratio_plot(variance_ratios:  np.ndarray, pair_labels: list,
                         df: None = None,rep = None,show: bool = False,caption: Optional[str] = None,) -> None:
    """
    Plots the explained variance ratio for each principal component as a bar plot.

    Args:
        variance_ratios (Sequence[float]): A sequence of explained variance ratios for each principal component.
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

def levenes_plot(
    levenes_results: pd.DataFrame,
    *,
    rep = None,            # optional StatsReporter
    caption: Optional[str] = None,
    show: bool = False,
    pair_labels: Optional[list] = None
) -> plt.Figure:
    """
    Plots Levene's test results as a bar plot with significance markers.

    Args:
        levenes_results dict: pairlabels (batchwise), statistics, pvalues for each unique pair, statistic and pvalue are 1D arrays length of num_features"""
    figs = []
    return figs

"""----------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------- Plotting functions for PCA correlation results ----------------------------------"""
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
):
    """
    Generate multiple PCA diagnostic plots and return a list of (caption, fig).

    Improvements / behavior:
      - covariates may be a numpy array (2D), a pandas.DataFrame, or a structured numpy array.
      - If covariates has column names (DataFrame.columns or structured dtype.names), those names are used.
      - If covariates is a plain ndarray, variable_names (if provided) will be used as covariate names.
      - variable_names may optionally include 'batch' as the first element: ['batch', 'Age', 'Sex'].
      - If no covariate names are available, defaults "Covariate1", "Covariate2", ...


      K-means clustering of PCA points by batch and covariates to be added in future edit, additionally, 
      silhouette score calculation for batch also added. (Future work may add similar implementation for covariates if needed).

    Args:
        PrincipleComponents (np.ndarray): 2D array of PCA components (samples x components).
        batch (np.ndarray): 1D array of batch labels for each sample.
        covariates (Optional[Union[np.ndarray, pd.DataFrame]]): Optional covariate data.
        variable_names (Optional[List[str]]): Optional list of variable names for batch and covariates.
        PC_correlations (bool): If True, generate correlation heatmap.
        show (bool): If True, display plots immediately.
        
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
    # --- 3) Correlation heatmap if requested ---
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
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=combined_names, yticklabels=combined_names, ax=ax)
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
    UMAP_embedding=True):

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


    # Perform UMAP embedding if data is provided and not too large (to avoid long runtime and memory issues)
    if data is not None and len(data) < 10000:
        reducer = umap.UMAP(random_state=42)
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
                figs.append((f"UMAP and PCA embedding by {name}", fig_cov))

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
    if len(data) > 10000:
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
        results dict containing:
          - 'per_batch_variance': {batch_label: 1D-array length k of variances per PC}
          - 'per_batch_frac_var': {batch_label: 1D-array length k of fraction variance (per-batch)}
          - 'pcs_used': k
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
        results dict containing:
          - 'cov_matrices': {batch_label: k x k covariance matrix}
          - 'pairwise_frobenius': DataFrame-like dict or 2D np.array of pairwise norms
          - 'pairwise_frobenius_normalized': same but normalized (if normalize=True)
          - 'pcs_used': k
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
def KS_plot(ks_results: dict,
             feature_names: list = None,
               rep = None,            # optional StatsReporter
                 caption: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """
    Plot the output of the two sample KS test as ordered plots of the -log10 p-values for each feature.

    Overall, returns two plots
        - one plot showing the pairwise KS test results for each feature as a dot plot (ordered as -log10 p-value)
        - One plot showing the batch vs whole dataset (excluding that batch), again as a dot plot ordered by -log10 p-value.

    Args:
        ks_results (dict): Output from TwoSampleKSTest(...)
            ks_results: keys are tuples like (b, 'overall') or (b1, b2)
        - each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2
            'summary': {'prop_significant': float, 'mean_D': float}
    Returns:
        figs (list): List of (caption, fig) tuples for each plot generated.

    """
    # ---- Validation ---- Structure of dictionary has batch vs over all and batch vs batch as keys

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.pyplot import gca    
# plot batch vs overall on one plot and code with the legend:
    fig= plt.figure(figsize=(12, 6))
    figs = []
    ax = gca()

    for key in ks_results:
        if len(key) == 2 and key[1] == 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax.plot(indices, -np.log10(sorted_pvals + 1e-10), '*',label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Overall")
    plt.grid(True)
    plt.legend()
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')
    figs.append((caption or "KS Test: Batch vs Overall", fig))

    # Repeat for batch vs batch on next figure:
    fig2 = plt.figure(figsize=(12, 6))
    ax2 = gca()
    for key in ks_results:
        if len(key) == 2 and key[1] != 'overall':
            b = key[0]
            res = ks_results[key]
            p_values = res['p_value']
            if feature_names is not None and len(feature_names) != len(p_values):
                raise ValueError("feature_names length must match number of features in ks_results.")
            n_features = len(p_values)
            indices = np.arange(n_features)
            # sort by -log10 p-value
            sorted_indices = np.argsort(-np.log10(p_values + 1e-10))  # add small value to avoid log(0)
            sorted_pvals = p_values[sorted_indices]
            sorted_features = feature_names[sorted_indices] if feature_names is not None else sorted_indices
            ax2.plot(indices, -np.log10(sorted_pvals + 1e-10),'.', label=f'Batch {b} vs Overall')
    plt.xlabel("Features (ordered by -log10 p-value)")
    plt.ylabel("-log10 p-value")
    plt.title("KS Test: Batch vs Batch")
    plt.grid(True)
    # Add an line to the plot to indicate significant threshold at 0.05 and 0.01 (Bonferroni corrected and uncorrected)
    sig_threshold_05 = -np.log10(0.05 / n_features)
    sig_threshold_01 = -np.log10(0.01 / n_features)
    plt.axhline(y=sig_threshold_05, color='r', linestyle='-', label='Significance Threshold (0.05 Bonferroni)')
    plt.axhline(y=sig_threshold_01, color='g', linestyle='-', label='Significance Threshold (0.01 Bonferroni)')

    plt.legend()
    figs.append((caption or "KS Test: Batch vs Batch", fig2))

    # Check if show is given, if so, display the plots
    for caption_i, fig in figs:
        if rep is not None:
            rep.log_plot(fig, caption_i)
            plt.close(fig)
        else: figs.append((caption_i, fig))
    if show:
        for _, fig in figs:
            fig.show()
    return rep if rep is not None else figs

##########################################################################
# Plotting for longitudinal metrics
###########################################################################
@rep_plot_wrapper
def plot_SubjectOrder(df,
                      idp_col='IDP',
                      time_a_col='TimeA',
                      time_b_col='TimeB',
                      rho_col='SpearmanRho',
                      p_col='pValue',
                      times_order=None,
                      significance=0.05,
                      ncols=2,
                      figsize_per_plot=(4,4),
                      cmap="icefire",
                      fmt=".2f",
                      center=0,
                      vmax_abs=None,
                      limit_idps=None,
                      sample_method='first',   # 'first' or 'random'
                      random_state=None,
                      rep=None,
                      show: bool = False,
                      combine_method: str = "stouffer",   # 'stouffer' or 'fisher'
                      p_correction: str = "fdr_bh"        # 'fdr_bh', 'bonferroni', or None
                      ):
    """
    Extended version of your function that *combines* p-values across IDPs (for each time-pair)
    and across time-pairs (for each IDP) using either Stouffer (signed) or Fisher, and optionally
    applies multiple-testing correction (BH or bonferroni).

    Notes:
      - combine_method='stouffer' uses the sign from rho (mean rho across IDPs for that cell)
        to create signed z-scores from two-sided p-values.
      - combine_method='fisher' uses scipy.stats.combine_pvalues(method='fisher') and ignores sign.
      - p_correction operates separately for the time×time summary matrix and for per-IDP combined p's.
      - For best statistical rigor with permutation-based tests, combining at the permutation-level
        (i.e. combining test stats per permutation and building an empirical null) is preferable.
    """
    import math
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- additional imports for combination ---
    try:
        from scipy.stats import combine_pvalues, norm
        from scipy.stats import chi2
    except Exception as e:
        raise ImportError("This function requires scipy. Please install scipy (pip install scipy).") from e

    # --- simple BH implementation and Bonferroni ---
    def bh_adjust(pvals):
        """Benjamini-Hochberg FDR correction. Returns adjusted p-values (same shape as input)."""
        p = np.asarray(pvals, dtype=float)
        flat = p.flatten()
        nanmask = np.isnan(flat)
        idx = np.where(~nanmask)[0]
        if idx.size == 0:
            return p  # nothing to do
        pv = flat[~nanmask]
        order = np.argsort(pv)
        ranked = np.empty_like(order)
        ranked[order] = np.arange(pv.size) + 1  # ranks 1..m
        m = pv.size
        adj = np.empty_like(pv)
        # BH adjusted p-values (step-up)
        adj_vals = pv * m / ranked
        # enforce monotonicity (cumulative minimum from largest to smallest)
        adj_vals_sorted = np.empty_like(adj_vals)
        adj_vals_sorted[order] = adj_vals
        # cumulative minimum from the end
        cummin = np.minimum.accumulate(adj_vals_sorted[::-1])[::-1]
        adj[order] = np.minimum(cummin, 1.0)
        flat_adj = flat.copy()
        flat_adj[~nanmask] = adj
        return flat_adj.reshape(p.shape)

    def bonferroni_adjust(pvals):
        p = np.asarray(pvals, dtype=float)
        flat = p.flatten()
        nanmask = np.isnan(flat)
        idx = np.where(~nanmask)[0]
        if idx.size == 0:
            return p
        pv = flat[~nanmask]
        m = pv.size
        adj = np.minimum(pv * m, 1.0)
        flat_adj = flat.copy()
        flat_adj[~nanmask] = adj
        return flat_adj.reshape(p.shape)

    def apply_correction(p_matrix, method):
        if method is None:
            return p_matrix
        if method == 'fdr_bh':
            return bh_adjust(p_matrix)
        elif method == 'bonferroni':
            return bonferroni_adjust(p_matrix)
        else:
            raise ValueError("p_correction must be 'fdr_bh', 'bonferroni', or None")

    # ---------- Validate and list IDPs ----------
    all_idps = sorted(df[idp_col].unique())
    if len(all_idps) == 0:
        raise ValueError("No IDPs found in dataframe.")

    # choose idps_to_plot according to limit_idps and sample_method
    if limit_idps is None:
        idps_to_plot = all_idps.copy()
    else:
        if not (isinstance(limit_idps, int) and limit_idps >= 1):
            raise ValueError("limit_idps must be None or a positive integer.")
        limit = min(limit_idps, len(all_idps))
        if sample_method == 'first':
            idps_to_plot = all_idps[:limit]
        elif sample_method == 'random':
            rng = random.Random(random_state)
            idps_to_plot = rng.sample(all_idps, limit)
        else:
            raise ValueError("sample_method must be 'first' or 'random'.")

    n_idp_plot = len(idps_to_plot)

    # ---------- Determine time ordering ----------
    if times_order is None:
        times = sorted(set(df[time_a_col].unique()) | set(df[time_b_col].unique()))
    else:
        times = list(times_order)
    n_times = len(times)
    if n_times == 0:
        raise ValueError("No time points found.")

    # ---------- Build matrices for ALL IDPs (used for summaries) ----------
    rho_mats_all = {}
    p_mats_all = {}
    all_rho_values = []
    for idp in all_idps:
        sub = df[df[idp_col] == idp].copy()
        rho = sub.pivot(index=time_a_col, columns=time_b_col, values=rho_col)
        pmat = sub.pivot(index=time_a_col, columns=time_b_col, values=p_col)
        rho = rho.reindex(index=times, columns=times)
        pmat = pmat.reindex(index=times, columns=times)
        # symmetrize if needed (use transpose to fill missing)
        rho = rho.combine_first(rho.T)
        pmat = pmat.combine_first(pmat.T)
        np.fill_diagonal(rho.values, np.nan)
        np.fill_diagonal(pmat.values, np.nan)
        rho_mats_all[idp] = rho.astype(float)
        p_mats_all[idp] = pmat.astype(float)
        all_rho_values.extend(rho.values.flatten()[~np.isnan(rho.values.flatten())])

    if len(all_rho_values) == 0:
        raise ValueError("No numeric SpearmanRho values found.")

    # ---------- Color scale ----------
    if vmax_abs is None:
        vmax_abs = max(abs(np.nanmin(all_rho_values)), abs(np.nanmax(all_rho_values)))
    vmin, vmax = -vmax_abs, vmax_abs

    # ---------- Summary calculations (use ALL IDPs) ----------
    stacked = np.stack([rho_mats_all[idp].values for idp in all_idps], axis=0)    # shape (n_idps, n_times, n_times)
    stacked_p = np.stack([p_mats_all[idp].values for idp in all_idps], axis=0)

    # Helper: ensure p in (0,1]; replace exact zeros by tiny value to avoid -inf/log issues
    tiny = 1e-300
    sp = stacked_p.copy()
    sp[np.isnan(sp)] = np.nan  # leave nans
    sp[sp == 0] = tiny

    # Combined p-values per time-pair across IDPs
    combined_p_matrix = np.full((n_times, n_times), np.nan)
    combined_rho_matrix = np.nanmean(stacked, axis=0)  # keep mean rho (for sign in Stouffer)
    if combine_method.lower() == 'fisher':
        # use scipy combine_pvalues for Fisher (ignores sign)
        for i in range(n_times):
            for j in range(n_times):
                pv = sp[:, i, j]
                pv = pv[~np.isnan(pv)]
                if pv.size == 0:
                    combined_p_matrix[i, j] = np.nan
                else:
                    # combine_pvalues returns (stat, p)
                    _, p_comb = combine_pvalues(pv, method='fisher')
                    combined_p_matrix[i, j] = p_comb
    elif combine_method.lower() == 'stouffer':
        # signed Stouffer: convert two-sided p to z, use sign of mean rho across IDPs for that cell
        # z_i = sign_i * norm.ppf(1 - p_i/2)
        for i in range(n_times):
            for j in range(n_times):
                pv = sp[:, i, j]
                pv = pv[~np.isnan(pv)]
                if pv.size == 0:
                    combined_p_matrix[i, j] = np.nan
                else:
                    # determine sign from mean rho across IDPs for that cell
                    rhos = stacked[:, i, j]
                    rhos_nonan = rhos[~np.isnan(rhos)]
                    sign_cell = 0
                    if rhos_nonan.size > 0:
                        mean_rho = np.nanmean(rhos_nonan)
                        sign_cell = np.sign(mean_rho) if not np.isnan(mean_rho) else 1.0
                        if sign_cell == 0:
                            sign_cell = 1.0
                    else:
                        sign_cell = 1.0
                    # convert p to z's, protect from p==0 or p==1
                    p_clip = np.clip(pv, tiny, 1 - 1e-16)
                    zs = norm.ppf(1.0 - p_clip / 2.0)  # two-sided -> two-tailed z magnitude
                    # apply sign
                    signed_zs = sign_cell * zs
                    # combine (equal weights)
                    z_comb = np.sum(signed_zs) / math.sqrt(zs.size)
                    # two-sided combined p:
                    p_comb = 2.0 * (1.0 - norm.cdf(abs(z_comb)))
                    combined_p_matrix[i, j] = float(np.clip(p_comb, tiny, 1.0))
    else:
        raise ValueError("combine_method must be 'stouffer' or 'fisher'")

    # Optional multiple-comparison correction across time×time combined p-matrix
    combined_p_matrix_adj = apply_correction(combined_p_matrix, p_correction)

    # ---------- Per-IDP: combine its off-diagonal p-values into a single p -----
    idp_combined_ps = []
    idp_mean_rhos = []
    for idp in all_idps:
        pmat = p_mats_all[idp].values
        rho = rho_mats_all[idp].values
        mask = ~np.eye(n_times, dtype=bool)  # exclude diagonal
        pv = pmat[mask]
        pv = pv[~np.isnan(pv)]
        rhos = rho[mask]
        rhos = rhos[~np.isnan(rhos)]
        if pv.size == 0:
            idp_combined_ps.append(np.nan)
            idp_mean_rhos.append(np.nan)
            continue
        if combine_method.lower() == 'fisher':
            # fisher combine
            _, p_comb = combine_pvalues(np.clip(pv, tiny, 1.0), method='fisher')
            idp_combined_ps.append(float(p_comb))
        else:  # stouffer signed
            mean_rho = np.nanmean(rho[mask])
            idp_mean_rhos.append(mean_rho)
            sign_cell = np.sign(mean_rho) if not np.isnan(mean_rho) else 1.0
            if sign_cell == 0:
                sign_cell = 1.0
            p_clip = np.clip(pv, tiny, 1.0 - 1e-16)
            zs = norm.ppf(1.0 - p_clip / 2.0)
            signed_zs = sign_cell * zs
            z_comb = np.sum(signed_zs) / math.sqrt(zs.size)
            p_comb = 2.0 * (1.0 - norm.cdf(abs(z_comb)))
            idp_combined_ps.append(float(np.clip(p_comb, tiny, 1.0)))
    # if mean rhos list wasn't filled (Fisher branch), compute idp_mean_rhos now
    if combine_method.lower() == 'fisher':
        idp_mean_rhos = []
        for idp in all_idps:
            rho = rho_mats_all[idp].values
            mask = ~np.eye(n_times, dtype=bool)
            idp_mean_rhos.append(np.nanmean(rho[mask]) if ~np.all(np.isnan(rho[mask])) else np.nan)
    else:
        # ensure lengths correct
        if len(idp_mean_rhos) != len(all_idps):
            # fallback compute
            idp_mean_rhos = []
            for idp in all_idps:
                rho = rho_mats_all[idp].values
                mask = ~np.eye(n_times, dtype=bool)
                idp_mean_rhos.append(np.nanmean(rho[mask]) if ~np.all(np.isnan(rho[mask])) else np.nan)

    idp_combined_ps = np.array(idp_combined_ps, dtype=float)
    idp_combined_ps_adj = idp_combined_ps.copy()
    if p_correction is not None:
        # apply correction across IDPs (treating them as a family)
        idp_combined_ps_adj = apply_correction(idp_combined_ps.reshape(-1, 1), p_correction).reshape(-1)

    # ---------- Prepare mean_rho_matrix (mean across IDPs) ----------
    mean_rho_matrix = np.nanmean(stacked, axis=0)
    np.fill_diagonal(mean_rho_matrix, np.nan)

    # ---------- Figure layout ----------
    nrows = math.ceil(n_idp_plot / ncols) if n_idp_plot > 0 else 0
    fig_w = figsize_per_plot[0] * ncols
    fig_h = max(1, nrows) * figsize_per_plot[1] + 1.6 * figsize_per_plot[1]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(max(1, nrows) + 2, ncols,
                          height_ratios=[1]*max(1, nrows) + [0.9, 0.9],
                          hspace=0.35, wspace=0.3)

    # ---------- Plot individual IDP heatmaps (ONLY idps_to_plot) ----------
    for idx, idp in enumerate(idps_to_plot):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        rho = rho_mats_all[idp]
        pmat = p_mats_all[idp]
        annot = np.full(rho.shape, "", dtype=object)
        for i in range(n_times):
            for j in range(n_times):
                val = rho.iat[i, j]
                pval = pmat.iat[i, j]
                if np.isnan(val):
                    annot[i, j] = ""
                else:
                    star = "*" if (not pd.isna(pval) and pval < significance) else ""
                    annot[i, j] = f"{val:{fmt}}{star}"
        sns.heatmap(rho,
                    ax=ax,
                    annot=annot,
                    fmt="",
                    cmap=cmap,
                    center=center,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.35,
                    linecolor="gray",
                    cbar=False,
                    square=False)
        ax.set_title(idp, fontsize=9)
        ax.set_xlabel("")   # explicit: no xlabel
        ax.set_ylabel("")   # explicit: no ylabel
        ax.set_xticklabels(times, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(times, rotation=0, fontsize=7)

    # hide unused axes inside the idp grid
    total_idp_slots = max(1, nrows) * ncols
    if n_idp_plot < total_idp_slots:
        for k in range(n_idp_plot, total_idp_slots):
            r = k // ncols
            c = k % ncols
            ax = fig.add_subplot(gs[r, c])
            ax.axis('off')

    # ---------- Summary 1: mean across ALL IDPs (time x time) ----------
    row_for_summaries = max(0, nrows)
    ax_mean_timepair = fig.add_subplot(gs[row_for_summaries, :])
    annot_mean = np.full(mean_rho_matrix.shape, "", dtype=object)
    # use combined_p_matrix_adj (corrected) to mark significance
    for i in range(n_times):
        for j in range(n_times):
            val = mean_rho_matrix[i, j]
            pval = combined_p_matrix_adj[i, j]
            if np.isnan(val):
                annot_mean[i, j] = ""
            else:
                star = "*" if (not np.isnan(pval) and pval < significance) else ""
                annot_mean[i, j] = f"{val:{fmt}}{star}"
    sns.heatmap(mean_rho_matrix,
                ax=ax_mean_timepair,
                annot=annot_mean,
                fmt="",
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.35,
                linecolor="gray",
                cbar=False,
                square=False)
    ax_mean_timepair.set_title(f"Mean across  {len(all_idps)} IDPs (per time-pair) — combined p: {combine_method}, correction: {p_correction}", fontsize=10)
    ax_mean_timepair.set_xlabel("")
    ax_mean_timepair.set_ylabel("")
    ax_mean_timepair.set_xticklabels(times, rotation=45, ha='right', fontsize=7)
    ax_mean_timepair.set_yticklabels(times, rotation=0, fontsize=7)

    # ---------- Summary 2: per-IDP mean across time-pairs (ALL IDPs) ----------
    ax_idp_mean = fig.add_subplot(gs[row_for_summaries+1, :])
    idp_mean_matrix = np.array(idp_mean_rhos).reshape(-1, 1)
    annot_idp = np.array([f"{v:{fmt}}{'*' if (not np.isnan(p) and p < significance) else ''}"
                          for v, p in zip(idp_mean_rhos, idp_combined_ps_adj)]).reshape(-1, 1)
    sns.heatmap(idp_mean_matrix,
                ax=ax_idp_mean,
                annot=annot_idp,
                fmt="",
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.35,
                linecolor="gray",
                cbar=False,
                yticklabels=all_idps,
                xticklabels=["MeanAcrossTimePairs"],
                square=False)
    ax_idp_mean.set_title(f"Per-IDP mean across time-pairs ({len(all_idps)} IDPs) — combined p: {combine_method}, correction: {p_correction}", fontsize=8)
    ax_idp_mean.set_xlabel("")
    ax_idp_mean.set_ylabel("")
    ax_idp_mean.set_xticklabels([""], rotation=0)

    # ---------- Shared colorbar ----------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Spearman Rho')

    plt.suptitle(f"Subject order consistency summaries computed from {len(all_idps)} IDPs\n('*' indicates p < {significance} from permutation testing)", fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.90, 0.96])

    if rep is not None:
        rep.log_plot(fig, "Subject order consistency")
        plt.close(fig)
        return None, None
    if show:
        plt.show()
    return fig

### plot within subject variability
import seaborn as sns
def build_style_registry(subjects,
                         idps,
                         subject_palette="tab10",
                         idp_palette="Set2",
                         subject_markers=None,
                         idp_markers=None):
    """
    Returns:
      subject_style: dict {subject: (color, marker)}
      idp_style: dict {idp: (color, marker)}
    """

    if subject_markers is None:
        subject_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h']
    if idp_markers is None:
        idp_markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '*']

    subj_colors = sns.color_palette(subject_palette, len(subjects))
    idp_colors = sns.color_palette(idp_palette, len(idps))

    subject_style = {
        s: (subj_colors[i % len(subj_colors)],
            subject_markers[i % len(subject_markers)])
        for i, s in enumerate(subjects)
    }

    idp_style = {
        i_name: (idp_colors[i % len(idp_colors)],
                 idp_markers[i % len(idp_markers)])
        for i, i_name in enumerate(idps)
    }

    return subject_style, idp_style


@rep_plot_wrapper
def plot_WithinSubjVar(
    df,
    subject_col='subject',
    idp_cols=None,
    subject_style=None,
    idp_style=None,
    limit_subjects=10,
    limit_idps_for_legend=10,
    figsize=(14,6),
    point_size=60,
    jitter=0.08,
    savepath=None,
    rep=None,
    show: bool = False):
    if idp_cols is None:
        idp_cols = [c for c in df.columns if c != subject_col]
    subjects = df['subject'].tolist()
    idps = [c for c in df.columns if c != 'subject']
    subject_style, idp_style = build_style_registry(
        subjects,
        idps,
        subject_palette="tab10",
        idp_palette="Set2"
)

    subjects = df[subject_col].unique().tolist()
    idps = list(idp_cols)

    if subject_style is None or idp_style is None:
        raise ValueError("Provide subject_style and idp_style from build_style_registry()")

    n_subjects = len(subjects)
    n_idps = len(idps)

    long = df.melt(id_vars=subject_col, value_vars=idps,
                   var_name='IDP', value_name='value')

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # -------- A: Per-IDP boxplots + subject points --------
    sns.boxplot(x='IDP', y='value', data=long, ax=axA, boxprops={'alpha':0.6})
    axA.set_title("Per-IDP distribution (subjects)")
    axA.set_xlabel("")
    axA.set_ylabel("WSV (%)")

    show_subject_legend = n_subjects <= limit_subjects
    idp_x = {idp: i for i, idp in enumerate(idps)}

    for subj in subjects:
        row = df[df[subject_col] == subj]
        xs, ys = [], []
        for idp in idps:
            xs.append(idp_x[idp] + np.random.uniform(-jitter, jitter))
            ys.append(row[idp].values[0])

        if show_subject_legend:
            color, marker = subject_style[subj]
            axA.scatter(xs, ys, s=point_size, marker=marker,
                        color=color, edgecolor='k', label=subj, zorder=3)
        else:
            axA.scatter(xs, ys, s=point_size*0.7, marker='o',
                        color='gray', edgecolor='k', alpha=0.8)

    if show_subject_legend:
        axA.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc='best')

    # -------- B: Mean across subjects per IDP --------
    idp_means = df[idps].mean(axis=0)
    sns.boxplot(x=idp_means.values, ax=axB, orient='h', boxprops={'alpha':0.6})
    axB.set_title("Per-IDP mean (across subjects)")
    axB.set_yticks([])

    show_idp_legend = n_idps <= limit_idps_for_legend
    for idp, mean_val in idp_means.items():
        if show_idp_legend:
            color, marker = idp_style[idp]
            axB.scatter(mean_val, 0, s=90, marker=marker,
                        color=color, edgecolor='k', label=idp)
        else:
            axB.scatter(mean_val, 0, s=70, marker='o',
                        color='gray', edgecolor='k')

    if show_idp_legend:
        axB.legend(title="IDP", bbox_to_anchor=(1.02, 1), loc='best')

    # -------- C: Mean across IDPs per subject --------
    subj_means = df.set_index(subject_col)[idps].mean(axis=1)
    sns.boxplot(x=subj_means.values, ax=axC, orient='h', boxprops={'alpha':0.6})
    axC.set_title("Per-subject mean (across IDPs)")
    axC.set_yticks([])

    show_subject_legend = n_subjects <= limit_subjects
    for subj, mean_val in subj_means.items():
        if show_subject_legend:
            color, marker = subject_style[subj]
            axC.scatter(mean_val, 0, s=90, marker=marker,
                        color=color, edgecolor='k', label=subj)
        else:
            axC.scatter(mean_val, 0, s=70, marker='o',
                        color='gray', edgecolor='k')

    # if show_subject_legend:
    #     axC.legend(title="Subject", bbox_to_anchor=(1.02, 1), loc='best')
    plt.suptitle("Within subject variability", fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if rep is not None:
        rep.log_plot(fig, "Within subject variability")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig

### Multivariate Site difference: MD
@rep_plot_wrapper
def plot_MultivariateBatchDifference(df,
                         batch_col='batch',
                         value_col='mdval',
                         avg_label='average_batch',
                         figsize=(8,6),
                         sort_by_value=True,
                         sort_rest_desc=True,
                         value_format="{:.1f}",
                         savepath=None,
                         rep=None,
                         show: bool = False):
    """
    Horizontal bar chart with:
      - average_batch always at the top
      - remaining batches optionally sorted by mdval
      - value labels rounded to 1 decimal
    """
    avg_color='tab:red'
    bar_color='tab:blue'
    if batch_col not in df.columns or value_col not in df.columns:
        raise ValueError("DataFrame must contain specified batch and value columns.")

    df_plot = df[[batch_col, value_col]].copy()

    # --- split average vs rest ---
    avg_df = df_plot[df_plot[batch_col] == avg_label]
    rest_df = df_plot[df_plot[batch_col] != avg_label]

    if sort_by_value:
        rest_df = rest_df.sort_values(value_col, ascending=not sort_rest_desc)

    # --- recombine: average always first ---
    plot_df = pd.concat([avg_df, rest_df], ignore_index=True)

    batches = plot_df[batch_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).tolist()
    y_pos = np.arange(len(batches))

    fig, ax = plt.subplots(figsize=figsize)

    colors = [avg_color if b == avg_label else bar_color for b in batches]
    bars = ax.barh(y_pos, values, color=colors, edgecolor='k', alpha=0.9)

    # --- numeric labels ---
    x_offset = max(values) * 0.01
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + x_offset,
                bar.get_y() + bar.get_height()/2,
                value_format.format(val),
                va='center', ha='left',
                fontsize=9, weight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(batches, fontsize=9)
    ax.invert_yaxis()  # average stays visually on top
    ax.set_xlabel(value_col)
    ax.set_ylabel(batch_col)
    ax.set_title(f"{value_col} per {batch_col}")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    #plt.show()
    
    if rep is not None:
        rep.log_plot(fig, "Multivariate batch differences (with overall distribution) using Mahalanobis distances")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig

### mixed effects models plots
def _build_default_idp_style(idps, palette="Set2", markers=None):
    if markers is None:
        markers = ['o','s','^','D','P','X','v','<','>','*','h','H']
    colors = sns.color_palette(palette, max(2, len(idps)))
    idp_style = {idp: (colors[i % len(colors)], markers[i % len(markers)]) for i, idp in enumerate(idps)}
    return idp_style

@rep_plot_wrapper
def plot_MixedEffectsPart1(df,
                          idp_col='IDP',
                          metrics=None,
                          plot_type='bar',           # 'bar' or 'box'
                          idp_style=None,           # dict {idp: (color, marker)}
                          limit_idps=10,            # for legend in box mode
                          figsize=(14,4),
                          value_format_float="{:.1f}",
                          value_format_int="{:d}",
                          seed=None,
                          savepath=None,
                          rep=None,
                          show: bool=False,
                          # new args
                          display: str = "subplots",   # 'subplots' (default), 'separate', or 'single'
                          metric: str = None           # required when display == 'single'
                          ):
    """
    Plots one figure per metric (subplots) OR separate figures per metric.

    - display='subplots' : original behaviour, 1 fig with n_metrics subplots (returns Figure).
    - display='separate' : create one Figure per metric, return dict {metric: Figure}.
    - display='single'   : create a single Figure for the metric named in `metric`, return Figure.

    Defaults: metrics excludes 'anova_batches' by design (use AdditiveEffect_long for omnibus).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if display not in ("subplots", "separate", "single"):
        raise ValueError("display must be 'subplots', 'separate' or 'single'")

    # sensible default metrics (remove anova_batches)
    if metrics is None:
        metrics = ['n_is_batchSig', 'ICC', 'WCV']  # default excludes 'anova_batches'

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metrics in df: {missing}")

    idps = list(df[idp_col].astype(str).values)
    n_idps = len(idps)
    if idp_style is None:
        idp_style = _build_default_idp_style(idps)

    rng = np.random.RandomState(seed) if seed is not None else np.random

    # helper to draw one metric into an axis
    def _draw_metric_ax(ax, metric_name):
        vals = pd.to_numeric(df[metric_name], errors='coerce').values

        if plot_type == 'bar':
            x_pos = np.arange(n_idps)
            colors = [idp_style[idp][0] for idp in idps]
            bars = ax.bar(x_pos, vals, color=colors, edgecolor='k', alpha=0.9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(idps, rotation=45, ha='right', fontsize=9)
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)

            # numeric labels above bars; compute margin
            if np.all(np.isnan(vals)) or np.nanmax(np.abs(vals)) == 0:
                y_top = 1.0
            else:
                y_top = np.nanmax(vals)
            margin = 0.12 * (y_top if y_top != 0 else 1.0)
            cur_ylim = ax.get_ylim()
            ax.set_ylim(cur_ylim[0], cur_ylim[1] + margin)

            for bar, v in zip(bars, vals):
                if pd.isna(v):
                    label = ""
                else:
                    if float(v).is_integer():
                        label = value_format_int.format(int(round(v)))
                    else:
                        label = value_format_float.format(float(v))
                x = bar.get_x() + bar.get_width()/2
                y = bar.get_height()
                ax.text(x, y + 0.01*(y_top if y_top!=0 else 1.0), label,
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

        elif plot_type == 'box':
            metric_vals = pd.Series(vals, index=idps)
            clean_vals = metric_vals.dropna().values
            if len(clean_vals) == 0:
                ax.text(0.5,0.5,"No data", ha='center')
                return
            bp = ax.boxplot(clean_vals, vert=True, widths=0.6, patch_artist=True)
            ax.set_xticks([])
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)

            show_idp_legend = (n_idps <= limit_idps)
            legend_handles = []
            legend_labels = []
            for i, idp in enumerate(idps):
                v = metric_vals.loc[idp]
                if pd.isna(v):
                    continue
                jitter_x = rng.uniform(-0.06, 0.06)
                x = 1.0 + jitter_x
                if show_idp_legend:
                    color, marker = idp_style[idp]
                    sc = ax.scatter(x, v, marker=marker, color=color, edgecolor='k', s=80, zorder=5)
                    if idp not in legend_labels:
                        legend_handles.append(sc)
                        legend_labels.append(idp)
                else:
                    ax.scatter(x, v, marker='o', color='gray', edgecolor='k', s=40, zorder=4)

            # autoscale y limits with margin
            vmin = np.nanmin(clean_vals)
            vmax = np.nanmax(clean_vals)
            rngv = vmax - vmin if vmax > vmin else max(abs(vmax), 1.0)
            ax.set_ylim(vmin - 0.06*rngv, vmax + 0.12*rngv)

            # If legend desired and small number of idps, add it to the bottom of this axis
            if show_idp_legend and len(legend_handles) > 0:
                # place legend underneath this axis (tight placement)
                ax_legend = ax.figure.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.12,
                                                ax.get_position().width, 0.08])
                ax_legend.axis('off')
                ax_legend.legend(legend_handles, legend_labels, ncol=min(6, len(legend_labels)),
                                 frameon=False, loc='center')

        ax.grid(axis='y', linestyle='--', alpha=0.25)

    # --- dispatch display modes ---
    if display == "subplots":
        n_metrics = len(metrics)
        # ensure at least one row/col layout works even for 1 metric
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for ax_idx, metric in enumerate(metrics):
            _draw_metric_ax(axes[ax_idx], metric)
        plt.suptitle(f"Number of pairs of batches significant ({plot_type} plot)", fontsize=13)
        plt.tight_layout(rect=[0,0,0.95,0.95])

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')

        if rep is not None:
            rep.log_plot(fig, "Results from the mixed effects models")
            plt.close(fig)
            return None, None
        if show:
            plt.show()
        return fig

    elif display == "single":
        if metric is None:
            raise ValueError("When display=='single' you must provide a metric name via `metric` argument.")
        if metric not in metrics:
            raise ValueError(f"Requested metric '{metric}' not present in metrics list or DataFrame.")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _draw_metric_ax(ax, metric)
        plt.suptitle(f"IDP metric: {metric} ({plot_type})", fontsize=13)
        plt.tight_layout(rect=[0,0,0.95,0.95])

        if savepath:
            # if savepath is provided and display is 'single', save with metric name suffix
            import os
            base, ext = os.path.splitext(savepath)
            spath = f"{base}_{metric}{ext or '.png'}"
            plt.savefig(spath, dpi=300, bbox_inches='tight')

        if rep is not None:
            rep.log_plot(fig, f"Mixed effects metric {metric}")
            plt.close(fig)
            return None, None
        if show:
            plt.show()
        return fig

    else:  # display == "separate"
        figs = {}
        for metric in metrics:
            fig_i, ax_i = plt.subplots(1, 1, figsize=figsize)
            _draw_metric_ax(ax_i, metric)
            plt.suptitle(f"IDP metric: {metric} ({plot_type})", fontsize=13)
            plt.tight_layout(rect=[0,0,0.95,0.95])
            figs[metric] = fig_i

            if savepath:
                import os
                base, ext = os.path.splitext(savepath)
                spath = f"{base}_{metric}{ext or '.png'}"
                fig_i.savefig(spath, dpi=300, bbox_inches='tight')

            if rep is not None:
                rep.log_plot(fig_i, f"Mixed effects metric {metric}")
                plt.close(fig_i)

        # If rep was set we already logged and closed each fig; return None, None consistent with rep
        if rep is not None:
            return None, None

        # If show requested, show all (note: may block depending on environment)
        if show:
            for fig_i in figs.values():
                fig_i.show()

        # Return dict of figs, caller can pick whichever one they want
        return figs


@rep_plot_wrapper
def plot_MixedEffectsPart2(df,
                                         idp_col='IDP',
                                         fix_eff=('age','sex'),
                                         p_thr=0.05,
                                         effect_style=None,    # dict eff -> (color, marker)
                                         idp_order=None,
                                         figsize=(10, 4),
                                         marker_size=80,
                                         cap_width=0.03,
                                         linewidth=2.0,
                                         xtick_rotation=45,
                                         highlight_color='red',
                                         title=None,
                                         savepath=None,
                                         rep=None,
                                         show: bool=False):
    """
    Plot each fixed-effect on its own subplot: IDPs on x-axis, est as marker, CI as vertical line.
    - df: dataframe containing rows with IDP and columns like '{eff}_est','{eff}_pval','{eff}_ciL','{eff}_ciU'
    - fix_eff: iterable of effect names (strings)
    - p_thr: p-value threshold to consider an estimate "significant" (filled highlight)
    - effect_style: optional dict mapping effect -> (color, marker); missing effects get default styling
    - idp_order: list of IDP names in desired order; if None, order is df[idp_col] appearance
    - highlight_color: color used to fill markers when p < p_thr
    Returns: fig, axes (axes is a list of axes objects matching fix_eff order)
    """
    import warnings

    effs = list(fix_eff)
    # idp ordering
    if idp_order is None:
        idps = list(df[idp_col].astype(str).values)
    else:
        missing_idps = [i for i in idp_order if i not in df[idp_col].values]
        if missing_idps:
            raise ValueError(f"idp_order contains unknown IDPs: {missing_idps}")
        idps = list(idp_order)
    n_idps = len(idps)
    if n_idps == 0:
        raise ValueError("No IDPs provided/available.")

    # build default styles for effects if needed
    if effect_style is None:
        palette = sns.color_palette("tab10", max(3, len(effs)))
        markers = ['o','s','D','^','v','P','X','*','h','<','>']
        effect_style = {eff: (palette[i % len(palette)], markers[i % len(markers)]) for i, eff in enumerate(effs)}
    else:
        # fill missing effects with defaults
        palette = sns.color_palette("tab10", max(3, len(effs)))
        markers = ['o','s','D','^','v','P','X','*','h','<','>']
        for i, eff in enumerate(effs):
            if eff not in effect_style:
                effect_style[eff] = (palette[i % len(palette)], markers[i % len(markers)])

    n_eff = len(effs)
    # layout: try a single row; if many effects stack into rows of up to 3
    ncols = min(3, n_eff)
    nrows = int(np.ceil(n_eff / ncols))
    fig, axes_grid = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows),
                                  squeeze=False)
    axes = axes_grid.flatten()

    legend_handles = []
    legend_labels = []

    for idx, eff in enumerate(effs):
        ax = axes[idx]
        est_col = f"{eff}_est"
        p_col = f"{eff}_pval"
        ciL_col = f"{eff}_ciL"
        ciU_col = f"{eff}_ciU"

        missing_cols = [c for c in (est_col, p_col, ciL_col, ciU_col) if c not in df.columns]
        if missing_cols:
            warnings.warn(f"Skipping effect '{eff}' — missing columns: {missing_cols}")
            ax.set_visible(False)
            continue

        color, marker = effect_style[eff]
        # gather arrays in idps order
        ests = []
        pvals = []
        ciLs = []
        ciUs = []
        for idp in idps:
            row = df[df[idp_col] == idp]
            if row.shape[0] == 0:
                ests.append(np.nan); pvals.append(np.nan); ciLs.append(np.nan); ciUs.append(np.nan)
            else:
                r0 = row.iloc[0]
                ests.append(pd.to_numeric(r0.get(est_col, np.nan), errors='coerce'))
                pvals.append(pd.to_numeric(r0.get(p_col, np.nan), errors='coerce'))
                ciLs.append(pd.to_numeric(r0.get(ciL_col, np.nan), errors='coerce'))
                ciUs.append(pd.to_numeric(r0.get(ciU_col, np.nan), errors='coerce'))

        ests = np.array(ests, dtype=float)
        pvals = np.array(pvals, dtype=float)
        ciLs = np.array(ciLs, dtype=float)
        ciUs = np.array(ciUs, dtype=float)

        x = np.arange(n_idps)
        # plot CI lines with caps
        for xi, low, high in zip(x, ciLs, ciUs):
            if np.isnan(low) and np.isnan(high):
                continue
            ax.plot([xi, xi], [low, high], color=color, linewidth=linewidth, zorder=1)
            # caps
            ax.plot([xi - cap_width, xi + cap_width], [low, low], color=color, linewidth=linewidth, zorder=1)
            ax.plot([xi - cap_width, xi + cap_width], [high, high], color=color, linewidth=linewidth, zorder=1)

        # plot estimates
        for xi, est, p in zip(x, ests, pvals):
            if np.isnan(est):
                continue
            if not np.isnan(p) and p < p_thr:
                ax.scatter(xi, est, marker=marker, s=marker_size, color=highlight_color, edgecolor='k', zorder=5)
            else:
                ax.scatter(xi, est, marker=marker, s=marker_size, facecolors='none',
                           edgecolors=color, linewidths=1.5, zorder=5)

        # aesthetics
        ax.set_xticks(x)
        ax.set_xticklabels(idps, rotation=xtick_rotation, ha='right')
        ax.set_xlim(-0.6, n_idps - 1 + 0.6)
        # autoscale y to include CIs comfortably
        finite_vals = np.concatenate([ciLs[~np.isnan(ciLs)], ciUs[~np.isnan(ciUs)], ests[~np.isnan(ests)]]) if (np.any(~np.isnan(ciLs)) or np.any(~np.isnan(ciUs))) else ests[~np.isnan(ests)]
        if finite_vals.size > 0:
            vmin = np.nanmin(finite_vals); vmax = np.nanmax(finite_vals)
            vrange = vmax - vmin if vmax != vmin else max(abs(vmax), 1.0)
            ax.set_ylim(vmin - 0.12*vrange, vmax + 0.12*vrange)

        ax.axhline(0, color='gray', linewidth=0.6, alpha=0.6)
        ax.set_title(eff)
        ax.set_ylabel("Estimate")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # prepare legend handle for this subplot (sample marker)
        lg = plt.Line2D([0], [0], marker=marker, color='w',
                        markerfacecolor=color, markeredgecolor='k', markersize=8, linestyle='None')
        legend_handles.append(lg)
        legend_labels.append(eff)

    # hide unused axes
    for j in range(len(effs), len(axes)):
        axes[j].set_visible(False)

    # overall title and legend (effects legend may be redundant since each subplot is labeled,
    # but keep an optional legend on the side if desired)
    if title is None:
        title = f"Fixed effects (p < {p_thr} highlighted)"
    fig.suptitle(title, fontsize=13)

    # place a small legend for effects colors/markers on the right if there are multiple effects
    if len(legend_handles) > 0 and len(effs) > 1:
        fig.legend(legend_handles, legend_labels, title='Effect', bbox_to_anchor=(0.95, 0.5),
                   loc='center left', frameon=False)

    plt.tight_layout(rect=[0,0,0.92,0.95])
   # if savepath:
   #    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    if rep is not None:
        rep.log_plot(fig, "Results from the mixed effects models")
        plt.close(fig)
        return None, None  # or return a small marker that it was logged
    if show:
        plt.show()
    return fig, axes[:len(effs)]

@rep_plot_wrapper
def plot_AddMultEffects(dfs,
                       feature_col='Feature',
                       p_col='p-value',
                       labels=None,
                       p_thr=0.05,
                       cmap='Reds',
                       annot_fmt="{:.3g}",
                       vmax_logp=10,
                       figsize=(4,8),
                       show_colorbar=True,
                       savepath=None,
                       # new: only two modes allowed
                       value_scale='p',   # 'p' or 'logp' (mutually exclusive)
                       rep=None,
                       show: bool=False,
                       # layout controls
                       annot_fontsize: float = 7.0,
                       tick_fontsize: float = 7.0,
                       cbar_shrink: float = 0.8,
                       linewidths: float = 0.5,
                       square: bool = False
                       ):
    """
    Plot matrix of p-values for features across one or more dfs.

    value_scale:
      - 'p'    : heatmap colors and annotations show raw p-values (range 0..1).
      - 'logp' : heatmap colors and annotations show -log10(p). (Base 10)
                 color scale is clipped at `vmax_logp`.

    Colorbar:
      - Shows the same scale as the heatmap.
      - Includes a tick marking p_thr (in 'p' mode) or -log10(p_thr) (in 'logp' mode).
        The tick label will indicate the p_thr value for clarity.

    Other layout params control fonts / size.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if value_scale not in ('p', 'logp'):
        raise ValueError("value_scale must be 'p' or 'logp'")

    # Normalize input to list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    if labels is None:
        labels = [f"col{i+1}" for i in range(len(dfs))]
    if len(labels) != len(dfs):
        raise ValueError("labels length must match number of dataframes")

    # Collect all features (sorted for stable order)
    all_features = []
    for df in dfs:
        all_features.extend(df[feature_col].astype(str).tolist())
    features = sorted(set(all_features), key=lambda x: x)

    # Build matrix of p-values (rows=features, cols=dataframes)
    pv_mat = pd.DataFrame(index=features, columns=labels, dtype=float)
    for df, lab in zip(dfs, labels):
        tmp = df[[feature_col, p_col]].drop_duplicates(subset=feature_col)
        tmp = tmp.set_index(feature_col)[p_col].astype(float)
        for feat in features:
            pv_mat.at[feat, lab] = tmp.get(feat, np.nan)

    # sanitise tiny zeros
    eps = 1e-300
    pv_safe = pv_mat.copy().astype(float)
    pv_safe = pv_safe.where(~np.isclose(pv_safe, 0.0), eps)

    # Build annotation matrices for the two modes
    annot_p = pv_mat.copy().astype(object)
    for r in features:
        for c in labels:
            v = pv_mat.at[r, c]
            if pd.isna(v):
                annot_p.at[r, c] = ""
            else:
                try:
                    txt = annot_fmt.format(float(v))
                except Exception:
                    txt = str(v)
                star = "*" if (not pd.isna(v) and v < p_thr) else ""
                annot_p.at[r, c] = f"{txt}{star}"

    # compute -log10(p) matrix (base 10) and annotations for it
    logp_mat = -np.log10(pv_safe)
    # clip to vmax_logp so extremely small p's don't blow up the scale
    logp_mat = logp_mat.clip(upper=vmax_logp)
    annot_logp = logp_mat.copy().astype(object)
    for r in features:
        for c in labels:
            v = logp_mat.at[r, c]
            if pd.isna(v):
                annot_logp.at[r, c] = ""
            else:
                try:
                    txt = "{:.3f}".format(float(v))
                except Exception:
                    txt = str(v)
                orig_p = pv_mat.at[r, c]
                star = "*" if (not pd.isna(orig_p) and orig_p < p_thr) else ""
                annot_logp.at[r, c] = f"{txt}{star}"

    # Select which matrix to plot and annotations
    if value_scale == 'p':
        plot_mat = pv_mat.copy().fillna(1.0).clip(lower=0.0, upper=1.0)
        annot_to_use = annot_p
        cbar_label = "p-value"
    else:  # 'logp'
        plot_mat = logp_mat.copy().fillna(0.0)
        annot_to_use = annot_logp
        cbar_label = "-log10(p-value)"

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # draw heatmap
    sns_heatmap = sns.heatmap(
        plot_mat,
        ax=ax,
        annot=annot_to_use,
        fmt="",
        cmap=cmap,
        cbar=show_colorbar,
        linewidths=linewidths,
        linecolor="gray",
        xticklabels=labels,
        yticklabels=features,
        square=square,
        annot_kws={"fontsize": annot_fontsize}
    )

    # adjust colorbar ticks and label: ensure p_thr is shown as tick
    if show_colorbar:
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label)
        # determine sensible tick locations depending on scale
        if value_scale == 'p':
            # prefer ticks that include p_thr; choose 5 ticks between 0 and 1 or max present
            max_val = min(1.0, np.nanmax(plot_mat.values))
            ticks = np.linspace(0.0, max_val, num=5)
            # ensure p_thr is included (or very near): insert it if missing
            if not np.isclose(ticks, p_thr).any():
                ticks = np.unique(np.concatenate((ticks, [p_thr])))
                ticks = np.sort(ticks)
            cbar.set_ticks(ticks)
            # label the p_thr tick specially
            tick_labels = [f"{t:.2g}" for t in ticks]
            # if p_thr present, annotate its label
            # find index
            idx = np.where(np.isclose(ticks, p_thr))[0]
            if idx.size > 0:
                tick_labels[idx[0]] = f"p_thr={p_thr:g}"
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=tick_fontsize)
        else:
            # value_scale == 'logp': ticks on -log10 scale, include thr_log
            max_val = min(vmax_logp, np.nanmax(plot_mat.values))
            ticks = np.linspace(0.0, max_val, num=5)
            thr_log = -np.log10(p_thr) if (p_thr is not None and p_thr > 0) else None
            if thr_log is not None and not np.isclose(ticks, thr_log).any():
                ticks = np.unique(np.concatenate((ticks, [thr_log])))
                ticks = np.sort(ticks)
            cbar.set_ticks(ticks)
            # convert ticks back to p for label alongside log value
            tick_labels = []
            for t in ticks:
                if thr_log is not None and np.isclose(t, thr_log):
                    # label with both representations for clarity
                    tick_labels.append(f"p_thr={p_thr:g}\n(-log10={t:.3f})")
                else:
                    tick_labels.append(f"{t:.2f}")
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=tick_fontsize)

        # shrink colorbar a bit to save room (works with Matplotlib colorbar)
        try:
            cb_axes = cbar.ax
            pos = cb_axes.get_position()
            cb_axes.set_position([pos.x0, pos.y0, pos.width * cbar_shrink, pos.height])
        except Exception:
            pass

    # axis formatting
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Feature p-values (* = p < {p_thr})")
    ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=45)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    if rep is not None:
        rep.log_plot(fig, "Results from the mixed effects models")
        plt.close(fig)
        return None, None
    if show:
        plt.show()
    return fig, ax
