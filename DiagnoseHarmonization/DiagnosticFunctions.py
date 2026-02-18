# DiagnosticFunctions.py
# Collection of diagnostic functions for harmonization assessment (pre and post)

import warnings
from collections import Counter
import re
from matplotlib.pylab import lstsq
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.stats import chi2
import argparse
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
from statsmodels.regression.mixed_linear_model import MixedLMResults
from numpy.random import default_rng
from scipy.stats import (
    chi2,
    fligner,
    pearsonr,
    spearmanr,
    rankdata,
    norm
)
from pandas.api.types import CategoricalDtype


"""
    Collection of statistical functions to assess and visualise batch effects in tabular data.
    Functions include:
    - Cohens_D: Calculate Cohen's d effect size between batches for each feature.
    - Mahalanobis_Distance: Calculate Mahalanobis distance between batches.
    - PC_Correlations: Perform PCA and correlate top PCs with batch and covariates.
    - fit_lmm_safe: Robustly fit a Linear Mixed Model with fallbacks and diagnostics.
    - Variance_Ratios: Calculate variance ratios between batches for each feature.
    - KS_Test: Performs two-sample Kolmogorov-Smirnov test between batches for each feature.

"""

def RobustOLS_Orig(data,covariates,batch,covariate_names,covariate_types,report=None):
    """Defining this function that can be called by cohen's d, variance ratio and KS test functions to residualise out covariate effects before calculating batch effects.
    Here, we support Dummy encoding of categorical covariates and mean-centering of continuous covariates
    Look for a variable which describes whether covariates are: 0 binary, 2 categorical, 3 Continous.
    If variable not given, we will attempt to infer from the unique observations
    Batch always categorical, create Dummy array for batch

    
    """
    # Check if covariate types are provided, if not, infer them from data and report mapping:
    if covariate_types is None:
        covariate_types = []
        for i in range(covariates.shape[1]):
            unique_vals = np.unique(covariates[:, i])
            if len(unique_vals) == 2:
                covariate_types.append(0)  # binary
            elif len(unique_vals) < 10:
                covariate_types.append(2)  # categorical
            else:
                covariate_types.append(3)  # continuous
                if report is not None and hasattr(report, "log_text"):
                    report.log_text(f"Covariate {covariate_names[i]} inferred as continuous with {len(unique_vals)} unique values.")
                    report.log_text(f"Please see https://link.springer.com/article/10.1007/s10459-010-9222-y for more information and consider passing the variable covariate_types to function"
                                    f" to ensure correct handling of covariates in the report. (list of length unique covariates, 0 binary, 2 categorical, 3 continuous)")

    # Create a DataFrame for OLS regression
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
    for i, (name, ctype) in enumerate(zip(covariate_names, covariate_types)):
        if ctype == 0:  # binary
            df[name] = covariates[:, i]
        elif ctype == 2:  # categorical
            df[name] = covariates[:, i].astype(str)  # convert to string for dummy encoding
        elif ctype == 3:  # continuous
            df[name] = covariates[:, i]
        else:
            raise ValueError(f"Unknown covariate type {ctype} for covariate {name}")

    df['batch'] = batch.astype(str)  # ensure batch is treated as categorical
    # Combine batch and covariates, fit OLS and add batch x beta back in
    # Create one array as design for OLS:
    design = np.ones((data.shape[0], 1))  # intercept
    for i, (name, ctype) in enumerate(zip(covariate_names, covariate_types)):
        if ctype == 0:  # binary, make first column of array
            design = np.column_stack((design, df[name]))
        elif ctype == 2:  # categorical
            design = np.column_stack((design, pd.get_dummies(df[name], drop_first=True)))  # dummy encode categorical covariates, drop first to avoid multicollinearity
        elif ctype == 3:  # continuous
            design = np.column_stack((design, df[name]))
        
    # Add dummy encoded batch as covariates to design matrix:
    design = np.column_stack((design, pd.get_dummies(df['batch'], drop_first=True)))
    # Use OLS and remove covariate effects, but add back in batch effects to ensure batch effects are preserved for downstream batch effect calculations
    from numpy.linalg import lstsq

    # Get residuals (data with covariate effects removed)
    B, *_ = lstsq(design, data, rcond=None)
    predicted = design @ B
    data = data - predicted + (design[:, 1+covariates.shape[1]:] @ B[1+covariates.shape[1]:,:])  # add back batch effects

        # Add back in batch effects to residuals to ensure batch effects are preserved for downstream batch effect calculations        



    return data

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

def RobustOLS(data, covariates, batch, covariate_names, covariate_types=None, report=None):
    """
    Minimal fixes to preserve batch while removing covariate effects.
    - Builds full design = [Intercept | covariate-expanded-columns | batch-dummies]
    - Fits full OLS, but subtracts only the covariate-predicted contribution
      (using the covariate betas estimated conditional on batch).
    - Returns residualized data (data - covariate_prediction).
    """
    # -------------------------
    # Infer covariate types if not provided (same heuristic you used)
    # -------------------------
    if covariate_types is None:
        covariate_types = []
        for i in range(covariates.shape[1]):
            unique_vals = np.unique(covariates[:, i])
            if len(unique_vals) == 2:
                covariate_types.append(0)  # binary
            elif len(unique_vals) < 10:
                covariate_types.append(2)  # categorical
            else:
                covariate_types.append(3)  # continuous
                if report is not None and hasattr(report, "log_text"):
                    report.log_text(
                        f"Covariate {covariate_names[i]} inferred as continuous with {len(unique_vals)} unique values.")
                    report.log_text(
                        "Please see https://link.springer.com/article/10.1007/s10459-010-9222-y for more information and consider passing the variable covariate_types to function"
                        " to ensure correct handling of covariates in the report. (list of length unique covariates, 0 binary, 2 categorical, 3 continuous)")

    # -------------------------
    # Create DataFrame (for dummy encoding) and prepare to build design matrix
    # -------------------------
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
    # put covariates into df similarly to your original code
    for i, (name, ctype) in enumerate(zip(covariate_names, covariate_types)):
        if ctype == 0:  # binary
            df[name] = covariates[:, i]
        elif ctype == 2:  # categorical
            df[name] = covariates[:, i].astype(str)
        elif ctype == 3:  # continuous
            df[name] = covariates[:, i]
        else:
            raise ValueError(f"Unknown covariate type {ctype} for covariate {name}")

    df['batch'] = batch.astype(str)  # ensure batch is treated as categorical

    # -------------------------
    # Build design matrix while tracking column ranges for each covariate and batch
    # -------------------------
    # Start with intercept
    design = np.ones((data.shape[0], 1))
    # We'll keep a list of (group_name, start_col_index, end_col_index) for mapping
    mapping = []   # each entry: (kind, name, start, end) end is exclusive
    cur_col = 1
    mapping.append(('intercept', 'Intercept', 0, 1))

    # Add covariates one-by-one, recording how many columns each expands to
    covariate_ranges = []  # list of (name, start, end) for covariate columns
    for i, (name, ctype) in enumerate(zip(covariate_names, covariate_types)):
        if ctype == 0:  # binary -> single numeric column
            col = df[name].astype(float).values.reshape(-1, 1)
            design = np.column_stack((design, col))
            start = cur_col
            end = cur_col + 1
            covariate_ranges.append((name, start, end))
            mapping.append(('covariate', name, start, end))
            cur_col = end
        elif ctype == 2:  # categorical -> dummy columns
            dummies = pd.get_dummies(df[name], drop_first=True)
            if dummies.shape[1] == 0:
                # no variation for this covariate: skip it
                if report is not None and hasattr(report, "log_text"):
                    report.log_text(f"Categorical covariate {name} has a single level; skipping.")
                continue
            design = np.column_stack((design, dummies.values))
            start = cur_col
            end = cur_col + dummies.shape[1]
            covariate_ranges.append((name, start, end))
            mapping.append(('covariate', name, start, end))
            cur_col = end
        elif ctype == 3:  # continuous -> single numeric column (same as original behavior)
            col = df[name].astype(float).values.reshape(-1, 1)
            design = np.column_stack((design, col))
            start = cur_col
            end = cur_col + 1
            covariate_ranges.append((name, start, end))
            mapping.append(('covariate', name, start, end))
            cur_col = end
        else:
            raise ValueError(f"Unknown covariate type {ctype} for covariate {name}")

    # Add batch dummies and record their range
    batch_dummies = pd.get_dummies(df['batch'], drop_first=True)
    if batch_dummies.shape[1] > 0:
        design = np.column_stack((design, batch_dummies.values))
        batch_start = cur_col
        batch_end = cur_col + batch_dummies.shape[1]
        mapping.append(('batch', 'batch', batch_start, batch_end))
        cur_col = batch_end
    else:
        # single batch; record empty range
        batch_start = batch_end = None
        mapping.append(('batch', 'batch', cur_col, cur_col))

    # -------------------------
    # Fit full model and remove covariate contribution using the covariate column indices
    # -------------------------
    # Solve least squares for full design: design @ B = data
    B_full, *_ = lstsq(design, data, rcond=None)  # shape: (n_design_cols, n_features)
    # Create a list of all covariate column indices (exclude intercept and batch)
    cov_indices = []
    for (name, start, end) in covariate_ranges:
        cov_indices.extend(list(range(start, end)))
    if len(cov_indices) == 0:
        # nothing to remove
        if report is not None and hasattr(report, "log_text"):
            report.log_text("No covariate columns detected — returning original data unchanged.")
        return data

    cov_indices = np.array(cov_indices, dtype=int)
    # covariate-only design slice and betas:
    cov_design = design[:, cov_indices]            # (n_samples, n_cov_design_cols)
    beta_cov = B_full[cov_indices, :]              # (n_cov_design_cols, n_features)
    pred_cov = cov_design @ beta_cov               # (n_samples, n_features)

    # Subtract only the covariate contribution (preserve intercept and batch terms)
    data_resid = data - pred_cov

    if report is not None and hasattr(report, "log_text"):
        report.log_text("RobustOLS: removed covariate contribution estimated conditional on batch (batch preserved).")

    return data_resid



def fit_lmm_safe(df, formula_fixed, group_col='batch', reml=False,
                 min_group_n=10, var_threshold=1e-8,
                 optimizers=('lbfgs', 'bfgs', 'powell', 'cg'),
                 maxiter=400):
    """
    Robust wrapper to fit a MixedLM to df for a single feature 'y'.

    Behavior:
      - If feature variance < var_threshold, returns success=False and notes include 'low_variance_feature'.
      - If any batch count < min_group_n, falls back to OLS (notes include 'small_group_count').
      - Scales numeric covariates (center+unit-std) to stabilize optimization.
      - Tries multiple optimizers. On success returns model results and computed stats.
      - On complete failure returns fallback OLS if possible, else None results.

    Args:
        df (pd.DataFrame): DataFrame containing 'y' and covariates.
        formula_fixed (str): Patsy formula for fixed effects (excluding random effects).
        group_col (str): Column name for grouping variable (random effect).
        reml (bool): Whether to use REML for LMM fitting.
        min_group_n (int): Minimum samples per group to attempt LMM.
        var_threshold (float): Minimum variance threshold to attempt LMM.
        optimizers (tuple): Optimizers to try for LMM fitting.
        maxiter (int): Maximum iterations for optimizer.
    Returns a dict:
        {
          'success': bool,         # True if LMM fit succeeded
          'mdf': MixedLMResults or None,
          'ols': OLSResults or None,  # fallback OLS fit for fixed part (if available)
          'notes': list[str],      # short tokens describing what happened
          'stats': dict            # summary values (var_fixed, var_batch, var_resid, R2_marginal, R2_conditional, ICC, LR_stat, pval_LRT)
        }

    """
    notes = []
    n = df.shape[0]
    if 'y' not in df.columns:
        raise ValueError("df must contain column 'y'.")
    # 1) low-variance check
    if np.nanvar(df['y']) <= var_threshold:
        notes.append('low_variance_feature')
        return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

    # 2) check minimal group sizes
    group_counts = df[group_col].value_counts()
    if (group_counts < min_group_n).any():
        notes.append('small_group_count')
        # Attempt OLS on fixed effects only and return as fallback
        try:
            import patsy
            y_vec, X = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
            ols_res = sm.OLS(y_vec, X).fit()
            notes.append('fallback_ols_used')
            stats = {
                'var_fixed': float(np.nanvar(np.dot(X.values, ols_res.params.values), ddof=0)),
                'var_batch': 0.0,
                'var_resid': float(ols_res.mse_resid),
                'R2_ols_fixed': float(ols_res.rsquared)
            }
            return {'success': False, 'mdf': None, 'ols': ols_res, 'notes': notes, 'stats': stats}
        except Exception as e:
            notes.append('fallback_ols_failed')
            return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

    # 3) scale numeric covariates
    for col in df.columns:
        if col in (group_col, 'y'):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            m = df[col].mean()
            s = df[col].std(ddof=0)
            if s == 0:
                df[col] = df[col] - m
            else:
                df[col] = (df[col] - m) / s

    # 4) try mixed model with multiple optimizers
    md = mixedlm(formula_fixed, groups=group_col, data=df, re_formula="1")
    last_exc = None
    for opt in optimizers:
        try:
            mdf = md.fit(reml=reml, method=opt, maxiter=maxiter, disp=False)
            # check convergence attribute when available
            converged = getattr(mdf, 'converged', True)
            if not converged:
                notes.append(f'optimizer_{opt}_no_converge')
                continue

            # try to extract random variance (handle formats)
            try:
                cov_re = mdf.cov_re
                # If cov_re is a DataFrame
                cov_re_arr = np.asarray(cov_re)
                # treat tiny elements as zero
                if cov_re_arr.size > 0 and np.nanmax(np.abs(cov_re_arr)) <= 1e-10:
                    notes.append('var_batch_near_zero')
            except Exception:
                notes.append('cov_re_inspect_failed')

            # compute fixed-effects linear predictor variance
            try:
                exog = mdf.model.exog
                fe_params = mdf.fe_params.values.reshape(-1, 1)
                linpred_fixed = np.dot(exog, fe_params).ravel()
                var_fixed = float(np.nanvar(linpred_fixed, ddof=0))
            except Exception:
                var_fixed = np.nan

            # random intercept variance
            try:
                # Some statsmodels versions return a DataFrame or scalar
                try:
                    var_batch = float(mdf.cov_re.iloc[0, 0])
                except Exception:
                    var_batch = float(np.asarray(mdf.cov_re).ravel()[0])
            except Exception:
                var_batch = np.nan

            # residual variance
            try:
                var_resid = float(mdf.scale)
            except Exception:
                var_resid = np.nan

            # totals and R2-ish metrics (Nakagawa style for Gaussian)
            total_var = var_fixed + (0.0 if np.isnan(var_batch) else var_batch) + var_resid
            R2_marginal = var_fixed / total_var if total_var != 0 and not np.isnan(var_fixed) else np.nan
            R2_conditional = (var_fixed + (0.0 if np.isnan(var_batch) else var_batch)) / total_var if total_var != 0 else np.nan
            ICC = var_batch / (var_batch + var_resid) if (not np.isnan(var_batch) and (var_batch + var_resid) != 0) else np.nan 

            # LRT vs fixed-only OLS
            try:
                # build OLS fixed-only design via patsy for the same formula
                import patsy
                y_vec, X_fixed = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
                ols_fixed = sm.OLS(y_vec, X_fixed).fit()
                llf_lmm = float(mdf.llf)
                llf_ols = float(ols_fixed.llf)
                LR_stat = 2.0 * (llf_lmm - llf_ols)
                pval_LRT = float(chi2.sf(LR_stat, 1)) if np.isfinite(LR_stat) else np.nan
            except Exception:
                LR_stat = np.nan
                pval_LRT = np.nan

            stats = {
                'var_fixed': var_fixed,
                'var_batch': var_batch,
                'var_resid': var_resid,
                'R2_marginal': R2_marginal,
                'R2_conditional': R2_conditional,
                'ICC': ICC,
                'LR_stat': LR_stat,
                'pval_LRT_random': pval_LRT
            }

            return {'success': True, 'mdf': mdf, 'ols': None, 'notes': notes, 'stats': stats}
        except Exception as e:
            last_exc = e
            notes.append(f'optimizer_{opt}_failed')
            continue

    # 5) All LMM attempts failed -> fallback to OLS (try to return useful stats)
    try:
        import patsy
        y_vec, X = patsy.dmatrices(formula_fixed, df, return_type='dataframe')
        ols_res = sm.OLS(y_vec, X).fit()
        stats = {
            'var_fixed': float(np.nanvar(np.dot(X.values, ols_res.params.values), ddof=0)),
            'var_batch': 0.0,
            'var_resid': float(ols_res.mse_resid),
            'R2_ols_fixed': float(ols_res.rsquared)
        }
        notes.append('all_lmm_optimizers_failed_fallback_ols')
        return {'success': False, 'mdf': None, 'ols': ols_res, 'notes': notes, 'stats': stats}
    except Exception:
        notes.append('all_lmm_and_ols_failed')
        return {'success': False, 'mdf': None, 'ols': None, 'notes': notes, 'stats': {}}

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations
# Cohens d function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations

def z_score(data):
    """
    Z-score normalization of the data matrix (samples x features).
    Use median centered by default as is more robust to outliers and non-normal distributions.
    """
    median = np.median(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    zscored = (data - median) / std
    return zscored
    

import numpy as np

def Cohens_D(Data, batch_indices, covariates=None, BatchNames=None, covariate_names=None, covariate_types=None):
    """
    Cohen's d for each batch vs "all other batches" using UNWEIGHTED batch averages.
    - mean_other: simple (unweighted) mean of the other-batch means
    - pooled/std: simple (unweighted) mean of the other-batch standard deviations (ddof=1 where possible)
    Returns:
      d_array: shape (num_batches, num_features)
      pair_labels: list of (BatchName_g, 'all_data')
    """
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch_indices, (list, np.ndarray)) or np.ndim(batch_indices) != 1:
        raise ValueError("batch_indices must be a 1D list or numpy array.")
    if Data.shape[0] != len(batch_indices):
        raise ValueError("Number of samples in Data must match length of batch_indices.")

    batch_indices = np.array(batch_indices, dtype=object)
    unique_groups = np.array(list(dict.fromkeys(batch_indices.tolist())))  # preserve order

    if len(unique_groups) < 2:
        raise ValueError("At least two unique groups are required to calculate Cohen's d")

    # Build BatchNames mapping flexibly
    if BatchNames is None:
        BatchNames_map = {g: str(g) for g in unique_groups}
    elif isinstance(BatchNames, dict):
        BatchNames_map = {g: BatchNames.get(g, str(g)) for g in unique_groups}
    elif isinstance(BatchNames, (list, tuple)):
        if len(BatchNames) != len(unique_groups):
            raise ValueError("When BatchNames is a list/tuple its length must equal the number of unique groups.")
        BatchNames_map = {g: name for g, name in zip(unique_groups, BatchNames)}
    else:
        raise ValueError("BatchNames must be a dict, list/tuple, or None.")

    # Use the helper fuction to residualise out covariate effects if covariates are provided, ensuring batch effects are preserved in the residuals for downstream calculations
    if covariates is not None:
        covariate_names = [f'cov_{i}' for i in range(covariates.shape[1])]
        Data_to_use = RobustOLS(Data, covariates, batch_indices, covariate_names, covariate_types)
    else:
        Data_to_use = Data

    # Compute per-batch stats: n, mean, std (ddof=1)
    per_batch = {}
    for g in unique_groups:
        mask = (batch_indices == g)
        Xg = Data_to_use[mask]
        n = Xg.shape[0]
        if n == 0:
            raise ValueError(f"No samples found for group {g}")
        mean = Xg.mean(axis=0)
        # std with ddof=1 when possible; else nan
        std = Xg.std(axis=0, ddof=1) if n > 1 else np.full(Data_to_use.shape[1], np.nan)
        per_batch[g] = {'n': n, 'mean': mean, 'std': std}

    pairwise_d = []
    pair_labels = []

    for g in unique_groups:
        stats_g = per_batch[g]
        mean_g = stats_g['mean']
        std_g = stats_g['std']

        # other batches (exclude g)
        other_groups = [h for h in unique_groups if h != g]
        means_other = np.stack([per_batch[h]['mean'] for h in other_groups], axis=0)  # (num_other, features)
        stds_other = np.stack([per_batch[h]['std'] for h in other_groups], axis=0)    # (num_other, features)

        # UNWEIGHTED averages across batches: simple arithmetic mean (ignore NaNs)
        mean_other = np.nanmean(means_other, axis=0)
        mean_std_other = np.nanmean(stds_other, axis=0)

        # If mean_std_other is zero or nan, set result to nan to avoid division by zero
        denom = mean_std_other.copy()
        denom[denom == 0] = np.nan

        d = (mean_g - mean_other) / denom

        pairwise_d.append(d)
        pair_labels.append((BatchNames_map[g], 'all_data'))

    return np.vstack(pairwise_d), pair_labels

# PC_Correlations performs PCA on data and computes Pearson correlation of the top N principal components with a batch variable.
def PC_Correlations(
    Data,
    batch,
    N_components=None,
    covariates=None,
    variable_names=None,
    *,
    enforce_min_components_for_plotting=True
):
    """
    Perform PCA and correlate top PCs with batch and optional covariates.

    Parameters
    ----------
    Data : np.ndarray (n_samples x n_features)
        Data matrix.
    batch : array-like (n_samples,)
        Batch labels (can be strings or numbers).
    N_components : int or None
        Requested number of principal components to compute/analyze.
        If None, default is 4 (but constrained by data size).
    covariates : None or array-like (n_samples x n_covariates)
        Optional covariate matrix.
    variable_names : None or list[str]
        Optional names for the variables (first element expected 'batch' if present).
    enforce_min_components_for_plotting : bool (default True)
        If True, the routine will try to ensure at least 2 components are returned
        when the dataset allows it (helps plotting PC1 vs PC2). If the data has
        fewer than 2 possible components (e.g., n_features==1), this is not possible
        and a warning will be emitted.

    Returns
    -------
    explained_variance : np.ndarray (n_components,)
        Percent variance explained by each returned component (sum <= 100).
    scores : np.ndarray (n_samples x n_components)
        PCA scores (projection of samples).
    PC_correlations : dict
        Mapping variable name -> {'correlation': array, 'p_value': array}
        where arrays have length == n_components.
    pca : sklearn.decomposition.PCA
        The fitted PCA object (useful for components_, explained_variance_ratio_, ...)
    """
    import numpy as np
    import pandas as pd
    import warnings
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    # --- Input checks & normalization ---
    if not isinstance(Data, np.ndarray) or Data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    n_samples, n_features = Data.shape

    batch = np.asarray(batch)
    if batch.ndim != 1:
        raise ValueError("batch must be a 1D array-like of length n_samples.")
    if batch.shape[0] != n_samples:
        raise ValueError("Number of samples in Data must match length of batch")

    # Covariates: if provided, convert to array and validate shape
    if covariates is not None:
        covariates = np.asarray(covariates)
        if covariates.ndim != 2:
            raise ValueError("covariates must be a 2D array (n_samples x n_covariates).")
        if covariates.shape[0] != n_samples:
            raise ValueError("covariates must have same number of rows as Data (samples).")

    # Decide how many components to compute
    # Default desired = 4 if not given
    desired = 4 if N_components is None else int(N_components)
    max_possible = min(n_samples, n_features)
    if max_possible <= 0:
        raise ValueError("Data must have at least one sample and one feature.")
    # Try to enforce at least 2 components for plotting when dataset allows it
    if enforce_min_components_for_plotting and max_possible >= 2:
        desired = max(desired, 2)
    # Final number of components we will compute
    n_comp = min(desired, max_possible)
    if n_comp < desired:
        warnings.warn(
            f"Requested N_components={desired} reduced to {n_comp} due to data size "
            f"(n_samples={n_samples}, n_features={n_features})."
        )

    # --- Fit PCA ---
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Data)   # shape (n_samples, n_comp)
    explained_variance = (pca.explained_variance_ratio_ * 100.0)  # % per PC

    # --- Prepare variables for correlation ---
    # Factorize batch robustly (works for strings / numbers / mixed)
    batch_codes, unique_batches = pd.factorize(batch)
    # convert to float for Pearson r
    batch_var = batch_codes.astype(float)

    variables = [batch_var]
    # Optional variables names (create defaults if not provided)
    if covariates is not None:
        # If covariates has column names (DataFrame), capture them
        if isinstance(covariates, pd.DataFrame):
            cov_arr = covariates.values
            cov_names = list(map(str, covariates.columns))
        else:
            cov_arr = np.asarray(covariates)
            cov_names = [f'covariate_{i+1}' for i in range(cov_arr.shape[1])]
        # extend variables with each covariate column
        variables.extend([cov_arr[:, i].astype(float) for i in range(cov_arr.shape[1])])
    else:
        cov_names = []

    # Build variable names list (first should be 'batch' unless user provided different)
    if variable_names is None:
        variable_names_out = ['batch'] + cov_names
    else:
        # Validate provided variable_names length matches number of variables
        # Accept possibility that user included 'batch' as first element
        if len(variable_names) == len(variables):
            variable_names_out = list(variable_names)
        elif len(variable_names) == len(variables) + 0 and str(variable_names[0]).lower() == "batch":
            # e.g. user provided ['batch', 'Age', 'Sex'] while variables is [batch, Age, Sex]
            variable_names_out = list(variable_names)
        elif len(variable_names) == len(variables) - 0:
            # allow case where user gave only covariate names (no 'batch')
            variable_names_out = ['batch'] + list(variable_names)
        else:
            raise ValueError(
                "variable_names length does not match number of variables (batch + covariates). "
                f"Got variable_names length {len(variable_names)} but need {len(variables)}."
            )

    # --- Compute Pearson correlations between each PC and each variable ---
    n_used_comps = scores.shape[1]
    PC_correlations = {}
    for name, var in zip(variable_names_out, variables):
        corrs = np.empty(n_used_comps, dtype=float)
        pvals = np.empty(n_used_comps, dtype=float)
        for i in range(n_used_comps):
            try:
                r, p = pearsonr(scores[:, i], var)
            except Exception:
                r, p = np.nan, np.nan
            corrs[i] = r
            pvals[i] = p
        PC_correlations[name] = {'correlation': corrs, 'p_value': pvals}

    # Return PCA object too so callers can access components_, mean_, etc.
    return explained_variance, scores, PC_correlations, pca

# MahalanobisDistance computes the Mahalanobis distance (multivariate difference between batch and global centroids)
def Mahalanobis_Distance(Data=None, batch=None, covariates=None):

    """
    Calculate the Mahalanobis distance between batches in the data.
    Takes optional covariates and returns distances between each batch pair
    both before and after regressing out covariates. Additionally provides
    distance of each batch to the overall centroid before and after residualizing
    covariates.

    Args:
        Data (np.ndarray): Data matrix where rows are samples (n) and columns are features (p).
        batch (np.ndarray): 1D array-like batch labels for each sample (length n).
        covariates (np.ndarray, optional): Covariate matrix (n x k). An intercept will be added automatically.

    Returns:
        dict: {
            "pairwise_raw": { (b1, b2): distance, ... },
            "pairwise_resid": { (b1, b2): distance, ... } or None if no covariates,
            "centroid_raw": { (b, 'global'): distance, ... },
            "centroid_resid": { (b, 'global'): distance, ... } or None if no covariates,
            "batches": list_of_unique_batches_in_order
        }

        Keys of the inner dicts are tuples like (b1, b2) for pairwise distances and (b, 'global') for
        distances to the overall centroid.

    Raises:
        ValueError: If inputs are invalid or less than two unique batches are provided.
    """
    # ---- validations ----
    if Data is None or batch is None:
        raise ValueError("Both Data and batch must be provided.")
    Data = np.asarray(Data, dtype=float)
    batch = np.asarray(batch)
    if Data.ndim != 2:
        raise ValueError("Data must be a 2D array (samples x features).")
    n, p = Data.shape
    if batch.shape[0] != n:
        raise ValueError("Batch length must match the number of rows in Data.")
    if np.isnan(Data).any():
        raise ValueError("Data contains NaNs; please impute or remove missing values first.")

    unique_batches = np.array(list(dict.fromkeys(batch.tolist())))  # stable order
    if unique_batches.size < 2:
        raise ValueError("At least two unique batches are required.")

    # Optional covariates handling
    have_covariates = covariates is not None
    if have_covariates:
        covariates = np.asarray(covariates, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates.reshape(-1, 1)
        if covariates.shape[0] != n:
            raise ValueError("Covariates must have the same number of rows as Data.")
        if np.isnan(covariates).any():
            raise ValueError("Covariates contain NaNs; please clean them first.")

    # ---- helpers ----
    def _batch_means(X):
        return {b: X[batch == b].mean(axis=0) for b in unique_batches}

    def _global_mean(X):
        return X.mean(axis=0)

    def _cov_pinv(X):
        # Sample covariance (unbiased). Use pseudo-inverse for stability (singular or p>n).
        S = np.cov(X, rowvar=False, bias=False)
        return np.linalg.pinv(S)

    def _mahal_sq(diff, Sinv):
        # Quadratic form; return sqrt for distance
        return float(np.sqrt(diff @ Sinv @ diff))

    def _pairwise_and_centroid_distances(X):
        means = _batch_means(X)
        gmean = _global_mean(X)
        Sinv = _cov_pinv(X)

        # pairwise
        pw = {}
        for (b1, b2) in combinations(unique_batches, 2):
            d = means[b1] - means[b2]
            pw[(b1, b2)] = _mahal_sq(d, Sinv)

        # centroid
        cent = {}
        for b in unique_batches:
            d = means[b] - gmean
            cent[(b, "global")] = _mahal_sq(d, Sinv)

        return pw, cent

    # ---- raw distances ----
    pairwise_raw, centroid_raw = _pairwise_and_centroid_distances(Data)

    # ---- residualize (if covariates) and compute distances again ----
    if have_covariates:
        # Add intercept
        X = np.column_stack([np.ones((n, 1)), covariates])
        # Solve least squares for each feature simultaneously
        # Data ≈ X @ B  => B = (X^T X)^+ X^T Data
        B, *_ = np.linalg.lstsq(X, Data, rcond=None)
        resid = Data - X @ B
        pairwise_resid, centroid_resid = _pairwise_and_centroid_distances(resid)
    else:
        pairwise_resid, centroid_resid = None, None

    return {
        "pairwise_raw": pairwise_raw,
        "pairwise_resid": pairwise_resid,
        "centroid_raw": centroid_raw,
        "centroid_resid": centroid_resid,
        "batches": unique_batches.tolist(),
    }

# Define a function to calculate the feature-wise ratio of variance between each unique batch pair
def Variance_Ratios(data, batch, covariates=None,covariate_names=None, covariate_types=None):
    # Define a function to calculate the feature-wise ratio of variance between each unique batch pair
    import numpy as np
    import pandas as pd
    from itertools import combinations
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) !=1:
        raise ValueError("batch must be a 1D list or numpy array.")
    if data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")
    
    batch = np.array(batch)
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to compute ratio of variance.")
    batch_data = {}
    ratio_of_variance = {}

    """If covariates are provided, remove their effects from the data using linear regression"""
    if covariates is not None:
        from numpy.linalg import inv, pinv
        data = RobustOLS(data, covariates, batch, covariate_names, covariate_types)
    # Calculate variances for each feature in each batch
    for b in unique_batches:
        batch_data[b] = data[batch == b]
    for b1, b2 in combinations(unique_batches, 2):
        var1 = np.var(batch_data[b1], axis=0, ddof=1)
        var2 = np.var(batch_data[b2], axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = var1 / var2
            ratio[np.isnan(ratio)] = 0  # Replace NaNs due to division by zero
        ratio_of_variance[(b1, b2)] = ratio
    return ratio_of_variance

# Define a function to perform the Levene's test for variance differences between each unique batch pair
def Levene_Test(data, batch, centre = 'median'):
    # Define a function to perform the Levene's test for variance differences between each unique batch pair
    """
    Args: data
    - data: subjects x features (np.ndarray)
    - batch: subjects x 1 (np.ndarray), batch labels
    - centre: str, optional, the center to use for the test, 'median' by default. See scipy.stats.levene for options.
    Returns:
        - levene_results: dictionary with Levene's test statistic and p-value for each pair of batches
    Raises:
        - ValueError: if Data is not a 2D array or batch is not a
        1D array, or if the number of samples in Data and batch do not match
    
    """
    import numpy as np
    from scipy.stats import levene
    from itertools import combinations
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) !=1:
        raise ValueError("batch must be a 1D list or numpy array.")
    
    if data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")
    batch = np.array(batch)
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required to perform Levene's test.")
    batch_data = {}
    levene_results = {}
    # Calculate variances for each feature in each batch
    for b in unique_batches:
        batch_data[b] = data[batch == b]
    for b1, b2 in combinations(unique_batches, 2):
        p_values = []
        statistics = []
        for feature_idx in range(data.shape[1]):
            stat, p_value = levene(batch_data[b1][:, feature_idx], batch_data[b2][:, feature_idx], center=centre)
            statistics.append(stat)
            p_values.append(p_value)
        levene_results[(b1, b2)] = {
            'statistic': np.array(statistics),
            'p_value': np.array(p_values)
        }
    return levene_results

# Define a function to perform two-sample Kolmogorov-Smirnov test for distribution differences between
# each unique batch pair and each batch with the overall distribution
def KS_Test(data,
                batch,
                feature_names=None,
                covariates=None,
                compare_pairs=True,
                compare_to_overall_excluding_batch=True,
                min_batch_n=3,
                alpha=0.05,
                do_fdr=True,
                residualize_covariates=True,
                covariate_names=None,
                covariate_types=None):
    """
    Improved two-sample KS testing for batch effect detection.

    Args:
      data: (n_samples, n_features) numpy array
      batch: array-like of length n_samples with batch labels
      feature_names: optional list of feature names
      compare_pairs: if True, include pairwise batch-vs-batch KS tests
      compare_to_overall_excluding_batch: if True, compare batch to overall excluding that batch
                                        (recommended). If False, compares to pooled overall (original behavior).
      min_batch_n: minimum samples required in each group to run KS for that feature (default 3)
      alpha: significance threshold for summary reporting
      do_fdr: whether to compute Benjamini-Hochberg FDR-corrected p-values per comparison

    Returns:
      dict:
        - keys are tuples like (b, 'overall') or (b1, b2)
        - each value is a dict with:
            'statistic': np.array of D statistics (length n_features)
            'p_value': np.array of p-values (nan where test not run)
            'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)
            'n_group1': array of sample counts per feature for group1 (same across features but kept for completeness)
            'n_group2': array of counts for group2
            'summary': {'prop_significant': float, 'mean_D': float}
        - 'feature_names': list of feature names
    """
    import numpy as np
    from scipy.stats import ks_2samp
    from itertools import combinations

    def benjamini_hochberg(pvals):
        """Simple BH FDR correction. pvals can contain np.nan; those are left as np.nan."""
        p = np.asarray(pvals)
        mask = ~np.isnan(p)
        p_nonan = p[mask]
        m = len(p_nonan)
        if m == 0:
            return np.full_like(p, np.nan, dtype=float)
        order = np.argsort(p_nonan)
        ranked = np.empty(m, dtype=float)
        # compute adjusted p in reverse order
        cummin = 1.0
        adj = np.empty(m, dtype=float)
        for i in range(m-1, -1, -1):
            rank = i + 1
            pval = p_nonan[order[i]]
            adj_val = min(cummin, pval * m / rank)
            cummin = adj_val
            adj[i] = adj_val
        # put back in original order
        adj_ordered = np.empty(m, dtype=float)
        adj_ordered[order] = adj
        out = np.full_like(p, np.nan, dtype=float)
        out[mask] = np.minimum(adj_ordered, 1.0)
        return out

    # ---- Validation ----
    if not hasattr(data, "ndim") or data.ndim != 2:
        raise ValueError("data must be a 2D numpy array (samples x features).")
    n_samples, n_features = data.shape
    batch = np.array(batch)
    if batch.ndim != 1 or len(batch) != n_samples:
        raise ValueError("batch must be 1D and match number of samples in data.")
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    if feature_names is None:
        feature_names = [f"feature_{i+1}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError("feature_names length must match number of features.")
    
    if residualize_covariates is True:
        if covariates is None: # Skip this whole step
            pass
        else:
            # Estimate betas for covariates and batch, remove covariate effects but keep batch effects
            from numpy.linalg import lstsq
            covariates = np.asarray(covariates, dtype=float)
            if covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            if covariates.shape[0] != n_samples:
                raise ValueError("Covariates must have the same number of rows as data.")
            data = RobustOLS(data, covariates, batch, covariate_names, covariate_types)


    ks_results = {}
    # Pre-slice batch indices to avoid repeated boolean masks
    batch_idx = {b: np.where(batch == b)[0] for b in unique_batches}

    # Helper to run KS for two index sets
    def run_ks_for_indices(idx1, idx2):
        stats = np.full(n_features, np.nan, dtype=float)
        pvals = np.full(n_features, np.nan, dtype=float)
        n1 = np.full(n_features, 0, dtype=int)
        n2 = np.full(n_features, 0, dtype=int)
        for fi in range(n_features):
            x = data[idx1, fi]
            y = data[idx2, fi]
            n1[fi] = x.size
            n2[fi] = y.size
            if x.size >= min_batch_n and y.size >= min_batch_n:
                try:
                    stat, p = ks_2samp(x, y, alternative='two-sided', mode='auto')
                    stats[fi] = stat
                    pvals[fi] = p
                except Exception:
                    stats[fi] = np.nan
                    pvals[fi] = np.nan
            else:
                # not enough samples to run KS reliably
                stats[fi] = np.nan
                pvals[fi] = np.nan
        return stats, pvals, n1, n2

    # Compare each batch to overall (pooled) or to overall excluding the batch
    for b in unique_batches:
        idx_b = batch_idx[b]
        if compare_to_overall_excluding_batch:
            # overall excluding this batch
            idx_other = np.setdiff1d(np.arange(n_samples), idx_b, assume_unique=True)
        else:
            # pooled overall includes the batch (original behavior)
            idx_other = np.arange(n_samples)
        if idx_b.size == 0 or idx_other.size == 0:
            # shouldn't happen, but guard
            stats = np.full(n_features, np.nan, dtype=float)
            pvals = np.full(n_features, np.nan, dtype=float)
            n1 = np.zeros(n_features, dtype=int)
            n2 = np.zeros(n_features, dtype=int)
        else:
            stats, pvals, n1, n2 = run_ks_for_indices(idx_b, idx_other)

        result = {
            'statistic': stats,
            'p_value': pvals,
            'p_value_fdr': None,
            'n_group1': n1,
            'n_group2': n2,
            'summary': {
                'prop_significant': float(np.sum((pvals < alpha) & ~np.isnan(pvals)) / np.sum(~np.isnan(pvals))) if np.any(~np.isnan(pvals)) else 0.0,
                'mean_D': float(np.nanmean(stats))
            }
        }
        if do_fdr:
            result['p_value_fdr'] = benjamini_hochberg(pvals)
        ks_results[(b, 'overall')] = result

    # Pairwise comparisons if requested
    if compare_pairs:
        for b1, b2 in combinations(unique_batches, 2):
            idx1 = batch_idx[b1]
            idx2 = batch_idx[b2]
            stats, pvals, n1, n2 = run_ks_for_indices(idx1, idx2)
            result = {
                'statistic': stats,
                'p_value': pvals,
                'p_value_fdr': None,
                'n_group1': n1,
                'n_group2': n2,
                'summary': {
                    'prop_significant': float(np.sum((pvals < alpha) & ~np.isnan(pvals)) / np.sum(~np.isnan(pvals))) if np.any(~np.isnan(pvals)) else 0.0,
                    'mean_D': float(np.nanmean(stats))
                }
            }
            if do_fdr:
                result['p_value_fdr'] = benjamini_hochberg(pvals)
            ks_results[(b1, b2)] = result

    ks_results['params'] = {
        'compare_pairs': compare_pairs,
        'compare_to_overall_excluding_batch': compare_to_overall_excluding_batch,
        'min_batch_n': min_batch_n,
        'alpha': alpha,
        'do_fdr': do_fdr
    }
    return ks_results

# Function to fit LMM safely with fallbacksdef fit_lmm_safe(df, formula_fixed, group_col, min_group_n=2, var_threshold=1e-8):
def Run_LMM_cross_sectional(Data, batch, covariates=None, feature_names=None, group_col_name='batch',
                  covariate_names=None, min_group_n=2, var_threshold=1e-8):
    """
    Runs LMM diagnostics for each feature and returns (results_df, summary).
    results_df columns: feature, success, var_fixed, var_batch, var_resid, R2_marginal, R2_conditional, ICC, notes
    summary: Counter of notes + counts
    """
    import numpy as np
    import pandas as pd
    import warnings
    from statsmodels.formula.api import mixedlm, ols
    import statsmodels.api as sm
    from scipy.stats import chi2
    import matplotlib.pyplot as plt

    Data = np.asarray(Data, dtype=float)
    n, p = Data.shape
    if feature_names is None:
        feature_names = [f'feature_{i+1}' for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length mismatch.")

    # Build base DataFrame for model inputs
    df_base = pd.DataFrame({group_col_name: np.asarray(batch)})
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cov_df = covariates.reset_index(drop=True)
        else:
            cov_arr = np.asarray(covariates)
            if cov_arr.ndim == 1:
                cov_df = pd.DataFrame({covariate_names[0] if covariate_names else 'cov1': cov_arr})
            else:
                names = covariate_names if covariate_names else [f'cov{i+1}' for i in range(cov_arr.shape[1])]
                cov_df = pd.DataFrame(cov_arr, columns=names)
        df_base = pd.concat([df_base, cov_df], axis=1)

    # build formula (fixed part)
    if covariates is not None and cov_df.shape[1] > 0:
        rhs = ' + '.join(list(cov_df.columns))
        formula_fixed = f"y ~ {rhs}"
    else:
        formula_fixed = "y ~ 1"

    rows = []
    notes_counter = Counter()
    for fi in range(p):
        df = df_base.copy()
        df['y'] = Data[:, fi]
        res = fit_lmm_safe(df, formula_fixed, group_col=group_col_name,
                           min_group_n=min_group_n, var_threshold=var_threshold)
        stats = res.get('stats', {})
        notes = res.get('notes', []) or []
        # ensure consistent fields
        row = {
            'feature': feature_names[fi],
            'success': bool(res.get('success', False)), # did LMM fit succeed?
            'var_fixed': stats.get('var_fixed', np.nan), # Variance explained by fixed effects
            'var_batch': stats.get('var_batch', np.nan), # Variance explained by batch random effect
            'var_resid': stats.get('var_resid', np.nan), # Residual variance
            'R2_marginal': stats.get('R2_marginal', np.nan), # Marginal R-squared (fixed effects)
            'R2_conditional': stats.get('R2_conditional', np.nan), # Conditional R-squared (fixed + random effects)
            'ICC': stats.get('ICC', np.nan), # Intraclass correlation coefficient (batch variance / total variance)
            'LR_stat': stats.get('LR_stat', np.nan),
            'pval_LRT_random': stats.get('pval_LRT_random', np.nan),
            'notes': ';'.join(notes)
        }
        rows.append(row)
        for ntag in notes:
            notes_counter[ntag] += 1
        # also tag success vs fallback
        notes_counter['succeeded_LMM' if res.get('success', False) else 'used_fallback'] += 1

    results_df = pd.DataFrame(rows)
    summary = dict(notes_counter)
    summary['n_features'] = p
    return results_df, summary

"""
"""
###################################
# FOR LONGITUDINAL DATA
####################################

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Sequence, Optional

### This function was written by Jake not Gaurav, adapted from above so may be redundant 
def Run_LMM_Longitudinal(Data, subject_ids, batch, covariates=None, feature_names=None,
                  subject_col_name='subject', batch_col_name='batch',
                  covariate_names=None, min_group_n=2, var_threshold=1e-8):
    """Runs LMM diagnostics, treating subject as random effect, batch and covariates as fixed effects
        Uses the LMM safe function for the fall backs and returns (results_df, summary).

    Args:
        Data: (n_samples, n_features) numpy array
        subject_ids: array-like of length n_samples with subject IDs
        batch: array-like of length n_samples with batch labels
        covariates: optional (n_samples, n_covariates) array-like
        feature_names: optional list of feature names
        subject_col_name: str, name for subject ID column in model
        batch_col_name: str, name for batch column in model
        covariate_names: optional list of covariate names
        min_group_n: minimum samples per group to fit LMM
        var_threshold: variance threshold to consider a variance component as non-zero

    
    """
    Data = np.asarray(Data, dtype=float)
    n, p = Data.shape
    if feature_names is None:
        feature_names = [f'feature_{i+1}' for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length mismatch.")

    # Build base DataFrame for model inputs, use same structure as cross sectional but include batch as fixed effect and subject as random effect
    df_base = pd.DataFrame({subject_col_name: np.asarray(subject_ids),
                            batch_col_name: np.asarray(batch)})
    if covariates is not None:
        if isinstance(covariates, pd.DataFrame):
            cov_df = covariates.reset_index(drop=True)
        else:
            cov_arr = np.asarray(covariates)
            if cov_arr.ndim == 1:
                cov_df = pd.DataFrame({covariate_names[0] if covariate_names else 'cov1': cov_arr})
            else:
                names = covariate_names if covariate_names else [f'cov{i+1}' for i in range(cov_arr.shape[1])]
                cov_df = pd.DataFrame(cov_arr, columns=names)
        df_base = pd.concat([df_base, cov_df], axis=1)
    
    # build formula (fixed part)
    fixed_terms = [batch_col_name]
    if covariates is not None and cov_df.shape[1] > 0:
        fixed_terms.extend(list(cov_df.columns))
    rhs = ' + '.join(fixed_terms)
    formula_fixed = f"y ~ {rhs}" # Here, y is column and rhs contains the fixed effects
    rows = []
    notes_counter = Counter()
    for fi in range(p):
        df = df_base.copy()
        df['y'] = Data[:, fi]
        res = fit_lmm_safe(df, formula_fixed, group_col=subject_col_name,
                           min_group_n=min_group_n, var_threshold=var_threshold)
        stats = res.get('stats', {})
        notes = res.get('notes', []) or []
        # ensure consistent fields
        row = {
            'feature': feature_names[fi],
            'success': bool(res.get('success', False)), # did LMM fit succeed?
            'var_fixed': stats.get('var_fixed', np.nan), # Variance explained by fixed effects
            'var_subject': stats.get('var_batch', np.nan), # Variance explained by subject random effect
            'var_resid': stats.get('var_resid', np.nan), # Residual variance
            'R2_marginal': stats.get('R2_marginal', np.nan), # Marginal R-squared (fixed effects)
            'R2_conditional': stats.get('R2_conditional', np.nan), # Conditional R-squared (fixed + random effects)
            'ICC': stats.get('ICC', np.nan), # Intraclass correlation coefficient (subject variance / total variance)
            'LR_stat': stats.get('LR_stat', np.nan),
            'pval_LRT_random': stats.get('pval_LRT_random', np.nan),
            'notes': ';'.join(notes)
        }
        rows.append(row)
        for ntag in notes:
            notes_counter[ntag] += 1
        # also tag success vs fallback: Same overall logic as cross-sectional, change what is fixed and what is random
        notes_counter['succeeded_LMM' if res.get('success', False) else 'used_fallback'] += 1
    results_df = pd.DataFrame(rows)
    summary = dict(notes_counter)
    summary['num_features_analyzed'] = p

    return results_df, summary

##########################################################################################################
# FOR LONGITUDINAL DATA: Subject Order Consistency using Spearman Correlations and Permutation Testing
##########################################################################################################

def _force_numeric_vector(series_like) -> np.ndarray:
    """Convert input to 1D float numpy array; non-convertible -> np.nan."""
    if isinstance(series_like, (pd.Series, pd.DataFrame)):
        arr = series_like.to_numpy().ravel()
    else:
        arr = np.asarray(series_like).ravel()
    try:
        return arr.astype(float)
    except Exception:
        out = np.empty(arr.shape, dtype=float)
        for i, v in enumerate(arr):
            try:
                out[i] = float(v)
            except Exception:
                out[i] = np.nan
        return out

def SubjectOrder_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    idp_names: Optional[Sequence[str]] = None,
    nPerm: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Pairwise Spearman with permutation testing using scipy.stats.spearmanr.

    Returns DataFrame with columns:
      ["TimeA","TimeB","IDP","nPairs","SpearmanRho","NullMeanRho","pValue"]
    pValue is computed with the recommended +1 correction:
      p = (1 + count_ge) / (1 + valid_null_count)
    """
    # Validate and normalize inputs
    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_idps = idp_matrix.shape

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")
    if len(timepoints) != n_samples:
        raise ValueError("Length of timepoints must match number of rows in idp_matrix.")

    if idp_names is None:
        idp_names = [f"idp_{i+1}" for i in range(n_idps)]
    else:
        idp_names = list(idp_names)
        if len(idp_names) != n_idps:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

    if not isinstance(nPerm, int) or nPerm < 1:
        raise ValueError("nPerm must be an integer >= 1.")

    subjects_arr = np.asarray(subjects).astype(str)
    timepoints_arr = np.asarray(timepoints).astype(str)

    # Preserve order of first appearance for timepoints
    tp_index = pd.Index(timepoints_arr)
    tp_levels = tp_index.unique().tolist()

    rng = default_rng(seed)

    rows = []
    for ia in range(len(tp_levels) - 1):
        for ib in range(ia + 1, len(tp_levels)):
            tpA = tp_levels[ia]; tpB = tp_levels[ib]

            idxA_all = np.nonzero(timepoints_arr == tpA)[0]
            idxB_all = np.nonzero(timepoints_arr == tpB)[0]

            if idxA_all.size == 0 or idxB_all.size == 0:
                for idp in idp_names:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp,
                                 "nPairs": 0, "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                continue

            Ta_subj = subjects_arr[idxA_all]
            Tb_subj = subjects_arr[idxB_all]

            # subjects in A that also appear in B
            maskA = np.isin(Ta_subj, Tb_subj)
            if not np.any(maskA):
                for idp in idp_names:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp,
                                 "nPairs": 0, "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                continue

            common_subj = Ta_subj[maskA]
            idxA = idxA_all[np.nonzero(maskA)[0]]

            # map first occurrence in Tb to global row indices
            tb_index_map = {}
            for i, val in enumerate(Tb_subj):
                if val not in tb_index_map:
                    tb_index_map[val] = idxB_all[i]
            idxB = np.array([tb_index_map[s] for s in common_subj], dtype=int)

            # iterate idps (columns)
            for j, idp_name in enumerate(idp_names):
                xa_raw = idp_matrix[idxA, j] if j < n_idps else np.array([], dtype=float)
                yb_raw = idp_matrix[idxB, j] if j < n_idps else np.array([], dtype=float)

                xa = _force_numeric_vector(xa_raw)
                yb = _force_numeric_vector(yb_raw)

                valid_mask = ~(np.isnan(xa) | np.isnan(yb))
                xa = xa[valid_mask]; yb = yb[valid_mask]
                nPairs = xa.size

                if nPairs < 3:
                    rows.append({"TimeA": tpA, "TimeB": tpB, "IDP": idp_name,
                                 "nPairs": int(nPairs), "SpearmanRho": np.nan,
                                 "NullMeanRho": np.nan, "pValue": np.nan})
                    continue

                # Compute observed Spearman directly (handles ties)
                obs_rho, _ = spearmanr(xa, yb, nan_policy="omit")
                abs_obs = None if np.isnan(obs_rho) else abs(obs_rho)

                # Permutation null distribution (permute yb within matched pairs)
                sum_null = 0.0
                valid_null_count = 0
                count_ge = 0

                for _ in range(nPerm):
                    perm_idx = rng.permutation(nPairs)
                    null_rho, _ = spearmanr(xa, yb[perm_idx], nan_policy="omit")
                    if not np.isnan(null_rho):
                        sum_null += null_rho
                        valid_null_count += 1
                        if abs_obs is not None and abs(null_rho) >= abs_obs:
                            count_ge += 1

                null_mean = float(sum_null / valid_null_count) if valid_null_count > 0 else np.nan
                pval = float((1 + count_ge) / (1 + valid_null_count)) if valid_null_count > 0 else float("nan")

                rows.append({
                    "TimeA": tpA,
                    "TimeB": tpB,
                    "IDP": idp_name,
                    "nPairs": int(nPairs),
                    "SpearmanRho": float(obs_rho) if not np.isnan(obs_rho) else np.nan,
                    "NullMeanRho": null_mean,
                    "pValue": pval,
                })

    results = pd.DataFrame(rows, columns=["TimeA", "TimeB", "IDP", "nPairs", "SpearmanRho", "NullMeanRho", "pValue"])
    return results

##########################################################################################################
# FOR LONGITUDINAL DATA: 2) Within subject variability: % difference (2 timepoints) OR CoV (>2 timepoints)
##########################################################################################################
# Add Tuple import to avoid error here:
from typing import Tuple, Dict, Any, List, Iterable

def WithinSubjVar_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    idp_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:

    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_idps = idp_matrix.shape

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")
    if len(timepoints) != n_samples:
        raise ValueError("Length of timepoints must match number of rows in idp_matrix.")

    if idp_names is None:
        idp_names = [f"idp_{i+1}" for i in range(n_idps)]
    else:
        idp_names = list(idp_names)
        if len(idp_names) != n_idps:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

    df = pd.DataFrame(idp_matrix, columns=idp_names)
    df["subject"] = list(subjects)
    # df["timepoint"] = list(timepoints)  # keep if you need it later

    out_rows = []
    for subj, g in df.groupby("subject", sort=False):
        row = {"subject": subj}
        for col in idp_names:
            arr = g[col].dropna().to_numpy(dtype=float)
            n = arr.size
            if n == 0:
                row[col] = np.nan
                continue
            mean_val = arr.mean()
            if mean_val == 0:
                row[col] = np.nan
                continue
            if n == 2:
                row[col] = float(abs(arr[0] - arr[1]) / mean_val * 100.0)
            elif n > 2:
                row[col] = float(arr.std(ddof=1) / mean_val * 100.0)
            else:
                row[col] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)

##########################################################################################################
# FOR LONGITUDINAL DATA: Multivariate pairwise site differences using Mahalanobis distances
##########################################################################################################

def MultiVariateBatchDifference_long(
    idp_matrix: np.ndarray,
    batch: pd.Series | Sequence,
    idp_names: Optional[Sequence[str]] = None,
    return_info: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute Mahalanobis distance of each batch/site mean from the overall mean in a numerically stable way.
    - idp_matrix: shape (n_samples, n_features) numeric
    - batch: Series-like of length n_samples giving the batch/site label for each row
    - idp_names: optional feature names
    - return_info: if True, also return diagnostic info dict
    Returns DataFrame with columns ["batch", "mdval"] (plus "average_batch" row).
    """
    # --- validate / coerce matrix ---
    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_features = idp_matrix.shape

    # --- validate batch ---
    if isinstance(batch, pd.Series):
        batch_ser = batch.astype("category")
    else:
        # try to create Series from provided sequence-like
        batch_ser = pd.Series(batch, dtype="category")
    if len(batch_ser) != n_samples:
        raise ValueError("Length of batch must match number of rows in idp_matrix.")

    # --- idp names ---
    if idp_names is None:
        idp_names = [f"idp_{i+1}" for i in range(n_features)]
    else:
        idp_names = list(idp_names)
        if len(idp_names) != n_features:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

    # build dataframe for convenience
    data = pd.DataFrame(idp_matrix, columns=idp_names)
    data["_batch"] = batch_ser.values  # keep parallel categorical

    cats = list(batch_ser.cat.categories)
    num_sites = len(cats)
    if num_sites == 0:
        raise ValueError("No batch categories found in `batch` input.")

    # prepare containers
    all_means = np.full((n_features, num_sites), np.nan, dtype=float)
    tmpCov = np.zeros((n_features, n_features), dtype=float)
    site_counts: List[int] = []

    # accumulate site covariances and means
    for i, lvl in enumerate(cats):
        mask = (data["_batch"] == lvl)
        site_df = data.loc[mask, idp_names]
        # drop rows with any NA across features (you may change policy)
        site_df_clean = site_df.dropna(axis=0, how="any")
        n_i = len(site_df_clean)
        site_counts.append(int(n_i))

        if n_i == 0:
            warnings.warn(f"Site '{lvl}' has zero retained samples after dropping NaNs.")
            cov_i = np.zeros((n_features, n_features), dtype=float)
            # leave all_means[:, i] as NaN
        else:
            mean_i = site_df_clean.to_numpy(dtype=float).mean(axis=0)
            all_means[:, i] = mean_i
            if n_i == 1:
                cov_i = np.zeros((n_features, n_features), dtype=float)
            else:
                cov_i = np.cov(site_df_clean.to_numpy(dtype=float), rowvar=False, ddof=1)
                # ensure shape is (n_features, n_features)
                cov_i = np.atleast_2d(cov_i)
                if cov_i.shape != (n_features, n_features):
                    # fallback to zeros if shapes mismatch
                    warnings.warn(f"Covariance for site '{lvl}' had unexpected shape {cov_i.shape}; using zeros.")
                    cov_i = np.zeros((n_features, n_features), dtype=float)

        tmpCov += cov_i

    # average covariance across sites (simple mean of site covariances)
    overallCov = tmpCov / float(num_sites)
    overallMean = np.nanmean(all_means, axis=1)  # shape (n_features,)

    # numeric stability check
    try:
        cond_number = np.linalg.cond(overallCov)
    except Exception:
        cond_number = np.inf

    info: Dict[str, Any] = {
        "site_categories": cats,
        "site_counts": site_counts,
        "cond_number": float(cond_number),
        "num_retained_svals": 0,
        "overallCov": overallCov,
    }

    # compute MD per site
    MD = np.full((num_sites,), np.nan, dtype=float)

    if not np.isfinite(cond_number) or cond_number > 1e15:
        # SVD-based pseudoinverse approach
        U, s, Vt = np.linalg.svd(overallCov, full_matrices=False)
        eps = np.finfo(float).eps
        tol = np.max(s) * max(overallCov.shape) * eps
        keep = s > tol
        s_inv = np.zeros_like(s)
        if keep.any():
            s_inv[keep] = 1.0 / s[keep]
        overallCov_pinv = (Vt.T * s_inv) @ U.T
        num_retained = int(np.sum(keep))
        info["num_retained_svals"] = num_retained
        # compute distances
        for i in range(num_sites):
            mu_i = all_means[:, i]
            if np.any(np.isnan(mu_i)):
                MD[i] = np.nan
                continue
            diff = mu_i - overallMean
            delta = float(diff.T @ overallCov_pinv @ diff)
            MD[i] = float(np.sqrt(max(delta, 0.0)))
    else:
        # stable to solve
        for i in range(num_sites):
            mu_i = all_means[:, i]
            if np.any(np.isnan(mu_i)):
                MD[i] = np.nan
                continue
            diff = mu_i - overallMean
            try:
                sol = np.linalg.solve(overallCov, diff)
                delta = float(diff.T @ sol)
                MD[i] = float(np.sqrt(max(delta, 0.0)))
            except np.linalg.LinAlgError:
                warnings.warn("overallCov singular during solve; falling back to pseudoinverse.")
                overallCov_pinv = np.linalg.pinv(overallCov)
                delta = float(diff.T @ overallCov_pinv @ diff)
                MD[i] = float(np.sqrt(max(delta, 0.0)))

    mean_md = float(np.nanmean(MD)) if np.any(np.isfinite(MD)) else np.nan
    site_labels = [str(c) for c in cats] + ["average_batch"]
    mdvals = np.concatenate([MD, np.array([mean_md], dtype=float)])
    fullMDtab = pd.DataFrame({"batch": site_labels, "mdval": mdvals})

    if return_info:
        return fullMDtab, info
    return fullMDtab

###############################################################################################################################################
# FOR LONGITUDINAL DATA: 1) Various mixed effects models - 
# mean comparison for - 
# 4) overall batch 
# 5) pairwise batches 
# 6) cross subject variability (ICC)
# 7) biological variability for fixed effects e.g., age, timepoint
################################################################################################################################################

def _force_categorical(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            if not isinstance(df[c].dtype, CategoricalDtype):
                df[c] = df[c].astype("category")

def _force_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            if not not isinstance(df[c].dtype, CategoricalDtype):
                s = df[c].astype(str)
                extracted = s.str.extract(r'(\d+)$', expand=False)
                if extracted.notna().all():
                    vals = extracted.astype(float)
                    vals = vals - vals.min()
                    df[c] = vals
                else:
                    df[c] = pd.Categorical(s).codes.astype(float)

def _zscore_columns(df: pd.DataFrame, vars_to_zscore: Iterable[str]) -> None:
    for v in vars_to_zscore:
        if v not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[v]):
            mu = df[v].mean(skipna=True); sigma = df[v].std(skipna=True)
            zname = f"zscore_{v}"
            if pd.isna(sigma) or sigma == 0:
                df[zname] = 0.0
            else:
                df[zname] = (df[v] - mu) / sigma

def build_mixed_formula(
    tbl_in: pd.DataFrame,
    response_var: str,
    fix_eff: Iterable[str],
    ran_eff: Iterable[str],
    batch_vars: Iterable[str],
    force_categorical: Iterable[str] = (),
    force_numeric: Iterable[str] = (),
    zscore_vars: Iterable[str] = (),
    zscore_response: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build three formulas:
      formulas[0] : full model (fixed terms + batch_vars + random terms)
      formulas[1] : subject-only/random-only (lhs ~ 1 + (1|<first_random_var>)) if random terms present,
                    otherwise lhs ~ 1
      formulas[2] : fixed-effects-only (without batch terms) optionally + random terms

    Returns modified df (with forced types / zscores) and list of formulas.
    """
    df = tbl_in.copy()
    fix_eff = list(fix_eff)
    ran_eff = list(ran_eff)
    batch_vars = list(batch_vars)
    force_categorical = list(force_categorical)
    force_numeric = list(force_numeric)
    zscore_vars = list(zscore_vars)

    _zscore_columns(df, zscore_vars)

    def present(name: str) -> bool:
        return name in df.columns

    def _maybe_use_zscore(v: str) -> str:
        zname = f"zscore_{v}"
        return zname if (zscore_response and zname in df.columns) else v

    lhs = _maybe_use_zscore(response_var)

    _force_categorical(df, batch_vars)
    _force_categorical(df, force_categorical)
    _force_numeric(df, force_numeric)

    # build fixed terms (include batch_vars by default after fix effs)
    fixed_terms: List[str] = []
    for v in fix_eff:
        use_name = _maybe_use_zscore(v)
        if present(use_name):
            fixed_terms.append(use_name)
    for v in batch_vars:
        use_name = _maybe_use_zscore(v)
        if present(use_name) and use_name not in fixed_terms:
            fixed_terms.append(use_name)

    # dedupe while preserving order
    seen = set()
    fixed_terms = [x for x in fixed_terms if not (x in seen or seen.add(x))]

    fixed_str_with_batch = "1" if len(fixed_terms) == 0 else " + ".join(fixed_terms)

    batch_like = set(batch_vars) | {f"zscore_{b}" for b in batch_vars}
    fixed_no_batch = [t for t in fixed_terms if t not in batch_like]
    fixed_str_no_batch = "1" if len(fixed_no_batch) == 0 else " + ".join(fixed_no_batch)

    # random terms -> (1|<var>)
    rand_terms = []
    for v in ran_eff:
        if present(v):
            if not isinstance(df[v].dtype, CategoricalDtype):
                df[v] = df[v].astype("category")
            rand_terms.append(f"(1|{v})")

    # construct formulas:
    if len(rand_terms) == 0:
        # no random effects
        formulas = [
            f"{lhs} ~ {fixed_str_with_batch}",
            f"{lhs} ~ 1",
            f"{lhs} ~ {fixed_str_no_batch}",
        ]
    else:
        rand_str = " + ".join(rand_terms)
        # For model 2, prefer a subject-only random intercept using the *first* valid ran_eff variable
        subj_rand = rand_terms[0]  # e.g. "(1|subjects)"
        formulas = [
            f"{lhs} ~ {fixed_str_with_batch} + {rand_str}",
            f"{lhs} ~ 1 + {subj_rand}",
            f"{lhs} ~ {fixed_str_no_batch} + {rand_str}",
        ]
    return df, formulas


def pairwise_site_tests(
    fit_result: MixedLMResults,
    group_var: str,
    data_frame: pd.DataFrame,
    alpha: float = 0.05,
    debug: bool = False,
) -> Tuple[int, pd.DataFrame]:
    if group_var not in data_frame.columns:
        raise KeyError(f"group var '{group_var}' not in data")
    if not isinstance(data_frame[group_var].dtype, CategoricalDtype):
        data_frame[group_var] = data_frame[group_var].astype("category")
    cats = list(data_frame[group_var].cat.categories)
    if len(cats) < 2:
        return 0, pd.DataFrame(columns=["siteA", "siteB", "p", "sig"])
    full_param_names = list(fit_result.params.index)
    exog_names = getattr(fit_result.model, "exog_names", None)
    if exog_names is None:
        exog_names = full_param_names.copy()
    if debug:
        print("PAIRWISE (WALD) DEBUG: full_param_names:", full_param_names)
        print("PAIRWISE (WALD) DEBUG: exog_names:", exog_names)
        print("PAIRWISE (WALD) DEBUG: categories:", cats)
    exog_to_idx = {name: i for i, name in enumerate(exog_names)}
    level_to_exog_idx = {}
    for lvl in cats:
        patt = f"[T.{lvl}]"
        found_exog = None
        for en in exog_names:
            if patt in en:
                found_exog = en; break
        if found_exog is None:
            for en in exog_names:
                if f"{group_var}_{lvl}" in en or en.endswith(f"_{lvl}") or re.search(rf"\b{re.escape(lvl)}\b", en):
                    found_exog = en; break
        level_to_exog_idx[lvl] = exog_to_idx[found_exog] if found_exog is not None else None
    if debug:
        print("PAIRWISE (WALD) DEBUG: level -> exog_idx mapping:", level_to_exog_idx)
    beta = fit_result.params.to_numpy(dtype=float)
    try:
        cov = fit_result.cov_params()
        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
    except Exception:
        raise RuntimeError("Could not obtain covariance matrix from fit_result.cov_params()")
    rows = []; sig_flags = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a = cats[i]; b = cats[j]
            ex_idx_a = level_to_exog_idx.get(a); ex_idx_b = level_to_exog_idx.get(b)
            contrast_exog = np.zeros(len(exog_names), dtype=float)
            if ex_idx_a is not None: contrast_exog[ex_idx_a] = 1.0
            if ex_idx_b is not None: contrast_exog[ex_idx_b] = -1.0
            contrast_full = np.zeros(len(full_param_names), dtype=float)
            for k, exog_name in enumerate(exog_names):
                if exog_name in full_param_names:
                    pidx = full_param_names.index(exog_name)
                else:
                    pidx = None
                    for t_i, pname in enumerate(full_param_names):
                        if re.search(rf"\b{re.escape(exog_name)}\b", str(pname)) or re.search(rf"\b{re.escape(exog_name.split('[')[0])}\b", str(pname)):
                            pidx = t_i; break
                if pidx is not None:
                    contrast_full[pidx] = contrast_exog[k]
            if np.allclose(contrast_full, 0):
                pval = float("nan")
            else:
                est = float(np.dot(contrast_full, beta))
                var = float(contrast_full @ cov_mat @ contrast_full.T)
                if var <= 0 or np.isnan(var): pval = float("nan")
                else:
                    z = est / np.sqrt(var); pval = 2.0 * (1.0 - norm.cdf(abs(z)))
            sig = int(pval < alpha) if (not np.isnan(pval)) else 0
            rows.append({"siteA": a, "siteB": b, "p": pval, "sig": sig})
            sig_flags.append(sig)
            if debug:
                print(f"PAIRWISE (WALD) DEBUG: {a} vs {b} -> ex_idx_a={ex_idx_a}, ex_idx_b={ex_idx_b}, p={pval}, sig={sig}")
    full_tab = pd.DataFrame(rows, columns=["siteA", "siteB", "p", "sig"])
    return int(np.nansum(sig_flags)), full_tab

def _extract_numeric_coeff_scalar(res, varname: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    params = res.params; pvals = res.pvalues; conf = res.conf_int()
    candidates = [varname, f"zscore_{varname}"]; found_name = None
    for cand in candidates:
        if cand in params.index:
            found_name = cand; break
    if found_name is None:
        for pname in params.index:
            if re.search(rf"\b{re.escape(varname)}\b", str(pname)):
                found_name = pname; break
    if found_name is None:
        exog_names = getattr(res.model, "exog_names", None)
        if exog_names:
            for en in exog_names:
                if re.search(rf"\b{re.escape(varname)}\b", str(en)):
                    if en in params.index:
                        found_name = en; break
    if found_name is None:
        out[f"{varname}_est"] = np.nan; out[f"{varname}_pval"] = np.nan; out[f"{varname}_ciL"] = np.nan; out[f"{varname}_ciU"] = np.nan
        return out
    est = float(params.get(found_name, np.nan))
    pval = float(pvals.get(found_name, np.nan)) if found_name in pvals.index else np.nan
    if found_name in conf.index:
        ciL, ciU = conf.loc[found_name].values
    else:
        ciL = ciU = np.nan
    out[f"{varname}_est"] = est; out[f"{varname}_pval"] = pval; out[f"{varname}_ciL"] = ciL; out[f"{varname}_ciU"] = ciU
    return out

model_defs = []
def MixedEffects_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    batches: Sequence,
    idp_names: Sequence,
    *,
    covariates: Optional[Dict[str, Sequence]] = None,
    fix_eff: Sequence = (),
    ran_eff: Sequence = (),
    force_categorical: Sequence = (),
    force_numeric: Sequence = (),
    zscore_var: Sequence = (),
    do_zscore: bool = True,
    p_thr: float = 0.05,
    p_corr: int = 1,
    reml: bool = True,
) -> pd.DataFrame:
    """
    Mixed-effects pipeline with canonical column names:
      - subjects -> 'subjects'
      - timepoints -> 'timepoints'
      - batches   -> 'batches'
    Users should refer to those exact names in fix_eff / ran_eff / force_* lists.

    Parameters
    ----------
    idp_matrix : (n_samples, n_idps) array-like
    subjects : sequence (length n_samples)
    timepoints : sequence (length n_samples)
    batches : sequence (length n_samples)
    idp_names : sequence of length n_idps
    covariates : optional dict name -> sequence (length n_samples)
    fix_eff, ran_eff, force_categorical, force_numeric, zscore_var : lists of names
    do_zscore : whether to use zscore_... columns when available
    p_thr, p_corr, reml : modeling options

    Returns
    -------
    pd.DataFrame with one row per IDP summarizing diagnostics and fixed-effect estimates.
    """
    # --- basic validation and coerce idp_matrix ---
    if not isinstance(idp_matrix, np.ndarray):
        idp_matrix = np.asarray(idp_matrix, dtype=float)
    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
    n_samples, n_features = idp_matrix.shape

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")
    if len(timepoints) != n_samples:
        raise ValueError("Length of timepoints must match number of rows in idp_matrix.")
    if len(batches) != n_samples:
        raise ValueError("Length of batches must match number of rows in idp_matrix.")

    idp_names = list(idp_names)
    if len(idp_names) != n_features:
        raise ValueError("idp_names length must match number of idp columns.")

    # canonical column names exposed to caller
    subjects_col = "subjects"
    timepoints_col = "timepoints"
    batches_col = "batches"

    # prepare covariates dict
    covariates = dict(covariates or {})
    for k, seq in covariates.items():
        if len(seq) != n_samples:
            raise ValueError(f"Covariate '{k}' length ({len(seq)}) does not match n_samples ({n_samples}).")

    # build master df using canonical names (so users can refer to those names later)
    df = pd.DataFrame(idp_matrix, columns=idp_names)
    df[subjects_col] = pd.Series(subjects).astype(str)
    df[timepoints_col] = pd.Series(timepoints).astype(str)
    df[batches_col] = pd.Series(batches).astype(str).astype("category")

    # insert covariates verbatim (keys used as column names)
    for k, seq in covariates.items():
        df[k] = pd.Series(seq)

    # normalize caller lists
    fix_eff = list(fix_eff or [])
    ran_eff = list(ran_eff or [])
    force_categorical = list(force_categorical or [])
    force_numeric = list(force_numeric or [])
    zscore_var = list(zscore_var or [])

    # Defaults:
    # - ran_eff defaults to ['subjects'] if user didn't provide any
    if len(ran_eff) == 0:
        if subjects_col in df.columns:
            ran_eff = [subjects_col]
        else:
            raise KeyError("ran_eff not provided and no 'subjects' column found in data.")

    # - fix_eff defaults to covariate keys + timepoints + batches (use exact names)
    if len(fix_eff) == 0:
        inferred_fix = list(covariates.keys())
        fix_eff = inferred_fix

    # - infer force_numeric / force_categorical from covariates if none provided
    if len(force_numeric) == 0 and len(force_categorical) == 0:
        for k in covariates.keys():
            ser = pd.Series(covariates[k])
            if pd.api.types.is_numeric_dtype(ser):
                force_numeric.append(k)
            else:
                force_categorical.append(k)
        # treat timepoints and batches as categorical by default
        if timepoints_col in df.columns and timepoints_col not in force_categorical:
            force_categorical.append(timepoints_col)
        if batches_col in df.columns and batches_col not in force_categorical:
            force_categorical.append(batches_col)

    # final validation: any referenced names must exist in df
    to_check = {
        "fix_eff": fix_eff,
        "ran_eff": ran_eff,
        "force_categorical": force_categorical,
        "force_numeric": force_numeric,
        "zscore_var": zscore_var,
    }
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}")

    outs: List[Dict[str, Any]] = []

    # iterate over IDPs and fit 3-model pipeline per IDP
    for tmpidp in idp_names:
        # pick needed columns: random effects, batch, fixed effects, idp
        cols_needed: List[str] = []
        cols_needed += [c for c in ran_eff if c in df.columns]
        cols_needed += [batches_col]
        cols_needed += [c for c in fix_eff if c in df.columns]
        cols_needed += [tmpidp]
        # dedupe while preserving order
        seen = set()
        cols_needed = [x for x in cols_needed if not (x in seen or seen.add(x))]

        all_data = df[cols_needed].copy()

        # include current idp in zscore list for local preprocessing
        zscore_vars_local = list(zscore_var) + [tmpidp]
        all_data, formulas = build_mixed_formula(
            all_data,
            response_var=tmpidp,
            fix_eff=fix_eff,
            ran_eff=ran_eff,
            batch_vars=[batches_col],
            force_categorical=force_categorical,
            force_numeric=force_numeric,
            zscore_vars=zscore_vars_local,
            zscore_response=do_zscore,
        )
        print(f"\nMixedEffects_long — IDP: {tmpidp}")
        print("  Model 1 (full):")
        print(f"    {formulas[0]}")
        print("  Model 2 (subject-only / null):")
        print(f"    {formulas[1]}")
        print("  Model 3 (no batch):")
        print(f"    {formulas[2]}")

        model_def = {
            "Feature": tmpidp,
            "models": {
                "full": formulas[0],
                "subject_only": formulas[1],
                "no_batch": formulas[2],
                }
                }

        model_defs.append(model_def)

        # reorder batch levels so largest group is reference
        all_data[batches_col] = all_data[batches_col].astype("category")
        counts = all_data[batches_col].value_counts()
        if len(counts) > 0:
            ref_site = counts.idxmax()
            current_cats = list(all_data[batches_col].cat.categories)
            if ref_site not in current_cats:
                ref_site = current_cats[0]
            new_categories = [ref_site] + [c for c in current_cats if c != ref_site]
            try:
                all_data[batches_col] = all_data[batches_col].cat.reorder_categories(new_categories, ordered=False)
            except Exception:
                all_data[batches_col] = all_data[batches_col].astype("category")

        # prepare output dict for this IDP
        rowd: Dict[str, Any] = {}
        rowd["IDP"] = tmpidp.replace("_", "-")
        rowd["batch"] = batches_col

        # Model 1: full (fixed with batch + random terms)
        fixed_formula_full = formulas[0]
        ml_formula_full = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", fixed_formula_full)

        res1 = None
        try:
            # groups: prefer first ran_eff if present, else use subjects_col
            if len(ran_eff) > 0 and ran_eff[0] in all_data.columns:
                groups_col = all_data[ran_eff[0]]
            elif subjects_col in all_data.columns:
                groups_col = all_data[subjects_col]
            else:
                groups_col = None

            if groups_col is None:
                raise RuntimeError("No valid grouping column for mixed model; skipping model fits.")

            mdl1 = smf.mixedlm(ml_formula_full, all_data, groups=groups_col)
            res1 = mdl1.fit(reml=reml, method="lbfgs")
        except Exception:
            # fill placeholders and continue to next IDP
            rowd.update({
                "n_is_batchSig": np.nan,
                "anova_batches": np.nan,
                "Subj_Var": np.nan,
                "Resid_Var": np.nan,
                "ICC": np.nan,
                "WCV": np.nan
            })
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan; rowd[f"{v}_pval"] = np.nan; rowd[f"{v}_ciL"] = np.nan; rowd[f"{v}_ciU"] = np.nan
            outs.append(rowd)
            continue

        # Pairwise batch/site tests
        try:
            n_sig, full_tab = pairwise_site_tests(res1, batches_col, all_data, alpha=p_thr, debug=False)
        except Exception:
            n_sig = 0; full_tab = pd.DataFrame(columns=["siteA", "siteB", "p", "sig"])

        # multiple-comparison handling
        if p_corr == 0:
            rowd["n_is_batchSig"] = int(n_sig)
        else:
            tmpsig = full_tab["p"].to_numpy(dtype=float)
            tmpsig_nonan = tmpsig[~np.isnan(tmpsig)]
            if len(tmpsig_nonan) > 0:
                p_corr_thr = 0.05 / len(tmpsig_nonan)
                rowd["n_is_batchSig"] = int(np.sum(tmpsig_nonan < p_corr_thr))
            else:
                rowd["n_is_batchSig"] = 0

        # ANOVA-like count of batch fixed-effect p < 0.05
        try:
            fe_pvals = res1.pvalues
            batch_mask = [bool(re.search(rf"{re.escape(str(batches_col))}", str(name))) for name in fe_pvals.index]
            anova_batches = int(np.sum(fe_pvals[batch_mask] < 0.05)) if any(batch_mask) else 0
        except Exception:
            anova_batches = np.nan
        rowd["anova_batches"] = anova_batches

        # Model 2: subject-only random -> variance components (ICC/WCV)
        try:
            formula2_raw = formulas[1]
            formula2_fixed = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formula2_raw)
            groups_col2 = all_data[ran_eff[0]] if (len(ran_eff) > 0 and ran_eff[0] in all_data.columns) else all_data[subjects_col] if subjects_col in all_data.columns else None
            if groups_col2 is None:
                raise RuntimeError("No grouping column available for subject-only random model.")
            mdl2 = smf.mixedlm(formula=formula2_fixed, data=all_data, groups=groups_col2)
            res2 = mdl2.fit(reml=reml, method="lbfgs")

            def _extract_subj_var(res_obj):
                try:
                    cov_re = getattr(res_obj, "cov_re", None)
                    if cov_re is None:
                        return np.nan
                    arr = np.asarray(cov_re)
                    if arr.size == 0:
                        return np.nan
                    return float(arr.ravel()[0])
                except Exception:
                    try:
                        return float(res_obj.cov_re.iloc[0, 0])
                    except Exception:
                        return np.nan

            subj_var = _extract_subj_var(res2)
            resid_var = float(getattr(res2, "scale", np.nan))

            # robust re-fit attempts if subj_var degenerate
            if subj_var == 0 or (isinstance(subj_var, float) and np.isfinite(subj_var) and subj_var < 1e-12):
                tried_ok = False
                for method_try in ("lbfgs", "powell", "nm"):
                    try:
                        res2_try = mdl2.fit(reml=False, method=method_try, maxiter=5000)
                        subj_var_try = _extract_subj_var(res2_try)
                        resid_var_try = float(getattr(res2_try, "scale", np.nan))
                        if np.isfinite(subj_var_try) and subj_var_try > 0 and not np.isnan(resid_var_try):
                            res2 = res2_try
                            subj_var = subj_var_try
                            resid_var = resid_var_try
                            tried_ok = True
                            break
                    except Exception:
                        continue
                if not tried_ok and subj_var == 0:
                    subj_var = np.nan

            rowd["Subj_Var"] = subj_var
            rowd["Resid_Var"] = resid_var
            try:
                if np.isfinite(subj_var) and np.isfinite(resid_var) and subj_var > 0:
                    rowd["ICC"] = subj_var / (subj_var + resid_var)
                    rowd["WCV"] = resid_var / subj_var
                else:
                    rowd["ICC"] = np.nan; rowd["WCV"] = np.nan
            except Exception:
                rowd["ICC"] = np.nan; rowd["WCV"] = np.nan
        except Exception:
            rowd["Subj_Var"] = np.nan; rowd["Resid_Var"] = np.nan; rowd["ICC"] = np.nan; rowd["WCV"] = np.nan

        # Model 3: fixed-effects only -> extract coefficients for fix_eff
        try:
            formula3_raw = formulas[2]
            formula3_fixed = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formula3_raw)
            groups_col3 = all_data[ran_eff[0]] if (len(ran_eff) > 0 and ran_eff[0] in all_data.columns) else all_data[subjects_col] if subjects_col in all_data.columns else None
            if groups_col3 is None:
                raise RuntimeError("No grouping column for model 3.")
            mdl3 = smf.mixedlm(formula=formula3_fixed, data=all_data, groups=groups_col3)
            res3 = mdl3.fit(reml=reml, method="lbfgs")

            for v in fix_eff:
                pname = f"zscore_{v}" if f"zscore_{v}" in res3.params.index else v
                coeff_dict = _extract_numeric_coeff_scalar(res3, pname)
                cleaned = {k.replace(pname, v): val for k, val in coeff_dict.items()}
                rowd.update(cleaned)
        except Exception:
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan; rowd[f"{v}_pval"] = np.nan; rowd[f"{v}_ciL"] = np.nan; rowd[f"{v}_ciU"] = np.nan

        outs.append(rowd)

    # assemble results DataFrame with stable column order
    if len(outs) == 0:
        return pd.DataFrame()
    first = outs[0]
    mdlnames = [k for k in first.keys() if k.endswith("_est") or k.endswith("_pval") or k.endswith("_ciL") or k.endswith("_ciU")]
    col_order = ["IDP", "batch", "n_is_batchSig", "anova_batches", "Subj_Var", "Resid_Var", "ICC", "WCV"] + mdlnames
    rows_df = pd.DataFrame(outs)
    for c in col_order:
        if c not in rows_df.columns:
            rows_df[c] = np.nan
    return rows_df[col_order], model_defs


##########################################################################################################
# FOR LONGITUDINAL DATA: 
# Additive batch effects - can 'batch' add additional variance in comparison to no 'batch'
# Multiplicative batch effects - comparing variance across batches
##########################################################################################################

# -------------------------
# Helper: build fixed terms
# -------------------------
def _build_fixed_formula_terms(fix_eff: Sequence[str], data: pd.DataFrame, do_zscore_predictors: bool = True) -> List[str]:
    """
    For each predictor v in fix_eff that is numeric in `data`, create zscore_v in `data` (per-feature/local df).
    Then return list of predictor column names to use in the formula, preferring zscore_v if present.
    """
    terms: List[str] = []
    if do_zscore_predictors and fix_eff is not None:
        for v in fix_eff:
            if v in data.columns and pd.api.types.is_numeric_dtype(data[v]):
                zname = f"zscore_{v}"
                if zname not in data.columns:
                    mu = data[v].mean(skipna=True)
                    sd = data[v].std(skipna=True)
                    if pd.isna(sd) or sd == 0:
                        data[zname] = 0.0
                    else:
                        data[zname] = (data[v] - mu) / sd
    for v in fix_eff or []:
        z = f"zscore_{v}"
        if do_zscore_predictors and z in data.columns:
            use = z
        else:
            use = v
        if use in data.columns:
            terms.append(use)
    return terms

# -------------------------
# Helper: safe fit wrapper
# -------------------------
def _safe_fit_mixedlm(formula_fixed: str, data: pd.DataFrame, group: str, reml: bool = False):
    if group not in data.columns:
        raise KeyError(f"group '{group}' not in data")
    data = data.copy()
    try:
        mdl = smf.mixedlm(formula_fixed, data, groups=data[group])
        res = mdl.fit(reml=reml, method="lbfgs")
        return res
    except Exception as e:
        warnings.warn(f"MixedLM fit failed for formula '{formula_fixed}': {e}")
        raise

# ----------------------------------------
# Helper: build input df
# ----------------------------------------
def _build_input_df_if_needed(
    data: Optional[pd.DataFrame],
    idp_matrix: Optional[np.ndarray],
    subjects: Optional[Sequence],
    timepoints: Optional[Sequence],
    batch_name: Optional[Sequence],
    idp_names: Optional[Iterable[str]],
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Build or return a DataFrame for modeling and return the actual column names used:
      (df, idvar, batchvar, timevar).
    Defaults to idvar='subjects', batchvar='batches', timevar='timepoints' if not provided.
    """
    idvar = idvar if idvar is not None else "subject_ids"
    batchvar = batchvar if batchvar is not None else "batch"
    timevar = timevar if timevar is not None else "timepoints"

    if data is not None:
        df = data.copy()
    else:
        if idp_matrix is None:
            raise ValueError("Either `data` or `idp_matrix` must be provided.")
        if subjects is None:
            raise ValueError("`subjects` must be provided when `data` is None.")
        if batch_name is None:
            raise ValueError("`batch_name` must be provided when `data` is None.")
        if not isinstance(idp_matrix, np.ndarray):
            idp_matrix = np.asarray(idp_matrix, dtype=float)
        if idp_matrix.ndim != 2:
            raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")
        n_samples, n_features = idp_matrix.shape
        if len(subjects) != n_samples:
            raise ValueError("Length of subjects must match idp_matrix rows.")
        if len(batch_name) != n_samples:
            raise ValueError("Length of batch_name must match idp_matrix rows.")
        if timepoints is not None and len(timepoints) != n_samples:
            raise ValueError("Length of timepoints must match idp_matrix rows.")
        if idp_names is None:
            idp_names = [f"idp_{i+1}" for i in range(n_features)]
        else:
            idp_names = list(idp_names)
            if len(idp_names) != n_features:
                raise ValueError("idp_names length must match idp_matrix.shape[1].")
        df = pd.DataFrame(idp_matrix, columns=idp_names)
        df[idvar] = pd.Series(subjects, dtype="object").values
        if timepoints is not None:
            df[timevar] = pd.Series(timepoints, dtype="object").values
        df[batchvar] = pd.Series(batch_name, dtype="object").values

    covariates = dict(covariates or {})
    if covariates:
        for cname, seq in covariates.items():
            if cname in df.columns:
                warnings.warn(f"Covariate '{cname}' already exists in DataFrame; skipping insertion.")
                continue
            seq = list(seq)
            if len(seq) != len(df):
                raise ValueError(f"Length of covariate '{cname}' ({len(seq)}) does not match number of rows ({len(df)}).")
            df[cname] = pd.Series(seq).values

    if batchvar in df.columns:
        try:
            df[batchvar] = df[batchvar].astype("category")
        except Exception:
            pass

    return df, idvar, batchvar, timevar

# ----------------------------------------
# Main: AdditiveEffect_long (per-feature zscore behavior)
# ----------------------------------------
model_defs_add=[]
def AdditiveEffect_long(
    data: Optional[pd.DataFrame] = None,
    idp_matrix: Optional[np.ndarray] = None,
    subjects: Optional[Sequence] = None,
    timepoints: Optional[Sequence] = None,
    batch_name: Optional[Sequence] = None,
    idp_names: Optional[Iterable[str]] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
    *,
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    fix_eff: Optional[Iterable[str]] = None,
    ran_eff: Optional[Iterable[str]] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test additive batch effect per feature (IDP), per-feature zscoring rules:

    - Numeric fixed predictors are z-scored per-feature (always).
    - If do_zscore == True: response (feature) is z-scored per-feature as well.
    - If do_zscore == False: response is left in original units.
    - Defaults for names: idvar='subjects', timevar='timepoints', batchvar='batches',
      unless the caller overrides idvar/batchvar/timevar.
    """
    # Build dataframe and canonical column names
    df, idcol, batchcol, tpcol = _build_input_df_if_needed(
        data=data,
        idp_matrix=idp_matrix,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batch_name,
        idp_names=idp_names,
        idvar=idvar,
        batchvar=batchvar,
        timevar=timevar,
        covariates=covariates,
    )

    covariates = dict(covariates or {})
    if fix_eff is None:
        fix_eff = list(covariates.keys())
    else:
        fix_eff = list(fix_eff)

    # RANDOM EFFECTS: infer only when caller omitted ran_eff (is None).
    if ran_eff is None:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    else:
        ran_eff = list(ran_eff)

    # Validate referenced names exist
    to_check = {"fix_eff": fix_eff, "ran_eff": ran_eff}
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}")

    # Determine feature columns (IDPs)
    exclude = {idcol, batchcol, tpcol}
    exclude |= set(covariates.keys())
    if idp_names is not None:
        feature_cols = list(idp_names)
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    V = len(feature_cols)
    if verbose:
        print(f"[AdditiveEffect_long] found {V} features")

    rows: List[Dict[str, Any]] = []
    for idx, feat in enumerate(feature_cols, 1):
        if verbose:
            print(f"[AdditiveEffect_long] ({idx}/{V}) testing additive batch effect for feature: {feat}")

        # Build local df for this feature (so per-feature zscoring is clean)
        # include feature, predictors, batch column, and grouping columns
        local_cols = [feat] + list(fix_eff) + [batchcol] + ran_eff
        local_df = df.loc[:, [c for c in local_cols if c in df.columns]].copy()

        # Drop rows where response is NaN (so zscores and fits use valid rows only)
        local_df = local_df[~local_df[feat].isna()].copy()
        if local_df.shape[0] < 3:
            if verbose:
                print(f"  skipping {feat}: too few non-NaN rows ({local_df.shape[0]})")
            rows.append({"Feature": feat, "TestStat": np.nan, "df": np.nan, "p-value": np.nan, "method": None})
            continue

        # Always z-score numeric predictors per-feature (local_df)
        # _build_fixed_formula_terms will create zscore_<v> for numeric fix_eff
        fixed_terms = _build_fixed_formula_terms(list(fix_eff or []), local_df, do_zscore_predictors=True)
        missing_fix = [vv for vv in (fix_eff or []) if (f"zscore_{vv}" not in local_df.columns) and (vv not in local_df.columns)]
        if missing_fix:
            warnings.warn(f"Fixed-effect columns not found in data (or zscore missing): {missing_fix}")

        # If do_zscore True -> z-score response per-feature and use z_<feat> as LHS
        if do_zscore:
            zresp = f"z_{feat}"
            if zresp not in local_df.columns:
                mu_r = local_df[feat].mean(skipna=True)
                sd_r = local_df[feat].std(skipna=True)
                if pd.isna(sd_r) or sd_r == 0:
                    local_df[zresp] = 0.0
                else:
                    local_df[zresp] = (local_df[feat] - mu_r) / sd_r
            lhs = zresp
        else:
            lhs = feat  # keep original units

        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{lhs} ~ {fixed_str} + C({batchcol})"
        reduced_fixed = f"{lhs} ~ {fixed_str}"

        # right before fitting
        if verbose:
            print("\n[MODEL]")
            print("Feature:", feat)
            print("Full model :", full_fixed)
            print("Reduced    :", reduced_fixed)
            print("Random eff :", ran_eff)

        model_defs_add.append ({
            "Feature": feat,
            "full_formula": full_fixed,
            "reduced_formula": reduced_fixed,
            "fix_eff": list(fix_eff),
            "ran_eff": list(ran_eff),})

        res_full = res_red = None
        group_name = ran_eff[0]
        try:
            res_full = _safe_fit_mixedlm(full_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            if verbose:
                print(f"  full fit failed for {feat}: {e}")
        try:
            res_red = _safe_fit_mixedlm(reduced_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            if verbose:
                print(f"  reduced fit failed for {feat}: {e}")

        LR = np.nan; df_stat = np.nan; pval = np.nan; used = None

        # Likelihood ratio test if both fits available
        if (res_full is not None) and (res_red is not None):
            try:
                llf_full = float(getattr(res_full, "llf", np.nan))
                llf_red = float(getattr(res_red, "llf", np.nan))
                if np.isfinite(llf_full) and np.isfinite(llf_red):
                    LR = 2.0 * (llf_full - llf_red)
                    try:
                        n_levels = int(pd.Categorical(local_df[batchcol]).nunique())
                        df_stat = max(n_levels - 1, 1)
                    except Exception:
                        df_stat = np.nan
                    if not np.isnan(df_stat):
                        pval = float(1.0 - chi2.cdf(LR, df_stat))
                    used = "LRT"
            except Exception as e:
                if verbose:
                    print(f"  LRT computation failed for {feat}: {e}")

        # If LRT didn't yield finite pval, try Wald-like test using full model params
        if not np.isfinite(pval) and (res_full is not None):
            try:
                pnames = list(res_full.params.index)
                batch_param_indices = []
                for i, pn in enumerate(pnames):
                    if (f"C({batchcol})" in pn) or (f"{batchcol}[T." in pn) or pn.startswith(f"{batchcol}_"):
                        batch_param_indices.append(i)
                if len(batch_param_indices) == 0:
                    cats = pd.Categorical(local_df[batchcol]).categories
                    for i, pn in enumerate(pnames):
                        for lvl in cats:
                            if f"{lvl}" in str(pn) and (batchcol in pn or f"C({batchcol})" in pn):
                                batch_param_indices.append(i)
                                break
                if len(batch_param_indices) > 0:
                    beta = res_full.params.to_numpy(dtype=float)
                    cov = res_full.cov_params()
                    cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
                    beta_b = beta[batch_param_indices]
                    Sigma_bb = cov_mat[np.ix_(batch_param_indices, batch_param_indices)]
                    try:
                        inv_Sigma_bb = np.linalg.inv(Sigma_bb)
                        W = float(beta_b.T @ inv_Sigma_bb @ beta_b)
                        df_w = len(beta_b)
                        p_w = float(1.0 - chi2.cdf(W, df_w))
                        LR, df_stat, pval = W, df_w, p_w
                        used = "Wald"
                    except np.linalg.LinAlgError:
                        Sigma_bb_pinv = np.linalg.pinv(Sigma_bb)
                        W = float(beta_b.T @ Sigma_bb_pinv @ beta_b)
                        df_w = len(beta_b)
                        p_w = float(1.0 - chi2.cdf(W, df_w))
                        LR, df_stat, pval = W, df_w, p_w
                        used = "Wald_pinv"
            except Exception as e:
                if verbose:
                    print(f"  Wald fallback failed for {feat}: {e}")

        rows.append({"Feature": feat, "TestStat": LR, "df": df_stat, "p-value": pval, "method": used})

    out = pd.DataFrame(rows)
    out = out.sort_values(by="TestStat", ascending=False).reset_index(drop=True)
    return out,model_defs_add

model_defs_mul = []
def MultiplicativeEffect_long(
    data: Optional[pd.DataFrame] = None,
    idp_matrix: Optional[np.ndarray] = None,
    subjects: Optional[Sequence] = None,
    timepoints: Optional[Sequence] = None,
    batch_name: Optional[Sequence] = None,
    idp_names: Optional[Iterable[str]] = None,
    covariates: Optional[Dict[str, Sequence]] = None,
    *,
    idvar: Optional[str] = None,
    batchvar: Optional[str] = None,
    timevar: Optional[str] = None,
    fix_eff: Optional[Iterable[str]] = None,
    ran_eff: Optional[Iterable[str]] = None,
    do_zscore: bool = True,
    reml: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Multiplicative (variability) batch test per feature.

    Robust to _build_input_df_if_needed returning either a DataFrame or (df, idcol, batchcol, tpcol).
    """
    # Call the helper and be defensive about the return type
    helper_out = _build_input_df_if_needed(
        data=data,
        idp_matrix=idp_matrix,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batch_name,
        idp_names=idp_names,
        idvar=idvar,
        batchvar=batchvar,
        timevar=timevar,
        covariates=covariates,
    )

    # Unpack in a robust way
    if isinstance(helper_out, tuple) and len(helper_out) >= 1:
        # New helper: expects (df, idcol, batchcol, tpcol)
        try:
            df = helper_out[0]
            # default fallback names if not provided
            idcol = helper_out[1] if len(helper_out) > 1 else (idvar if idvar is not None else "subjects")
            batchcol = helper_out[2] if len(helper_out) > 2 else (batchvar if batchvar is not None else "batches")
            tpcol = helper_out[3] if len(helper_out) > 3 else (timevar if timevar is not None else "timepoints")
        except Exception as e:
            raise RuntimeError(f"_build_input_df_if_needed returned an unexpected tuple shape: {e}")
    elif isinstance(helper_out, pd.DataFrame):
        # Old helper style: returned only a DataFrame
        df = helper_out
        idcol = idvar if idvar is not None else "subjects"
        batchcol = batchvar if batchvar is not None else "batches"
        tpcol = timevar if timevar is not None else "timepoints"
        if verbose:
            warnings.warn("Detected old _build_input_df_if_needed signature (returned DataFrame). Falling back to conventional column names.")
    else:
        raise RuntimeError(f"_build_input_df_if_needed returned unsupported type: {type(helper_out)}")

    # now df is a DataFrame and idcol/batchcol/tpcol are set
    covariates = dict(covariates or {})
    if fix_eff is None:
        fix_eff = list(covariates.keys())
    else:
        fix_eff = list(fix_eff)

    # RANDOM EFFECTS: infer only when caller omitted ran_eff (is None).
    if ran_eff is None:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    else:
        ran_eff = list(ran_eff)

    # Ensure batch column is categorical (use returned batchcol)
    if batchcol in df.columns:
        try:
            df[batchcol] = df[batchcol].astype("category")
        except Exception:
            pass

    # Defaults: ran_eff -> [idcol], fix_eff -> covariate keys + tpcol + batchcol
    if len(ran_eff) == 0:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    if len(fix_eff) == 0:
        inferred_fix = list(covariates.keys())
        if tpcol in df.columns and tpcol not in inferred_fix:
            inferred_fix.append(tpcol)
        if batchcol in df.columns and batchcol not in inferred_fix:
            inferred_fix.append(batchcol)
        fix_eff = inferred_fix

    # Validate referenced names exist
    to_check = {"fix_eff": fix_eff, "ran_eff": ran_eff}
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}")

    # Determine feature columns (IDPs)
    exclude = {idcol, batchcol, tpcol}
    exclude |= set(covariates.keys())
    if idp_names is not None:
        feature_cols = list(idp_names)
    else:
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    V = len(feature_cols)
    if verbose:
        print(f"[MultEffect_long] found {V} features")

    rows: List[Dict[str, Any]] = []

    for idx, feat in enumerate(feature_cols, 1):
        if verbose:
            print(f"[MultEffect_long] ({idx}/{V}) testing multiplicative batch effect for feature: {feat}")

        # Build local df for this feature
        local_cols = [feat] + list(fix_eff) + [batchcol] + ran_eff
        local_df = df.loc[:, [c for c in local_cols if c in df.columns]].copy()

        # Drop rows where response is NaN
        local_df = local_df[~local_df[feat].isna()].copy()
        if local_df.shape[0] < 3:
            if verbose:
                print(f"  skipping {feat}: too few non-NaN rows ({local_df.shape[0]})")
            rows.append({"Feature": feat, "ChiSq": np.nan, "DF": np.nan, "p-value": np.nan, "method": None})
            continue

        # Always z-score numeric predictors per-feature (in local_df)
        fixed_terms = _build_fixed_formula_terms(list(fix_eff or []), local_df, do_zscore_predictors=True)
        missing_fix = [vv for vv in (fix_eff or []) if (f"zscore_{vv}" not in local_df.columns) and (vv not in local_df.columns)]
        if missing_fix:
            warnings.warn(f"Fixed-effect columns not found in data (or zscore missing): {missing_fix}")

        # Optionally z-score response per-feature
        if do_zscore:
            zresp = f"z_{feat}"
            if zresp not in local_df.columns:
                mu_r = local_df[feat].mean(skipna=True)
                sd_r = local_df[feat].std(skipna=True)
                if pd.isna(sd_r) or sd_r == 0:
                    local_df[zresp] = 0.0
                else:
                    local_df[zresp] = (local_df[feat] - mu_r) / sd_r
            lhs = zresp
        else:
            lhs = feat

        fixed_str = " + ".join(fixed_terms) if len(fixed_terms) > 0 else "1"
        full_fixed = f"{lhs} ~ {fixed_str} + C({batchcol})"

        if verbose:
            print("\n[MODEL]")
            print("Feature     :", feat)
            print("Full model  :", full_fixed)
            print("Random eff  :", ran_eff)

        model_defs_mul.append({
            "Feature": feat,
            "formula": full_fixed,
            "ran_eff": list(ran_eff),
            "fix_eff": list(fix_eff),})


        # Fit full mixed model to obtain residuals
        res_full = None
        group_name = ran_eff[0]
        try:
            res_full = _safe_fit_mixedlm(full_fixed, local_df, group=group_name, reml=reml)
        except Exception as e:
            rows.append({"Feature": feat, "ChiSq": np.nan, "DF": np.nan, "p-value": np.nan, "method": None})
            if verbose:
                print(f"  fit failed for {feat}: {e}")
            continue

        # Obtain residuals aligned with local_df indices
        try:
            resid = res_full.resid
            resid = pd.Series(np.asarray(resid), index=local_df.index)
        except Exception:
            resid = pd.Series(np.asarray(getattr(res_full, "resid", np.asarray([]))), index=local_df.index if len(local_df) == len(getattr(res_full, "resid", [])) else None)
        

        # Group residuals by batch level and run Fligner test
        try:
            cats = pd.Categorical(local_df[batchcol]).categories
            groups = [resid[local_df[batchcol] == lvl].dropna().values for lvl in cats]
            if len(groups) < 2 or any(len(g) == 0 for g in groups):
                raise ValueError("Not enough observations per batch group for Fligner test.")
            stat, pval = fligner(*groups, center="median")
            df_stat = len(groups) - 1
            method = "Fligner"
        except Exception as e:
            if verbose:
                print(f"  fligner test failed for {feat}: {e}")
            stat = np.nan; pval = np.nan; df_stat = np.nan; method = None

        rows.append({
            "Feature": feat,
            "ChiSq": float(stat) if not np.isnan(stat) else np.nan,
            "DF": df_stat,
            "p-value": float(pval) if not np.isnan(pval) else np.nan,
            "method": method
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(by="ChiSq", ascending=False).reset_index(drop=True)
    return out, model_defs_mul

"""

------------------ CLI Help Only Setup ------------------
 Help functions are set up to provide descriptions of the available functions without executing them.
"""
# call the help functions for each diagnostic function, for example in terminal use `python DiagnosticFunctions.py -h Cohens_D`
def setup_help_only_parser():
    parser = argparse.ArgumentParser(
        prog='DiagnosticFunctions',
        description='Diagnostic function library (use -h with a function name to view its help).'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available functions')

    # Help entry for Cohens_D
    parser_cd = subparsers.add_parser(
        'Cohens_D',
        help='Compute Cohen\'s d for two datasets',
        description="""
        Computes Cohen's d effect size per feature.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Cohens_D.py --Data1 <data1.npy> --Data2 <data2.npy>

        Returns a list of Cohen's d values for each feature.
        Data1 and Data2 should be numpy arrays with shape (features, samples).
        Each feature's Cohen's d is calculated as (mean1 - mean2) / pooled_std,
        where pooled_std is the square root of the average of the variances of both groups

        Note: This function does not handle missing values or NaNs.
        Ensure that Data1 and Data2 are preprocessed accordingly.

        '''
    )
    # Help entry for PcaCorr
    parser_pca = subparsers.add_parser(
        'PcaCorr',
        help='Perform PCA and correlate top PCs with batch',
        description="""
        Performs PCA on data and computes correlation of top N principal components with batch variable.
        Returns Pearson correlations, explained variance, PCA scores, and PC-batch correlations.
        Optional parameter:
        --N_components (default=3): Number of PCs to analyze.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.PcaCorr --Data <data.npy> --batch <batch.npy>
        Returns:
        - Pearson correlation coefficients for each PC with the batch variable.
        - Explained variance for each PC.
        - PCA scores for each sample.
        - Correlation of the first N_components PCs with the batch variable.'''
    )
    parser_mahalanobis = subparsers.add_parser(
        'mahalanobis_distance',
        help='Calculate Mahalanobis distance between batches',
        description="""
        Calculates Mahalanobis distance between pairs of batches in the data.
        If covariates are provided, it will regress each feature on the covariates and return residuals from which the Mahalanobis distance is calculated.
        Args:
            Data (np.ndarray): Data matrix where rows are samples and columns are features.
            batch (np.ndarray): Batch labels for each sample.
            Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
            covariates (np.ndarray, optional): Covariates to regress out from the data.
        Returns:
            dict: A dictionary with Mahalanobis distances for each pair of batches.
        Raises:
            ValueError: If less than two unique batches are provided.
        Example:
            mahalanobis_distance(Data=data, batch=batch_labels, Cov=cov_matrix, covariates=covariates)
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.mahalanobis_distance --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Mahalanobis distances for each pair of batches.
        '''
    )

    parser_variance_ratios = subparsers.add_parser(

        'Variance_Ratios',
        help='Calculate variance ratios between batches',
        description="""
        Calculates the feature-wise ratio of variance between each unique batch pair,
        optionally removing covariate effects via linear regression.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Variance_ratios --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with variance ratios for each pair of batches.
        '''
    )
    parser_ks_test = subparsers.add_parser(
        'KS_Test',
        help='Perform KS test between batches',
        description="""
        Performs two-sample Kolmogorov-Smirnov test for distribution differences between
        each unique batch pair and each batch with the overall distribution.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.KS_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with KS test statistics and p-values for each pair of batches.
        '''
    )
    parser_levene_test = subparsers.add_parser(
        'Levene_Test',
        help='Perform Levene\'s test between batches',
        description="""
        Performs Levene's test for variance differences between each unique batch pair.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Levene_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Levene's test statistics and p-values for each pair of batches.
        '''
    )
    parser_run_lmm = subparsers.add_parser(
        'Run_LMM',
        help='Run linear mixed model diagnostics',
        description="""
        Runs linear mixed model diagnostics for each feature in the data,
        returning variance components, R-squared values, ICC, and fitting notes.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.run_lmm --Data <data.npy> --batch <batch.npy>
        Returns a DataFrame with LMM diagnostics for each feature.
        '''
    )

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()
"""







------------------ CLI Help Only Setup ------------------
 Help functions are set up to provide descriptions of the available functions without executing them.
"""
# call the help functions for each diagnostic function, for example in terminal use `python DiagnosticFunctions.py -h Cohens_D`
def setup_help_only_parser():
    parser = argparse.ArgumentParser(
        prog='DiagnosticFunctions',
        description='Diagnostic function library (use -h with a function name to view its help).'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available functions')

    # Help entry for Cohens_D
    parser_cd = subparsers.add_parser(
        'Cohens_D',
        help='Compute Cohen\'s d for two datasets',
        description="""
        Computes Cohen's d effect size per feature.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Cohens_D.py --Data1 <data1.npy> --Data2 <data2.npy>

        Returns a list of Cohen's d values for each feature.
        Data1 and Data2 should be numpy arrays with shape (features, samples).
        Each feature's Cohen's d is calculated as (mean1 - mean2) / pooled_std,
        where pooled_std is the square root of the average of the variances of both groups

        Note: This function does not handle missing values or NaNs.
        Ensure that Data1 and Data2 are preprocessed accordingly.

        '''
    )
    # Help entry for PcaCorr
    parser_pca = subparsers.add_parser(
        'PcaCorr',
        help='Perform PCA and correlate top PCs with batch',
        description="""
        Performs PCA on data and computes correlation of top N principal components with batch variable.
        Returns Pearson correlations, explained variance, PCA scores, and PC-batch correlations.
        Optional parameter:
        --N_components (default=3): Number of PCs to analyze.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.PcaCorr --Data <data.npy> --batch <batch.npy>
        Returns:
        - Pearson correlation coefficients for each PC with the batch variable.
        - Explained variance for each PC.
        - PCA scores for each sample.
        - Correlation of the first N_components PCs with the batch variable.'''
    )
    parser_mahalanobis = subparsers.add_parser(
        'mahalanobis_distance',
        help='Calculate Mahalanobis distance between batches',
        description="""
        Calculates Mahalanobis distance between pairs of batches in the data.
        If covariates are provided, it will regress each feature on the covariates and return residuals from which the Mahalanobis distance is calculated.
        Args:
            Data (np.ndarray): Data matrix where rows are samples and columns are features.
            batch (np.ndarray): Batch labels for each sample.
            Cov (np.ndarray, optional): Covariance matrix. If None, it will be computed from Data.
            covariates (np.ndarray, optional): Covariates to regress out from the data.
        Returns:
            dict: A dictionary with Mahalanobis distances for each pair of batches.
        Raises:
            ValueError: If less than two unique batches are provided.
        Example:
            mahalanobis_distance(Data=data, batch=batch_labels, Cov=cov_matrix, covariates=covariates)
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.mahalanobis_distance --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Mahalanobis distances for each pair of batches.
        '''
    )

    parser_variance_ratios = subparsers.add_parser(

        'Variance_Ratios',
        help='Calculate variance ratios between batches',
        description="""
        Calculates the feature-wise ratio of variance between each unique batch pair,
        optionally removing covariate effects via linear regression.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Variance_ratios --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with variance ratios for each pair of batches.
        '''
    )
    parser_ks_test = subparsers.add_parser(
        'KS_Test',
        help='Perform KS test between batches',
        description="""
        Performs two-sample Kolmogorov-Smirnov test for distribution differences between
        each unique batch pair and each batch with the overall distribution.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.KS_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with KS test statistics and p-values for each pair of batches.
        '''
    )
    parser_levene_test = subparsers.add_parser(
        'Levene_Test',
        help='Perform Levene\'s test between batches',
        description="""
        Performs Levene's test for variance differences between each unique batch pair.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.Levene_test --Data <data.npy> --batch <batch.npy>
        Returns a dictionary with Levene's test statistics and p-values for each pair of batches.
        '''
    )
    parser_run_lmm = subparsers.add_parser(
        'Run_LMM',
        help='Run linear mixed model diagnostics',
        description="""
        Runs linear mixed model diagnostics for each feature in the data,
        returning variance components, R-squared values, ICC, and fitting notes.
        """,
        epilog = '''
        Example usage:
        DiagnosticFunctions.run_lmm --Data <data.npy> --batch <batch.npy>
        Returns a DataFrame with LMM diagnostics for each feature.
        '''
    )

    return parser

if __name__ == '__main__':
    parser = setup_help_only_parser()
    parser.parse_args()