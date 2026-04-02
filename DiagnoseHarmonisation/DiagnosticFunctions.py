# DiagnosticFunctions.py
# Collection of diagnostic functions for harmonisation assessment (pre and post)

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from collections import Counter
import re
from matplotlib.pylab import lstsq
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.stats import chi2
import argparse
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
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


def _coerce_covariates(
    covariates: Optional[Any],
    covariate_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert covariates into a clean DataFrame. This is a helper function for preparing the data for LMM fitting.

    - DataFrame input: returned as-is (index reset).
    - 1D array-like: converted to a single-column DataFrame.
    - 2D array-like: converted to a DataFrame with provided names or default cov1..covk.
    """
    if covariates is None:
        return pd.DataFrame()

    if isinstance(covariates, pd.DataFrame):
        cov_df = covariates.reset_index(drop=True).copy()
        return cov_df

    cov_arr = np.asarray(covariates)

    if cov_arr.ndim == 1:
        if covariate_names is not None and len(covariate_names) > 0:
            col_name = str(covariate_names[0])
        else:
            col_name = "cov1"
        return pd.DataFrame({col_name: cov_arr})

    if cov_arr.ndim != 2:
        raise ValueError("covariates must be None, a DataFrame, a 1D array, or a 2D array.")

    n_cov = cov_arr.shape[1]
    if covariate_names is not None:
        if len(covariate_names) != n_cov:
            raise ValueError(
                f"covariate_names length mismatch: expected {n_cov}, got {len(covariate_names)}."
            )
        names = [str(x) for x in covariate_names]
    else:
        names = [f"cov{i+1}" for i in range(n_cov)]

    return pd.DataFrame(cov_arr, columns=names)


def _prepare_base_dataframe(
    batch: Any,
    covariates: Optional[Any] = None,
    covariate_names: Optional[List[str]] = None,
    group_col_name: str = "batch",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the base dataframe used for each feature. This is a helper function for Run_LMM_cross_sectional to prepare the data for LMM fitting.
    Returns the dataframe and the fixed-effects RHS terms.
    """
    df_base = pd.DataFrame({group_col_name: np.asarray(batch)})
    cov_df = _coerce_covariates(covariates, covariate_names=covariate_names)

    if len(cov_df) > 0:
        df_base = pd.concat([df_base.reset_index(drop=True), cov_df.reset_index(drop=True)], axis=1)
        rhs_terms = list(cov_df.columns)
    else:
        rhs_terms = []

    return df_base, rhs_terms


def _build_fixed_formula(rhs_terms: List[str]) -> str:
    """Construct the fixed-effects formula for statsmodels. This is a helper function for Run_LMM_cross_sectional to prepare the formula for LMM fitting."""
    if len(rhs_terms) == 0:
        return "y ~ 1"
    return "y ~ " + " + ".join(rhs_terms)


def _standardize_numeric_covariates(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Standardise numeric covariates only. Leaves batch/group and y untouched.
    This is a helper function for Run_LMM_cross_sectional to prepare the data for LMM fitting.
    """
    out = df.copy()

    for col in out.columns:
        if col in (group_col, "y"):
            continue

        if pd.api.types.is_numeric_dtype(out[col]):
            m = out[col].mean()
            s = out[col].std(ddof=0)
            if pd.isna(s) or s == 0:
                out[col] = out[col] - m
            else:
                out[col] = (out[col] - m) / s

    return out


def _fit_ols_fixed_only(df: pd.DataFrame, formula_fixed: str) -> Dict[str, Any]:
    """
    Fit a fixed-effects-only model and return fallback stats.
    Thus is a helper function for fit_lmm_safe to provide a fallback when LMM fitting fails or is inappropriate.
    """
    import patsy

    y_vec, X = patsy.dmatrices(formula_fixed, df, return_type="dataframe")
    ols_res = sm.OLS(y_vec, X).fit()

    fitted = np.dot(X.values, ols_res.params.values)
    var_fixed = float(np.nanvar(fitted, ddof=0))

    return {
        "success": False,
        "mdf": None,
        "ols": ols_res,
        "optimizer_used": None,
        "notes": ["fallback_ols_used"],
        "stats": {
            "var_fixed": var_fixed,
            "var_batch": 0.0,
            "var_resid": float(ols_res.mse_resid),
            "R2_ols_fixed": float(ols_res.rsquared),
            "R2_marginal": np.nan,
            "R2_conditional": np.nan,
            "ICC": 0.0,
            "delta_R2": np.nan,
            "LR_stat": np.nan,
            "pval_LRT_random": np.nan,
            "pval_LRT_random_mixture": np.nan,
        },
        "warning_types": [],
        "warning_messages": [],
        "status": "fallback_ols",
    }


def fit_lmm_safe(
    df: pd.DataFrame,
    formula_fixed: str,
    group_col: str = "batch",
    reml: bool = False,
    min_group_n: int = 10,
    var_threshold: float = 1e-8,
    optimizers: Iterable[str] = ("lbfgs", "bfgs", "powell", "cg"),
    maxiter: int = 400,
    boundary_pvalue: bool = True,
) -> Dict[str, Any]:
    """
    Fit a random-intercept LMM with warnings captured and safe fallbacks.
    This is a helper function for Run_LMM_cross_sectional to fit the LMM for each feature with robust error handling and diagnostics.

    Returns a dictionary with:
      - success
      - mdf / ols
      - optimizer_used
      - notes
      - warning_types / warning_messages
      - stats
      - status
    """
    notes: List[str] = []
    warning_types: List[str] = []
    warning_messages: List[str] = []

    if "y" not in df.columns:
        raise ValueError("df must contain column 'y'.")

    # 1) low-variance feature check
    if np.nanvar(df["y"].to_numpy(dtype=float)) <= var_threshold:
        notes.append("low_variance_feature")
        return {
            "success": False,
            "mdf": None,
            "ols": None,
            "optimizer_used": None,
            "notes": notes,
            "stats": {},
            "warning_types": warning_types,
            "warning_messages": warning_messages,
            "status": "skipped_low_variance",
        }

    # 2) group size check
    group_counts = df[group_col].value_counts(dropna=False)
    if (group_counts < min_group_n).any():
        notes.append("small_group_count")
        try:
            fallback = _fit_ols_fixed_only(df, formula_fixed)
            fallback["notes"] = notes + fallback["notes"]
            fallback["status"] = "fallback_ols_small_group"
            return fallback
        except Exception:
            notes.append("fallback_ols_failed")
            return {
                "success": False,
                "mdf": None,
                "ols": None,
                "optimizer_used": None,
                "notes": notes,
                "stats": {},
                "warning_types": warning_types,
                "warning_messages": warning_messages,
                "status": "failed_small_group",
            }

    # 3) standardise numeric covariates
    df_fit = _standardize_numeric_covariates(df, group_col=group_col)

    # 4) fit mixed model with optimizer sequence
    md = mixedlm(formula_fixed, data=df_fit, groups=df_fit[group_col], re_formula="1")
    last_exc = None
    chosen_optimizer = None

    for opt in optimizers:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mdf = md.fit(reml=reml, method=opt, maxiter=maxiter, disp=False)

            if len(caught) > 0:
                for w in caught:
                    warning_types.append(type(w.message).__name__)
                    warning_messages.append(str(w.message))

            converged = bool(getattr(mdf, "converged", True))
            if not converged:
                notes.append(f"optimizer_{opt}_no_converge")
                continue

            chosen_optimizer = opt

            # random intercept variance
            try:
                var_batch = float(np.asarray(mdf.cov_re).ravel()[0])
            except Exception:
                var_batch = np.nan
                notes.append("cov_re_inspect_failed")

            # fixed-effects linear predictor variance
            try:
                exog = mdf.model.exog
                fe_params = mdf.fe_params.values.reshape(-1, 1)
                linpred_fixed = np.dot(exog, fe_params).ravel()
                var_fixed = float(np.nanvar(linpred_fixed, ddof=0))
            except Exception:
                var_fixed = np.nan
                notes.append("var_fixed_compute_failed")

            # residual variance
            try:
                var_resid = float(mdf.scale)
            except Exception:
                var_resid = np.nan
                notes.append("var_resid_compute_failed")

            # total variance and summary metrics
            batch_term = 0.0 if np.isnan(var_batch) else var_batch
            total_var = var_fixed + batch_term + var_resid

            R2_marginal = var_fixed / total_var if np.isfinite(total_var) and total_var > 0 else np.nan
            R2_conditional = (var_fixed + batch_term) / total_var if np.isfinite(total_var) and total_var > 0 else np.nan
            delta_R2 = R2_conditional - R2_marginal if np.isfinite(R2_conditional) and np.isfinite(R2_marginal) else np.nan
            ICC = batch_term / (batch_term + var_resid) if np.isfinite(batch_term) and np.isfinite(var_resid) and (batch_term + var_resid) > 0 else np.nan

            if np.isfinite(var_batch) and abs(var_batch) <= 1e-10:
                notes.append("var_batch_near_zero")

            # LRT against fixed-only OLS
            LR_stat = np.nan
            pval_LRT = np.nan
            pval_LRT_mixture = np.nan
            try:
                import patsy
                y_vec, X_fixed = patsy.dmatrices(formula_fixed, df_fit, return_type="dataframe")
                ols_fixed = sm.OLS(y_vec, X_fixed).fit()

                llf_lmm = float(mdf.llf)
                llf_ols = float(ols_fixed.llf)
                LR_stat = 2.0 * (llf_lmm - llf_ols)

                if np.isfinite(LR_stat) and LR_stat >= 0:
                    # naive chi-square p-value
                    pval_LRT = float(chi2.sf(LR_stat, 1))
                    # mixture approximation for variance-on-boundary testing
                    pval_LRT_mixture = float(0.5 * chi2.sf(LR_stat, 1)) if boundary_pvalue else np.nan
            except Exception:
                notes.append("lrt_failed")

            stats = {
                "var_fixed": var_fixed,
                "var_batch": var_batch,
                "var_resid": var_resid,
                "R2_marginal": R2_marginal,
                "R2_conditional": R2_conditional,
                "delta_R2": delta_R2,
                "ICC": ICC,
                "LR_stat": LR_stat,
                "pval_LRT_random": pval_LRT,
                "pval_LRT_random_mixture": pval_LRT_mixture,
            }

            return {
                "success": True,
                "mdf": mdf,
                "ols": None,
                "optimizer_used": chosen_optimizer,
                "notes": notes,
                "stats": stats,
                "warning_types": sorted(set(warning_types)),
                "warning_messages": list(dict.fromkeys(warning_messages)),  # preserve order, unique
                "status": "lmm",
            }

        except Exception as e:
            last_exc = e
            notes.append(f"optimizer_{opt}_failed")
            continue

    # 5) all LMM attempts failed -> OLS fallback
    try:
        fallback = _fit_ols_fixed_only(df_fit, formula_fixed)
        fallback["notes"] = notes + ["all_lmm_optimizers_failed_fallback_ols"]
        fallback["status"] = "fallback_ols_all_lmm_failed"
        fallback["warning_types"] = sorted(set(warning_types))
        fallback["warning_messages"] = list(dict.fromkeys(warning_messages))
        fallback["optimizer_used"] = None
        return fallback
    except Exception:
        notes.append("all_lmm_and_ols_failed")
        if last_exc is not None:
            notes.append(f"last_exception={type(last_exc).__name__}")
        return {
            "success": False,
            "mdf": None,
            "ols": None,
            "optimizer_used": None,
            "notes": notes,
            "stats": {},
            "warning_types": sorted(set(warning_types)),
            "warning_messages": list(dict.fromkeys(warning_messages)),
            "status": "failed",
        }


def Run_LMM_cross_sectional(
    Data,
    batch,
    covariates=None,
    feature_names=None,
    group_col_name="batch",
    covariate_names=None,
    min_group_n=2,
    var_threshold=1e-8,
    reml=False,
    optimizers=("lbfgs", "bfgs", "powell", "cg"),
    maxiter=400,
    boundary_pvalue=True,
):
    """
    Run a random-intercept Linear Mixed Model for each feature in Data with batch as the grouping variable and optional covariates as fixed effects.
    Parameters:
        - Data: array-like of shape (n_samples, n_features) - the data matrix to analyze.
        - batch: array-like of shape (n_samples,) - batch labels for each sample.
        - covariates: optional array-like of shape (n_samples, n_covariates) or DataFrame - covariate data to include as fixed effects.
        - feature_names: optional list of length n_features - names for each feature.
        - group_col_name: str - name for the batch grouping column in the DataFrame.
        - covariate_names: optional list of length n_covariates - names for each covariate.
        - min_group_n: int - minimum number of samples per batch group to attempt LMM fitting.
        - var_threshold: float - variance threshold below which a feature is considered low-variance and skipped.
        - reml: bool - whether to use REML for LMM fitting.
        - optimizers: iterable of str - sequence of optimizers to try for LMM fitting.
        - maxiter: int - maximum iterations for each optimizer.
        - boundary_pvalue: bool - whether to calculate mixture p-value for variance-on-boundary testing in LRT.
    
    Returns:
        - results_df: DataFrame with one row per feature containing LMM fitting results and diagnostics.
        - summary: dict summarising counts of various notes and warnings across features.
    
    Notes:
        - If a feature has variance below var_threshold, it is skipped and noted as "low_variance_feature".
        - If any batch group has fewer than min_group_n samples, LMM fitting is skipped and an OLS fallback is attempted, with notes "small_group_count" and "fallback_ols_used" or "fallback_ols_failed".
        - If LMM fitting fails to converge or raises an exception for all optimizers, an OLS fallback is attempted with note "all_lmm_optimizers_failed_fallback_ols".
        - Various warnings during fitting are captured and recorded in the results.

    """
    Data = np.asarray(Data, dtype=float)
    if Data.ndim != 2:
        raise ValueError("Data must be a 2D array-like object with shape (n_samples, n_features).")

    n, p = Data.shape

    if feature_names is None:
        feature_names = [f"feature_{i+1}" for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length mismatch.")

    df_base, rhs_terms = _prepare_base_dataframe(
        batch=batch,
        covariates=covariates,
        covariate_names=covariate_names,
        group_col_name=group_col_name,
    )
    formula_fixed = _build_fixed_formula(rhs_terms)

    rows = []
    notes_counter = Counter()

    for fi in range(p):
        df = df_base.copy()
        df["y"] = Data[:, fi]

        res = fit_lmm_safe(
            df=df,
            formula_fixed=formula_fixed,
            group_col=group_col_name,
            reml=reml,
            min_group_n=min_group_n,
            var_threshold=var_threshold,
            optimizers=optimizers,
            maxiter=maxiter,
            boundary_pvalue=boundary_pvalue,
        )

        stats = res.get("stats", {}) or {}
        notes = res.get("notes", []) or []

        row = {
            "feature_index": fi,
            "feature": feature_names[fi],
            "success": bool(res.get("success", False)),
            "status": res.get("status", "unknown"),
            "optimizer_used": res.get("optimizer_used", None),
            "var_fixed": stats.get("var_fixed", np.nan),
            "var_batch": stats.get("var_batch", np.nan),
            "var_resid": stats.get("var_resid", np.nan),
            "R2_marginal": stats.get("R2_marginal", np.nan),
            "R2_conditional": stats.get("R2_conditional", np.nan),
            "delta_R2": stats.get("delta_R2", np.nan),
            "ICC": stats.get("ICC", np.nan),
            "LR_stat": stats.get("LR_stat", np.nan),
            "pval_LRT_random": stats.get("pval_LRT_random", np.nan),
            "pval_LRT_random_mixture": stats.get("pval_LRT_random_mixture", np.nan),
            "notes": ";".join(notes),
            "warning_types": ";".join(res.get("warning_types", []) or []),
        }
        rows.append(row)

        for ntag in notes:
            notes_counter[ntag] += 1

        status = res.get("status", "unknown")
        notes_counter[f"status_{status}"] += 1
        notes_counter["succeeded_LMM" if res.get("success", False) else "used_fallback"] += 1

    results_df = pd.DataFrame(rows)
    summary = dict(notes_counter)
    summary["n_features"] = p

    return results_df, summary


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
    This is a helper function for residualising out covariate effects while preserving batch effects:
    It is used in; Cohens_D, Variance_Ratios and KS_Test functions to ensure that the batch effect calculations are not confounded by covariate effects.

    Args:
        data: np.ndarray of shape (n_samples, n_features) - the data matrix to be residualised
        covariates: np.ndarray of shape (n_samples, n_covariates) - the covariate matrix
        batch: array-like of shape (n_samples,) - batch labels for each sample
        covariate_names: list of length n_covariates - names for each covariate
        covariate_types: list of length n_covariates with values 0 (binary), 2 (categorical), 3 (continuous) - optional, if not provided will be inferred
        report: optional object with method log_text(str) for logging messages about covariate type inference and processing steps.
    Returns:
        data_resid: np.ndarray of shape (n_samples, n_features) - the data matrix with covariate effects removed but batch effects preserved.

    """
    # -------------------------
    # Infer covariate types if not provided (same heuristic you used)
    # -------------------------
        # Check if covariate names are provided,
    if covariate_names is None:
        covariate_names = [f'cov_{i}' for i in range(covariates.shape[1])]

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

# ------------------ Diagnostic Functions ------------------
# Cohens D function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations
# Cohens d function calculates the effect size between two groups for each feature.
import numpy as np
from itertools import combinations

def z_score(data,MAD=False):
    """
    Z-score normalization of the data matrix (samples x features).
    Use median centered by default as is more robust to outliers and non-normal distributions.
    """
    median = np.median(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    zscored = (data - median) / std
    if MAD == True:
        MAD = np.median(np.abs(data - median), axis=0)
        zscored = (data - median) / MAD

    return zscored

import numpy as np

def robust_z_score(data, method="mad", eps=1e-12):
    """
    Robust normalization of a data matrix (samples x features).

    Parameters
    ----------
    data : array-like
        Input data, shape (n_samples, n_features) or (n_samples,)
    method : {"mad", "iqr", "std"}
        Scaling method.
        - "mad": median absolute deviation, scaled to be comparable to std
        - "iqr": interquartile range, scaled to be comparable to std
        - "std": standard z-score, but still median-centered here
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    zscored : ndarray
        Normalized data.
    center : ndarray
        Per-feature center used.
    scale : ndarray
        Per-feature scale used.
    """
    data = np.asarray(data)

    center = np.median(data, axis=0)

    if method == "mad":
        scale = np.median(np.abs(data - center), axis=0)
        scale = 1.4826 * scale  # makes it comparable to std for normal data
    elif method == "iqr":
        q75 = np.percentile(data, 75, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        scale = q75 - q25
        scale = scale / 1.349  # optional normalizing constant
    elif method == "std":
        scale = np.std(data, axis=0, ddof=1)
    else:
        raise ValueError("method must be one of {'mad', 'iqr', 'std'}")

    scale = np.where(scale < eps, 1.0, scale)
    zscored = (data - center) / scale
    return zscored

import numpy as np

def Cohens_D(Data, batch_indices, covariates=None, BatchNames=None, covariate_names=None, covariate_types=None):
    """
    Cohen's d for each batch vs "all other batches" using UNWEIGHTED batch averages.
    In some instances, one may want to see batch vs batch specific comparisons, but this function focuses on batch vs rest to give a more global picture of how each batch differs from the overall distribution of the data (after covariate effects are removed if covariates are provided).

    Args:
        Data: np.ndarray of shape (n_samples, n_features) - the data matrix.
        batch_indices: array-like of shape (n_samples,) - batch labels for each sample (can be strings or numbers).
        covariates: optional np.ndarray of shape (n_samples, n_covariates) - covariate matrix to residualise out before calculating Cohen's d.
        BatchNames: optional dict mapping batch label to display name, or list/tuple of names in order of unique batches, or None to use batch labels as names.
        covariate_names: optional list of length n_covariates with names for each covariate (used for internal processing and reporting).
        covariate_types: optional list of length n_covariates with values 0 (binary), 2 (categorical), 3 (continuous) indicating the type of each covariate (used for internal processing and reporting). If not provided, types will be inferred from the data.
    
    Returns:
        pairwise_d: np.ndarray of shape (n_batches, n_features) with Cohen's d values for each batch vs rest comparison.
        pair_labels: list of tuples with labels for each comparison (e.g., [('Batch1', 'all_data'), ('Batch2', 'all_data'), ...]).
    
    Note:
        - Cohen's d is calculated as (mean_batch - mean_other) / std_other, where mean_other and std_other are the UNWEIGHTED averages of the means and stds of the other batches (i.e., simple arithmetic mean across batches, NaNs replace by batch specific noise).
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
    Perform PCA and correlate top PCs with batch and covariates if given, returning explained variance, scores, and correlation results.

    Args:
        Data: np.ndarray of shape (n_samples, n_features) - the data matrix.
        batch: array-like of shape (n_samples,) - batch labels for each sample (can
        N_components: int or None - number of principal components to compute (default None means min(n_samples, n_features)).
        covariates: optional np.ndarray of shape (n_samples, n_covariates)
        variable_names: optional list of length covariates
    Returns:
        explained_variance: np.ndarray of shape (n_components,) with percentage of variance explained by each PC.
        scores: np.ndarray of shape (n_samples, n_components) with the PCA scores for each sample.
        PC_correlations: dict mapping variable name to dict with keys 'correlation' (array of shape (n_components,)) and 'p_value' (array of shape (n_components,)) for the Pearson correlation of each PC with that variable.


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
    else: # Have extended vraiables by batch by now:
        # check if they match, if not add batch to as first variable name if not already present
        if len(variable_names) != len(variables):
            variable_names = ['batch'] + list(cov_names)
    
    # Final check on variable names length check after adding batch flag:
    if len(variable_names) != len(variables):
        raise ValueError(f"Length of variable_names ({len(variable_names)}) must match number of variables ({len(variables)}).")

    variable_names_out = variable_names # Final variable names to use in output (after ensuring batch is included and covariate names
    
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

def Variance_Ratios(data, batch, covariates=None,
                    covariate_names=None, covariate_types=None,
                    mode='rest'):
    """
    Calculate feature-wise ratio(s) of variance for batches. Can calculate in multiple modes depending on the desired comparisons:

    Modes:
      - 'pairwise'       : (default) compute variance ratio for each unique pair (b1, b2) -> var(b1)/var(b2)
      - 'rest'           : for each batch b compute var(b) / var(all_samples_except_b)  (pooled rest)
      - 'unweighted_mean': for each batch b compute var(b) / mean(var(other_batches))
      - 'weighted_mean'  : for each batch b compute var(b) / weighted_mean(var(other_batches),
                                                           weights = sample sizes of other batches)

    Args:
        data: numpy array (n_samples, n_features)
        batch: 1D array-like of length n_samples (batch labels)
        covariates: optional, passed to RobustOLS for adjustment (same as your current workflow)
        covariate_names, covariate_types: passed to RobustOLS
        mode: one of {'pairwise', 'rest', 'unweighted_mean', 'weighted_mean'}

    Returns:
        dict:
          - if mode == 'pairwise': keys are tuples (b1, b2) (same as your original function)
          - else: keys are batch labels and values are arrays (n_features,) of var(batch)/var(reference)
    """

    import numpy as np
    from itertools import combinations

    # ---------------- Validation ----------------
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (samples x features).")
    if not isinstance(batch, (list, np.ndarray)) or np.ndim(batch) != 1:
        raise ValueError("batch must be a 1D list or numpy array.")
    if data.shape[0] != len(batch):
        raise ValueError("Number of samples in Data must match length of batch")

    batch = np.array(batch)
    unique_batches = np.unique(batch)
    if len(unique_batches) < 2:
        raise ValueError("At least two unique batches are required.")

    # ------------- Optional Covariate Removal -------------
    if covariates is not None:
        data = RobustOLS(data, covariates, batch,
                         covariate_names, covariate_types)

    # ------------- Precompute Batch Stats -------------
    batch_data = {}
    batch_counts = {}
    batch_var = {}

    for b in unique_batches:
        Xb = data[batch == b]
        batch_data[b] = Xb
        batch_counts[b] = Xb.shape[0]

        if Xb.shape[0] > 1:
            batch_var[b] = np.var(Xb, axis=0, ddof=1)
        else:
            batch_var[b] = np.zeros(data.shape[1], dtype=float)

    variance_rows = []
    pair_labels = []

    # ==========================================================
    # MODE 1: Pairwise (original behaviour)
    # ==========================================================
    if mode == 'pairwise':
        for b1, b2 in combinations(unique_batches, 2):
            var1 = batch_var[b1]
            var2 = batch_var[b2]

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = var1 / var2
                ratio[np.isnan(ratio)] = 0

            variance_rows.append(ratio)
            pair_labels.append(f"{b1} / {b2}")

    # ==========================================================
    # MODE 2: Pooled Rest
    # ==========================================================
    elif mode == 'rest':
        for b in unique_batches:
            rest_data = data[batch != b]

            if rest_data.shape[0] > 1:
                var_rest = np.var(rest_data, axis=0, ddof=1)
            else:
                var_rest = np.zeros(data.shape[1], dtype=float)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = batch_var[b] / var_rest
                ratio[np.isnan(ratio)] = 0

            variance_rows.append(ratio)
            pair_labels.append(f"{b} / Rest")

    # ==========================================================
    # MODE 3 & 4: Mean of Other Batches
    # ==========================================================
    elif mode in ('unweighted_mean', 'weighted_mean'):

        vars_stack = np.vstack([batch_var[b] for b in unique_batches])
        counts = np.array([batch_counts[b] for b in unique_batches], dtype=float)

        for i, b in enumerate(unique_batches):

            mask = np.arange(len(unique_batches)) != i
            other_vars = vars_stack[mask]

            if mode == 'unweighted_mean':
                comparator = np.mean(other_vars, axis=0)
                label_suffix = "Mean(Other)"
            else:
                other_counts = counts[mask]
                comparator = np.average(other_vars, axis=0,
                                        weights=other_counts)
                label_suffix = "WeightedMean(Other)"

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = batch_var[b] / comparator
                ratio[np.isnan(ratio)] = 0

            variance_rows.append(ratio)
            pair_labels.append(f"{b} / {label_suffix}")

    else:
        raise ValueError("mode must be one of: "
                         "'pairwise', 'rest', "
                         "'unweighted_mean', 'weighted_mean'")

    variance_ratios = np.vstack(variance_rows)

    return variance_ratios, pair_labels

# Define a function to perform the Levene's test for variance differences between each unique batch pair
def Levene_Test(data, batch, centre = 'median'):
    # Define a function to perform the Levene's test for variance differences between each unique batch pair
    """
    Perform Levene's test for variance differences between each unique batch pair.
    Args:
        data: np.ndarray of shape (n_samples, n_features) - the data matrix.
        batch: np.ndarray of shape (n_samples,) - the batch labels.
        centre: str, optional - the method to calculate the center for Levene's test ('median', 'mean', or 'trimmed'). Default is 'median' which is more robust to outliers and non-normal distributions.
    Returns:
        dict: A dictionary where keys are tuples of batch pairs (batch1, batch2) and values are dictionaries containing 'statistic' and 'p_value' arrays of shape (n_features
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
                compare_pairs=False,
                compare_to_overall_excluding_batch=True,
                min_batch_n=3,
                alpha=0.05,
                do_fdr=True,
                residualize_covariates=True,
                covariate_names=None,
                covariate_types=None):
    """
    Perform two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair and optionally compare each batch to the overall distribution (either including or excluding that batch).

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
      residualize_covariates: if True, residualize covariates before testing (default True)
      covariate_names: optional list of covariate names (used for internal processing and reporting)
      covariate_types: optional list of covariate types (0=binary, 2=categorical, 3=continuous) (used for internal processing and reporting)
    
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


"""
"""

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