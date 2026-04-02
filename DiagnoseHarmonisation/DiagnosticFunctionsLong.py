###################################
# FOR LONGITUDINAL DATA
####################################

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Sequence, Optional
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
    Compute pairwise Spearman correlations between matched subjects across all
    ordered timepoint pairs, with permutation-based significance testing.

    For each pair of timepoints (order preserved from first appearance), subjects
    present at both timepoints are matched and Spearman’s rho is computed for each
    IDP (column). Significance is assessed via permutation testing by shuffling
    the second-timepoint values within matched pairs (nPerm iterations). P-values
    use the +1 correction: p = (1 + count_ge) / (1 + valid_null_count).

    Args:
    
        idp_matrix : array-like, shape (n_samples, n_idps)
            Numeric matrix of IDP values.
        subjects : sequence of length n_samples
            Subject identifiers (matched across timepoints).
        timepoints : sequence of length n_samples
            Timepoint labels (order defines comparison order).
        idp_names : sequence of length n_idps, optional
            Names of IDPs; defaults to ["idp_1", ...].
        nPerm : int, default=10000
            Number of permutations (>=1).
        seed : int, optional
            Random seed for reproducibility.

    Returns:
        pd.DataFrame
            Columns: ["TimeA","TimeB","IDP","nPairs",
                    "SpearmanRho","NullMeanRho","pValue"].
            Rows with fewer than 3 matched pairs return NaNs for statistics.
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
    """
    Compute within-subject variability (percent) for each IDP across timepoints.

    For each subject, variability is calculated per IDP using available (non-NaN)
    observations:

    - If exactly 2 timepoints: absolute percent difference relative to the mean,
      ``|x1 - x2| / mean * 100``.
    - If >2 timepoints: coefficient of variation (sample SD, ddof=1) relative
      to the mean, ``SD / mean * 100``.
    - If mean is 0 or no valid data: returns NaN.

    Parameters
    ----------
    idp_matrix : array-like, shape (n_samples, n_idps)
        Numeric matrix of IDP values.
    subjects : sequence of length n_samples
        Subject identifiers used to group repeated measurements.
    timepoints : sequence of length n_samples
        Timepoint labels (not directly used in calculation but required for
        input alignment).
    idp_names : sequence of length n_idps, optional
        Names of IDPs; defaults to ["idp_1", ...].

    Returns
    -------
    pd.DataFrame
        One row per subject with columns ``["subject", <IDP1>, <IDP2>, ...]``,
        where each IDP value represents within-subject percent variability.

    Raises
    ------
    ValueError
        If ``idp_matrix`` is not 2-D, or if input sequence lengths do not
        match the number of rows, or if ``idp_names`` length does not match
        the number of columns.
    """


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
    Compute multivariate batch/site differences as Mahalanobis distances of site
    means from the overall mean, with numerically-stable handling of covariance
    estimation and inversion.

    For each batch (site) this routine:

    - Computes the site mean vector after dropping rows with any NaN across features.
    - Estimates each site's covariance (zero matrix if n_samples_retained <= 1).
    - Averages site covariances to form an overall covariance.
    - Computes the Mahalanobis distance (MD) between each site mean and the overall
      mean. If the overall covariance is ill-conditioned or singular, the function
      falls back to an SVD-based pseudoinverse (with tolerance-based truncation)
      for numeric stability.

    Parameters
    ----------
    idp_matrix : array-like, shape (n_samples, n_features)
        Numeric matrix of features / IDPs (rows are observations).
    batch : pandas.Series or sequence of length n_samples
        Batch / site labels for each row. Will be converted to a categorical
        Series; categories (in order) determine output site order.
    idp_names : sequence of length n_features, optional
        Feature names used for intermediate DataFrame columns. Defaults to
        ["idp_1", "idp_2", ...].
    return_info : bool, default False
        If True, also return a diagnostics dict with keys:

        - ``"site_categories"``: list of category labels.
        - ``"site_counts"``: list of retained (post-NaN-drop) sample counts per site.
        - ``"cond_number"``: condition number of the averaged covariance.
        - ``"num_retained_svals"``: number of singular values retained (SVD fallback).
        - ``"overallCov"``: the averaged covariance matrix (numpy array).

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, dict)
        DataFrame with columns ``["batch", "mdval"]``. Rows are ordered by batch
        categories followed by a final row ``"average_batch"`` containing the mean
        MD across sites. If ``return_info`` is True, a tuple
        ``(DataFrame, info_dict)`` is returned.

    Raises
    ------
    ValueError
        If ``idp_matrix`` is not 2-D, or if ``batch`` length does not match the
        number of rows, or if ``idp_names`` length does not match the number of
        features.

    Notes
    -----
    - Rows with any NaN across features are dropped when computing a site's mean
      and covariance. If a site has zero retained rows, a warning is emitted and
      its mean is left as NaN (MD will be NaN). If a site has one retained row,
      its covariance is taken to be the zero matrix.
    - The averaged covariance is the simple mean of per-site covariances.
    - Mahalanobis distances are computed as
      ``sqrt((mu_i - mu_overall)' Σ^{-1} (mu_i - mu_overall))``.
      For numerical stability, the function attempts a direct linear solve when
      the averaged covariance is well-conditioned; otherwise it uses an SVD-based
      pseudoinverse with a tolerance derived from machine epsilon.
    - Returns NaN for MD if a site's mean vector is NaN (e.g., site had no
      retained observations after NaN-dropping).
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

    Returns: modified df (with forced types / zscores) and list of formulas.
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
) -> Tuple[pd.DataFrame, list]:
    """
    Run a mixed-effects modeling pipeline per-IDP for longitudinal, multi-site data.

    For each IDP (column) this function:

    1. Builds three formulas (full, subject-only / null, no-batch) via
       ``build_mixed_formula``, honoring forced categorical/numeric conversions
       and optional z-scoring of variables.
    2. Fits a full mixed model (fixed effects including batch + specified random
       effects) using ``statsmodels`` MixedLM.
    3. Runs pairwise Wald contrasts between batch levels to count significant
       site differences.
    4. Fits a subject-only random-intercept model to extract subject (between)
       variance and residual (within) variance, then computes ICC and WCV.
    5. Fits a fixed-effects-only model (no batch terms) to extract coefficient
       estimates, p-values and confidence intervals for requested fixed effects.
    6. Collects diagnostics and returns one summary row per IDP. Model failures
       yield NaNs for that IDP but do not stop the pipeline.

    Parameters
    ----------
    idp_matrix : array-like, shape (n_samples, n_idps)
        Numeric matrix of IDP values (rows = observations).
    subjects : sequence of length n_samples
        Subject identifiers (canonical column name used internally: ``'subjects'``).
    timepoints : sequence of length n_samples
        Timepoint labels (canonical name: ``'timepoints'``).
    batches : sequence of length n_samples
        Batch / site labels (canonical name: ``'batches'``); converted to
        categorical.
    idp_names : sequence of length n_idps
        Names for IDP columns.
    covariates : dict, optional
        Mapping of ``name -> sequence`` for additional covariates; each
        sequence length must match ``n_samples``.
    fix_eff : sequence, optional
        Fixed-effect variable names to include. If empty, defaults to the keys
        of ``covariates`` (batch/timepoint are still included by the formula
        builder as ``batch_vars``).
    ran_eff : sequence, optional
        Random-effect grouping variables. If empty, defaults to
        ``['subjects']``.
    force_categorical : sequence, optional
        Columns to coerce to categorical dtype.
    force_numeric : sequence, optional
        Columns to coerce to numeric dtype.
    zscore_var : sequence, optional
        Variables to z-score before model fitting.
    do_zscore : bool, default True
        Use ``zscore_...`` columns for response/predictors when available.
    p_thr : float, default 0.05
        Nominal alpha for pairwise Wald tests.
    p_corr : int, default 1
        If ``0``, disable multiple-comparison correction for pairwise tests;
        otherwise apply Bonferroni-style correction at
        ``alpha = 0.05 / n_tests_nonNaN``.
    reml : bool, default True
        Whether to fit MixedLM using REML.

    Returns
    -------
    tuple of (pd.DataFrame, list)
        A tuple ``(results_df, model_defs)`` where:

        - ``results_df`` contains one row per IDP with columns (stable order):
          ``["IDP", "batch", "n_is_batchSig", "anova_batches", "Subj_Var",
          "Resid_Var", "ICC", "WCV", <fixed-effect estimate/pval/ci columns>]``.

          - ``n_is_batchSig``: count of pairwise contrasts judged significant
            (after correction when ``p_corr != 0``).
          - ``anova_batches``: number of batch-related fixed-effect terms with
            p < 0.05 as reported by the full model's p-values.
          - ``Subj_Var``, ``Resid_Var``: variance components from the
            subject-only model (may be NaN on fit failure).
          - ``ICC``: intra-class correlation
            ``(Subj_Var / (Subj_Var + Resid_Var))``.
          - ``WCV``: within/between variance ratio ``(Resid_Var / Subj_Var)``.
          - For every variable in ``fix_eff``, the DataFrame will include
            ``<var>_est``, ``<var>_pval``, ``<var>_ciL``, ``<var>_ciU``.

        - ``model_defs`` is a list of dicts (one per IDP) capturing the three
          model formulas used for that feature.

    Raises
    ------
    ValueError
        If ``idp_matrix`` is not 2-D or if input sequence lengths mismatch the
        number of rows.
    KeyError
        If requested variables in ``fix_eff``, ``ran_eff``, or ``force_*``
        lists are not present in the assembled DataFrame.

    Notes
    -----
    - Column names exposed to users (for ``fix_eff`` / ``ran_eff`` /
      ``force_*``) are exactly: ``'subjects'``, ``'timepoints'``,
      ``'batches'`` — these names are inserted into the working DataFrame so
      callers should use them when referring to these variables.
    - The function reorders batch categories so the largest group becomes the
      reference level before fitting (helps stable parameterization of
      contrasts).
    - The primary grouping column for mixed models is the first valid entry of
      ``ran_eff`` (or ``'subjects'`` when ``ran_eff`` was not specified).
    - If model fitting fails for an IDP, the pipeline records NaNs for that
      IDP and continues (failures do not stop the whole run).
    - Pairwise contrasts are computed with ``pairwise_site_tests`` using the
      fit object's parameters and covariance; p-values are two-sided Wald
      z-tests.
    - Confidence intervals and p-values are extracted from the fitted
      ``statsmodels`` result objects when available; missing names or
      extraction failures result in NaNs for those fields.
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

# -----
# Helper: build fixed terms
# -----
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

# -----
# Helper: safe fit wrapper
# -----
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

# 
# Helper: build input df
# 
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

# 
# Main: AdditiveEffect_long (per-feature zscore behavior)
# 
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
    Test for additive (mean/location) batch effects per feature using mixed models.

    For each feature (IDP) this routine:

    - Builds a per-feature local DataFrame including the response, specified fixed
      predictors and the batch column; numeric predictors are z-scored per-feature.
    - Optionally z-scores the response per-feature when ``do_zscore=True``
      (default).
    - Fits a full mixed model ``lhs ~ <fixed_terms> + C(batch)`` with random
      effects given by ``ran_eff`` (defaults to ``idvar`` when ``ran_eff`` is
      None).
    - Fits a reduced mixed model ``lhs ~ <fixed_terms>`` (same random structure).
    - Primary test: likelihood-ratio test (LRT) using model log-likelihoods:
      ``LR = 2 * (llf_full - llf_reduced)``,
      ``df = n_levels(batch) - 1`` (fallback to 1 if unknown). If LRT is not
      available or fails, falls back to a multivariate Wald test on the
      batch-related parameters (or a pseudoinverse-based Wald if the covariance
      is singular).
    - Records test statistic, degrees of freedom, p-value and which method was
      used: ``"LRT"``, ``"Wald"``, or ``"Wald_pinv"``.

    Parameters
    ----------
    data : pd.DataFrame, optional
        If provided, used directly; otherwise ``idp_matrix`` + ``subjects`` +
        ``batch_name`` (and optional ``timepoints``) are used to construct the
        DataFrame.
    idp_matrix : ndarray, optional
        Shape ``(n_samples, n_features)``. Required when ``data`` is None.
    subjects : sequence, optional
        Subject/grouping IDs (required when ``data`` is None).
    timepoints : sequence, optional
        Timepoint labels (used only when building the DataFrame from arrays).
    batch_name : sequence, optional
        Batch labels (required when ``data`` is None).
    idp_names : iterable of str, optional
        Feature names when building from ``idp_matrix``.
    covariates : dict, optional
        Mapping of ``name -> sequence`` for additional covariates.
    idvar : str, optional
        Column name for subject IDs; defaults to ``'subject_ids'``.
    batchvar : str, optional
        Column name for batch labels; defaults to ``'batch'``.
    timevar : str, optional
        Column name for timepoints; defaults to ``'timepoints'``.
    fix_eff : iterable, optional
        Fixed-effect predictors; defaults to ``covariates.keys()`` when not
        supplied.
    ran_eff : iterable, optional
        Random-effect grouping variables; defaults to ``[idvar]``.
    do_zscore : bool, default True
        Z-score the response per-feature when True.
    reml : bool, default False
        Whether to fit MixedLM using REML.
    verbose : bool, default True
        Print progress / model formulas.

    Returns
    -------
    tuple of (pd.DataFrame, list)
        A tuple ``(results_df, model_defs)`` where ``results_df`` contains one
        row per feature with columns:

        - ``"Feature"``: feature name.
        - ``"TestStat"``: LR (or Wald) statistic.
        - ``"df"``: degrees of freedom used for the test.
        - ``"p-value"``: p-value for the test.
        - ``"method"``: ``"LRT"``, ``"Wald"``, ``"Wald_pinv"``, or ``None``
          if the test was not performed.

        ``model_defs`` is a list of dicts capturing per-feature formulas and
        settings.

    Raises
    ------
    KeyError
        If ``ran_eff`` variables are not found in the assembled DataFrame.
    ValueError
        If ``idp_matrix`` is not 2-D, or if input sequence lengths do not
        match the number of rows.

    Notes
    -----
    - Per-feature predictor z-scoring is always applied to numeric ``fix_eff``
      (local to each feature) via ``_build_fixed_formula_terms``.
    - ``do_zscore=True`` (default): z-scores the response per feature and uses
      the z-scored response (``z_<feature>``) as LHS. Set ``do_zscore=False``
      to keep original units.
    - ``reml=False`` (default): mixed models are fitted with REML disabled.
      Pass ``reml=True`` to use REML.
    - Rows with NaN responses are dropped per-feature. Features with fewer than
      3 retained rows are skipped and returned with NaNs.
    - If the full or reduced mixed fit fails, that feature is reported with NaNs.
    - The Wald fallback constructs contrasts for batch-related parameters found
      in the fitted parameter names and uses the parameter covariance matrix to
      compute a chi-square statistic; pseudoinverse is used if needed.
    - Because predictors are z-scored per-feature, coefficient magnitudes are
      comparable across features only in the z-scored scale (unless
      ``do_zscore=False``).
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
    Test for multiplicative (variance / heteroskedasticity) batch effects per feature.

    For each feature (IDP) this routine:

    - Builds a per-feature local DataFrame including the response, specified fixed
      predictors and the batch column; numeric predictors are z-scored per-feature.
    - Optionally z-scores the response per-feature when ``do_zscore=True``
      (default).
    - Fits a full mixed model ``lhs ~ <fixed_terms> + C(batch)`` with random
      effects given by ``ran_eff`` (defaults to ``idvar`` when ``ran_eff`` is
      None) — the residuals from this fit are used for variance comparisons.
    - Tests whether residual variability differs across batches using Fligner's
      test (a robust, non-parametric test for homogeneity of variances). The
      reported statistic is the Fligner chi-square and the p-value is from that
      test.
    - Records test statistic, DF (n_groups - 1), p-value and method
      ``"Fligner"``.

    Parameters
    ----------
    data : pd.DataFrame, optional
        If provided, used directly; otherwise ``idp_matrix`` + ``subjects`` +
        ``batch_name`` (and optional ``timepoints``) are used to construct the
        DataFrame.
    idp_matrix : ndarray, optional
        Shape ``(n_samples, n_features)``. Required when ``data`` is None.
    subjects : sequence, optional
        Subject/grouping IDs (required when ``data`` is None).
    timepoints : sequence, optional
        Timepoint labels (used only when building the DataFrame from arrays).
    batch_name : sequence, optional
        Batch labels (required when ``data`` is None).
    idp_names : iterable of str, optional
        Feature names when building from ``idp_matrix``.
    covariates : dict, optional
        Mapping of ``name -> sequence`` for additional covariates.
    idvar : str, optional
        Column name for subject IDs; defaults to ``'subject_ids'``.
    batchvar : str, optional
        Column name for batch labels; defaults to ``'batch'``.
    timevar : str, optional
        Column name for timepoints; defaults to ``'timepoints'``.
    fix_eff : iterable, optional
        Fixed-effect predictors; defaults to covariate keys plus
        ``timevar``/``batchvar`` when not supplied.
    ran_eff : iterable, optional
        Random-effect grouping variables; defaults to ``[idvar]``.
    do_zscore : bool, default True
        Z-score the response per-feature when True.
    reml : bool, default False
        Whether to fit MixedLM using REML.
    verbose : bool, default True
        Print progress / model formulas.

    Returns
    -------
    tuple of (pd.DataFrame, list)
        A tuple ``(results_df, model_defs)`` where ``results_df`` contains one
        row per feature with columns:

        - ``"Feature"``: feature name.
        - ``"ChiSq"``: Fligner test statistic (chi-square).
        - ``"DF"``: degrees of freedom (n_groups - 1).
        - ``"p-value"``: p-value from Fligner's test.
        - ``"method"``: ``"Fligner"`` (or ``None`` if the test could not be
          run).

        ``model_defs`` is a list of dicts capturing per-feature formula and
        settings.

    Raises
    ------
    KeyError
        If ``fix_eff`` or ``ran_eff`` variables are not found in the assembled
        DataFrame.
    ValueError
        If ``idp_matrix`` is not 2-D, or if input sequence lengths do not
        match the number of rows.

    Notes
    -----
    - Per-feature predictor z-scoring is always applied to numeric ``fix_eff``
      (local to each feature) via ``_build_fixed_formula_terms``.
    - ``do_zscore=True`` (default): z-scores the response per feature and uses
      the z-scored response (``z_<feature>``) as LHS. Set ``do_zscore=False``
      to keep original units.
    - ``reml=False`` (default): MixedLM fits are run with REML disabled.
      Pass ``reml=True`` to change that.
    - ``ran_eff`` defaults to ``[idvar]`` (where ``idvar`` defaults to
      ``'subject_ids'``).
    - Residuals used for the variance test are extracted from the full
      mixed-model fit. If fitting fails for a given feature, that feature is
      returned with NaNs.
    - The Fligner test requires at least two groups and non-empty residual
      samples for each group; otherwise the test is not run for that feature.
    - Rows with NaN responses are dropped per-feature. Features with fewer than
      3 retained rows are skipped and returned with NaNs.
    - Because predictors and possibly responses are z-scored per-feature, the
      residuals used for heteroskedasticity testing are on the z-scored scale
      when ``do_zscore=True``.
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
