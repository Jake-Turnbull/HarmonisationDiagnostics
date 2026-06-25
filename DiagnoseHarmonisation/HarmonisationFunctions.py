"""

Script containing self contained harmonisation functions that can be used in conjunction with the diagnostic tools:
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import math
import re
import sys
import warnings

import numpy as np
import numpy.linalg as la
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.stats import chi2, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

 

def design_mat(mod, numerical_covariates, batch_levels):
    """Construct design matrix for ComBat, ensuring batch levels are in the correct order and handling numerical covariates."""
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design

# --------------------- Helper functions ---------------------
# Translated from MATLAB, need to have concistency checked with NeuroComBat
def aprior(delta_hat):
    """Calculate the aprior parameter for the inverse gamma distribution based on the method of moments."""
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    """Calculate the bprior parameter for the inverse gamma distribution based on the method of moments."""
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    """Calculate the posterior mean for the batch effect parameters."""
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    """Calculate the posterior variance for the batch effect parameters."""
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def itSol(sdat_batch, gamma_hat, delta_hat, gamma_bar, t2, a, b,
          conv=0.001, return_hist=False):
    """
    Iteratively solve for posterior mean and variance of batch-effect parameters.

    If return_hist=True, also returns:
        count : number of iterations
        hist  : dictionary storing EB values at each iteration
    """
    import numpy as np

    g_old = gamma_hat.copy()
    d_old = delta_hat.copy()
    change = 1
    count = 0
    n = sdat_batch.shape[1]

    hist = {
        "iter": [],
        "g": [],
        "d": [],
        "sum2": [],
        "delta_hat": [],
        "change": []
    }

    while change > conv:
        g_new = postmean(gamma_hat, gamma_bar, n, d_old, t2)

        sum2 = np.sum((sdat_batch - g_new[:, None]) ** 2, axis=1)

        d_new = postvar(sum2, n, a, b)

        change = max(
            np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), np.finfo(float).eps)),
            np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), np.finfo(float).eps))
        )

        count += 1

        hist["iter"].append(count)
        hist["g"].append(g_new.copy())
        hist["d"].append(d_new.copy())
        hist["sum2"].append(sum2.copy())

        if n > 1:
            hist["delta_hat"].append(sum2 / (n - 1))
        else:
            hist["delta_hat"].append(np.full_like(sum2, np.nan))

        hist["change"].append(change)

        g_old = g_new
        d_old = d_new

        if count > 100:
            print("Warning: itSol did not converge after 100 iterations")
            break

    adjust = np.vstack([g_new, d_new])

    if return_hist:
        return adjust, count, hist

    return adjust

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

# ----------------------------- Main functions -----------------------------
# Define ComBat harmonisation function
# Translated from MATLAB, need to have concistency checked with NeuroComBat
def combat(data, batch, mod=[], parametric=True,
           DeltaCorrection=True, UseEB=True, ReferenceBatch=None,
           RegressCovariates=False, GammaCorrection=True, covbat_mode=False, return_priors=False):
    """
    Run ComBat harmonisation on the data and return the harmonized data.

    This version accepts numpy arrays or pandas DataFrame/Series for data, batch, and mod.
    If a DataFrame is supplied, columns are treated as samples (so data.shape == (n_features, n_samples)).
    The function will auto-transpose data or mod if it detects that samples were provided as rows.
    The returned bayesdata is the same type as input data (DataFrame -> DataFrame, ndarray -> ndarray).

    Note: helper functions aprior, bprior, itSol must be defined in scope.

    Args:
        data (np.array or pd.DataFrame): The data matrix to be harmonized, with shape (n_features, n_samples).
        batch (np.array or pd.Series): A vector of batch labels for each sample, with length n_samples.
        mod (np.array or pd.DataFrame, optional): An optional design matrix of covariates to adjust for, with shape (n_samples, n_covariates).
        parametric (bool, optional): Whether to use parametric adjustments. Default is True.
        DeltaCorrection (bool, optional): Whether to apply delta (scale) correction. Default is True.
        UseEB (bool, optional): Whether to use empirical Bayes adjustments. Default is True.
        ReferenceBatch (str or int, optional): If provided, the name or index of the reference batch to use for fitting priors. Default is None (no reference).
        RegressCovariates (bool, optional): Whether to regress out covariate effects before harmonisation. Default is False.
        GammaCorrection (bool, optional): Whether to apply gamma (mean) correction. Default is True.
        covbat_mode (bool, optional): Whether to run in CovBat mode which includes additional covariance correction steps. Default is False.
        return_priors (bool, optional): Whether to return the estimated parameters from the ComBat model along with the harmonized data. Default is False.

    Returns:
        bayesdata (np.array or pd.DataFrame): The harmonized data, in the same format as the input data.
        priors (dict, optional): A dictionary containing the estimated parameters from the ComBat model, including:
            - gamma_hat: raw batch effect mean estimates (n_batch, n_features)
            - delta_hat: raw batch effect variance estimates (n_batch, n_features)
            - gamma_star: empirical Bayes adjusted batch effect means (n_batch, n_features)
            - delta_star: empirical Bayes adjusted batch effect variances (n_batch, n_features)
            - gamma_bar: mean of gamma_hat across batches (n_batch,)
            - t2: variance of gamma_hat across batches (n_batch,)
            - a_prior: aprior parameters for each batch (n_batch,)
            - b_prior: bprior parameters for each batch (n_batch,)  
 
    Note:
    If using this version of ComBat, please cite:

    Jean-Philippe Fortin, Drew Parker, Birkan Tunc, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, Robert T Schultz, Ragini Verma, Russell T Shinohara. Harmonisation Of Multi-Site Diffusion Tensor Imaging Data. NeuroImage, 161, 149-170, 2017
    Jean-Philippe Fortin, Nicholas Cullen, Yvette I. Sheline, Warren D. Taylor, Irem Aselcioglu, Philip A. Cook, Phil Adams, Crystal Cooper, Maurizio Fava, Patrick J. McGrath, Melvin McInnis, Mary L. Phillips, Madhukar H. Trivedi, Myrna M. Weissman, Russell T. Shinohara. Harmonisation of cortical thickness measurements across scanners and sites. NeuroImage, 167, 104-120, 2018
    W. Evan Johnson and Cheng Li, Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1):118-127, 2007.

    """
    import pandas as pd
    import numpy as np
    
    # Remember whether inputs were pandas objects so we can restore types/labels on output
    dat_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # Keep original labels (if any) to restore later
    dat_orig_index = data.index if dat_was_df else None
    dat_orig_columns = data.columns if dat_was_df else None
    batch_index = batch.index if batch_was_series else None
    mod_orig_index = mod.index if mod_was_df else None
    mod_orig_columns = mod.columns if mod_was_df else None

    # Convert pandas -> numpy, but allow transposing if the user supplied samples as rows
    # For data: desired internal shape = (n_features, n_samples) (rows=features, cols=samples)
    if dat_was_df:
        dat_np = data.values.astype(float)
        # if batch length matches number of rows, assume user gave samples as rows and transpose
        len_batch = len(batch)
        if len_batch == dat_np.shape[0] and len_batch != dat_np.shape[1]:
            dat_np = dat_np.T
            dat_transposed = True
        else:
            dat_transposed = False
    else:
        dat_np = np.asarray(data, dtype=float)
        dat_transposed = False

    # Normalize batch into 1D numpy array
    if batch_was_series:
        batch_np = batch.values
    else:
        batch_np = np.asarray(batch)
    batch_np = batch_np.ravel()

    # If dat_np and batch lengths mismatch, try to detect transposed data (samples as rows)
    if dat_np.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')

    # If batch length matches rows instead of columns, transpose dat_np
    if batch_np.shape[0] == dat_np.shape[0] and batch_np.shape[0] != dat_np.shape[1]:
        dat_np = dat_np.T
        dat_transposed = not dat_transposed  # flip if we already flipped earlier

    # Now dat_np shape[1] should equal batch length
    if dat_np.shape[1] != batch_np.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')

    # Handle mod (design covariates). Desired internal shape: (n_samples, n_covariates)
    if mod is None:
        mod_np = None
    else:
        if mod_was_df:
            mod_np = mod.values.astype(float)
            # If mod rows equal n_samples -> OK; else if mod.columns equal n_samples -> transpose
            n_samples = dat_np.shape[1]
            if mod_np.shape[0] == n_samples:
                pass
            elif mod_np.shape[1] == n_samples:
                mod_np = mod_np.T
            else:
                raise ValueError('Design matrix "mod" shape not compatible with data samples.')
        else:
            mod_np = np.asarray(mod, dtype=float)
            if mod_np.ndim == 1:
                # single covariate vector
                if mod_np.shape[0] == dat_np.shape[1]:
                    mod_np = mod_np.reshape(-1, 1)
                elif mod_np.shape[0] == dat_np.shape[0]:
                    # maybe passed per-feature by mistake
                    mod_np = mod_np.reshape(1, -1).T
                else:
                    raise ValueError('Design matrix "mod" length is incompatible with number of samples.')
            else:
                # 2D array: check orientation
                if mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] == dat_np.shape[1]:
                    # mod provided as (n_covariates x n_samples) -> transpose
                    mod_np = mod_np.T
                elif mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] != dat_np.shape[1]:
                    raise ValueError('Design matrix "mod" rows must match number of samples in "data".')

    # Use these working arrays from now on
    data = dat_np
    batch = batch_np
    mod = mod_np

    # Check the given parameters and print status messages
    if ReferenceBatch is None:
        print('Reference batch not given, defaulting to no reference')
    else:
        print(f'ReferenceBatch = {ReferenceBatch} -- fitting prior estimates using this batch and leaving batch unchanged')

    if not UseEB:
        print('Empirical Bayes set to false, using first estimates from raw mean and variances')
    else:
        print('Empirical Bayes set to true')

    if RegressCovariates:
        print('Regress Covariates set to true, skipping re-addition of OLS covariate estimates ')

    if not DeltaCorrection:
        print('Delta correction set to False, applying no delta (scale) correction on data')

    if not GammaCorrection:
        print('Gamma correction set to False, applying no gamma (mean) correction on data')

    # Basic input validation (after conversions)
    if data.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')
    if batch.ndim != 1:
        raise ValueError('Batch vector "batch" must be 1-dimensional (samples,).')
    if data.shape[1] != batch.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')
    if mod is not None:
        if mod.ndim != 2:
            raise ValueError('Design matrix "mod" must be 2-dimensional (samples x covariates).')
        if mod.shape[0] != data.shape[1]:
            raise ValueError('Number of samples in "data" must match number of rows in "mod" design matrix.')

    # --------------------- Begin ComBat core logic ---------------------

    # Compute SDs across samples for each feature (row)
    sds = np.std(data, axis=1, ddof=1)
    wh = np.where(sds == 0)[0]
    if wh.size > 0:
        raise ValueError('Error. There are rows with constant values across samples. Remove these rows and rerun ComBat.')

    # Convert batch vector to categorical and create dummy variables
    batch_cat = pd.Categorical(batch)
    batchmod = pd.get_dummies(batch_cat, drop_first=False).values  # shape (n_samples, n_batch)

    # Number of batches
    n_batch = batchmod.shape[1]
    levels = np.array(batch_cat.categories)
    print(f'[combat] Found {n_batch} batches')

    # Create list of arrays each containing sample indices for a batch
    batches = [np.where(batch == lev)[0] for lev in levels]

    # Size of each batch and total number of samples
    n_batches = np.array([len(b) for b in batches])
    n_array = np.sum(n_batches)

    # Construct design matrix including batch and additional covariates (mod)
    if mod is None:
        mod_arr = np.zeros((data.shape[1], 0))
    else:
        mod_arr = np.asarray(mod, dtype=float)
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)

    design = np.hstack([batchmod, mod_arr])  # shape (n_samples, n_batch + n_cov)

    # Remove intercept column if present
    intercept = np.ones((n_array, 1))
    cols_to_keep = []
    for j in range(design.shape[1]):
        if not np.allclose(design[:, j], intercept.ravel()):
            cols_to_keep.append(j)
    design = design[:, cols_to_keep]

    print(f'[combat] Adjusting for {design.shape[1] - n_batch} covariate(s) of covariate level(s)')

    # Check for confounding between batch and covariates
    if np.linalg.matrix_rank(design) < design.shape[1]:
        nn = design.shape[1]
        if nn == (n_batch + 1):
            raise ValueError('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        if nn > (n_batch + 1):
            temp = design[:, (n_batch):nn]
            if np.linalg.matrix_rank(temp) < temp.shape[1]:
                raise ValueError('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else:
                raise ValueError('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')

    print('[combat] Standardizing Data across features')

    # Estimate coefficients B_hat using least squares: B_hat = inv(design' * design) * design' * data'
    XtX = design.T @ design
    inv_XtX = np.linalg.pinv(XtX) # Find the pseudo-inverse in case XtX is singular
    B_hat = inv_XtX @ design.T @ data.T  # shape (k, n_features) 

    # Reference batch handling
    # Storage for EB iteration history
    if ReferenceBatch is not None:
        try:
            ref_idx = int(np.where(levels == ReferenceBatch)[0][0])
        except Exception:
            raise ValueError('ReferenceBatch not found in batch levels.')

        ref_samples = batches[ref_idx]
        ref_batch_effect = B_hat[ref_idx, :]

        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            Cov_effects = (tmp @ B_hat).T
        else:
            Cov_effects = np.zeros((data.shape[0], data.shape[1]))

        design_ref = design[ref_samples, :]
        predicted_ref = (design_ref @ B_hat).T
        residuals_ref = data[:, ref_samples] - predicted_ref
        var_ref = np.mean(residuals_ref ** 2, axis=1)

        stand_mean = np.tile(ref_batch_effect[:, None], (1, n_array))
        stand_mean = stand_mean + Cov_effects
        var_pooled = var_ref.copy()
        print(f'The size of the var_pooled array is {var_pooled.shape}')
    else:
        n_features = data.shape[0]
        n_samples = data.shape[1]
        XtX = design.T @ design
        inv_XtX = np.linalg.pinv(XtX)
        B_hat = inv_XtX @ design.T @ data.T
        grand_mean = (n_batches / n_array) @ B_hat[0:n_batch, :]
        predicted = (design @ B_hat).T
        resid = data - predicted
        var_pooled = np.mean(resid ** 2, axis=1)
        if np.any(var_pooled == 0):
            nonzeros = var_pooled[var_pooled != 0]
            if nonzeros.size > 0:
                var_pooled[var_pooled == 0] = np.median(nonzeros)
            else:
                var_pooled[var_pooled == 0] = 1e-6

        stand_mean = np.tile(grand_mean[:, None], (1, n_array))
        if design.shape[1] > n_batch:
            tmp = design.copy()
            tmp[:, :n_batch] = 0
            stand_mean = stand_mean + (tmp @ B_hat).T

    # Optional: regress covariates
    if design.shape[1] > n_batch:
        X_cov = design[:, n_batch:]
        X_cov = X_cov - np.mean(X_cov, axis=0, keepdims=True)
        B_cov = B_hat[n_batch:, :]
        Cov_effects = (X_cov @ B_cov).T
    else:
        Cov_effects = np.zeros_like(data)

    # Standardize the data, adding in small constant to avoid division by zero
    s_data = (data - stand_mean) / (np.sqrt(var_pooled)[:, None] + 1e-8)

    # Estimate batch effect parameters using least squares
    print('[combat] Fitting L/S model and finding priors')
    batch_design = design[:, :n_batch]  # samples x n_batch
    XtX_b = batch_design.T @ batch_design
    inv_XtX_b = np.linalg.pinv(XtX_b)
    gamma_hat = inv_XtX_b @ batch_design.T @ s_data.T  # shape (n_batch, n_features)
    print(f'Size of gamma hat: {gamma_hat.shape}')

    # Estimate batch-specific variances
    delta_hat = np.zeros((n_batch, data.shape[0]))
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) > 1:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=1)
        else:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=0) + 1e-6

    print(f'Size of delta hat: {delta_hat.shape}')

    # Compute hyperparameters
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    t2[t2 == 0] = 1e-6

    a_prior = np.zeros(n_batch)
    b_prior = np.zeros(n_batch)
    for i in range(n_batch):
        a_prior[i] = aprior(delta_hat[i, :])
        b_prior[i] = bprior(delta_hat[i, :])

    # Apply empirical Bayes estimates (parametric)
    # Storage for EB iteration history
    eb_hist = {
        "by_batch": {},
        "counts": {},
        "levels": levels.copy()
    }

    # Apply empirical Bayes estimates (parametric)
    if parametric:
        print('[combat] Finding parametric adjustments')
        gamma_star = np.zeros_like(gamma_hat)
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for i in range(n_batch):
            indices = batches[i]
            if len(indices) == 0:
                continue
            temp, count, hist = itSol(
                s_data[:, indices],
                gamma_hat[i, :],
                delta_hat[i, :],
                gamma_bar[i],
                t2[i],
                a_prior[i],
                b_prior[i],
                conv=0.001,
                return_hist=True
            )

            gamma_star[i, :] = temp[0, :]
            delta_star[i, :] = temp[1, :]

            batch_label = levels[i]
            eb_hist["by_batch"][batch_label] = hist
            eb_hist["counts"][batch_label] = count

        if ReferenceBatch is not None:
            gamma_star[ref_idx, :] = np.zeros(data.shape[0])
            delta_star[ref_idx, :] = np.ones(data.shape[0])
    else:
        gamma_star = gamma_hat.copy()
        delta_star = delta_hat.copy()

    print('Size of gamma_star:', gamma_star.shape)
    bayesdata = s_data.copy()


    if not UseEB:
        print('Discounting the EB adjustments and using Raw estimates, this is not advised')
        delta_star = delta_hat.copy()
        gamma_star = gamma_hat.copy()
    if DeltaCorrection:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None]) / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
        else:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = bayesdata[:, indices] / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
    else:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None])
        else:
            print('Warning: Both Gamma and delta have been set to false, no ComBat adjustments have been applied')
    if covbat_mode:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # --- assume these are provided:
        # bayesdata : numpy array shape (n_features, n_samples)
        # batch : whatever your function needs
        # stand_mean : either shape (n_features,) or (n_features, 1)
        # data (optional) : original pandas DataFrame if you want to keep sample names
        # combat_for_covbat : your function (may accept numpy or pandas)

        # Optional: keep sample names in an array if you still want them
        # If you don't have `data`, just omit this.
        try:
            sample_names = np.asarray(data.columns)
        except Exception:
            sample_names = None

        print('[covbat] Adjusting the Data')

        # CovBat adjustment via PCA
        comdata = bayesdata.T                      # shape: (n_samples, n_features)
        bmu = np.mean(comdata, axis=0)             # mean across samples -> shape (n_features,)

        # standardize data before PCA
        scaler = StandardScaler()
        comdata_std = scaler.fit_transform(comdata)  # (n_samples, n_features)

        pca = PCA()
        pca.fit(comdata_std)
        pc_comp = pca.components_                    # (n_components, n_features)

        # full_scores as numpy array: shape (n_components, n_samples)
        full_scores = pca.transform(comdata_std).T   # pca.transform -> (n_samples, n_components) -> .T -> (n_components, n_samples)

        # Hard code pct_var for now:
        pct_var = 0.95
        var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
        # npc: number of PCs needed to exceed pct_var
        npc = int(np.min(np.where(var_exp > pct_var))) + 1

        # slice the scores to the first npc components
        scores = full_scores[:npc, :]               # shape (npc, n_samples)

        # If combat_for_covbat accepts numpy arrays, call directly:
        
        scores_com = combat(scores, batch, mod=None, parametric=True,UseEB=False)
        
            # If it expects a pandas DataFrame, convert temporarily:
        import pandas as pd
        # Output is a numpy array:
        
        full_scores[:npc, :] = scores_com

        # prepare output array (same shape as bayesdata)
        x_covbat = np.zeros_like(bayesdata)         # shape (n_features, n_samples)

        # project back to original space
        # full_scores.T shape (n_samples, n_components)
        # pc_comp shape (n_components, n_features)
        # np.dot -> (n_samples, n_features) -> .T -> (n_features, n_samples)
        proj = np.dot(full_scores.T, pc_comp).T     # shape (n_features, n_samples)

        # inverse transform the standardization
        # scaler was fit on comdata (n_samples, n_features) so we need to pass proj.T (n_samples, n_features)
        x_recon = scaler.inverse_transform(proj.T).T  # back to shape (n_features, n_samples)

        # add reconstructed signal and the stored mean (stand_mean)
        x_covbat += x_recon

        # ensure stand_mean broadcasts across columns: make it (n_features, 1) if it's (n_features,)
        """stand_mean_arr = np.asarray(stand_mean)
        if stand_mean_arr.ndim == 1:
            stand_mean_arr = stand_mean_arr.reshape(-1, 1)   # (n_features, 1)

        x_covbat += stand_mean_arr    # broadcasting across columns
        """

        # final output
        bayesdata = x_covbat.copy()
        print('[covbat] Finished CovBat adjustment')

    # Transform data back to original scale
    if RegressCovariates:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + (stand_mean - Cov_effects)
    else:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + stand_mean
    
    # Flip bayes data back if we transposed at the start
    if dat_transposed:
        bayesdata = bayesdata.T
    if return_priors:
        priors = {
            "levels": levels,
            "gamma_bar": gamma_bar,
            "t2": t2,
            "a_prior": a_prior,
            "b_prior": b_prior,
            "gamma_hat": gamma_hat,
            "delta_hat": delta_hat,
            "gamma_star": gamma_star,
            "delta_star": delta_star,
            "num_iter": eb_hist["counts"],
            "hist": eb_hist
        }

        output = {
            "bayesdata": bayesdata,
            "B_hat": B_hat,
            "priors": priors,

            # Optional flat copies for backwards compatibility
            "gamma_bar": gamma_bar,
            "t2": t2,
            "a_prior": a_prior,
            "b_prior": b_prior,
            "delta_hat": delta_hat,
            "gamma_hat": gamma_hat,
            "delta_star": delta_star,
            "gamma_star": gamma_star,
            "hist": eb_hist
        }

        return output
    else:
        return bayesdata


def combat_modular(
    data,
    batch,
    mod=None,
    mean_model: str = "ols",
    gam_opts: Optional[Dict[str, Any]] = None,
    prior_mode: str = "global",
    prior_weight_methods: Optional[Sequence[str]] = None,
    prior_weight_opts: Optional[Dict[str, Any]] = None,
    parametric: bool = True,
    DeltaCorrection: bool = True,
    UseEB: bool = True,
    ReferenceBatch: Optional[Any] = None,
    RegressCovariates: bool = False,
    GammaCorrection: bool = True,
    covbat_mode: bool = False,
    return_priors: bool = False,
    **kwargs,
):
    """
    Modular ComBat entrypoint.

    This function is a modular wrapper around the existing `combat` implementation.
    By default (`mean_model='ols'` and `prior_mode='global'`) it calls the
    original `combat` function to preserve exact baseline behaviour. Experimental
    modular paths (GAM mean models, local priors) will be implemented here
    incrementally.
    
        Parameters (additional options):
            - `mean_model`: 'ols' (default) or 'gam' to fit flexible spline-based covariate effects.
            - `prior_mode`: 'global' (default) or 'local' to enable feature-wise local priors.
            - `prior_weight_methods`: sequence of weight method names (see `_construct_local_priors`).
            - `prior_weight_opts`: dict of options passed to `_construct_local_priors` (method weights, spatial coords, directional params).

        Returns:
            - If `return_priors=True`, a dict with keys `bayesdata`, `B_hat`, `priors` (includes `local_priors` when requested).
            - Otherwise returns the harmonized data array.
    """
    # Fast-path: preserve baseline behaviour exactly by delegating to legacy combat
    if (mean_model == "ols") and (prior_mode == "global"):
        return combat(
            data=data,
            batch=batch,
            mod=mod,
            parametric=parametric,
            DeltaCorrection=DeltaCorrection,
            UseEB=UseEB,
            ReferenceBatch=ReferenceBatch,
            RegressCovariates=RegressCovariates,
            GammaCorrection=GammaCorrection,
            covbat_mode=covbat_mode,
            return_priors=return_priors,
        )

    # For non-default configurations (GAM mean model or local priors) we perform
    # the modular pipeline here so we can construct and apply feature-wise priors.
    import pandas as pd
    from patsy import dmatrix

    # --- Normalize inputs (similar to `combat`) ---
    dat_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # Convert data to internal shape (n_features, n_samples)
    if dat_was_df:
        dat_np = data.values.astype(float)
        len_batch = len(batch)
        if len_batch == dat_np.shape[0] and len_batch != dat_np.shape[1]:
            dat_np = dat_np.T
            dat_transposed = True
        else:
            dat_transposed = False
    else:
        dat_np = np.asarray(data, dtype=float)
        dat_transposed = False

    # Normalize batch
    if batch_was_series:
        batch_np = batch.values
    else:
        batch_np = np.asarray(batch)
    batch_np = batch_np.ravel()

    if dat_np.ndim != 2:
        raise ValueError('Data matrix "data" must be 2-dimensional (features x samples).')

    if batch_np.shape[0] == dat_np.shape[0] and batch_np.shape[0] != dat_np.shape[1]:
        dat_np = dat_np.T
        dat_transposed = not dat_transposed

    if dat_np.shape[1] != batch_np.shape[0]:
        raise ValueError('Number of samples in "data" must match length of "batch" vector.')

    # Prepare covariate DataFrame (possibly spline-expanded for GAM)
    if mod is None:
        mod_df = None
    else:
        if mean_model == "gam":
            # ensure mod is samples x covariates
            if mod_was_df:
                mod_df = mod.copy()
                if mod_df.shape[0] != dat_np.shape[1] and mod_df.shape[1] == dat_np.shape[1]:
                    mod_df = mod_df.T
            else:
                mod_np = np.asarray(mod)
                if mod_np.ndim == 1:
                    mod_np = mod_np.reshape(-1, 1)
                elif mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] == dat_np.shape[1]:
                    mod_np = mod_np.T
                mod_df = pd.DataFrame(mod_np, columns=[f"cov{i}" for i in range(mod_np.shape[1])])

            # expand numeric covariates with spline bases
            basis_list = []
            for col in mod_df.columns:
                if pd.api.types.is_numeric_dtype(mod_df[col].dtype):
                    n_splines = (gam_opts.get("n_splines") if gam_opts else None) or 6
                    degree = (gam_opts.get("degree") if gam_opts else None) or 3
                    # use bs() basis via patsy
                    formula = f"bs({col}, df={n_splines}, degree={degree}, include_intercept=False) - 1"
                    try:
                        basis = dmatrix(formula, data=mod_df, return_type="dataframe")
                    except Exception:
                        basis = mod_df[[col]]
                    basis_list.append(basis)
                else:
                    dummies = pd.get_dummies(mod_df[col], prefix=col, drop_first=False)
                    basis_list.append(dummies)

            if len(basis_list) > 0:
                mod_df = pd.concat(basis_list, axis=1)
            else:
                mod_df = pd.DataFrame(index=mod_df.index)
        else:
            # OLS: ensure mod is samples x covariates DataFrame
            if mod_was_df:
                mod_df = mod.copy()
                if mod_df.shape[0] != dat_np.shape[1] and mod_df.shape[1] == dat_np.shape[1]:
                    mod_df = mod_df.T
            else:
                mod_np = np.asarray(mod)
                if mod_np.ndim == 1:
                    mod_np = mod_np.reshape(-1, 1)
                elif mod_np.shape[0] != dat_np.shape[1] and mod_np.shape[1] == dat_np.shape[1]:
                    mod_np = mod_np.T
                mod_df = pd.DataFrame(mod_np, columns=[f"cov{i}" for i in range(mod_np.shape[1])])

    # Build batch design and full design matrix
    batch_cat = pd.Categorical(batch_np)
    batchmod = pd.get_dummies(batch_cat, drop_first=False).values
    n_batch = batchmod.shape[1]
    levels = np.array(batch_cat.categories)
    batches = [np.where(batch_np == lev)[0] for lev in levels]

    if mod_df is None:
        mod_arr = np.zeros((dat_np.shape[1], 0))
    else:
        mod_arr = mod_df.values
        if mod_arr.ndim == 1:
            mod_arr = mod_arr.reshape(-1, 1)

    design = np.hstack([batchmod, mod_arr])

    # Remove intercept columns (constant columns)
    intercept = np.ones((design.shape[0], 1))
    cols_to_keep = []
    for j in range(design.shape[1]):
        if not np.allclose(design[:, j], intercept.ravel()):
            cols_to_keep.append(j)
    design = design[:, cols_to_keep]

    # Check for confounding
    if np.linalg.matrix_rank(design) < design.shape[1]:
        nn = design.shape[1]
        if nn == (n_batch + 1):
            raise ValueError('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
        if nn > (n_batch + 1):
            temp = design[:, (n_batch):nn]
            if np.linalg.matrix_rank(temp) < temp.shape[1]:
                raise ValueError('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
            else:
                raise ValueError('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')

    # Fit mean model and compute standardization
    B_hat, stand_mean, var_pooled, Cov_effects = _fit_mean_model(dat_np, design, n_batch, batches, ReferenceBatch)
    s_data = _standardize(dat_np, stand_mean, var_pooled)

    # Estimate raw batch parameters
    gamma_hat, delta_hat = _estimate_raw_batch(s_data, design, n_batch, batches)

    # Global priors
    gamma_bar_global, t2_global, a_prior_global, b_prior_global = _construct_priors(gamma_hat, delta_hat)

    # If local prior mode requested, construct local priors using weighted pooling
    if prior_mode == "local":
        local_priors = _construct_local_priors(
            s_data,
            gamma_hat,
            delta_hat,
            var_pooled,
            weight_methods=(prior_weight_methods or ["correlation_similarity"]),
            weight_opts=(prior_weight_opts or {}),
            global_priors=(gamma_bar_global, t2_global, a_prior_global, b_prior_global),
        )
        gamma_bar_used = local_priors["gamma_bar_local"]
        t2_used = local_priors["t2_local"]
        a_prior_used = local_priors["a_prior_local"]
        b_prior_used = local_priors["b_prior_local"]
    else:
        # broadcast global scalars per-feature for compatibility with itSol
        gamma_bar_used = np.tile(gamma_bar_global[:, None], (1, gamma_hat.shape[1]))
        t2_used = np.tile(t2_global[:, None], (1, gamma_hat.shape[1]))
        a_prior_used = np.tile(a_prior_global[:, None], (1, gamma_hat.shape[1]))
        b_prior_used = np.tile(b_prior_global[:, None], (1, gamma_hat.shape[1]))
        # no local priors requested; global priors will be broadcast per-feature
    # Prepare storage for EB-adjusted parameters (common path)
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    eb_hist = {"by_batch": {}, "counts": {}}
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) == 0:
            continue
        g_bar_i = gamma_bar_used[i, :]
        t2_i = t2_used[i, :]
        a_i = a_prior_used[i, :]
        b_i = b_prior_used[i, :]
        temp, count, hist = itSol(
            s_data[:, indices],
            gamma_hat[i, :],
            delta_hat[i, :],
            g_bar_i,
            t2_i,
            a_i,
            b_i,
            conv=0.001,
            return_hist=True,
        )
        gamma_star[i, :] = temp[0, :]
        delta_star[i, :] = temp[1, :]
        eb_hist["by_batch"][levels[i]] = hist
        eb_hist["counts"][levels[i]] = count

    # Reconstruct final harmonized data (add covariate effects back if required)
    stand_mean_with_cov = stand_mean
    bayesdata = _reconstruct(s_data, gamma_star, delta_star, batches, var_pooled, stand_mean_with_cov, Cov_effects, DeltaCorrection=DeltaCorrection, GammaCorrection=GammaCorrection, RegressCovariates=RegressCovariates)

    if dat_transposed:
        bayesdata = bayesdata.T

    # Build priors dict including both global and local when requested
    priors = {
        "levels": levels,
        "gamma_bar": gamma_bar_global,
        "t2": t2_global,
        "a_prior": a_prior_global,
        "b_prior": b_prior_global,
        "gamma_hat": gamma_hat,
        "delta_hat": delta_hat,
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "num_iter": eb_hist.get("counts", {}),
        "hist": eb_hist,
    }

    if prior_mode == "local":
        priors["local_priors"] = local_priors

    output = {
        "bayesdata": bayesdata,
        "B_hat": B_hat,
        "priors": priors,
        # legacy flat copies
        "gamma_bar": gamma_bar_global,
        "t2": t2_global,
        "a_prior": a_prior_global,
        "b_prior": b_prior_global,
        "delta_hat": delta_hat,
        "gamma_hat": gamma_hat,
        "delta_star": delta_star,
        "gamma_star": gamma_star,
        "hist": eb_hist,
    }

    if return_priors:
        return output
    return bayesdata


def _fit_mean_model(data, design, n_batch, batches, ReferenceBatch=None):
    """
    Fit an OLS mean model and compute standardization terms.

    Parameters
    - data: (n_features, n_samples)
    - design: (n_samples, n_covariates) with first `n_batch` cols as batch dummies
    - n_batch: number of batch columns in design
    - batches: list of index arrays for each batch

    Returns
    - B_hat: (n_covariates, n_features)
    - stand_mean: (n_features, n_samples)
    - var_pooled: (n_features,)
    - Cov_effects: (n_features, n_samples)
    """
    # Estimate coefficients B_hat using least squares
    XtX = design.T @ design
    inv_XtX = np.linalg.pinv(XtX)
    B_hat = inv_XtX @ design.T @ data.T  # shape (k, n_features)

    # Compute grand mean and pooled variance
    n_batches = np.array([len(b) for b in batches])
    n_array = np.sum(n_batches)
    grand_mean = (n_batches / n_array) @ B_hat[0:n_batch, :]
    predicted = (design @ B_hat).T
    resid = data - predicted
    var_pooled = np.mean(resid ** 2, axis=1)
    if np.any(var_pooled == 0):
        nonzeros = var_pooled[var_pooled != 0]
        if nonzeros.size > 0:
            var_pooled[var_pooled == 0] = np.median(nonzeros)
        else:
            var_pooled[var_pooled == 0] = 1e-6

    stand_mean = np.tile(grand_mean[:, None], (1, n_array))
    if design.shape[1] > n_batch:
        tmp = design.copy()
        tmp[:, :n_batch] = 0
        stand_mean = stand_mean + (tmp @ B_hat).T

    # Covariate effects (if any)
    if design.shape[1] > n_batch:
        X_cov = design[:, n_batch:]
        X_cov = X_cov - np.mean(X_cov, axis=0, keepdims=True)
        B_cov = B_hat[n_batch:, :]
        Cov_effects = (X_cov @ B_cov).T
    else:
        Cov_effects = np.zeros_like(data)

    return B_hat, stand_mean, var_pooled, Cov_effects


def _standardize(data, stand_mean, var_pooled):
    """Standardize data using provided mean and pooled variance."""
    s_data = (data - stand_mean) / (np.sqrt(var_pooled)[:, None] + 1e-8)
    return s_data


def _estimate_raw_batch(s_data, design, n_batch, batches):
    """Estimate raw batch-wise means and variances from standardized data."""
    batch_design = design[:, :n_batch]
    XtX_b = batch_design.T @ batch_design
    inv_XtX_b = np.linalg.pinv(XtX_b)
    gamma_hat = inv_XtX_b @ batch_design.T @ s_data.T  # shape (n_batch, n_features)

    delta_hat = np.zeros((n_batch, s_data.shape[0]))
    for i in range(n_batch):
        indices = batches[i]
        if len(indices) > 1:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=1)
        else:
            delta_hat[i, :] = np.var(s_data[:, indices], axis=1, ddof=0) + 1e-6

    return gamma_hat, delta_hat


def _construct_priors(gamma_hat, delta_hat):
    """Construct global priors (gamma_bar, t2, aprior, bprior) per batch."""
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    t2[t2 == 0] = 1e-6

    n_batch = gamma_hat.shape[0]
    a_prior = np.zeros(n_batch)
    b_prior = np.zeros(n_batch)
    for i in range(n_batch):
        a_prior[i] = aprior(delta_hat[i, :])
        b_prior[i] = bprior(delta_hat[i, :])

    return gamma_bar, t2, a_prior, b_prior


def _eb_shrinkage(s_data, gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior, batches, parametric=True, conv=0.001):
    """Apply empirical Bayes shrinkage per batch using `itSol` iterator."""
    if not parametric:
        return gamma_hat.copy(), delta_hat.copy(), {}

    n_batch = gamma_hat.shape[0]
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    eb_hist = {"by_batch": {}, "counts": {}}

    for i in range(n_batch):
        indices = batches[i]
        if len(indices) == 0:
            continue
        temp, count, hist = itSol(
            s_data[:, indices],
            gamma_hat[i, :],
            delta_hat[i, :],
            gamma_bar[i],
            t2[i],
            a_prior[i],
            b_prior[i],
            conv=conv,
            return_hist=True,
        )

        gamma_star[i, :] = temp[0, :]
        delta_star[i, :] = temp[1, :]
        eb_hist["by_batch"][i] = hist
        eb_hist["counts"][i] = count

    return gamma_star, delta_star, eb_hist


def _reconstruct(s_data, gamma_star, delta_star, batches, var_pooled, stand_mean, Cov_effects, DeltaCorrection=True, GammaCorrection=True, RegressCovariates=False):
    """Apply batch corrections and inverse-standardize to reconstruct harmonized data."""
    bayesdata = s_data.copy()
    n_batch = gamma_star.shape[0]

    if DeltaCorrection:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None]) / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
        else:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = bayesdata[:, indices] / (np.sqrt(delta_star[i, :])[:, None] + 1e-8)
    else:
        if GammaCorrection:
            for i in range(n_batch):
                indices = batches[i]
                if len(indices) == 0:
                    continue
                bayesdata[:, indices] = (bayesdata[:, indices] - (gamma_star[i, :])[:, None])

    if RegressCovariates:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + (stand_mean - Cov_effects)
    else:
        bayesdata = (bayesdata * (np.sqrt(var_pooled)[:, None])) + stand_mean

    return bayesdata


def _fit_gam_covariates(data, mod_df, gam_opts=None):
    """Fit covariate effects using GAM or spline-basis fallback.

    Returns covariate effect estimates with shape (n_features, n_samples)
    to be removed from data prior to batch estimation.
    """
    import pandas as pd
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import Ridge

    n_features, n_samples = data.shape

    # Try to use pygam if available
    try:
        from pygam import LinearGAM, s

        gam_available = True
    except Exception:
        gam_available = False

    X = mod_df.values
    # If pygam available, fit per-feature GAMs
    cov_effects = np.zeros_like(data)
    if gam_available:
        for fi in range(n_features):
            y = data[fi, :]
            gam = LinearGAM()
            try:
                gam.gridsearch(X, y)
                mu = gam.predict(X)
            except Exception:
                # fallback to linear prediction if gam fails
                mu = np.linalg.lstsq(np.hstack([np.ones((X.shape[0], 1)), X]), y, rcond=None)[0]
                mu = (np.hstack([np.ones((X.shape[0], 1)), X]) @ mu).ravel()
            cov_effects[fi, :] = mu
        return cov_effects

    # Fallback: build spline basis for each numeric covariate and fit ridge regression
    # Build spline basis
    n_cov = X.shape[1]
    transformer = SplineTransformer(degree=3, n_knots=5, include_bias=False)
    try:
        X_spline = transformer.fit_transform(X)
    except Exception:
        # If spline transform fails (e.g., constant columns), fall back to original X
        X_spline = X

    # Add intercept
    X_design = np.hstack([np.ones((X_spline.shape[0], 1)), X_spline])

    # Fit ridge per feature (vectorized solution)
    # Solve (X'X + alpha I) W = X'Y  => W = inv(X'X + alpha I) X'Y
    alpha = 1.0 if gam_opts is None else gam_opts.get("alpha", 1.0)
    XtX = X_design.T @ X_design
    reg = alpha * np.eye(XtX.shape[0])
    inv = np.linalg.pinv(XtX + reg)
    W = inv @ X_design.T @ data.T

    cov_effects = (X_design @ W).T
    return cov_effects


def _construct_local_priors(s_data, gamma_hat, delta_hat, var_pooled, weight_methods=None, weight_opts=None, global_priors=None):
    """Construct feature-wise, batch-wise local priors using weighted pooling.

    Supports multiple weight methods and parameter controls via ``weight_methods``
    and ``weight_opts``. Allowed methods include:
      - 'correlation_similarity' : |corr(feature_i, feature_j)| across samples
      - 'variance_similarity'    : similarity of pooled variances
      - 'magnitude_similarity'   : similarity of mean absolute gamma across features
      - 'spatial_proximity'      : Gaussian kernel on provided feature coordinates
      - 'directional_bias'       : per-batch sign/magnitude agreement of raw gamma
      - 'custom' or 'similarity_matrix': use precomputed matrix from ``weight_opts``

    ``weight_opts`` supports:
      - 'method_weights': dict mapping method -> weight (floats). Defaults to equal weights.
      - 'min_effective': int minimum effective sample size before falling back to global priors (default 5).
      - 'feature_coords': ndarray (n_features, d) for spatial kernel.
      - 'sigma': float bandwidth for spatial kernel (optional).
      - 'similarity_matrix': ndarray (n_features, n_features) custom similarity.
      - 'normalize': bool row-normalize weight matrices (default True).

    Returns dict with arrays 'gamma_bar_local', 't2_local', 'a_prior_local', 'b_prior_local'
    (shape (n_batch, n_features)) and 'weights' (shape (n_batch, n_features, n_features)).
    """
    eps = 1e-12
    if weight_methods is None:
        weight_methods = ["correlation_similarity"]
    if weight_opts is None:
        weight_opts = {}

    n_batch, n_features = gamma_hat.shape

    # Method weights (how to combine different similarity matrices)
    method_weights = weight_opts.get("method_weights")
    if method_weights is None:
        method_weights = {m: 1.0 / len(weight_methods) for m in weight_methods}

    min_effective = int(weight_opts.get("min_effective", 5))
    normalize = bool(weight_opts.get("normalize", True))

    # --- Precompute base similarity matrices (feature x feature) ---
    # Correlation-based similarity
    try:
        corr = np.corrcoef(s_data)
        corr = np.nan_to_num(corr)
        corr_abs = np.abs(corr)
    except Exception:
        corr_abs = np.eye(n_features)

    # Variance similarity (closer pooled variances -> higher similarity)
    v = np.asarray(var_pooled).ravel()
    if v.size != n_features:
        v = np.ones(n_features)
    vdiff = np.abs(v[:, None] - v[None, :])
    maxdiff = np.max(vdiff)
    if maxdiff <= 0:
        var_sim = np.ones_like(vdiff)
    else:
        var_sim = 1.0 - (vdiff / (maxdiff + eps))
    var_sim = np.clip(var_sim, 0.0, 1.0)

    # Magnitude similarity based on mean absolute gamma across batches
    mag = np.mean(np.abs(gamma_hat), axis=0)
    mdiff = np.abs(mag[:, None] - mag[None, :])
    maxmdiff = np.max(mdiff)
    if maxmdiff <= 0:
        mag_sim = np.ones_like(mdiff)
    else:
        mag_sim = 1.0 - (mdiff / (maxmdiff + eps))
    mag_sim = np.clip(mag_sim, 0.0, 1.0)

    # Spatial proximity (optional)
    coords = weight_opts.get("feature_coords", None)
    if coords is not None:
        coords = np.asarray(coords)
        try:
            distsq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
            sigma = float(weight_opts.get("sigma", np.std(np.sqrt(distsq)) if np.std(np.sqrt(distsq)) > eps else 1.0))
            spatial_sim = np.exp(-distsq / (2.0 * (sigma ** 2) + eps))
        except Exception:
            spatial_sim = np.ones((n_features, n_features))
    else:
        spatial_sim = np.ones((n_features, n_features))

    # Custom similarity matrix (optional)
    sim_matrix = weight_opts.get("similarity_matrix", None)
    if sim_matrix is not None:
        sim_matrix = np.asarray(sim_matrix)
        if sim_matrix.shape != (n_features, n_features):
            sim_matrix = None

    # Storage for per-batch weight matrices
    weights_all = np.zeros((n_batch, n_features, n_features), dtype=float)

    # Build per-batch weight matrices combining requested methods
    for b in range(n_batch):
        W = np.zeros((n_features, n_features), dtype=float)
        for method in weight_methods:
            wgt = float(method_weights.get(method, 0.0))
            if wgt == 0.0:
                continue
            if method == "correlation_similarity":
                M = corr_abs
            elif method == "variance_similarity":
                M = var_sim
            elif method == "magnitude_similarity":
                M = mag_sim
            elif method == "spatial_proximity":
                M = spatial_sim
            elif method in ("custom", "similarity_matrix"):
                M = sim_matrix if sim_matrix is not None else np.zeros((n_features, n_features))
            elif method == "directional_bias":
                # Directional bias with configurable parameters. We compute sign
                # agreement and a magnitude-kernel, optionally downweighting
                # features whose sign disagrees with a global sign.
                dir_opts = weight_opts.get("directional", {}) if isinstance(weight_opts, dict) else {}
                dir_min_abs = float(dir_opts.get("dir_min_abs", weight_opts.get("dir_min_abs", 1e-8)))
                dir_sigma = dir_opts.get("dir_sigma", weight_opts.get("dir_sigma", None))
                dir_power = float(dir_opts.get("dir_power", weight_opts.get("dir_power", 1.0)))
                use_global_sign = bool(dir_opts.get("use_global_sign", weight_opts.get("use_global_sign", False)))
                global_sign_strength = float(dir_opts.get("global_sign_strength", weight_opts.get("global_sign_strength", 0.5)))

                mag_b = np.abs(gamma_hat[b, :])
                signs = np.sign(gamma_hat[b, :])
                # Mask small magnitudes as noise
                mask = mag_b >= dir_min_abs

                # sign agreement matrix (1 if same sign and not noisy)
                sign_agree = (signs[:, None] * signs[None, :] > 0).astype(float)
                sign_agree = sign_agree * (mask[:, None] * mask[None, :])

                # magnitude similarity kernel (Gaussian on abs gamma differences)
                if dir_sigma is None:
                    # robust default: std of non-negligible magnitudes, fallback to 1.0
                    nonzero_mag = mag_b[mask]
                    if nonzero_mag.size > 1:
                        dir_sigma_val = float(np.std(nonzero_mag))
                    else:
                        dir_sigma_val = float(np.std(mag_b))
                    if dir_sigma_val <= 0:
                        dir_sigma_val = 1.0
                else:
                    dir_sigma_val = float(dir_sigma)

                mdiff_b = np.abs(mag_b[:, None] - mag_b[None, :])
                Mmag = np.exp(- (mdiff_b ** 2) / (2.0 * (dir_sigma_val ** 2) + eps))

                M = sign_agree * (Mmag ** dir_power)

                if use_global_sign:
                    global_signs = np.sign(np.mean(gamma_hat, axis=0))
                    align = (signs == global_signs).astype(float)
                    # features misaligned with global sign are downweighted by global_sign_strength
                    factor = np.where(align == 1.0, 1.0, global_sign_strength)
                    M = M * (factor[:, None] * factor[None, :])
            else:
                M = np.zeros((n_features, n_features), dtype=float)

            W += wgt * M

        # Force non-negative and zero diagonal
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)

        # Row-normalize to make weights sum to 1 per target feature
        if normalize:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            W = W / row_sums

        weights_all[b, :, :] = W

    # --- Compute weighted priors per batch and per feature ---
    gamma_bar_local = np.zeros((n_batch, n_features), dtype=float)
    t2_local = np.zeros((n_batch, n_features), dtype=float)
    a_prior_local = np.zeros((n_batch, n_features), dtype=float)
    b_prior_local = np.zeros((n_batch, n_features), dtype=float)

    for b in range(n_batch):
        gh = gamma_hat[b, :]
        dh = delta_hat[b, :]
        Wb = weights_all[b, :, :]
        for f in range(n_features):
            w = Wb[f, :].copy()
            w[f] = 0.0
            if np.allclose(w.sum(), 0.0):
                w = np.ones(n_features, dtype=float)
                w[f] = 0.0
            s = w.sum()
            if s <= eps:
                w = np.zeros_like(w)
            else:
                w = w / s

            ess = 1.0 / (np.sum(w ** 2) + eps)
            if ess < min_effective and global_priors is not None:
                gamma_bar_local[b, f] = global_priors[0][b]
                t2_local[b, f] = global_priors[1][b]
                a_prior_local[b, f] = global_priors[2][b]
                b_prior_local[b, f] = global_priors[3][b]
                continue

            gamma_bar_local[b, f] = float(np.sum(w * gh))
            mu = gamma_bar_local[b, f]
            t2_local[b, f] = float(np.sum(w * (gh - mu) ** 2))
            if t2_local[b, f] <= eps:
                t2_local[b, f] = 1e-6

            m = float(np.sum(w * dh))
            s2 = float(np.sum(w * (dh - m) ** 2))
            if s2 <= eps and global_priors is not None:
                a_prior_local[b, f] = global_priors[2][b]
                b_prior_local[b, f] = global_priors[3][b]
            else:
                a_prior_local[b, f] = (2.0 * s2 + m ** 2) / (s2 + eps)
                b_prior_local[b, f] = (m * s2 + m ** 3) / (s2 + eps)

    return {
        "gamma_bar_local": gamma_bar_local,
        "t2_local": t2_local,
        "a_prior_local": a_prior_local,
        "b_prior_local": b_prior_local,
        "weights": weights_all,
    }

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    """Iteratively solve for the posterior mean and variance of the batch effect parameters.
    This version is used by CovBat and taken from: https://github.com/andy1764/CovBat_Harmonisation
    Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., Shou, H., & Initiative, T. A. D. N. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. Human Brain Mapping, 43(4), 1179–1195. https://doi.org/10.1002/hbm.25688)
    """



    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]


# Define CovBat harmonisation function: from Chen et al. 2022

# Define harmonisation via mixed effects model (Regression analysis)
import numpy as np
import pandas as pd
import warnings
from statsmodels.formula.api import mixedlm

def lme_harmonisation(data, batch, mod, variable_names):
    """
    Fits a per feature linear mixed model to harmonize data across batches while adjusting for covariates. This function is an alternative to ComBat that uses mixed effects modeling to estimate and remove batch effects.

    Args:
        data (pd.DataFrame or np.array): The data matrix to be harmonized, with shape (n_samples, n_features).
        batch (pd.Series or np.array): A vector of batch labels for each sample, with length n_samples.
        mod (pd.DataFrame or np.array): A design matrix of covariates to adjust for, with shape (n_samples, n_covariates).
        variable_names (list of str): A list of column names corresponding to the covariates in `mod`, used for formula construction.
    Returns:
        np.array: A harmonized data matrix of the same shape as the input `data`, with batch effects removed according to the fitted mixed models.
    
    Note:
        This function fits a separate linear mixed model for each feature (column) in the data matrix, with the batch variable as a random effect and the covariates as fixed effects. The residuals from these models are returned as the harmonized data.

    """
    # ----------------------------
    # Normalize inputs & keep labels
    # ----------------------------
    data_was_df = isinstance(data, pd.DataFrame)
    batch_was_series = isinstance(batch, (pd.Series, pd.Index))
    mod_was_df = isinstance(mod, pd.DataFrame)

    # keep labels to restore later
    data_index = data.index if data_was_df else None
    data_columns = data.columns if data_was_df else None

    # Convert inputs to numpy arrays of expected orientation:
    # internal working shape for `data_np` is (n_samples, n_features)
    if data_was_df:
        data_np = data.values.astype(float)
    else:
        data_np = np.asarray(data, dtype=float)

    # batch -> 1D array of length n_samples
    if batch_was_series:
        batch_np = np.asarray(batch.values)
    else:
        batch_np = np.asarray(batch).ravel()

    # mod -> (n_samples, n_covariates) or None
    if mod is None:
        mod_np = None
    else:
        if mod_was_df:
            mod_np = mod.values.astype(float)
        else:
            mod_np = np.asarray(mod, dtype=float)
        # if 1D, make column vector
        if mod_np.ndim == 1:
            mod_np = mod_np.reshape(-1, 1)

    # Basic validation
    if data_np.ndim != 2:
        raise ValueError("Data must be a 2D array (samples x features).")
    n_samples, n_features = data_np.shape

    # Check batch is numeric, if not convert to categorical codes
    if not np.issubdtype(batch_np.dtype, np.number):
        batch_cat = pd.Categorical(batch_np)
        batch_np = batch_cat.codes  # integer codes for categories

    if batch_np.ndim != 1 or batch_np.shape[0] != n_samples:
        print(batch_np.shape, n_samples)
        raise ValueError("Batch must be a 1D array-like with length equal to number of samples (rows of data).")

    if mod_np is not None:
        if mod_np.ndim != 2:
            raise ValueError("mod must be a 2D array (samples x covariates).")
        if mod_np.shape[0] != n_samples:
            raise ValueError("mod must have the same number of rows (samples) as data.")
        if len(variable_names) != mod_np.shape[1]:
            raise ValueError("variable_names length must equal number of covariates (columns of mod).")

    # ----------------------------
    # Build a base DataFrame with batch and covariates used for every per-feature fit
    # ----------------------------
    # We create a DataFrame with one row per sample. For each feature we will assign
    # the response column temporarily and fit a mixed model formula on that DataFrame.
    base_df = pd.DataFrame(index=range(n_samples))
    base_df['batch'] = batch_np  # grouping factor (categorical)

    if mod_np is not None:
        for i, var in enumerate(variable_names):
            base_df[var] = mod_np[:, i]

    # Prepare interaction terms string for the formula: batch:cov1 + batch:cov2 + ...
    interaction_terms = []
    if mod_np is not None:
        for var in variable_names:
            interaction_terms.append(f'batch:{var}')
    interaction_str = ' + '.join(interaction_terms) if interaction_terms else ''

    # Fixed part: batch + covariates
    fixed_parts = ['batch'] + (variable_names if variable_names else [])
    fixed_str = ' + '.join(fixed_parts)

    # full RHS for formula (skip empty pieces)
    if interaction_str:
        rhs = f"{fixed_str} + {interaction_str}"
    else:
        rhs = fixed_str

    # ----------------------------
    # Fit per-feature mixed-effects model and collect residuals
    # ----------------------------
    # We will fit: Y ~ <rhs> with groups=batch (random intercept).
    # This is done separately for each feature (column) in data_np.
    residuals = np.zeros_like(data_np, dtype=float)  # shape (n_samples, n_features)
    warnings.filterwarnings("ignore")  # suppress fit warnings; you may remove this

    for feat_idx in range(n_features):
        # create a temporary response column name that doesn't conflict with others
        resp_col = '_y_response'
        base_df[resp_col] = data_np[:, feat_idx]

        # formula example: '_y_response ~ batch + age + sex + batch:age + batch:sex'
        formula = f'Q("{resp_col}") ~ {rhs}'

        # Fit mixed model with random intercept for batch
        try:
            model = mixedlm(formula, base_df, groups=base_df['batch'])
            # try default fit; if convergence issues occur, fallback handled below
            result = model.fit()
        except Exception:
            # fallback fit with different options if default fails (method and reml off)
            model = mixedlm(formula, base_df, groups=base_df['batch'])
            result = model.fit(method='lbfgs', reml=False, maxiter=2000, disp=False)

        # result.resid is length n_samples
        residuals[:, feat_idx] = result.resid.values

        # drop temporary response column (next iteration will overwrite)
        base_df.drop(columns=[resp_col], inplace=True)

    # ----------------------------
    # Restore output type & labels to match input
    # ----------------------------
    if data_was_df:
        # preserve original index and column names
        residuals_df = pd.DataFrame(residuals, index=data_index if data_index is not None else range(n_samples),
                                    columns=data_columns if data_columns is not None else range(n_features))
        return residuals_df
    else:
        return residuals

# Define LME_IQM harmonisation functions
def lme_iqm_harmonise(
    data: pd.DataFrame,
    idp_list: Sequence[str],
    qc_list: Sequence[str],
    preserve_covars: Sequence[str] = ("age",),
    adjust_covars: Sequence[str] = ("timepoint",),
    reverse_guard_covars: Optional[Sequence[str]] = None,
    categorical_covars: Sequence[str] = ("timepoint", "batch"),
    batch_col: str = "batch",
    subject_col: str = "subjectID",
    age_source_col: str = "Final_Age",
    batch_source_col: str = "Site",
    age_col: str = "age",
    iqm_variance: float = 95,
    p_thr: float = 0.05,
    max_qcs: float = float("inf"),
    apply_pca: bool = True,
    outfilename: str = "iqm_harmonised.csv",
    summary_csv: str = "qc_selection_summary.csv",
    additive_detail_csv: str = "qc_selection_additive_details.csv",
    multiplicative_detail_csv: str = "qc_selection_multiplicative_details.csv",
    selected_additive_qcs_csv: str = "selected_additive_qcs_by_volume.csv",
    selected_multiplicative_qcs_csv: str = "selected_multiplicative_qcs_by_volume.csv",
    allow_ols_fallback: bool = True,
    optimizer_order: Sequence[str] = ("lbfgs", "powell", "cg", "nm"),
    maxiter: int = 2000,
    verbose_model_fits: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Harmonise IDPs using IQM/QC-based additive and multiplicative correction.

    The function selects QC features that:
    - are associated with batch/site effects,
    - do not encode preserve covariates,
    - do not encode the response proxy,
    - and pass reverse-guard tests when enabled.

    Depending on `apply_pca`, QC variables are either:
    - z-scored and reduced with PCA, or
    - used directly after standardisation.

    Parameters
    ----------
    data : pandas.DataFrame
        Main input dataframe containing IDPs, QC metrics, subject IDs, and covariates.
    idp_list : Sequence[str]
        List of IDP columns to harmonise.
    qc_list : Sequence[str]
        List of QC metric columns used as candidate harmonisation regressors.
    preserve_covars : Sequence[str], default=("age",)
        Covariates that QC variables are not allowed to encode.
    adjust_covars : Sequence[str], default=("timepoint",)
        Covariates included as adjustment variables in the models.
    reverse_guard_covars : Sequence[str] | None, default=None
        Optional subset of preserve covariates to test with reverse-response guards.
        If None, numeric preserve covariates are used automatically.
    categorical_covars : Sequence[str], default=("timepoint", "batch")
        Covariates treated as categorical in regression formulas.
    batch_col : str, default="batch"
        Batch/site variable used as the harmonisation target.
    subject_col : str, default="subjectID"
        Subject identifier used for random intercepts in MixedLM.
    age_source_col : str, default="Final_Age"
        Source column used to create `age_col` if `age_col` is missing.
    batch_source_col : str, default="Site"
        Source column used to create `batch_col` if `batch_col` is missing.
    age_col : str, default="age"
        Working age column used in the models.
    iqm_variance : float, default=95
        Percentage of QC variance to retain when PCA is applied.
    p_thr : float, default=0.05
        P-value threshold used for QC selection and multiplicative testing.
    max_qcs : float, default=inf
        Maximum number of QCs allowed per volume.
    apply_pca : bool, default=True
        If True, run PCA on z-scored QC variables and retain enough PCs to
        explain `iqm_variance` percent variance. If False, use z-scored raw QCs.
    outfilename : str, default="iqm_harmonised.csv"
        Output CSV for the harmonised dataframe.
    summary_csv : str, default="qc_selection_summary.csv"
        Output CSV for the per-volume summary table.
    additive_detail_csv : str, default="qc_selection_additive_details.csv"
        Output CSV for additive QC selection diagnostics.
    multiplicative_detail_csv : str, default="qc_selection_multiplicative_details.csv"
        Output CSV for multiplicative QC selection diagnostics.
    selected_additive_qcs_csv : str, default="selected_additive_qcs_by_volume.csv"
        Output CSV listing selected additive QCs by volume.
    selected_multiplicative_qcs_csv : str, default="selected_multiplicative_qcs_by_volume.csv"
        Output CSV listing selected multiplicative QCs by volume.
    allow_ols_fallback : bool, default=True
        If MixedLM fitting fails, fall back to OLS.
    optimizer_order : Sequence[str], default=("lbfgs", "powell", "cg", "nm")
        Optimizers attempted in order for MixedLM fitting.
    maxiter : int, default=2000
        Maximum number of optimizer iterations for MixedLM.
    verbose_model_fits : bool, default=False
        If True, print detailed model fitting diagnostics.

    Returns
    -------
    data : pandas.DataFrame
        Input dataframe with `harmonised_<IDP>` columns added.
    qc_selection : dict[str, Any]
        Summary of selected QCs and correction decisions.
    add_detail_df : pandas.DataFrame
        Additive QC selection diagnostic table.
    mult_detail_df : pandas.DataFrame
        Multiplicative QC selection diagnostic table.
    summary_df : pandas.DataFrame
        Per-volume summary table.

    Examples
    --------
    >>> df = pd.read_csv("test_data/alldata.csv")
    >>> idp_list = pd.read_csv("test_data/IDP_list.csv", header=None).iloc[:, 0].dropna().astype(str).str.strip().tolist()
    >>> iqm_list = pd.read_csv("test_data/IQM_list.csv", header=None).iloc[:, 0].dropna().astype(str).str.strip().tolist()
    >>> data_out, qc_selection, add_detail_df, mult_detail_df, summary_df = lme_iqm_harmonise(
    ...     data=df,
    ...     idp_list=idp_list,
    ...     qc_list=iqm_list,
    ...     preserve_covars=("age",),
    ...     adjust_covars=("timepoint",),
    ...     reverse_guard_covars=("age",),
    ...     categorical_covars=("timepoint", "scan_session"),
    ...     batch_col="scan_session",
    ...     subject_col="subject",
    ...     age_source_col="age",
    ...     batch_source_col="scan_session",
    ...     age_col="age",
    ...     iqm_variance=95,
    ...     p_thr=0.05,
    ...     apply_pca=True,
    ...     verbose_model_fits=True,
    ... )
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def ordered_unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    preserve_covars = ordered_unique(list(preserve_covars))
    adjust_covars = ordered_unique(list(adjust_covars))
    categorical_covars = ordered_unique(list(categorical_covars))

    if batch_col not in categorical_covars:
        categorical_covars.append(batch_col)

    categorical_covars_set = set(categorical_covars)

    if reverse_guard_covars is None:
        reverse_guard_covars = [c for c in preserve_covars if c not in categorical_covars_set]
    else:
        reverse_guard_covars = ordered_unique(list(reverse_guard_covars))
        reverse_guard_covars = [c for c in reverse_guard_covars if c in preserve_covars]
        reverse_guard_covars = [c for c in reverse_guard_covars if c not in categorical_covars_set]

    def term_expr(var: str) -> str:
        return f"C({var})" if var in categorical_covars_set else var

    def rhs_expr(covars: Sequence[str]) -> str:
        covars = ordered_unique([c for c in covars if c is not None and c != ""])
        if len(covars) == 0:
            return "1"
        return " + ".join(term_expr(c) for c in covars)

    def strip_random_effect(formula: str) -> str:
        out = re.sub(r"\s*\+\s*\(\s*1\s*\|\s*[\w]+\s*\)\s*", " ", formula)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def complete_case(df_in: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        cols = list(dict.fromkeys(cols))
        return df_in.loc[:, cols].dropna().copy()

    def fixed_params(result):
        return getattr(result, "fe_params", result.params)

    def _fit_mixed_formula(df_in: pd.DataFrame, formula: str, group_col: str, method: str):
        fixed_formula = strip_random_effect(formula)
        model = smf.mixedlm(fixed_formula, data=df_in, groups=df_in[group_col])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
        return res

    def fit_mixed_then_ols(df_in: pd.DataFrame, formula: str, group_col: str):
        """
        Try multiple MixedLM optimizers first.
        If all fail and allow_ols_fallback=True, fall back to OLS.
        Returns result, family, used_method.
        """
        fixed_formula = strip_random_effect(formula)

        if verbose_model_fits:
            print("\n--------------------------------------------------")
            print("Requested formula:")
            print(" ", formula)
            print("Fixed-effects formula used:")
            print(" ", fixed_formula)
            print("Random effects:")
            print(f"  (1 | {group_col})")
            print("Trying MixedLM optimizers:", list(optimizer_order))

        for method in optimizer_order:
            try:
                if verbose_model_fits:
                    print(f"  Trying optimizer: {method}")
                res = _fit_mixed_formula(df_in, formula, group_col, method)
                if getattr(res, "converged", True):
                    if verbose_model_fits:
                        print("  SUCCESS")
                        print("  Fit family: MixedLM")
                        print("  Optimizer:", method)
                        print("  Converged:", getattr(res, "converged", True))
                        print("--------------------------------------------------")
                    return res, "mixed", method
            except Exception as exc:
                if verbose_model_fits:
                    print(f"  FAILED ({method})")
                    print("   Reason:", exc)

        if not allow_ols_fallback:
            raise RuntimeError(f"MixedLM failed for formula: {fixed_formula}")

        if verbose_model_fits:
            print("\n  All MixedLM optimizers failed.")
            print("  Falling back to OLS.")
            print("  WARNING: subject random effect removed.")
            print("--------------------------------------------------")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_res = smf.ols(fixed_formula, data=df_in).fit()
        return ols_res, "ols", "ols"

    def fit_pair_same_family(df_in: pd.DataFrame, full_formula: str, reduced_formula: str, group_col: str):
        """
        Fit a nested pair using the same family and same optimizer when possible.
        First tries MixedLM for both, then OLS for both.
        Returns full_fit, reduced_fit, family, method_full, method_reduced.
        """
        full_fixed = strip_random_effect(full_formula)
        red_fixed = strip_random_effect(reduced_formula)

        for method in optimizer_order:
            try:
                full_fit = _fit_mixed_formula(df_in, full_formula, group_col, method)
                red_fit = _fit_mixed_formula(df_in, reduced_formula, group_col, method)
                if getattr(full_fit, "converged", True) and getattr(red_fit, "converged", True):
                    return full_fit, red_fit, "mixed", method, method
            except Exception:
                pass

        if not allow_ols_fallback:
            raise RuntimeError(
                "MixedLM failed for nested comparison:\n"
                f"FULL: {full_fixed}\nRED : {red_fixed}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_ols = smf.ols(full_fixed, data=df_in).fit()
            red_ols = smf.ols(red_fixed, data=df_in).fit()
        return full_ols, red_ols, "ols", "ols", "ols"

    def lrt_pvalue(full_fit, reduced_fit, family: str) -> float:
        lr_stat = 2.0 * (full_fit.llf - reduced_fit.llf)
        lr_stat = max(float(lr_stat), 0.0)

        if family == "mixed":
            df_full = int(getattr(full_fit, "k_fe", len(full_fit.params)))
            df_red = int(getattr(reduced_fit, "k_fe", len(reduced_fit.params)))
        else:
            df_full = int(getattr(full_fit, "df_model", len(full_fit.params) - 1))
            df_red = int(getattr(reduced_fit, "df_model", len(reduced_fit.params) - 1))

        df_diff = max(df_full - df_red, 1)
        return float(chi2.sf(lr_stat, df_diff))

    def evaluate_qc_stage(
        df_stage: pd.DataFrame,
        response: str,
        qcname: str,
        proxy_term: str,
        stage_name: str,
    ):
        """
        Evaluate one QC feature for additive or multiplicative selection.
        Returns (passed_bool, row_dict).
        """
        fixed_terms = ordered_unique(list(preserve_covars) + list(adjust_covars) + [batch_col, proxy_term])

        row = {
            "response": response,
            "volume": response,   # keep a "volume" field for CSV compatibility
            "qc": qcname,
            "stage": stage_name,
        }

        passed = True
        first_family = None
        first_method = None

        # Preserve covariates: QC should not explain them
        for cov in preserve_covars:
            full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
            red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != cov])} + (1|{subject_col})"
            full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
            p_cov = lrt_pvalue(full_fit, red_fit, fam)
            row[f"p_preserve_{cov}"] = p_cov
            passed = passed and (p_cov > p_thr)
            if first_family is None:
                first_family = fam
                first_method = method_full

        # batch must be significant
        full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
        red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != batch_col])} + (1|{subject_col})"
        full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
        p_batch = lrt_pvalue(full_fit, red_fit, fam)
        row["p_batch"] = p_batch
        passed = passed and (p_batch < p_thr)
        if first_family is None:
            first_family = fam
            first_method = method_full

        # proxy term must not drive QC
        full_formula = f"{qcname} ~ {rhs_expr(fixed_terms)} + (1|{subject_col})"
        red_formula = f"{qcname} ~ {rhs_expr([t for t in fixed_terms if t != proxy_term])} + (1|{subject_col})"
        full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
        p_proxy = lrt_pvalue(full_fit, red_fit, fam)
        row["p_proxy"] = p_proxy
        passed = passed and (p_proxy > p_thr)
        if first_family is None:
            first_family = fam
            first_method = method_full

        # reverse guards for numeric preserve covariates only
        for cov in reverse_guard_covars:
            reverse_terms = ordered_unique([qcname] + list(preserve_covars) + list(adjust_covars) + [batch_col, proxy_term])
            full_formula = f"{cov} ~ {rhs_expr(reverse_terms)} + (1|{subject_col})"
            red_formula = f"{cov} ~ {rhs_expr([t for t in reverse_terms if t != qcname])} + (1|{subject_col})"
            full_fit, red_fit, fam, method_full, method_red = fit_pair_same_family(df_stage, full_formula, red_formula, subject_col)
            p_rev = lrt_pvalue(full_fit, red_fit, fam)
            row[f"p_reverse_{cov}"] = p_rev
            passed = passed and (p_rev > p_thr)
            if first_family is None:
                first_family = fam
                first_method = method_full

        row["passed"] = int(passed)
        row["fit_family"] = first_family
        row["fit_method"] = first_method
        return passed, row

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    data = data.copy()
    data.columns = data.columns.astype(str).str.strip()

    print("=== IQM Harmonization (Flexible Version) ===")
    print(f"P-value threshold: {p_thr:.4f}")
    print(f"Max QCs per volume: {max_qcs}")
    print(f"PCA variance threshold: {iqm_variance:.1f}%")
    print(f"Apply PCA: {apply_pca}")
    print(f"Preserve covars: {list(preserve_covars)}")
    print(f"Adjust covars: {list(adjust_covars)}")
    print(f"Categorical covars: {list(categorical_covars)}")
    print(f"Reverse-guard covars: {list(reverse_guard_covars)}")

    # prepare age / batch columns
    if age_col not in data.columns:
        if age_source_col not in data.columns:
            raise ValueError(f"Missing '{age_col}' and source column '{age_source_col}'")
        data[age_col] = zscore(data[age_source_col].astype(float))

    if batch_col not in data.columns:
        if batch_source_col not in data.columns:
            raise ValueError(f"Missing '{batch_col}' and source column '{batch_source_col}'")
        data[batch_col] = data[batch_source_col]

    # required columns
    for c in [subject_col] + list(adjust_covars) + list(preserve_covars):
        if c not in data.columns:
            raise ValueError(f"Missing required covariate column: {c}")

    missing_idps = [c for c in idp_list if c not in data.columns]
    missing_qcs = [c for c in qc_list if c not in data.columns]
    if missing_idps:
        raise ValueError(f"Missing IDP columns: {missing_idps}")
    if missing_qcs:
        raise ValueError(f"Missing QC columns: {missing_qcs}")

    # categorical casting
    data[subject_col] = data[subject_col].astype("category")
    for c in categorical_covars:
        if c in data.columns:
            data[c] = data[c].astype("category")

    # ------------------------------------------------------------------
    # STEP 1: QC basis (PCA or raw standardized QCs)
    # ------------------------------------------------------------------
    print("\nStep 1: Preparing QC metrics...")

    qc_df = data.loc[:, list(qc_list)].copy()
    if qc_df.isna().any().any():
        raise ValueError("QC columns contain missing values. Please clean/impute before running.")

    qc_matrix = qc_df.to_numpy(dtype=float)

    # remove zero-variance QCs
    stds = np.std(qc_matrix, axis=0, ddof=1)
    zero_var_mask = stds == 0
    removed = [q for q, keep in zip(qc_list, ~zero_var_mask) if not keep]
    if removed:
        print(f"  Removing {len(removed)} zero-variance QCs")
        print(" ", removed[:20])

    qc_matrix = qc_matrix[:, ~zero_var_mask]
    qc_list_clean = [q for q, keep in zip(qc_list, ~zero_var_mask) if keep]

    if qc_matrix.shape[1] == 0:
        raise ValueError("No QC variables left after removing zero-variance columns.")

    qc_z = StandardScaler().fit_transform(qc_matrix)

    if apply_pca:
        pca = PCA()
        scores = pca.fit_transform(qc_z)
        explained = pca.explained_variance_ratio_ * 100.0
        cumvar = np.cumsum(explained)
        num_qc_features = int(np.argmax(cumvar >= iqm_variance) + 1)

        qc_feature_names = [f"QC{i+1}" for i in range(num_qc_features)]
        for i, name in enumerate(qc_feature_names):
            data[name] = scores[:, i]

        qc_basis = "PCA"
        print(f"  PCA input shape: {qc_z.shape}")
        print(f"  Retaining PCs: {num_qc_features} ({cumvar[num_qc_features - 1]:.2f}%)")
    else:
        num_qc_features = qc_z.shape[1]
        qc_feature_names = list(qc_list_clean)
        for i, name in enumerate(qc_feature_names):
            data[name] = qc_z[:, i]

        qc_basis = "RAW"
        cumvar = None
        print(f"  Using raw QC variables directly: {num_qc_features} features")

    # ------------------------------------------------------------------
    # STEP 2: ADDITIVE QC SELECTION
    # ------------------------------------------------------------------
    print("\nStep 2: Selecting QCs for ADDITIVE correction...")
    print("  Criteria: batch-driven, not preserve-covariate-driven")

    good_pcs_add = np.zeros((num_qc_features, len(idp_list)), dtype=int)
    all_out_add = []
    additive_detail_rows = []

    for v_idx, volname in enumerate(idp_list):
        print(f"\n=== Additive selection: {volname} ===")
        rows = []

        for qc_i in range(num_qc_features):
            qcname = qc_feature_names[qc_i]

            df_stage = complete_case(
                data,
                [subject_col] + list(preserve_covars) + list(adjust_covars) + [batch_col, volname, qcname],
            )

            try:
                passed, row = evaluate_qc_stage(df_stage, volname, qcname, proxy_term=volname, stage_name="additive")
                good_pcs_add[qc_i, v_idx] = int(passed)
            except Exception as exc:
                print(f"  Additive QC check failed for {qcname}: {exc}")
                row = {
                    "response": volname,
                    "volume": volname,
                    "qc": qcname,
                    "stage": "additive",
                    "p_batch": np.nan,
                    "p_proxy": np.nan,
                    "passed": 0,
                    "fit_family": "failed",
                    "fit_method": "failed",
                }
                for cov in preserve_covars:
                    row[f"p_preserve_{cov}"] = np.nan
                for cov in reverse_guard_covars:
                    row[f"p_reverse_{cov}"] = np.nan

            rows.append(row)
            additive_detail_rows.append(row)

        all_out_add.append([volname, rows])

    n_add = good_pcs_add.sum(axis=0)
    print(f"  QCs selected: min={n_add.min()}, max={n_add.max()}, mean={n_add.mean():.1f}")

    # ------------------------------------------------------------------
    # STEP 3: MULTIPLICATIVE QC SELECTION
    # ------------------------------------------------------------------
    print("\nStep 3: Selecting QCs for MULTIPLICATIVE correction...")
    print("  KEY: Using ORIGINAL volumes (not additive-corrected) for variance proxy")
    print("  Criteria: batch-driven, NOT strongly variance-driven")

    good_pcs_mult = np.zeros((num_qc_features, len(idp_list)), dtype=int)
    all_out_mult = []
    multiplicative_detail_rows = []

    for v_idx, volname in enumerate(idp_list):
        print(f"\n=== Multiplicative selection: {volname} ===")
        rows = []

        # Biology-only residuals from the original volume
        bio_df = complete_case(
            data,
            [subject_col] + list(preserve_covars) + list(adjust_covars) + [volname],
        )
        bio_formula = f"{volname} ~ {rhs_expr(list(preserve_covars) + list(adjust_covars))} + (1|{subject_col})"
        bio_fit, _, _ = fit_mixed_then_ols(bio_df, bio_formula, subject_col)

        res = np.asarray(bio_fit.resid, dtype=float)
        log_r2_series = pd.Series(
            np.log(np.maximum(res ** 2, np.finfo(float).eps)),
            index=bio_df.index,
            name="log_r2_tmp",
        )

        for qc_i in range(num_qc_features):
            qcname = qc_feature_names[qc_i]

            df_stage = data.loc[:, [subject_col] + list(preserve_covars) + list(adjust_covars) + [batch_col, qcname]].copy()
            df_stage = df_stage.join(log_r2_series, how="inner").dropna().copy()

            try:
                passed, row = evaluate_qc_stage(
                    df_stage,
                    volname,
                    qcname,
                    proxy_term="log_r2_tmp",
                    stage_name="multiplicative",
                )
                good_pcs_mult[qc_i, v_idx] = int(passed)
            except Exception as exc:
                print(f"  Multiplicative QC check failed for {qcname}: {exc}")
                row = {
                    "response": volname,
                    "volume": volname,
                    "qc": qcname,
                    "stage": "multiplicative",
                    "p_batch": np.nan,
                    "p_proxy": np.nan,
                    "passed": 0,
                    "fit_family": "failed",
                    "fit_method": "failed",
                }
                for cov in preserve_covars:
                    row[f"p_preserve_{cov}"] = np.nan
                for cov in reverse_guard_covars:
                    row[f"p_reverse_{cov}"] = np.nan

            rows.append(row)
            multiplicative_detail_rows.append(row)

        all_out_mult.append([volname, rows])

    n_mult = good_pcs_mult.sum(axis=0)
    print(f"  QCs selected: min={n_mult.min()}, max={n_mult.max()}, mean={n_mult.mean():.1f}")

    # ------------------------------------------------------------------
    # STEP 4: APPLY CORRECTIONS
    # ------------------------------------------------------------------
    print("\nStep 4: Applying corrections...")

    qc_selection: Dict[str, Any] = {
        "volumes": list(idp_list),
        "qc_basis": qc_basis,
        "apply_pca": apply_pca,
        "additive_qcs": [],
        "multiplicative_qcs": [],
        "additive_count": [],
        "multiplicative_count": [],
        "multiplicative_model_pval": [],
        "multiplicative_applied": [],
        "p_threshold": p_thr,
        "max_qcs": max_qcs,
        "preserve_covars": list(preserve_covars),
        "adjust_covars": list(adjust_covars),
        "categorical_covars": list(categorical_covars),
        "reverse_guard_covars": list(reverse_guard_covars),
        "qc_feature_names": list(qc_feature_names),
        "pca_num_components": num_qc_features if apply_pca else None,
        "pca_variance_explained": float(cumvar[num_qc_features - 1]) if apply_pca else None,
    }

    y_harmonised = np.full((len(data), len(idp_list)), np.nan, dtype=float)

    for v_idx, volname in enumerate(idp_list):
        print(f"\n=== Volume: {volname} ===")

        # ----------------------------
        # ADDITIVE CORRECTION
        # ----------------------------
        qc_mask_add = list(np.where(good_pcs_add[:, v_idx] == 1)[0] + 1)
        if not math.isinf(max_qcs) and len(qc_mask_add) > int(max_qcs):
            qc_mask_add = qc_mask_add[: int(max_qcs)]

        if len(qc_mask_add) == 0:
            print("  Additive correction: skipped (no QCs passed)")
            tmp_no_add = data[volname].astype(float).copy()
            qc_selection["additive_qcs"].append([])
            qc_selection["additive_count"].append(0)
        else:
            qc_vars_add = [qc_feature_names[i - 1] for i in qc_mask_add]
            print(f"  Additive correction: applying {len(qc_vars_add)} QC(s) -> {qc_vars_add}")

            dfm = complete_case(
                data,
                [subject_col] + list(preserve_covars) + list(adjust_covars) + [volname] + qc_vars_add,
            )
            formula_add = f"{volname} ~ {rhs_expr(list(preserve_covars) + list(adjust_covars) + qc_vars_add)} + (1|{subject_col})"

            fit_add, family_add, method_add = fit_mixed_then_ols(dfm, formula_add, subject_col)
            beta = fixed_params(fit_add)
            beta_qc = beta.reindex(qc_vars_add).fillna(0.0).to_numpy(dtype=float)
            add_effect = dfm.loc[:, qc_vars_add].to_numpy(dtype=float) @ beta_qc

            tmp_no_add = data[volname].astype(float).copy()
            tmp_no_add.loc[dfm.index] = dfm[volname].to_numpy(dtype=float) - add_effect

            qc_selection["additive_qcs"].append(qc_vars_add)
            qc_selection["additive_count"].append(len(qc_mask_add))

        tmp_no_add = tmp_no_add.clip(lower=1e-6)

        # ----------------------------
        # MULTIPLICATIVE CORRECTION
        # ----------------------------
        print("  Multiplicative selection: computing variance proxy from original volume")

        bio_df = complete_case(
            data,
            [subject_col] + list(preserve_covars) + list(adjust_covars) + [volname],
        )
        bio_formula = f"{volname} ~ {rhs_expr(list(preserve_covars) + list(adjust_covars))} + (1|{subject_col})"
        bio_fit, _, _ = fit_mixed_then_ols(bio_df, bio_formula, subject_col)

        res = np.asarray(bio_fit.resid, dtype=float)
        log_r2_series = pd.Series(
            np.log(np.maximum(res ** 2, np.finfo(float).eps)),
            index=bio_df.index,
            name="log_r2_tmp",
        )

        qc_mask_mult = list(np.where(good_pcs_mult[:, v_idx] == 1)[0] + 1)
        if not math.isinf(max_qcs) and len(qc_mask_mult) > int(max_qcs):
            qc_mask_mult = qc_mask_mult[: int(max_qcs)]

        if len(qc_mask_mult) == 0:
            print("  Multiplicative correction: skipped (no QCs passed)")
            y_h = tmp_no_add.copy()
            qc_selection["multiplicative_qcs"].append([])
            qc_selection["multiplicative_count"].append(0)
            qc_selection["multiplicative_model_pval"].append(np.nan)
            qc_selection["multiplicative_applied"].append(False)
        else:
            qc_vars_mult = [qc_feature_names[i - 1] for i in qc_mask_mult]
            print(f"  Multiplicative selection: {len(qc_vars_mult)} QC(s) passed -> {qc_vars_mult}")

            dfm = data.loc[:, [subject_col] + list(preserve_covars) + list(adjust_covars) + qc_vars_mult].copy()
            dfm["tmp_noAdditive"] = tmp_no_add
            dfm = dfm.join(log_r2_series, how="inner").dropna().copy()
            dfm["log_tmp_noAdditive"] = np.log(dfm["tmp_noAdditive"].astype(float))

            formula_red = f"log_tmp_noAdditive ~ {rhs_expr(list(preserve_covars) + list(adjust_covars))} + (1|{subject_col})"
            formula_full = f"log_tmp_noAdditive ~ {rhs_expr(list(preserve_covars) + list(adjust_covars) + qc_vars_mult)} + (1|{subject_col})"

            fit_full, fit_red, family_mult, method_full, method_red = fit_pair_same_family(dfm, formula_full, formula_red, subject_col)
            model_pval = lrt_pvalue(fit_full, fit_red, family_mult)
            qc_selection["multiplicative_model_pval"].append(model_pval)

            print(f"  Multiplicative model comparison p-value: {model_pval:.6g}")

            if model_pval < p_thr:
                beta = fixed_params(fit_full)
                beta_qc = beta.reindex(qc_vars_mult).fillna(0.0).to_numpy(dtype=float)
                lp = dfm.loc[:, qc_vars_mult].to_numpy(dtype=float) @ beta_qc
                lp_centered = lp - np.nanmean(lp)
                multiplicative_effect = np.exp(lp_centered)

                y_h = tmp_no_add.copy()
                y_h.loc[dfm.index] = dfm["tmp_noAdditive"].to_numpy(dtype=float) / multiplicative_effect

                qc_selection["multiplicative_qcs"].append(qc_vars_mult)
                qc_selection["multiplicative_count"].append(len(qc_mask_mult))
                qc_selection["multiplicative_applied"].append(True)
                print("  Multiplicative correction: applied")
            else:
                y_h = tmp_no_add.copy()
                qc_selection["multiplicative_qcs"].append([])
                qc_selection["multiplicative_count"].append(0)
                qc_selection["multiplicative_applied"].append(False)
                print("  Multiplicative correction: skipped (model p >= threshold)")

        data[f"harmonised_{volname}"] = y_h.to_numpy(dtype=float)
        y_harmonised[:, v_idx] = y_h.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # STEP 5: SAVE AND SUMMARISE
    # ------------------------------------------------------------------
    print("\nStep 5: Saving results...")

    data.to_csv(outfilename, index=False)
    print("  Data saved:", outfilename)

    summary_df = pd.DataFrame({
        "volume": qc_selection["volumes"],
        "additive_count": qc_selection["additive_count"],
        "multiplicative_count": qc_selection["multiplicative_count"],
        "multiplicative_applied": qc_selection["multiplicative_applied"],
        "multiplicative_model_pval": qc_selection["multiplicative_model_pval"],
    })
    summary_df.to_csv(summary_csv, index=False)
    print("  Summary saved:", summary_csv)

    add_detail_df = pd.DataFrame(additive_detail_rows)
    add_detail_df.to_csv(additive_detail_csv, index=False)
    print("  Additive details saved:", additive_detail_csv)

    mult_detail_df = pd.DataFrame(multiplicative_detail_rows)
    mult_detail_df.to_csv(multiplicative_detail_csv, index=False)
    print("  Multiplicative details saved:", multiplicative_detail_csv)

    pd.DataFrame({
        "volume": qc_selection["volumes"],
        "selected_qcs": [";".join(x) for x in qc_selection["additive_qcs"]],
    }).to_csv(selected_additive_qcs_csv, index=False)

    pd.DataFrame({
        "volume": qc_selection["volumes"],
        "selected_qcs": [";".join(x) for x in qc_selection["multiplicative_qcs"]],
    }).to_csv(selected_multiplicative_qcs_csv, index=False)

    print("  Selected QC lists saved.")

    print("\n=== SUMMARY ===")
    for i, vol in enumerate(qc_selection["volumes"]):
        print(
            f"{vol}: "
            f"additive={qc_selection['additive_count'][i]} QC(s), "
            f"multiplicative={qc_selection['multiplicative_count'][i]} QC(s), "
            f"multiplicative_applied={qc_selection['multiplicative_applied'][i]}"
        )

    print("\nDone!")
    return data, qc_selection, add_detail_df, mult_detail_df, summary_df