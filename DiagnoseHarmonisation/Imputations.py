"""This is a script containing functions that perform imputations of missing values in a dataset. 
The imputations are built off of the sklearn library and are designed to handle missing values in a dataset. 
For specific documentation, visit the sklearn documentation: https://scikit-learn.org/stable/modules/impute.html
The following imputation methods are implemented:


1. Mean imputation: missing values are replaced with the mean of the observed values for each feature.
2. Median imputation: missing values are replaced with the median of the observed values for each feature.
3. KNN imputation: missing values are replaced with the mean of the k-nearest neighbors for each feature.
4. Regression imputation: missing values are replaced with predictions from a regression model trained on the observed values for each feature.
5. PCA_imputation: missing values are replaced with predictions from a PCA model in an iterative manner.


Other imputation methods, along with cross-validation and hyperparameter tuning may be added in the future.
All methods require the data to be a pandas DataFrame, and the output will also be a pandas DataFrame with the same index and columns as the input data.

"""

import pandas as pd

def mean_imputation(data):
    """Impute missing values with the mean of each feature."""
    return data.fillna(data.mean())

def median_imputation(data):
    """Impute missing values with the median of each feature."""
    return data.fillna(data.median())

def knn_imputation(data, n_neighbors=5):
    """Impute missing values using k-nearest neighbors."""
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

def regression_imputation_feature(data,iter = False):
    """Impute missing values using regression models trained on all other features.

    Args:
        data (pd.DataFrame): The input dataset with missing values.
        iter (bool): If True, perform iterative imputation until convergence or a maximum number of iterations is reached. If False, perform a single imputation step.

    Returns:
        pd.DataFrame: The dataset with imputed values.
    
    """
    from sklearn.linear_model import LinearRegression

    if iter:
        # Iteratively impute missing values using regression models, updating the dataset until convergence or a maximum number of iterations is reached.
        max_iter = 10
        tol = 1e-3
        for i in range(max_iter):
            data_prev = data.copy()
            for column in data.columns:
                if data[column].isnull().any():
                    predictor_cols = [c for c in data.columns if c != column]
                    train_mask = data[column].notna() & data[predictor_cols].notna().all(axis=1)
                    predict_mask = data[column].isna() & data[predictor_cols].notna().all(axis=1)

                    if train_mask.sum() < 2 or predict_mask.sum() == 0:
                        continue

                    model = LinearRegression()
                    model.fit(data.loc[train_mask, predictor_cols], data.loc[train_mask, column])
                    data.loc[predict_mask, column] = model.predict(data.loc[predict_mask, predictor_cols])
            # Check for convergence
            if (data - data_prev).abs().max().max() < tol:
                break
        data_imputed = data.copy()
        return data_imputed
    else:
        data_imputed = data.copy()
        for column in data.columns:
            if data[column].isnull().any():
                predictor_cols = [c for c in data.columns if c != column]
                train_mask = data[column].notna() & data[predictor_cols].notna().all(axis=1)
                predict_mask = data[column].isna() & data[predictor_cols].notna().all(axis=1)

                if train_mask.sum() < 2 or predict_mask.sum() == 0:
                    continue

                model = LinearRegression()
                model.fit(data.loc[train_mask, predictor_cols], data.loc[train_mask, column])
                data_imputed.loc[predict_mask, column] = model.predict(data.loc[predict_mask, predictor_cols])
    return data_imputed


def regression_imputation_covariates(data, covariates, iter=False):
    """Impute missing values using regression models with covariates.

    Args:
        data (pd.DataFrame): The input dataset with missing values.
        covariates (list): List of column names to be used as covariates for imputation.

    Returns:
        pd.DataFrame: The dataset with imputed values.
    """
    from sklearn.linear_model import LinearRegression

    if iter:
        # Iteratively impute missing values using regression models, updating the dataset until convergence or a maximum number of iterations is reached.
        max_iter = 10
        tol = 1e-3
        for i in range(max_iter):
            data_prev = data.copy()
            for column in data.columns:
                if data[column].isnull().any():
                    train_mask = data[column].notna() & data[covariates].notna().all(axis=1)
                    predict_mask = data[column].isna() & data[covariates].notna().all(axis=1)

                    if train_mask.sum() < 2 or predict_mask.sum() == 0:
                        continue

                    model = LinearRegression()
                    model.fit(data.loc[train_mask, covariates], data.loc[train_mask, column])
                    data.loc[predict_mask, column] = model.predict(data.loc[predict_mask, covariates])
            # Check for convergence
            if (data - data_prev).abs().max().max() < tol:
                break
        data_imputed = data.copy()
        return data_imputed

    data_imputed = data.copy()
    for column in data.columns:
        if data[column].isnull().any():
            train_mask = data[column].notna() & data[covariates].notna().all(axis=1)
            predict_mask = data[column].isna() & data[covariates].notna().all(axis=1)

            if train_mask.sum() < 2 or predict_mask.sum() == 0:
                continue

            model = LinearRegression()
            model.fit(data.loc[train_mask, covariates], data.loc[train_mask, column])
            data_imputed.loc[predict_mask, column] = model.predict(data.loc[predict_mask, covariates])
    return data_imputed

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh


def nets_svds(X, n):
    """
    Python equivalent of MATLAB nets_svds from https://git.fmrib.ox.ac.uk/steve/fslnets_matlab
    This is included as a helper function for PCA_imputation and is not used elsewhere in the codebase.

    Args:
        X (ndarray): Input data matrix with shape (m, n).
        n (int): Number of components to compute. If n <= 0, compute rank+n components.

    Returns:
        U (ndarray): Left singular vectors with shape (m, n).
        S (ndarray): Singular values with shape (n, n).
        V (ndarray): Right singular vectors with shape (p, n).
    """

    m, p = X.shape

    if n < 1:
        n = max(min(m, p) + n, 1)

    if m < p:
        A = X @ X.T
        A = np.asarray(A, dtype=float)
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0

        if n < m:
            vals, U = eigsh(A, k=n, which='LM')
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            U = U[:, order]
        else:
            vals, U = eigh(A)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            U = U[:, order]

        s = np.sqrt(np.abs(vals))
        S = np.diag(s)

        invs = np.divide(1.0, s, out=np.zeros_like(s), where=s > 1e-12)
        V = X.T @ (U * invs)

    else:
        A = X.T @ X
        A = np.asarray(A, dtype=float)
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0

        if n < p:
            vals, V = eigsh(A, k=n, which='LM')
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            V = V[:, order]
        else:
            vals, V = eigh(A)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            V = V[:, order]

        s = np.sqrt(np.abs(vals))
        S = np.diag(s)

        invs = np.divide(1.0, s, out=np.zeros_like(s), where=s > 1e-12)
        U = X @ (V * invs)

    return U, S, V


def PCA_imputation(impute_input, Npca, Nrand=0):
    """
    Impute missing values using PCA-based imputation. Based on original code in MATLAB by Stephen Smith, modified for python:
    https://git.fmrib.ox.ac.uk/steve/fslnets_matlab/

    Args:
        impute_input (array-like): Input data with missing values (NaNs).
        Npca (int): Number of principal components to use for imputation. If Npca > 0, use Npca components. If Npca < 0, use -Npca components.
        Nrand (int): Number of random initializations for imputation. If Nrand > 0, perform Nrand random initializations and average the results. If Nrand = 0, perform a single imputation.
    
    Returns:
        impute_replace (ndarray): Imputed data with missing values replaced.
        impute_noreplace (ndarray): Imputed data without replacing missing values.
        pcaU (ndarray): Left singular vectors from PCA.
        pcaS (ndarray): Singular values from PCA.
        pcaV (ndarray): Right singular vectors from PCA.

    """

    # check if input is a pandas DataFrame and convert to numpy array
    original_data = None
    if isinstance(impute_input, pd.DataFrame):
        original_data = impute_input.copy()
        impute_input = impute_input.values

    X = np.asarray(impute_input, dtype=float).copy()

    if Npca > 0:
        NpcaINT = min(2 * Npca, min(X.shape))
    else:
        NpcaINT = min(-Npca, min(X.shape))

    OrigMean = np.nanmean(X, axis=0)
    X -= OrigMean

    if X.shape[1] == 1 or Npca == 0:
        impute_replace = X.copy()
        impute_replace[np.isnan(impute_replace)] = 0

        impute_noreplace = impute_replace.copy()
        pcaU = impute_replace
        pcaS = np.array([[1.]])
        pcaV = np.array([[1.]])

    else:

        grot = []

        for r in range(max(1, Nrand)):

            nanmask = np.isnan(X)

            if Nrand == 0:
                impute_noreplace = np.zeros_like(X)
            else:
                std = np.nanstd(X, axis=0)
                noise = np.random.randn(*X.shape)
                impute_noreplace = noise * std / 2

            impute_corr = 0

            while impute_corr < 0.9999 or np.isnan(impute_corr):

                impute_replace = X.copy()
                impute_replace[nanmask] = impute_noreplace[nanmask]

                pcaU, pcaS, pcaV = nets_svds(impute_replace, NpcaINT)

                if Npca > 0:
                    s = np.diag(pcaS)
                    smin = s.min()

                    shrink = ((np.arange(len(s)) /
                               max(len(s)-1, 1))**4) * smin

                    pcaSnew = np.diag(s - shrink)

                else:
                    pcaSnew = pcaS

                impute_noreplace = pcaU @ pcaSnew @ pcaV.T

                if nanmask.sum() < 2:
                    impute_corr += 0.3
                else:
                    old = impute_replace[nanmask]
                    new = impute_noreplace[nanmask]

                    if np.std(old) == 0 or np.std(new) == 0:
                        impute_corr = 0
                    else:
                        impute_corr = np.corrcoef(old, new)[0, 1]

            if Nrand > 0:
                grot.append(impute_replace)

        if Nrand > 0:
            impute_replace = np.mean(grot, axis=0)

        if Npca != 0:
            k = abs(Npca)
            pcaU = pcaU[:, :k]
            pcaS = pcaS[:k, :k]
            pcaV = pcaV[:, :k]

    impute_replace += OrigMean
    impute_noreplace += OrigMean

    # If the input was a pandas DataFrame, convert the output back to a DataFrame with the same index and columns
    if original_data is not None:
        impute_replace = pd.DataFrame(impute_replace, index=original_data.index, columns=original_data.columns)
        impute_noreplace = pd.DataFrame(impute_noreplace, index=original_data.index, columns=original_data.columns)
        pcaU = pd.DataFrame(pcaU, index=original_data.index, columns=[f'PC{i+1}' for i in range(pcaU.shape[1])])
        pcaS = pd.DataFrame(pcaS, index=[f'PC{i+1}' for i in range(pcaS.shape[0])], columns=[f'PC{i+1}' for i in range(pcaS.shape[1])])
        pcaV = pd.DataFrame(pcaV, index=original_data.columns, columns=[f'PC{i+1}' for i in range(pcaV.shape[1])])

    return impute_replace, impute_noreplace, pcaU, pcaS, pcaV
