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
from statsmodels.stats.multitest import multipletests
import json
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
    batch: Sequence,
    idp_names: Optional[Sequence[str]] = None,
    nPerm: int = 10000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute subject-order consistency across batches/scanners.

    For each pair of batches, subjects represented in both batches
    are matched. If a subject has multiple observations within the
    same batch, those observations are averaged first so that each
    subject contributes one value per batch and IDP.

    Spearman rank correlation is then calculated across matched
    subjects for each IDP.

    Significance is assessed using permutation testing by shuffling
    the values from the second batch across matched subjects.

    Parameters
    ----------
    idp_matrix : array-like, shape (n_samples, n_idps)
        Numeric matrix of IDP values.

    subjects : sequence of length n_samples
        Subject identifiers.

    batch : sequence of length n_samples
        Batch/scanner/site/session identifier for each observation.

    idp_names : sequence of length n_idps, optional
        IDP names. If None, names are generated as:

            idp_1
            idp_2
            ...

    nPerm : int, default=10000
        Number of permutations.

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per batch-pair x IDP.

        Columns:

            TimeA
            TimeB
            IDP
            nPairs
            SpearmanRho
            NullMeanRho
            pValue

        TimeA and TimeB contain batch labels and are retained for
        compatibility with existing output and plotting functions.

        nPairs is the number of matched subjects contributing to
        the correlation after removal of missing IDP values.

        Rows with fewer than three matched subjects return NaN for
        SpearmanRho, NullMeanRho and pValue.
    """

    # ============================================================
    # VALIDATE IDP MATRIX
    # ============================================================

    idp_matrix = np.asarray(
        idp_matrix,
        dtype=float,
    )

    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_idps).")

    n_samples, n_idps = idp_matrix.shape

    if n_samples == 0:
        raise ValueError("idp_matrix contains no observations.")

    if n_idps == 0:
        raise ValueError("idp_matrix contains no IDPs.")

    # ============================================================
    # VALIDATE SUBJECTS
    # ============================================================

    if len(subjects) != n_samples:
        raise ValueError("Length of subjects must match number of rows in idp_matrix.")

    subjects_arr = np.asarray(subjects).astype(str)

    # ============================================================
    # VALIDATE BATCH
    # ============================================================

    if len(batch) != n_samples:
        raise ValueError("Length of batch must match number of rows in idp_matrix.")

    batch_arr = np.asarray(batch).astype(str)

    # ============================================================
    # VALIDATE IDP NAMES
    # ============================================================

    if idp_names is None:
        idp_names = [f"idp_{i + 1}" for i in range(n_idps)]

    else:
        idp_names = list(idp_names)

        if len(idp_names) != n_idps:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

        if len(set(idp_names)) != len(idp_names):
            raise ValueError("idp_names must be unique.")

    # ============================================================
    # VALIDATE PERMUTATIONS
    # ============================================================

    if (
        not isinstance(
            nPerm,
            int,
        )
        or nPerm < 1
    ):
        raise ValueError("nPerm must be an integer >= 1.")

    # ============================================================
    # CHECK SUBJECT / BATCH LABELS
    # ============================================================

    if np.any(pd.isna(subjects)):
        raise ValueError("subjects contains missing values.")

    if np.any(pd.isna(batch)):
        raise ValueError("batch contains missing values.")

    # ============================================================
    # BATCH ORDER
    #
    # Preserve order of first appearance.
    # ============================================================

    batch_levels = pd.Index(batch_arr).unique().tolist()

    if len(batch_levels) < 2:
        raise ValueError(
            "At least two unique batches are required for subject-order consistency."
        )

    # ============================================================
    # RANDOM NUMBER GENERATOR
    # ============================================================

    rng = default_rng(seed)

    # ============================================================
    # CREATE OBSERVATION-LEVEL DATAFRAME
    # ============================================================

    idp_df = pd.DataFrame(
        idp_matrix,
        columns=idp_names,
    )

    idp_df.insert(
        0,
        "_batch",
        batch_arr,
    )

    idp_df.insert(
        0,
        "_subject",
        subjects_arr,
    )

    # ============================================================
    # AVERAGE REPEATED OBSERVATIONS WITHIN SUBJECT x BATCH
    #
    # This gives one row for each subject under each batch.
    #
    # If there is only one observation for a subject x batch,
    # that observation is unchanged.
    # ============================================================

    subject_batch_df = idp_df.groupby(
        [
            "_subject",
            "_batch",
        ],
        sort=False,
        as_index=False,
    )[idp_names].mean()

    # ============================================================
    # COUNT REPEATED SUBJECT x BATCH OBSERVATIONS
    # ============================================================

    subject_batch_counts = idp_df.groupby(
        [
            "_subject",
            "_batch",
        ],
        sort=False,
    ).size()

    n_repeated_subject_batch = int((subject_batch_counts > 1).sum())

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 70)

    print("SUBJECT-ORDER CONSISTENCY")

    print("=" * 70)

    print(
        "Observations:",
        n_samples,
    )

    print(
        "Subjects:",
        len(np.unique(subjects_arr)),
    )

    print(
        "Batches:",
        len(batch_levels),
    )

    print(
        "Subject x batch combinations:",
        len(subject_batch_df),
    )

    print(
        "Subject x batch combinations with >1 observation:",
        n_repeated_subject_batch,
    )

    if n_repeated_subject_batch > 0:
        print(
            "Repeated observations within subject x batch "
            "were averaged before calculating correlations."
        )

    # ============================================================
    # OUTPUT STORAGE
    # ============================================================

    rows = []

    # ============================================================
    # BATCH-PAIR LOOP
    # ============================================================

    for ia in range(len(batch_levels) - 1):
        for ib in range(
            ia + 1,
            len(batch_levels),
        ):
            batch_A = batch_levels[ia]

            batch_B = batch_levels[ib]

            # ====================================================
            # EXTRACT SUBJECTS FROM EACH BATCH
            # ====================================================

            data_A = subject_batch_df[subject_batch_df["_batch"] == batch_A].copy()

            data_B = subject_batch_df[subject_batch_df["_batch"] == batch_B].copy()

            # ====================================================
            # MATCH SUBJECTS ACROSS BATCHES
            #
            # Inner merge guarantees that only subjects observed
            # under both batches contribute.
            # ====================================================

            matched = data_A.merge(
                data_B,
                on="_subject",
                how="inner",
                suffixes=(
                    "_A",
                    "_B",
                ),
                validate="one_to_one",
                sort=False,
            )

            n_matched_subjects = len(matched)

            # ====================================================
            # NO MATCHED SUBJECTS
            # ====================================================

            if n_matched_subjects == 0:
                for idp_name in idp_names:
                    rows.append(
                        {
                            "TimeA": batch_A,
                            "TimeB": batch_B,
                            "IDP": idp_name,
                            "nPairs": 0,
                            "SpearmanRho": np.nan,
                            "NullMeanRho": np.nan,
                            "pValue": np.nan,
                        }
                    )

                continue

            # ====================================================
            # IDP LOOP
            # ====================================================

            for idp_name in idp_names:
                column_A = f"{idp_name}_A"

                column_B = f"{idp_name}_B"

                # ------------------------------------------------
                # Extract matched subject values
                # ------------------------------------------------

                xa = pd.to_numeric(
                    matched[column_A],
                    errors="coerce",
                ).to_numpy(dtype=float)

                yb = pd.to_numeric(
                    matched[column_B],
                    errors="coerce",
                ).to_numpy(dtype=float)

                # ------------------------------------------------
                # Remove missing / non-finite values
                # ------------------------------------------------

                valid_mask = np.isfinite(xa) & np.isfinite(yb)

                xa = xa[valid_mask]

                yb = yb[valid_mask]

                nPairs = int(len(xa))

                # =================================================
                # MINIMUM NUMBER OF MATCHED SUBJECTS
                # =================================================

                if nPairs < 3:
                    rows.append(
                        {
                            "TimeA": batch_A,
                            "TimeB": batch_B,
                            "IDP": idp_name,
                            "nPairs": nPairs,
                            "SpearmanRho": np.nan,
                            "NullMeanRho": np.nan,
                            "pValue": np.nan,
                        }
                    )

                    continue

                # =================================================
                # OBSERVED SPEARMAN CORRELATION
                # =================================================

                obs_rho, _ = spearmanr(
                    xa,
                    yb,
                    nan_policy="omit",
                )

                if np.isnan(obs_rho):
                    abs_obs = None

                else:
                    abs_obs = abs(float(obs_rho))

                # =================================================
                # PERMUTATION NULL DISTRIBUTION
                #
                # Shuffle values from batch B across matched
                # subjects while keeping batch A fixed.
                # =================================================

                sum_null = 0.0

                valid_null_count = 0

                count_ge = 0

                for _ in range(nPerm):
                    perm_idx = rng.permutation(nPairs)

                    null_rho, _ = spearmanr(
                        xa,
                        yb[perm_idx],
                        nan_policy="omit",
                    )

                    if np.isnan(null_rho):
                        continue

                    null_rho = float(null_rho)

                    sum_null += null_rho

                    valid_null_count += 1

                    if abs_obs is not None and abs(null_rho) >= abs_obs:
                        count_ge += 1

                # =================================================
                # NULL MEAN
                # =================================================

                if valid_null_count > 0:
                    null_mean = float(sum_null / valid_null_count)

                else:
                    null_mean = np.nan

                # =================================================
                # PERMUTATION P-VALUE
                #
                # +1 correction:
                #
                #     p = (1 + count_ge)
                #         / (1 + valid_null_count)
                # =================================================

                if valid_null_count > 0:
                    pval = float((1 + count_ge) / (1 + valid_null_count))

                else:
                    pval = np.nan

                # =================================================
                # STORE RESULT
                # =================================================

                rows.append(
                    {
                        "TimeA": batch_A,
                        "TimeB": batch_B,
                        "IDP": idp_name,
                        "nPairs": nPairs,
                        "SpearmanRho": (
                            float(obs_rho) if not np.isnan(obs_rho) else np.nan
                        ),
                        "NullMeanRho": null_mean,
                        "pValue": pval,
                    }
                )

    # ============================================================
    # CREATE OUTPUT DATAFRAME
    # ============================================================

    columns = [
        "TimeA",
        "TimeB",
        "IDP",
        "nPairs",
        "SpearmanRho",
        "NullMeanRho",
        "pValue",
    ]

    results = pd.DataFrame(
        rows,
        columns=columns,
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    valid_rho = pd.to_numeric(
        results["SpearmanRho"],
        errors="coerce",
    )

    print(
        "Batch pairs:",
        (len(batch_levels) * (len(batch_levels) - 1) // 2),
    )

    print(
        "IDPs:",
        len(idp_names),
    )

    print(
        "Result rows:",
        len(results),
    )

    print(
        "Valid Spearman correlations:",
        int(np.isfinite(valid_rho).sum()),
    )

    print("=" * 70)

    # ============================================================
    # RETURN
    # ============================================================

    return results

##########################################################################################################
# FOR LONGITUDINAL DATA: 2) Within subject variability: % difference (2 timepoints) OR CoV (>2 timepoints)
##########################################################################################################
from itertools import combinations


def _pairwise_rpd(arr: np.ndarray) -> float:
    """
    Median pairwise relative percent difference (RPD) across all pairs.

    RPD(i, j) = abs(xi - xj) / ((xi + xj) / 2) * 100

    Parameters
    ----------
    arr : np.ndarray
        1D array of numeric values.

    Returns
    -------
    float
        Median pairwise RPD in percent, or NaN if fewer than 2 valid values
        or if no valid denominator is available.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size

    if n < 2:
        return np.nan

    rpds = []
    for i, j in combinations(range(n), 2):
        denom = (arr[i] + arr[j]) / 2.0
        if denom == 0 or np.isnan(denom):
            continue
        rpds.append(abs(arr[i] - arr[j]) / denom * 100.0)

    if len(rpds) == 0:
        return np.nan

    return float(np.median(rpds))


def WithinSubjVar_long(
    idp_matrix: np.ndarray,
    subjects: Sequence,
    timepoints: Sequence,
    idp_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute within-subject variability (percent) for each IDP across timepoints.

    Returns both the median pairwise RPD and its standard deviation.

    This version uses one consistent metric for all subjects:
    median pairwise RPD across all available non-missing measurements.

    Output columns:
    - subject
    - n_obs
    - metric_type
    - one column per IDP (each containing a tuple of median RPD and its standard deviation)

    Parameters
    ----------
    idp_matrix : np.ndarray
        Numeric matrix of shape (n_samples, n_idps).
    subjects : Sequence
        Subject identifiers aligned to rows of idp_matrix.
    timepoints : Sequence
        Timepoint labels aligned to rows of idp_matrix. Kept for validation.
    idp_names : Optional[Sequence[str]]
        Optional list of IDP names. Defaults to idp_1, idp_2, ...

    Returns
    -------
    pd.DataFrame
        One row per subject with within-subject variability values.
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
    # df["timepoint"] = list(timepoints) 

    out_rows = []
    for subj, g in df.groupby("subject", sort=False):
        row = {
            "subject": subj,
            "n_obs": len(g),
            "metric_type": "Median Pairwise RPD",
        }

        for col in idp_names:
            arr = g[col].dropna().to_numpy(dtype=float)
            row[col] = _pairwise_rpd(arr)  


        out_rows.append(row)

    return pd.DataFrame(out_rows)

##########################################################################################################
# FOR LONGITUDINAL DATA: Multivariate pairwise site differences using Mahalanobis distances
##########################################################################################################
from typing import Any, Dict, Optional, Sequence

def MultiVariateBatchDifference_long(
    idp_matrix: np.ndarray,
    batch: pd.Series | Sequence,
    subject: pd.Series | Sequence | None = None,
    timepoint: pd.Series | Sequence | None = None,
    idp_names: Sequence[str] | None = None,
    covariates: dict[str, pd.Series | Sequence | np.ndarray] | None = None,
    return_info: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    
    """
    Multivariate batch/site differences using Mahalanobis distances
    for regular and repeated-measures data.

    The analysis representation is selected from the observed
    repeated-measures structure.

    Analysis hierarchy
    ------------------
    1. If batch varies within subject:
           subject-centred observations are used.

    2. Otherwise, if subjects are repeated across multiple
       timepoints:
           consecutive longitudinal changes are used:
               Y(i,t) - Y(i,t-1)

    3. Otherwise:
           original observations are used.

    Covariates are supplied as a dictionary:

        covariates = {
            "age": age_array,
            "sex": sex_array,
        }

    Numeric covariates are retained as numeric.

    Two-level categorical covariates are converted to a single
    0/1 variable.

    Categorical covariates with more than two levels are expanded
    into k-1 dummy variables using the first observed level as the
    reference category.

    Both raw and covariate-adjusted Mahalanobis distances are
    returned when covariates are supplied.

    Parameters
    ----------
    idp_matrix : np.ndarray
        Numeric matrix with shape (n_samples, n_features).

    batch : sequence
        Batch/site/session label for each observation.

    subject : sequence, optional
        Subject identifier for each observation.

    timepoint : sequence, optional
        Timepoint/visit identifier for each observation.

    idp_names : sequence of str, optional
        IDP/feature names.

    covariates : dict, optional
        Covariates to residualize before constructing the
        repeated-measures representation.

    return_info : bool
        If True, return:

            results, info

    Returns
    -------
    results : dict
        Contains:

            pairwise_raw
            pairwise_resid
            centroid_raw
            centroid_resid
            batches

    info : dict
        Returned when return_info=True.
    """

    # ============================================================
    # 1. VALIDATE / COERCE IDP MATRIX
    # ============================================================

    if not isinstance(
        idp_matrix,
        np.ndarray,
    ):
        idp_matrix = np.asarray(
            idp_matrix,
            dtype=float,
        )

    if idp_matrix.ndim != 2:
        raise ValueError("idp_matrix must be 2D (n_samples, n_features).")

    idp_matrix = idp_matrix.astype(
        float,
        copy=False,
    )

    n_samples, n_features = idp_matrix.shape

    if n_samples == 0:
        raise ValueError("idp_matrix contains no observations.")

    if n_features == 0:
        raise ValueError("idp_matrix contains no features.")

    # ============================================================
    # 2. VALIDATE BATCH
    # ============================================================

    if isinstance(
        batch,
        pd.Series,
    ):
        batch_ser = batch.reset_index(drop=True)
    else:
        batch_ser = pd.Series(batch)

    if len(batch_ser) != n_samples:
        raise ValueError("Length of batch must match number of rows in idp_matrix.")

    if batch_ser.isna().any():
        raise ValueError("batch contains missing values.")

    # ============================================================
    # 3. VALIDATE SUBJECT
    # ============================================================

    if subject is not None:
        if isinstance(
            subject,
            pd.Series,
        ):
            subject_ser = subject.reset_index(drop=True)

        else:
            subject_ser = pd.Series(subject)

        if len(subject_ser) != n_samples:
            raise ValueError(
                "Length of subject must match number of rows in idp_matrix."
            )

        if subject_ser.isna().any():
            raise ValueError("subject contains missing values.")

    else:
        subject_ser = None

    # ============================================================
    # 4. VALIDATE TIMEPOINT
    # ============================================================

    if timepoint is not None:
        if isinstance(
            timepoint,
            pd.Series,
        ):
            timepoint_ser = timepoint.reset_index(drop=True)

        else:
            timepoint_ser = pd.Series(timepoint)

        if len(timepoint_ser) != n_samples:
            raise ValueError(
                "Length of timepoint must match number of rows in idp_matrix."
            )

        if timepoint_ser.isna().any():
            raise ValueError("timepoint contains missing values.")

    else:
        timepoint_ser = None

    # ============================================================
    # 5. IDP NAMES
    # ============================================================

    if idp_names is None:
        idp_names = [f"idp_{i + 1}" for i in range(n_features)]

    else:
        idp_names = list(idp_names)

        if len(idp_names) != n_features:
            raise ValueError("idp_names length must match idp_matrix.shape[1].")

        if len(set(idp_names)) != len(idp_names):
            raise ValueError("idp_names must be unique.")

    # ============================================================
    # 6. PREPARE COVARIATES
    # ============================================================

    have_covariates = covariates is not None

    covariate_encoding = {}

    if have_covariates:
        if not isinstance(
            covariates,
            dict,
        ):
            raise TypeError(
                "covariates must be a dictionary of the form "
                "{'covariate_name': values}."
            )

        if len(covariates) == 0:
            raise ValueError("covariates dictionary is empty.")

        encoded_parts = []

        for name, values in covariates.items():
            cov_name = str(name)

            # ----------------------------------------------------
            # Convert to Series
            # ----------------------------------------------------

            if isinstance(
                values,
                pd.Series,
            ):
                series = values.reset_index(drop=True)

            else:
                try:
                    series = pd.Series(values).reset_index(drop=True)

                except Exception as exc:
                    raise ValueError(
                        f"Could not convert covariate '{cov_name}' to a 1D sequence."
                    ) from exc

            if len(series) != n_samples:
                raise ValueError(
                    f"Covariate '{cov_name}' has "
                    f"{len(series)} values, but idp_matrix "
                    f"has {n_samples} observations."
                )

            if series.isna().any():
                raise ValueError(
                    f"Covariate '{cov_name}' contains missing "
                    "values. Please clean or impute them first."
                )

            # ----------------------------------------------------
            # Boolean
            # ----------------------------------------------------

            if pd.api.types.is_bool_dtype(series):
                encoded = pd.DataFrame({cov_name: series.astype(float)})

                covariate_encoding[cov_name] = {
                    "type": "binary",
                    "levels": [
                        False,
                        True,
                    ],
                    "reference_level": False,
                    "columns": [cov_name],
                }

                encoded_parts.append(encoded)

                continue

            # ----------------------------------------------------
            # Numeric
            # ----------------------------------------------------

            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if numeric.isna().any():
                    raise ValueError(
                        f"Covariate '{cov_name}' contains values "
                        "that cannot be converted to numeric."
                    )

                unique_values = list(pd.unique(numeric))

                if len(unique_values) == 1:
                    warnings.warn(
                        f"Covariate '{cov_name}' has only one "
                        "unique value and therefore contributes "
                        "no variation to the model."
                    )

                encoded = pd.DataFrame({cov_name: numeric.astype(float)})

                if len(unique_values) == 2:
                    covariate_encoding[cov_name] = {
                        "type": "binary_numeric",
                        "levels": [float(x) for x in unique_values],
                        "reference_level": float(unique_values[0]),
                        "columns": [cov_name],
                    }

                else:
                    covariate_encoding[cov_name] = {
                        "type": "continuous",
                        "columns": [cov_name],
                    }

                encoded_parts.append(encoded)

                continue

            # ----------------------------------------------------
            # Categorical/string
            # ----------------------------------------------------

            series_string = series.astype(str)

            levels = list(pd.unique(series_string))

            n_levels = len(levels)

            if n_levels < 2:
                raise ValueError(
                    f"Covariate '{cov_name}' has fewer than "
                    "two unique levels and cannot be used "
                    "as a covariate."
                )

            # ----------------------------------------------------
            # Binary categorical
            # ----------------------------------------------------

            if n_levels == 2:
                reference = levels[0]

                positive = levels[1]

                encoded_series = series_string.map(
                    {
                        reference: 0.0,
                        positive: 1.0,
                    }
                )

                encoded = pd.DataFrame({cov_name: encoded_series})

                covariate_encoding[cov_name] = {
                    "type": "binary_categorical",
                    "levels": [
                        reference,
                        positive,
                    ],
                    "reference_level": reference,
                    "columns": [cov_name],
                }

                encoded_parts.append(encoded)

            # ----------------------------------------------------
            # Multi-level categorical
            # ----------------------------------------------------

            else:
                categorical = pd.Categorical(
                    series_string,
                    categories=levels,
                    ordered=False,
                )

                dummy_df = pd.get_dummies(
                    categorical,
                    prefix=cov_name,
                    drop_first=True,
                    dtype=float,
                )

                dummy_df.columns = [str(col) for col in dummy_df.columns]

                covariate_encoding[cov_name] = {
                    "type": "categorical",
                    "levels": levels,
                    "reference_level": levels[0],
                    "columns": dummy_df.columns.tolist(),
                }

                encoded_parts.append(dummy_df)

        if len(encoded_parts) == 0:
            raise ValueError("No usable covariates remained after encoding.")

        cov_df_encoded = pd.concat(
            encoded_parts,
            axis=1,
        )

        covariates_array = cov_df_encoded.to_numpy(dtype=float)

        if not np.isfinite(covariates_array).all():
            raise ValueError("Encoded covariates contain NaN or infinite values.")

        covariates = covariates_array

    else:
        covariate_encoding = {}

        covariates = None

    # ============================================================
    # 7. REPEATED-MEASURES STRUCTURE AND TIMEPOINT ORDER
    # ============================================================

    def _timepoint_sort_key(value):

        text = str(value).strip().lower()

        baseline_labels = {
            "baseline",
            "pre",
            "pretest",
            "pre_test",
            "t0",
            "time0",
            "time_0",
        }

        if text in baseline_labels:
            return (
                0,
                0.0,
                text,
            )

        try:
            return (
                1,
                float(text),
                text,
            )

        except ValueError:
            pass

        numbers = re.findall(
            r"[-+]?\d*\.?\d+",
            text,
        )

        if numbers:
            try:
                return (
                    1,
                    float(numbers[-1]),
                    text,
                )

            except ValueError:
                pass

        return (
            2,
            0.0,
            text,
        )

    # ------------------------------------------------------------
    # Timepoint structure
    # ------------------------------------------------------------

    if timepoint_ser is None:
        ordered_timepoints = None

        n_timepoints = 1

    else:
        if isinstance(
            timepoint_ser.dtype,
            pd.CategoricalDtype,
        ):
            observed = set(timepoint_ser.tolist())

            ordered_timepoints = [
                tp for tp in (timepoint_ser.cat.categories) if tp in observed
            ]

        else:
            ordered_timepoints = sorted(
                pd.unique(timepoint_ser).tolist(),
                key=_timepoint_sort_key,
            )

        n_timepoints = len(ordered_timepoints)

    # ------------------------------------------------------------
    # Repeated subjects
    # ------------------------------------------------------------

    if subject_ser is None:
        has_repeated_subjects = False

    else:
        has_repeated_subjects = bool(subject_ser.duplicated().any())

    # ------------------------------------------------------------
    # Batch variation within subject
    # ------------------------------------------------------------

    batch_varies_within_subject = False

    batches_per_subject = None

    n_subjects = 0

    n_subjects_multiple_batches = 0

    n_subjects_single_batch = 0

    if subject_ser is not None:
        structure_df = pd.DataFrame(
            {
                "_subject": subject_ser.to_numpy(),
                "_batch": batch_ser.to_numpy(),
            }
        )

        batches_per_subject = structure_df.groupby(
            "_subject",
            sort=False,
        )["_batch"].nunique()

        batch_varies_within_subject = bool((batches_per_subject > 1).any())
        n_subjects_multiple_batches = int((batches_per_subject > 1).sum())
        n_subjects_single_batch = int((batches_per_subject == 1).sum())

    # ------------------------------------------------------------
    # Select analysis representation
    # ------------------------------------------------------------

    if (
        subject_ser is not None
        and has_repeated_subjects
        and batch_varies_within_subject
    ):
        analysis_mode = "repeated"

        longitudinal_case = "batch_varies_within_subject"

    elif subject_ser is not None and has_repeated_subjects and n_timepoints > 1:
        analysis_mode = "longitudinal"

        longitudinal_case = "batch_constant_within_subject"

    else:
        analysis_mode = "regular"

        longitudinal_case = None

    # ============================================================
    # 8. HELPER: RESIDUALIZATION
    # ============================================================

    def _residualize(
        X: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:

        design = np.column_stack(
            [
                np.ones(
                    (
                        C.shape[0],
                        1,
                    )
                ),
                C,
            ]
        )

        B, *_ = np.linalg.lstsq(
            design,
            X,
            rcond=None,
        )

        return X - design @ B

    # ============================================================
    # 9. HELPER: DESIGN RANK
    # ============================================================

    def _design_rank(
        C: np.ndarray,
    ) -> int:

        design = np.column_stack(
            [
                np.ones(
                    (
                        C.shape[0],
                        1,
                    )
                ),
                C,
            ]
        )

        return int(np.linalg.matrix_rank(design))

    # ============================================================
    # 10. HELPER: COVARIANCE
    # ============================================================

    def _covariance(
        X: np.ndarray,
    ) -> np.ndarray:

        if X.shape[0] <= 1:
            return np.zeros(
                (
                    X.shape[1],
                    X.shape[1],
                ),
                dtype=float,
            )

        cov = np.cov(
            X,
            rowvar=False,
            ddof=1,
        )

        cov = np.atleast_2d(
            np.asarray(
                cov,
                dtype=float,
            )
        )

        if cov.shape != (
            X.shape[1],
            X.shape[1],
        ):
            raise ValueError("Unexpected covariance matrix shape.")

        return cov

    # ============================================================
    # 11. HELPER: STABLE PSEUDOINVERSE
    # ============================================================

    def _stable_pinv(
        S: np.ndarray,
    ):

        U, s, Vt = np.linalg.svd(
            S,
            full_matrices=False,
        )

        if len(s) == 0:
            return (
                np.zeros_like(S),
                0,
            )

        tol = np.max(s) * max(S.shape) * np.finfo(float).eps

        keep = s > tol

        s_inv = np.zeros_like(s)

        s_inv[keep] = 1.0 / s[keep]

        pinv = (Vt.T * s_inv) @ U.T

        return (
            pinv,
            int(np.sum(keep)),
        )

    # ============================================================
    # 12. HELPER: MAHALANOBIS DISTANCES
    # ============================================================

    def _calculate_distances(
        X: np.ndarray,
        batches: np.ndarray,
        covariance_mode: str,
        centroid_mode: str,
    ):

        X = np.asarray(
            X,
            dtype=float,
        )

        batches = np.asarray(batches)

        if X.ndim != 2:
            raise ValueError("Analysis matrix must be 2D.")

        if X.shape[0] != len(batches):
            raise ValueError("Batch length does not match analysis matrix.")

        if X.shape[0] < 2:
            raise ValueError("At least two observations are required.")

        if not np.isfinite(X).all():
            raise ValueError("Analysis matrix contains non-finite values.")

        unique_batches = list(dict.fromkeys(batches.tolist()))

        if len(unique_batches) < 2:
            raise ValueError("At least two unique batches are required.")

        # --------------------------------------------------------
        # Batch means
        # --------------------------------------------------------

        batch_means = {b: X[batches == b].mean(axis=0) for b in unique_batches}

        # --------------------------------------------------------
        # Covariance
        # --------------------------------------------------------

        if covariance_mode == "site_average":
            site_covariances = []

            for b in unique_batches:
                Xb = X[batches == b]

                site_covariances.append(_covariance(Xb))

            overall_cov = np.mean(
                site_covariances,
                axis=0,
            )

        elif covariance_mode == "pooled":
            overall_cov = _covariance(X)

        else:
            raise ValueError("Unknown covariance_mode.")

        # --------------------------------------------------------
        # Inverse covariance
        # --------------------------------------------------------

        try:
            cond_number = float(np.linalg.cond(overall_cov))

        except Exception:
            cond_number = np.inf

        use_pinv = not np.isfinite(cond_number) or cond_number > 1e15

        retained_svals = 0

        if use_pinv:
            (
                covariance_inv,
                retained_svals,
            ) = _stable_pinv(overall_cov)

        else:
            try:
                covariance_inv = np.linalg.inv(overall_cov)

            except np.linalg.LinAlgError:
                warnings.warn(
                    "Covariance matrix was singular during "
                    "inversion; using pseudoinverse."
                )

                (
                    covariance_inv,
                    retained_svals,
                ) = _stable_pinv(overall_cov)

                use_pinv = True

        # --------------------------------------------------------
        # Pairwise distances
        # --------------------------------------------------------

        pairwise = {}

        for b1, b2 in combinations(
            unique_batches,
            2,
        ):
            diff = batch_means[b1] - batch_means[b2]

            delta = float(diff @ covariance_inv @ diff)

            pairwise[
                (
                    b1,
                    b2,
                )
            ] = float(
                np.sqrt(
                    max(
                        delta,
                        0.0,
                    )
                )
            )

        # --------------------------------------------------------
        # Global centroid
        # --------------------------------------------------------

        if centroid_mode == "site_mean":
            global_mean = np.mean(
                np.vstack([batch_means[b] for b in unique_batches]),
                axis=0,
            )

        elif centroid_mode == "global":
            global_mean = X.mean(axis=0)

        else:
            raise ValueError("Unknown centroid_mode.")

        centroid = {}

        for b in unique_batches:
            diff = batch_means[b] - global_mean

            delta = float(diff @ covariance_inv @ diff)

            centroid[
                (
                    b,
                    "global",
                )
            ] = float(
                np.sqrt(
                    max(
                        delta,
                        0.0,
                    )
                )
            )

        diagnostics = {
            "cond_number": cond_number,
            "used_pseudoinverse": bool(use_pinv),
            "num_retained_svals": int(retained_svals),
            "covariance_rank": int(np.linalg.matrix_rank(overall_cov)),
            "overallCov": overall_cov,
            "n_observations": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }

        return (
            pairwise,
            centroid,
            diagnostics,
        )

    # ============================================================
    # 13. REGULAR DATA
    # ============================================================

    if analysis_mode == "regular":
        complete_rows = np.isfinite(idp_matrix).all(axis=1)

        if not complete_rows.any():
            raise ValueError("No complete IDP observations remain.")

        X_raw = idp_matrix[complete_rows]

        B_raw = batch_ser.to_numpy()[complete_rows]

        (
            pairwise_raw,
            centroid_raw,
            raw_diag,
        ) = _calculate_distances(
            X_raw,
            B_raw,
            covariance_mode="site_average",
            centroid_mode="site_mean",
        )

        if have_covariates:
            C_valid = covariates[complete_rows]

            X_resid = _residualize(
                X_raw,
                C_valid,
            )

            (
                pairwise_resid,
                centroid_resid,
                resid_diag,
            ) = _calculate_distances(
                X_resid,
                B_raw,
                covariance_mode="site_average",
                centroid_mode="site_mean",
            )

        else:
            pairwise_resid = None

            centroid_resid = None

            resid_diag = None

        results = {
            "pairwise_raw": pairwise_raw,
            "pairwise_resid": pairwise_resid,
            "centroid_raw": centroid_raw,
            "centroid_resid": centroid_resid,
            "batches": list(dict.fromkeys(B_raw.tolist())),
        }

        info = {
            "mode": "regular",
            "longitudinal_case": None,
            "analysis_representation": "original observations",
            "n_timepoints": int(n_timepoints),
            "timepoints": (
                None if ordered_timepoints is None else list(ordered_timepoints)
            ),
            "has_repeated_subjects": bool(has_repeated_subjects),
            "batch_varies_within_subject": bool(batch_varies_within_subject),
            "n_subjects_multiple_batches": int(n_subjects_multiple_batches),
            "n_subjects_single_batch": int(n_subjects_single_batch),
            "n_samples_input": int(n_samples),
            "n_samples_retained": int(X_raw.shape[0]),
            "n_samples_excluded": int(n_samples - X_raw.shape[0]),
            "raw_diagnostics": raw_diag,
            "residual_diagnostics": resid_diag,
            "idp_names": list(idp_names),
            "covariates_used": bool(have_covariates),
            "n_covariates_after_encoding": (
                int(covariates.shape[1]) if have_covariates else 0
            ),
            "covariate_encoding": covariate_encoding,
        }

        if return_info:
            return (
                results,
                info,
            )

        return results

    # ============================================================
    # 14. LONGITUDINAL / REPEATED DATAFRAME
    # ============================================================

    data = pd.DataFrame(
        idp_matrix,
        columns=idp_names,
    )

    data["_subject"] = subject_ser.to_numpy()

    if timepoint_ser is None:
        data["_timepoint"] = np.repeat(
            "__single_timepoint__",
            n_samples,
        )

    else:
        data["_timepoint"] = timepoint_ser.to_numpy()

    data["_batch"] = batch_ser.to_numpy()

    # ============================================================
    # 15. OBSERVATION-LEVEL RESIDUALIZATION
    # ============================================================

    if have_covariates:
        complete_for_resid = np.isfinite(idp_matrix).all(axis=1)

        if complete_for_resid.sum() <= covariates.shape[1] + 1:
            raise ValueError(
                "Insufficient complete observations for covariate residualization."
            )

        design_rank = _design_rank(covariates[complete_for_resid])

        if design_rank < covariates.shape[1] + 1:
            warnings.warn(
                "The covariate design matrix is rank deficient. "
                "Residualization will use the least-squares "
                "solution, but some covariates may be linearly "
                "dependent."
            )

        resid_matrix = np.full_like(
            idp_matrix,
            np.nan,
            dtype=float,
        )

        resid_matrix[complete_for_resid] = _residualize(
            idp_matrix[complete_for_resid],
            covariates[complete_for_resid],
        )

    else:
        resid_matrix = None

        design_rank = None

    # ============================================================
    # 16. BATCH CONSTANT WITHIN SUBJECT
    #
    # Consecutive longitudinal changes:
    #
    #     T2 - T1
    #     T3 - T2
    #     ...
    # ============================================================

    if longitudinal_case == ("batch_constant_within_subject"):
        raw_vectors = []

        raw_batches = []

        resid_vectors = []

        resid_batches = []

        retained_subjects = []

        for (
            subject_id,
            subject_df,
        ) in data.groupby(
            "_subject",
            sort=False,
        ):
            rows_by_timepoint = []

            complete_subject = True

            for tp in ordered_timepoints:
                rows = subject_df[subject_df["_timepoint"] == tp]

                if len(rows) != 1:
                    complete_subject = False

                    break

                rows_by_timepoint.append(rows.iloc[0])

            if not complete_subject:
                continue

            subject_batches = subject_df["_batch"].unique()

            if len(subject_batches) != 1:
                continue

            subject_batch = subject_batches[0]

            # ----------------------------------------------------
            # Raw longitudinal changes
            # ----------------------------------------------------

            values_by_timepoint = []

            for row in rows_by_timepoint:
                values = row[idp_names].to_numpy(dtype=float)

                if not np.isfinite(values).all():
                    complete_subject = False

                    break

                values_by_timepoint.append(values)

            if not complete_subject:
                continue

            change_blocks = []

            for j in range(
                1,
                len(values_by_timepoint),
            ):
                change_blocks.append(
                    values_by_timepoint[j] - values_by_timepoint[j - 1]
                )

            subject_vector = np.concatenate(change_blocks)

            raw_vectors.append(subject_vector)

            raw_batches.append(subject_batch)

            # ----------------------------------------------------
            # Residualized longitudinal changes
            # ----------------------------------------------------

            if have_covariates:
                resid_values_by_timepoint = []

                for row in rows_by_timepoint:
                    original_index = row.name

                    values = resid_matrix[original_index]

                    if not np.isfinite(values).all():
                        complete_subject = False

                        break

                    resid_values_by_timepoint.append(values)

                if complete_subject:
                    resid_change_blocks = []

                    for j in range(
                        1,
                        len(resid_values_by_timepoint),
                    ):
                        resid_change_blocks.append(
                            resid_values_by_timepoint[j]
                            - resid_values_by_timepoint[j - 1]
                        )

                    resid_vector = np.concatenate(resid_change_blocks)

                    resid_vectors.append(resid_vector)

                    resid_batches.append(subject_batch)

            retained_subjects.append(subject_id)

        if len(raw_vectors) < 2:
            raise ValueError(
                "Fewer than two subjects have complete data "
                "across all required timepoints. "
                "Longitudinal Mahalanobis analysis cannot "
                "be performed."
            )

        # --------------------------------------------------------
        # Raw distances
        # --------------------------------------------------------

        X_raw = np.vstack(raw_vectors)

        B_raw = np.asarray(raw_batches)

        (
            pairwise_raw,
            centroid_raw,
            raw_diag,
        ) = _calculate_distances(
            X_raw,
            B_raw,
            covariance_mode="pooled",
            centroid_mode="global",
        )

        # --------------------------------------------------------
        # Residual distances
        # --------------------------------------------------------

        if have_covariates:
            if len(resid_vectors) >= 2:
                X_resid = np.vstack(resid_vectors)

                B_resid = np.asarray(resid_batches)

                (
                    pairwise_resid,
                    centroid_resid,
                    resid_diag,
                ) = _calculate_distances(
                    X_resid,
                    B_resid,
                    covariance_mode="pooled",
                    centroid_mode="global",
                )

            else:
                pairwise_resid = None

                centroid_resid = None

                resid_diag = None

                warnings.warn(
                    "Insufficient complete subjects for the "
                    "covariate-adjusted longitudinal analysis."
                )

        else:
            pairwise_resid = None

            centroid_resid = None

            resid_diag = None

        results = {
            "pairwise_raw": pairwise_raw,
            "pairwise_resid": pairwise_resid,
            "centroid_raw": centroid_raw,
            "centroid_resid": centroid_resid,
            "batches": list(dict.fromkeys(B_raw.tolist())),
        }

        info = {
            "mode": ("two_timepoints" if n_timepoints == 2 else "multiple_timepoints"),
            "longitudinal_case": "batch_constant_within_subject",
            "analysis_representation": (
                "within-subject changes between consecutive timepoints"
            ),
            "has_repeated_subjects": bool(has_repeated_subjects),
            "batch_varies_within_subject": False,
            "n_subjects_multiple_batches": int(n_subjects_multiple_batches),
            "n_subjects_single_batch": int(n_subjects_single_batch),
            "n_timepoints": int(n_timepoints),
            "timepoints": list(ordered_timepoints),
            "n_subjects_input": int(subject_ser.nunique()),
            "n_subjects_retained": int(len(retained_subjects)),
            "n_subjects_excluded": int(subject_ser.nunique() - len(retained_subjects)),
            "retained_subjects": retained_subjects,
            "representation": ("within-subject changes between consecutive timepoints"),
            "change_definition": "Y(i,t) - Y(i,t-1)",
            "n_analysis_features": int(X_raw.shape[1]),
            "raw_diagnostics": raw_diag,
            "residual_diagnostics": resid_diag,
            "idp_names": list(idp_names),
            "covariates_used": bool(have_covariates),
            "n_covariates_after_encoding": (
                int(covariates.shape[1]) if have_covariates else 0
            ),
            "covariate_design_rank": (
                int(design_rank) if design_rank is not None else None
            ),
            "covariate_encoding": covariate_encoding,
        }

        if return_info:
            return (
                results,
                info,
            )

        return results

    # ============================================================
    # 17. BATCH VARIES WITHIN SUBJECT
    #
    # Subject-centred observations are used.
    # ============================================================

    raw_rows = []
    raw_batches = []
    resid_rows = []
    resid_batches = []
    retained_subjects = []

    for (
        subject_id,
        subject_df,
    ) in data.groupby(
        "_subject",
        sort=False,
    ):
        subject_indices = subject_df.index.to_numpy(dtype=int)
        complete_subject_rows = []
        for idx in subject_indices:
            values = idp_matrix[idx]

            if np.isfinite(values).all():
                complete_subject_rows.append(idx)

        if len(complete_subject_rows) < 2:
            continue

        # --------------------------------------------------------
        # Raw subject-centering
        # --------------------------------------------------------
        X_subject = idp_matrix[complete_subject_rows]
        subject_mean = X_subject.mean(axis=0)
        X_subject_centered = X_subject - subject_mean
        B_subject = batch_ser.iloc[complete_subject_rows].to_numpy()
        raw_rows.append(X_subject_centered)
        raw_batches.append(B_subject)

        # --------------------------------------------------------
        # Residualized subject-centering
        # --------------------------------------------------------

        if have_covariates:
            X_subject_resid = resid_matrix[complete_subject_rows]
            if not np.isfinite(X_subject_resid).all():
                continue
            resid_subject_mean = X_subject_resid.mean(axis=0)
            X_subject_resid_centered = X_subject_resid - resid_subject_mean
            resid_rows.append(X_subject_resid_centered)
            resid_batches.append(B_subject)
        retained_subjects.append(subject_id)

    if len(raw_rows) == 0:
        raise ValueError(
            "No subjects have at least two complete observations "
            "for the batch-varying repeated-measures analysis."
        )

    X_raw = np.vstack(raw_rows)
    B_raw = np.concatenate(raw_batches)

    # ------------------------------------------------------------
    # Raw distances
    # ------------------------------------------------------------
    (
        pairwise_raw,
        centroid_raw,
        raw_diag,
    ) = _calculate_distances(
        X_raw,
        B_raw,
        covariance_mode="pooled",
        centroid_mode="global",
    )

    # ------------------------------------------------------------
    # Residual distances
    # ------------------------------------------------------------

    if have_covariates:
        if len(resid_rows) > 0:
            X_resid = np.vstack(resid_rows)
            B_resid = np.concatenate(resid_batches)
            if X_resid.shape[0] >= 2:
                (
                    pairwise_resid,
                    centroid_resid,
                    resid_diag,
                ) = _calculate_distances(
                    X_resid,
                    B_resid,
                    covariance_mode="pooled",
                    centroid_mode="global",
                )

            else:
                pairwise_resid = None
                centroid_resid = None
                resid_diag = None

        else:
            pairwise_resid = None
            centroid_resid = None
            resid_diag = None
            warnings.warn(
                "No complete observations remain for the "
                "covariate-adjusted batch-varying analysis."
            )

    else:
        pairwise_resid = None
        centroid_resid = None
        resid_diag = None

    # ============================================================
    # RESULTS
    # ============================================================

    results = {
        "pairwise_raw": pairwise_raw,
        "pairwise_resid": pairwise_resid,
        "centroid_raw": centroid_raw,
        "centroid_resid": centroid_resid,
        "batches": list(dict.fromkeys(B_raw.tolist())),
    }

    # ============================================================
    # INFO
    # ============================================================

    if n_timepoints <= 1:
        mode_label = "single_timepoint_repeated"

    elif n_timepoints == 2:
        mode_label = "two_timepoints"

    else:
        mode_label = "multiple_timepoints"

    info = {
        "mode": mode_label,
        "longitudinal_case": "batch_varies_within_subject",
        "analysis_representation": "subject-centred observations",
        "has_repeated_subjects": bool(has_repeated_subjects),
        "batch_varies_within_subject": True,
        "n_subjects_multiple_batches": int(n_subjects_multiple_batches),
        "n_subjects_single_batch": int(n_subjects_single_batch),
        "n_timepoints": int(n_timepoints),
        "timepoints": (
            None if ordered_timepoints is None else list(ordered_timepoints)
        ),
        "n_subjects_input": int(subject_ser.nunique()),
        "n_subjects_retained": len(retained_subjects),
        "n_subjects_excluded": int(subject_ser.nunique() - len(retained_subjects)),
        "retained_subjects": retained_subjects,
        "representation": "subject-centred observations",
        "change_definition": None,
        "n_analysis_features": int(X_raw.shape[1]),
        "raw_diagnostics": raw_diag,
        "residual_diagnostics": resid_diag,
        "idp_names": list(idp_names),
        "covariates_used": bool(have_covariates),
        "n_covariates_after_encoding": (
            int(covariates.shape[1]) if have_covariates else 0
        ),
        "covariate_design_rank": (
            int(design_rank) if design_rank is not None else None
        ),
        "covariate_encoding": covariate_encoding,
    }

    if return_info:
        return (
            results,
            info,
        )

    return results

###############################################################################################################################################
# FOR LONGITUDINAL DATA: 1) Various mixed effects models -
# mean comparison for -
# 4) overall batch
# 5) pairwise batches
# 6) cross subject variability (ICC)
# 7) biological variability for fixed effects e.g., age, timepoint
#
# Statistical intent:
#   - Full model:      Y ~ fixed effects + batch + random intercept(s)
#   - Subject-only:    Y ~ 1 + subject random intercept
#   - No-batch model:  Y ~ fixed effects + random intercept(s)
#
# Notes:
#   - The overall/additive batch-effect test is handled separately by
#     AdditiveEffect_long. The legacy `anova_batches` field below is therefore
#     retained only for backward compatibility and is NOT an omnibus batch test.
#   - Timepoint is NOT automatically added as a fixed effect. The user must
#     specify it (e.g. in `covariates` / `fix_eff`) because timepoint can be
#     structurally confounded with site/batch in some longitudinal designs.
#   - Biological fixed-effect estimates are intentionally obtained from the
#     no-batch model so that users can compare biological effects before and
#     after harmonisation.
#   - ICC is intentionally based on a subject-only random-intercept model so
#     that it represents the proportion of total model variance attributable
#     to between-subject variability.
################################################################################################################################################
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pandas.api.types import CategoricalDtype
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

def _force_categorical(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Convert requested columns to pandas categorical dtype when present."""
    for c in cols:
        if c in df.columns and not isinstance(df[c].dtype, CategoricalDtype):
            df[c] = df[c].astype("category")


def _force_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """
    Convert requested columns to numeric values when they are not already numeric.

    For string labels ending in digits (e.g. 'timepoint1', 'timepoint2'), the
    trailing numeric component is extracted and shifted to start at zero.
    Otherwise categorical codes are used, preserving the original helper's
    intended fallback behavior.
    """
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype(str)
            extracted = s.str.extract(r'(\d+)$', expand=False)
            if extracted.notna().all():
                vals = extracted.astype(float)
                vals = vals - vals.min()
                df[c] = vals
            else:
                df[c] = pd.Categorical(s).codes.astype(float)


def _zscore_columns(df: pd.DataFrame, vars_to_zscore: Iterable[str]) -> None:
    """Create per-column z-scored versions in-place for numeric variables."""
    for v in vars_to_zscore:
        if v not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[v]):
            mu = df[v].mean(skipna=True)
            sigma = df[v].std(skipna=True)
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
    zscore_covariates: Iterable[str] = (),
    zscore_response: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the longitudinal mixed-model formulas used by the pipeline.

    Returns formulas ordered as:
      1. full model:        fixed effects + batch + random effects
      2. subject-only model: intercept + first random grouping term
      3. no-batch model:    fixed effects without batch + random effects

    Z-scoring behavior:
      - If ``zscore_response=True``, the response is z-scored.
      - Numeric fixed-effect covariates are standardized automatically by
        ``zscore_covariates`` is honored for explicit standardization.
      - Categorical variables are not z-scored.
    """
    df = tbl_in.copy()
    fix_eff = list(fix_eff)
    ran_eff = list(ran_eff)
    batch_vars = list(batch_vars)
    force_categorical = list(force_categorical)
    force_numeric = list(force_numeric)
    zscore_covariates = list(zscore_covariates)

    # Coerce requested types BEFORE z-scoring so that a user-requested numeric
    # variable can actually be standardized after conversion.
    _force_categorical(df, batch_vars)
    _force_categorical(df, force_categorical)
    _force_numeric(df, force_numeric)

    # Z-score explicitly requested variables.
    _zscore_columns(df, zscore_covariates)

    def present(name: str) -> bool:
        return name in df.columns

    def _maybe_use_zscore(v: str) -> str:
        zname = f"zscore_{v}"
        # Predictor selection depends on whether v was requested/selected for
        # z-scoring, not on whether the response itself is z-scored.
        return zname if (v in zscore_covariates and zname in df.columns) else v

    if zscore_response:
        zresp_name = f"zscore_{response_var}"
        if zresp_name not in df.columns and response_var in df.columns:
            _zscore_columns(df, [response_var])
        lhs = zresp_name if zresp_name in df.columns else response_var
    else:
        lhs = response_var

    # Build fixed terms; batch is included in the full model and removed from
    # the no-batch model below.
    fixed_terms: List[str] = []
    for v in fix_eff:
        use_name = _maybe_use_zscore(v)
        if present(use_name):
            fixed_terms.append(use_name)

    for v in batch_vars:
        use_name = _maybe_use_zscore(v)
        if present(use_name) and use_name not in fixed_terms:
            fixed_terms.append(use_name)

    # Dedupe while preserving order.
    seen = set()
    fixed_terms = [x for x in fixed_terms if not (x in seen or seen.add(x))]

    fixed_str_with_batch = "1" if len(fixed_terms) == 0 else " + ".join(fixed_terms)

    batch_like = set(batch_vars) | {f"zscore_{b}" for b in batch_vars}
    fixed_no_batch = [t for t in fixed_terms if t not in batch_like]
    fixed_str_no_batch = "1" if len(fixed_no_batch) == 0 else " + ".join(fixed_no_batch)

    # Random terms -> (1|<var>). Current MixedEffects_long fitting uses the
    # first valid grouping variable as the actual statsmodels group argument.
    rand_terms: List[str] = []
    for v in ran_eff:
        if present(v):
            if not isinstance(df[v].dtype, CategoricalDtype):
                df[v] = df[v].astype("category")
            rand_terms.append(f"(1|{v})")

    if len(rand_terms) == 0:
        formulas = [
            f"{lhs} ~ {fixed_str_with_batch}",
            f"{lhs} ~ 1",
            f"{lhs} ~ {fixed_str_no_batch}",
        ]
    else:
        rand_str = " + ".join(rand_terms)
        subj_rand = rand_terms[0]
        formulas = [
            f"{lhs} ~ {fixed_str_with_batch} + {rand_str}",
            f"{lhs} ~ 1 + {subj_rand}",
            f"{lhs} ~ {fixed_str_no_batch} + {rand_str}",
        ]

    return df, formulas


def pairwise_site_tests(
    fit_result,
    group_var: str,
    data_frame: pd.DataFrame,
    alpha: float = 0.05,
    debug: bool = False,
) -> Tuple[int, pd.DataFrame]:
    """
    Pairwise Wald tests for batch/site fixed-effect differences in a fitted model.

    Returns the number of raw-p significant pairwise comparisons and a table
    containing raw p-values. Multiple-comparison correction is applied by the
    caller (Bonferroni by default in MixedEffects_long).
    """
    if group_var not in data_frame.columns:
        raise KeyError(f"group var '{group_var}' not in data")

    local_group = data_frame[group_var]
    if not isinstance(local_group.dtype, CategoricalDtype):
        local_group = local_group.astype("category")

    cats = list(local_group.cat.categories)
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
    level_to_exog_idx: Dict[Any, Optional[int]] = {}

    for lvl in cats:
        patt = f"[T.{lvl}]"
        found_exog = None
        for en in exog_names:
            if patt in en:
                found_exog = en
                break
        if found_exog is None:
            for en in exog_names:
                if (
                    f"{group_var}_{lvl}" in en
                    or en.endswith(f"_{lvl}")
                    or re.search(rf"\b{re.escape(str(lvl))}\b", en)
                ):
                    found_exog = en
                    break
        level_to_exog_idx[lvl] = exog_to_idx[found_exog] if found_exog is not None else None

    if debug:
        print("PAIRWISE (WALD) DEBUG: level -> exog_idx mapping:", level_to_exog_idx)

    beta = fit_result.params.to_numpy(dtype=float)
    try:
        cov = fit_result.cov_params()
        cov_mat = cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
    except Exception as exc:
        raise RuntimeError("Could not obtain covariance matrix from fit_result.cov_params()") from exc

    rows = []
    sig_flags = []

    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a = cats[i]
            b = cats[j]
            ex_idx_a = level_to_exog_idx.get(a)
            ex_idx_b = level_to_exog_idx.get(b)

            contrast_exog = np.zeros(len(exog_names), dtype=float)
            if ex_idx_a is not None:
                contrast_exog[ex_idx_a] = 1.0
            if ex_idx_b is not None:
                contrast_exog[ex_idx_b] = -1.0

            contrast_full = np.zeros(len(full_param_names), dtype=float)
            for k, exog_name in enumerate(exog_names):
                if exog_name in full_param_names:
                    pidx = full_param_names.index(exog_name)
                else:
                    pidx = None
                    for t_i, pname in enumerate(full_param_names):
                        if (
                            re.search(rf"\b{re.escape(exog_name)}\b", str(pname))
                            or re.search(rf"\b{re.escape(exog_name.split('[')[0])}\b", str(pname))
                        ):
                            pidx = t_i
                            break
                if pidx is not None:
                    contrast_full[pidx] = contrast_exog[k]

            if np.allclose(contrast_full, 0):
                pval = float("nan")
            else:
                est = float(np.dot(contrast_full, beta))
                var = float(contrast_full @ cov_mat @ contrast_full.T)
                if var <= 0 or np.isnan(var):
                    pval = float("nan")
                else:
                    z = est / np.sqrt(var)
                    pval = 2.0 * (1.0 - norm.cdf(abs(z)))

            sig = int(pval < alpha) if np.isfinite(pval) else 0
            rows.append({"siteA": a, "siteB": b, "p": pval, "sig": sig})
            sig_flags.append(sig)

            if debug:
                print(
                    f"PAIRWISE (WALD) DEBUG: {a} vs {b} -> "
                    f"ex_idx_a={ex_idx_a}, ex_idx_b={ex_idx_b}, p={pval}"
                )

    full_tab = pd.DataFrame(rows, columns=["siteA", "siteB", "p", "sig"])
    return int(np.nansum(sig_flags)), full_tab


def _extract_numeric_coeff_scalar(res, varname: str) -> Dict[str, float]:
    """Extract one fixed-effect estimate, p-value and confidence interval."""
    out: Dict[str, float] = {}
    params = res.params
    pvals = res.pvalues
    conf = res.conf_int()

    candidates = [varname, f"zscore_{varname}"]
    found_name = None

    for cand in candidates:
        if cand in params.index:
            found_name = cand
            break

    if found_name is None:
        for pname in params.index:
            if re.search(rf"\b{re.escape(varname)}\b", str(pname)):
                found_name = pname
                break

    if found_name is None:
        exog_names = getattr(res.model, "exog_names", None)
        if exog_names:
            for en in exog_names:
                if re.search(rf"\b{re.escape(varname)}\b", str(en)) and en in params.index:
                    found_name = en
                    break

    if found_name is None:
        out[f"{varname}_est"] = np.nan
        out[f"{varname}_pval"] = np.nan
        out[f"{varname}_ciL"] = np.nan
        out[f"{varname}_ciU"] = np.nan
        return out

    est = float(params.get(found_name, np.nan))
    pval = float(pvals.get(found_name, np.nan)) if found_name in pvals.index else np.nan

    if found_name in conf.index:
        ciL, ciU = conf.loc[found_name].values
    else:
        ciL = ciU = np.nan

    out[f"{varname}_est"] = est
    out[f"{varname}_pval"] = pval
    out[f"{varname}_ciL"] = ciL
    out[f"{varname}_ciU"] = ciU
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
    zscore_covariates: Sequence = (),
    zscore_response: bool = True,
    # Backward-compatible aliases for the previous parameter names.
    do_zscore: Optional[bool] = None,
    zscore_var: Optional[Sequence] = None,
    p_thr: float = 0.05,
    p_corr: int = 1,
    reml: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Fit several longitudinal mixed-effects models per IDP.

    The main outputs are intended for harmonisation/biological comparison:
      * pairwise site/batch differences from the full model;
      * subject-level variance, residual variance, ICC and within/between
        variance ratio from a subject-only model;
      * biological fixed-effect estimates from the no-batch model.

    Z-scoring defaults:
      * ``zscore_response=True`` standardizes each response/IDP by default.
      * Numeric fixed-effect covariates remain in their original units by default.
      * Covariates are standardized only when explicitly listed in
        ``zscore_covariates``.
      * Categorical fixed effects are left categorical.
      * ``do_zscore`` and ``zscore_var`` are accepted as backward-compatible aliases
        for the previous API.

    Important design choices:
      * Batch/site is included in the full model but not the biological-effect
        model, so users can compare biological estimates before vs after
        harmonisation without forcing batch adjustment in this analysis.
      * Timepoint is NOT automatically added. Users must specify it among the
        fixed effects/covariates when appropriate.
      * The ICC is deliberately based on ``Y ~ 1 + (1|subject)`` so it measures
        between-subject variance as a fraction of total subject + residual
        variance.
      * ``p_corr=1`` (default) uses Bonferroni-adjusted pairwise p-values for
        ``n_is_batchSig``; ``p_corr=0`` uses raw pairwise p-values.

    ``ran_eff`` currently supports one actual grouping variable in statsmodels:
    the first valid element is supplied as ``groups=...``. Additional grouping
    terms are retained in the constructed formula for compatibility, but are
    not implemented as separate statsmodels grouping factors.
    """
    global model_defs
    model_defs = []

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

    subjects_col = "subjects"
    timepoints_col = "timepoints"
    batches_col = "batches"

    covariates = dict(covariates or {})
    for k, seq in covariates.items():
        if len(seq) != n_samples:
            raise ValueError(
                f"Covariate '{k}' length ({len(seq)}) does not match n_samples ({n_samples})."
            )

    df = pd.DataFrame(idp_matrix, columns=idp_names)
    df[subjects_col] = pd.Series(subjects).astype(str)
    df[timepoints_col] = pd.Series(timepoints).astype(str)
    df[batches_col] = pd.Series(batches).astype(str).astype("category")

    for k, seq in covariates.items():
        df[k] = pd.Series(seq)

    fix_eff = list(fix_eff or [])
    ran_eff = list(ran_eff or [])
    force_categorical = list(force_categorical or [])
    force_numeric = list(force_numeric or [])
    # Backward compatibility with the previous API names.
    if do_zscore is not None:
        zscore_response = bool(do_zscore)
    if zscore_var is not None:
        if len(zscore_covariates) > 0:
            raise ValueError("Use either `zscore_covariates` or legacy `zscore_var`, not both.")
        zscore_covariates = zscore_var
    zscore_covariates = list(zscore_covariates or [])

    if len(ran_eff) == 0:
        ran_eff = [subjects_col]

    if len(fix_eff) == 0:
        # Deliberately do NOT auto-add batch or timepoint. The user controls
        # biological covariates, and additive batch testing is handled separately.
        fix_eff = list(covariates.keys())

    # Infer types for covariates only when the user has not supplied explicit
    # type overrides. Timepoint and batch are categorical by default.
    if len(force_numeric) == 0 and len(force_categorical) == 0:
        for k in covariates.keys():
            ser = pd.Series(covariates[k])
            if pd.api.types.is_numeric_dtype(ser):
                force_numeric.append(k)
            else:
                force_categorical.append(k)
        if timepoints_col not in force_categorical:
            force_categorical.append(timepoints_col)
        if batches_col not in force_categorical:
            force_categorical.append(batches_col)

    to_check = {
        "fix_eff": fix_eff,
        "ran_eff": ran_eff,
        "force_categorical": force_categorical,
        "force_numeric": force_numeric,
        "zscore_covariates": zscore_covariates,
    }
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(
            f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Apply requested type coercions once to the master data. build_mixed_formula
    # also applies them to each local frame, which keeps it safe as a standalone helper.
    _force_categorical(df, force_categorical)
    _force_numeric(df, force_numeric)
    _force_categorical(df, [batches_col])
    _force_categorical(df, [timepoints_col])

    outs: List[Dict[str, Any]] = []

    for tmpidp in idp_names:
        cols_needed: List[str] = []
        cols_needed += [c for c in ran_eff if c in df.columns]
        cols_needed += [batches_col]
        cols_needed += [c for c in fix_eff if c in df.columns]
        cols_needed += [tmpidp]

        seen = set()
        cols_needed = [x for x in cols_needed if not (x in seen or seen.add(x))]

        all_data = df[cols_needed].copy()

        # Remove rows with missing response before feature-specific standardisation/fitting.
        all_data = all_data.loc[all_data[tmpidp].notna()].copy()

        rowd: Dict[str, Any] = {}
        rowd["IDP"] = tmpidp.replace("_", "-")
        rowd["batch"] = batches_col

        if len(all_data) < 3:
            rowd.update({
                "n_is_batchSig": np.nan,
                "n_is_batchSig_raw": np.nan,
                "n_is_batchSig_bonf": np.nan,
                "anova_batches": np.nan,
                "Subj_Var": np.nan,
                "Resid_Var": np.nan,
                "ICC": np.nan,
                "WithinBetweenVarRatio": np.nan,
                "WCV": np.nan,
                "pairwise_site_tests": [],
            })
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan
                rowd[f"{v}_pval"] = np.nan
                rowd[f"{v}_ciL"] = np.nan
                rowd[f"{v}_ciU"] = np.nan
            outs.append(rowd)
            model_defs.append({"Feature": tmpidp, "models": {"full": None, "subject_only": None, "no_batch": None}})
            continue

        # Standardize the response by default. Covariates are standardized only
        # when explicitly requested through zscore_covariates.
        zscore_covariates_local = list(zscore_covariates)

        all_data, formulas = build_mixed_formula(
            all_data,
            response_var=tmpidp,
            fix_eff=fix_eff,
            ran_eff=ran_eff,
            batch_vars=[batches_col],
            force_categorical=force_categorical,
            force_numeric=force_numeric,
            zscore_covariates=zscore_covariates_local,
            zscore_response=zscore_response,
        )

        model_def = {
            "Feature": tmpidp,
            "models": {
                "full": formulas[0],
                "subject_only": formulas[1],
                "no_batch": formulas[2],
            },
        }
        model_defs.append(model_def)

        # Choose the most frequent observed batch as reference.
        all_data[batches_col] = all_data[batches_col].astype("category")
        observed_counts = all_data[batches_col].value_counts()
        if len(observed_counts) > 0:
            ref_site = observed_counts.idxmax()
            current_cats = list(all_data[batches_col].cat.categories)
            if ref_site not in current_cats:
                ref_site = current_cats[0]
            new_categories = [ref_site] + [c for c in current_cats if c != ref_site]
            try:
                all_data[batches_col] = all_data[batches_col].cat.reorder_categories(
                    new_categories, ordered=False
                )
            except Exception:
                all_data[batches_col] = all_data[batches_col].astype("category")

        # -----------------------------
        # Full model: pairwise site tests
        # -----------------------------
        fixed_formula_full = formulas[0]
        ml_formula_full = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", fixed_formula_full)

        try:
            if len(ran_eff) > 0 and ran_eff[0] in all_data.columns:
                groups_col = all_data[ran_eff[0]]
            else:
                groups_col = all_data[subjects_col]

            mdl1 = smf.mixedlm(ml_formula_full, all_data, groups=groups_col)
            res1 = mdl1.fit(reml=reml, method="lbfgs")
        except Exception:
            rowd.update({
                "n_is_batchSig": np.nan,
                "n_is_batchSig_raw": np.nan,
                "n_is_batchSig_bonf": np.nan,
                "anova_batches": np.nan,
                "Subj_Var": np.nan,
                "Resid_Var": np.nan,
                "ICC": np.nan,
                "WithinBetweenVarRatio": np.nan,
                "WCV": np.nan,
                "pairwise_site_tests": [],
            })
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan
                rowd[f"{v}_pval"] = np.nan
                rowd[f"{v}_ciL"] = np.nan
                rowd[f"{v}_ciU"] = np.nan
            outs.append(rowd)
            continue

        # Pairwise batch/site tests.
        try:
            n_sig_raw, full_tab = pairwise_site_tests(
                res1, batches_col, all_data, alpha=p_thr, debug=False
            )
        except Exception:
            n_sig_raw = 0
            full_tab = pd.DataFrame(columns=["siteA", "siteB", "p", "sig"])

        full_tab = full_tab.copy()
        if len(full_tab) > 0:
            full_tab["pair_label"] = (
                full_tab["siteA"].astype(str) + " vs " + full_tab["siteB"].astype(str)
            )
            pvals = pd.to_numeric(full_tab["p"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(pvals)

            full_tab["p_adj_bonf"] = np.nan
            full_tab["sig_bonf"] = False

            if valid.any():
                _, p_adj, _, _ = multipletests(
                    pvals[valid], alpha=p_thr, method="bonferroni"
                )
                full_tab.loc[valid, "p_adj_bonf"] = p_adj
                full_tab.loc[valid, "sig_bonf"] = p_adj < p_thr

        n_sig_bonf = (
            int(np.nansum(full_tab["sig_bonf"].astype(int))) if len(full_tab) else 0
        )

        # Keep backward compatibility: n_is_batchSig follows p_corr.
        if p_corr == 0:
            rowd["n_is_batchSig"] = int(n_sig_raw)
        else:
            rowd["n_is_batchSig"] = int(n_sig_bonf)

        rowd["n_is_batchSig_raw"] = int(n_sig_raw)
        rowd["n_is_batchSig_bonf"] = int(n_sig_bonf)

        # Legacy diagnostic retained for backward compatibility. This is the
        # number of individual batch coefficients with p < 0.05; it is NOT an
        # omnibus/overall batch-effect test.
        try:
            fe_pvals = res1.pvalues
            batch_mask = [
                bool(re.search(rf"{re.escape(str(batches_col))}", str(name)))
                for name in fe_pvals.index
            ]
            anova_batches = int(np.sum(fe_pvals[batch_mask] < 0.05)) if any(batch_mask) else 0
        except Exception:
            anova_batches = np.nan
        rowd["anova_batches"] = anova_batches

        # -----------------------------
        # Subject-only model for ICC / within-between variability
        # -----------------------------
        try:
            formula2_raw = formulas[1]
            formula2_fixed = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formula2_raw)
            groups_col2 = (
                all_data[ran_eff[0]]
                if (len(ran_eff) > 0 and ran_eff[0] in all_data.columns)
                else all_data[subjects_col]
            )
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
            rowd["Subj_Var"] = subj_var
            rowd["Resid_Var"] = resid_var

            if np.isfinite(subj_var) and np.isfinite(resid_var) and subj_var > 0:
                rowd["ICC"] = subj_var / (subj_var + resid_var)
                rowd["WithinBetweenVarRatio"] = resid_var / subj_var
            else:
                # With subject variance estimated at zero, the ICC/ratio are
                # boundary quantities and are not reported as finite values.
                rowd["ICC"] = np.nan
                rowd["WithinBetweenVarRatio"] = np.nan

            # Backward-compatible alias; prefer WithinBetweenVarRatio going forward.
            rowd["WCV"] = rowd["WithinBetweenVarRatio"]
        except Exception:
            rowd["Subj_Var"] = np.nan
            rowd["Resid_Var"] = np.nan
            rowd["ICC"] = np.nan
            rowd["WithinBetweenVarRatio"] = np.nan
            rowd["WCV"] = np.nan

        # -----------------------------
        # Biological fixed effects from the NO-BATCH model
        # -----------------------------
        try:
            formula3_raw = formulas[2]
            formula3_fixed = re.sub(r"\s*\+\s*\(1\|[^)]+\)", "", formula3_raw)
            groups_col3 = (
                all_data[ran_eff[0]]
                if (len(ran_eff) > 0 and ran_eff[0] in all_data.columns)
                else all_data[subjects_col]
            )
            mdl3 = smf.mixedlm(formula=formula3_fixed, data=all_data, groups=groups_col3)
            res3 = mdl3.fit(reml=reml, method="lbfgs")

            for v in fix_eff:
                pname = f"zscore_{v}" if f"zscore_{v}" in res3.params.index else v
                coeff_dict = _extract_numeric_coeff_scalar(res3, pname)
                cleaned = {k.replace(pname, v): val for k, val in coeff_dict.items()}
                rowd.update(cleaned)
        except Exception:
            for v in fix_eff:
                rowd[f"{v}_est"] = np.nan
                rowd[f"{v}_pval"] = np.nan
                rowd[f"{v}_ciL"] = np.nan
                rowd[f"{v}_ciU"] = np.nan

        rowd["pairwise_site_tests"] = full_tab.to_dict(orient="records")
        outs.append(rowd)

    if len(outs) == 0:
        return pd.DataFrame(), model_defs

    mdlnames = []
    seen_names = set()
    for out_row in outs:
        for k in out_row.keys():
            if (
                k.endswith("_est")
                or k.endswith("_pval")
                or k.endswith("_ciL")
                or k.endswith("_ciU")
            ) and k not in seen_names:
                seen_names.add(k)
                mdlnames.append(k)

    col_order = [
        "IDP",
        "batch",
        "n_is_batchSig",
        "n_is_batchSig_raw",
        "n_is_batchSig_bonf",
        "anova_batches",
        "Subj_Var",
        "Resid_Var",
        "ICC",
        "WithinBetweenVarRatio",
        "WCV",
        "pairwise_site_tests",
    ] + mdlnames

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
    alpha: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Test for additive (mean/location) batch effects per feature using mixed models.

    The user-facing result is intentionally simple: one row per feature with the
    batch-effect test, p-value, FDR-adjusted q-value, and a boolean BatchEffect flag.
    Model diagnostics (including singular random-effect covariance) are handled
    internally and do not automatically invalidate a fitted feature.

    For each feature (IDP) this routine:

    - Builds a per-feature local DataFrame including the response, specified fixed
      predictors and the batch column; numeric predictors are z-scored per-feature.
    - Optionally z-scores the response per-feature when ``do_zscore=True``
      (default).
    - Fits a full mixed model ``lhs ~ <fixed_terms> + C(batch)`` with the random
      grouping variable given by ``ran_eff`` (defaults to ``idvar``).
    - Fits a reduced mixed model ``lhs ~ <fixed_terms>`` with the same grouping.
    - Uses an ML likelihood-ratio test as the primary batch-effect test.
    - Falls back to a Wald test only when the LRT cannot be obtained.
      If the batch covariance is singular, a rank-aware pseudoinverse Wald test is
      used and labelled ``Wald_pinv_rank``.
    - Applies Benjamini-Hochberg FDR correction across finite feature-level p-values.

    Notes:
        ``reml=False`` is the default because the full and reduced models differ in
        fixed effects and are therefore compared using ML likelihoods.
        A singular random-effects covariance (for example, an estimated subject
        variance of approximately zero) is not treated as an automatic model
        failure. It is a valid boundary solution when the remaining fixed-effect
        model and likelihood are usable.
        ``ran_eff`` currently supports one grouping variable; additional entries
        are ignored for fitting and generate a warning for clarity.
    """
    if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

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

    if len(ran_eff) == 0:
        raise ValueError("ran_eff must contain at least one grouping variable.")
    if len(ran_eff) > 1:
        warnings.warn(
            "Multiple ran_eff variables were supplied, but MixedLM currently uses "
            "only the first grouping variable."
        )

    # Validate referenced names exist
    to_check = {"fix_eff": fix_eff, "ran_eff": ran_eff}
    missing = {k: [x for x in v if x not in df.columns] for k, v in to_check.items()}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise KeyError(
            f"Variables not found in data columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Determine feature columns (IDPs)
    exclude = {idcol, batchcol, tpcol}
    exclude |= set(covariates.keys())
    if idp_names is not None:
        feature_cols = list(idp_names)
    else:
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    V = len(feature_cols)
    if verbose:
        print(f"[AdditiveEffect_long] found {V} features")

    rows: List[Dict[str, Any]] = []

    def _fit_converged(result) -> bool:
        value = getattr(result, "converged", True)
        return bool(value) if value is not None else True

    def _random_effect_singular(result, tol: float = 1e-8) -> bool:
        """Detect a boundary/singular random-effect covariance without failing the fit."""
        try:
            cov_re = getattr(result, "cov_re", None)
            if cov_re is None:
                return False
            arr = cov_re.to_numpy(dtype=float) if hasattr(cov_re, "to_numpy") else np.asarray(cov_re, dtype=float)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return False
            scale = max(1.0, float(np.max(np.abs(arr))))
            rank = np.linalg.matrix_rank(arr, tol=tol * scale)
            return rank < min(arr.shape)
        except Exception:
            return False

    def _random_effect_variance(result):
        try:
            cov_re = getattr(result, "cov_re", None)
            if cov_re is None:
                return np.nan
            arr = cov_re.to_numpy(dtype=float) if hasattr(cov_re, "to_numpy") else np.asarray(cov_re, dtype=float)
            if arr.ndim == 2 and arr.shape == (1, 1):
                return float(arr[0, 0])
        except Exception:
            pass
        return np.nan

    for idx, feat in enumerate(feature_cols, 1):
        if verbose:
            print(
                f"[AdditiveEffect_long] ({idx}/{V}) testing additive batch effect for feature: {feat}"
            )

        # Build local df for this feature (so per-feature z-scoring is clean)
        local_cols = [feat] + list(fix_eff) + [batchcol] + ran_eff
        local_df = df.loc[:, [c for c in local_cols if c in df.columns]].copy()

        # Drop rows where response is NaN
        local_df = local_df[~local_df[feat].isna()].copy()
        if local_df.shape[0] < 3:
            if verbose:
                print(f"  skipping {feat}: too few non-NaN rows ({local_df.shape[0]})")
            rows.append({
                "Feature": feat,
                "TestStat": np.nan,
                "df": np.nan,
                "p-value": np.nan,
                "method": "Unavailable",
                "BatchEffect": False,
            })
            continue

        # Always z-score numeric predictors per-feature.
        fixed_terms = _build_fixed_formula_terms(
            list(fix_eff or []), local_df, do_zscore_predictors=True
        )
        missing_fix = [
            vv for vv in (fix_eff or [])
            if (f"zscore_{vv}" not in local_df.columns) and (vv not in local_df.columns)
        ]
        if missing_fix:
            warnings.warn(
                f"Fixed-effect columns not found in data (or zscore missing): {missing_fix}"
            )

        # Optional per-feature response z-score.
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
        reduced_fixed = f"{lhs} ~ {fixed_str}"

        if verbose:
            print("\n[MODEL]")
            print("Feature:", feat)
            print("Full model :", full_fixed)
            print("Reduced    :", reduced_fixed)
            print("Random eff :", ran_eff)

        model_defs_add.append({
            "Feature": feat,
            "full_formula": full_fixed,
            "reduced_formula": reduced_fixed,
            "fix_eff": list(fix_eff),
            "ran_eff": list(ran_eff),
        })

        res_full = res_red = None
        full_error = reduced_error = None
        group_name = ran_eff[0]

        try:
            res_full = _safe_fit_mixedlm(
                full_fixed, local_df, group=group_name, reml=reml
            )
        except Exception as e:
            full_error = str(e)
            if verbose:
                print(f"  full fit failed for {feat}: {e}")

        try:
            res_red = _safe_fit_mixedlm(
                reduced_fixed, local_df, group=group_name, reml=reml
            )
        except Exception as e:
            reduced_error = str(e)
            if verbose:
                print(f"  reduced fit failed for {feat}: {e}")

        LR = np.nan
        df_stat = np.nan
        pval = np.nan
        used = None
        full_converged = bool(res_full is not None and _fit_converged(res_full))
        reduced_converged = bool(res_red is not None and _fit_converged(res_red))
        full_singular = bool(res_full is not None and _random_effect_singular(res_full))
        reduced_singular = bool(res_red is not None and _random_effect_singular(res_red))

        # Primary test: likelihood-ratio test.
        if (res_full is not None) and (res_red is not None):
            try:
                llf_full = float(getattr(res_full, "llf", np.nan))
                llf_red = float(getattr(res_red, "llf", np.nan))
                if np.isfinite(llf_full) and np.isfinite(llf_red):
                    LR = max(0.0, 2.0 * (llf_full - llf_red))
                    n_levels = int(local_df[batchcol].nunique(dropna=True))
                    df_stat = max(n_levels - 1, 1)
                    pval = float(chi2.sf(LR, df_stat))
                    used = "LRT"
            except Exception as e:
                if verbose:
                    print(f"  LRT computation failed for {feat}: {e}")

        # Fallback: Wald test on batch coefficients.
        if not np.isfinite(pval) and (res_full is not None):
            try:
                pnames = list(res_full.params.index)
                batch_param_indices = []
                for i, pn in enumerate(pnames):
                    if (
                        f"C({batchcol})" in pn
                        or f"{batchcol}[T." in pn
                        or pn.startswith(f"{batchcol}_")
                    ):
                        batch_param_indices.append(i)

                if len(batch_param_indices) == 0:
                    cats = pd.Categorical(local_df[batchcol]).categories
                    for i, pn in enumerate(pnames):
                        for lvl in cats:
                            if f"{lvl}" in str(pn) and (
                                batchcol in pn or f"C({batchcol})" in pn
                            ):
                                if i not in batch_param_indices:
                                    batch_param_indices.append(i)
                                break

                if len(batch_param_indices) > 0:
                    beta = res_full.params.to_numpy(dtype=float)
                    cov = res_full.cov_params()
                    cov_mat = (
                        cov.to_numpy() if hasattr(cov, "to_numpy") else np.asarray(cov)
                    )
                    beta_b = beta[batch_param_indices]
                    Sigma_bb = cov_mat[np.ix_(batch_param_indices, batch_param_indices)]
                    Sigma_bb = 0.5 * (Sigma_bb + Sigma_bb.T)

                    scale = max(1.0, float(np.max(np.abs(Sigma_bb))))
                    rank = int(np.linalg.matrix_rank(Sigma_bb, tol=1e-8 * scale))

                    if rank > 0:
                        if rank < len(beta_b):
                            inv_Sigma_bb = np.linalg.pinv(Sigma_bb, rcond=1e-8)
                            used = "Wald_pinv_rank"
                        else:
                            inv_Sigma_bb = np.linalg.inv(Sigma_bb)
                            used = "Wald"

                        W = float(beta_b.T @ inv_Sigma_bb @ beta_b)
                        df_w = rank
                        p_w = float(chi2.sf(W, df_w))
                        LR, df_stat, pval = W, df_w, p_w
            except Exception as e:
                if verbose:
                    print(f"  Wald fallback failed for {feat}: {e}")

        rows.append({
            "Feature": feat,
            "TestStat": LR,
            "df": df_stat,
            "p-value": pval,
            "method": used if used is not None else "Unavailable",
            # User-facing summary. Missing/unavailable tests are never called
            # "No batch effect"; they are simply marked False here.
            "BatchEffect": bool(np.isfinite(pval) and pval < alpha),
            # Internal diagnostics are retained in the dataframe for programmatic
            # access, but do not need to be shown in a simple UI.
            "_full_converged": full_converged,
            "_reduced_converged": reduced_converged,
            "_full_singular": full_singular,
            "_reduced_singular": reduced_singular,
            "_full_random_variance": _random_effect_variance(res_full) if res_full is not None else np.nan,
            "_reduced_random_variance": _random_effect_variance(res_red) if res_red is not None else np.nan,
            "_fit_error_full": full_error,
            "_fit_error_reduced": reduced_error,
        })

    out = pd.DataFrame(rows)

    # Multiple-testing correction across features. This remains internal-friendly:
    # q-values are returned, while BatchEffect is still a simple boolean result.
    out["q-value"] = np.nan
    valid = out["p-value"].notna() & np.isfinite(out["p-value"].to_numpy(dtype=float))
    if valid.any():
        pvals = out.loc[valid, "p-value"].to_numpy(dtype=float)
        m = len(pvals)
        order = np.argsort(pvals)
        ranked = pvals[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        ranked = np.clip(ranked, 0.0, 1.0)
        qvals = np.empty(m, dtype=float)
        qvals[order] = ranked
        out.loc[valid, "q-value"] = qvals

    # Use FDR-adjusted significance as the main batch-effect call when possible.
    out["BatchEffect"] = out["q-value"].lt(alpha) & out["q-value"].notna()

    # Keep the familiar sort order from the original implementation.
    out = out.sort_values(by="TestStat", ascending=False, na_position="last").reset_index(drop=True)
    return out, model_defs_add


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
    alpha: float = 0.05,
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
      ``"Fligner"``. Benjamini-Hochberg FDR correction is applied across
      features, and the user-facing ``BatchEffect`` flag is based on the
      FDR-adjusted q-value.

    Args:
        data: Optional DataFrame used directly when provided.
        idp_matrix: Optional feature matrix used when `data` is not supplied.
        subjects: Optional subject IDs used when building a DataFrame from
            arrays.
        timepoints: Optional timepoint labels used when building a DataFrame
            from arrays.
        batch_name: Optional batch labels used when building a DataFrame from
            arrays.
        idp_names: Optional feature names for `idp_matrix`.
        covariates: Optional mapping of `name -> sequence` for additional
            covariates.
        idvar: Column name for subject IDs.
        batchvar: Column name for batch labels.
        timevar: Column name for timepoints.
        fix_eff: Fixed-effect predictors.
        ran_eff: Random-effect grouping variables.
        do_zscore: Whether to z-score the response per feature.
        reml: Whether to fit `MixedLM` using REML.
        verbose: Whether to print progress and model formulas.
        alpha: FDR significance threshold used for the user-facing ``BatchEffect`` flag.

    Returns:
        pd.DataFrame: One row per feature with multiplicative batch-effect test
        statistics, degrees of freedom, raw p-values, FDR-adjusted q-values,
        and a user-facing ``BatchEffect`` flag.

    Raises:
        KeyError: If `fix_eff` or `ran_eff` variables are not found in the assembled DataFrame.
        ValueError: If `idp_matrix` is not 2-D or if input sequence lengths do not match the number of rows.

    Notes:
        Per-feature predictor z-scoring is always applied to numeric `fix_eff`
      (local to each feature) via ``_build_fixed_formula_terms``.
        ``do_zscore=True`` (default): z-scores the response per feature and uses
      the z-scored response (``z_<feature>``) as LHS. Set ``do_zscore=False``
      to keep original units.
        ``reml=False`` (default): MixedLM fits are run with REML disabled.
      Pass ``reml=True`` to change that.
        ``ran_eff`` defaults to ``[idvar]`` (where ``idvar`` defaults to
      ``'subject_ids'``).
        Residuals used for the variance test are extracted from the full
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

    # Defaults: ran_eff -> [idcol], fix_eff -> covariate keys + tpcol. Batch is
    # added separately below, so it must not also be placed in fix_eff.
    if len(ran_eff) == 0:
        if idcol in df.columns:
            ran_eff = [idcol]
        else:
            raise KeyError("ran_eff not provided and idvar column not found in data.")
    if len(fix_eff) == 0:
        inferred_fix = list(covariates.keys())
        if tpcol in df.columns and tpcol not in inferred_fix:
            inferred_fix.append(tpcol)
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
        

        # Group residuals by observed batch level and run Fligner test.
        # Restrict to levels actually observed for this feature; categorical columns
        # can otherwise retain unused levels after per-feature row filtering.
        try:
            observed_batch = local_df[batchcol].dropna()
            cats = pd.unique(observed_batch)
            groups = [resid[local_df[batchcol] == lvl].dropna().values for lvl in cats]
            if len(groups) < 2 or any(len(g) == 0 for g in groups):
                raise ValueError("Not enough observations per batch group for Fligner test.")
            stat, pval = fligner(*groups, center="median")
            df_stat = len(groups) - 1
            method = "Fligner"

        except Exception as e:
            if verbose:
                print(f"  fligner test failed for {feat}: {e}")
            stat = np.nan
            pval = np.nan
            df_stat = np.nan
            method = None

        rows.append({
            "Feature": feat,
            "ChiSq": float(stat) if not np.isnan(stat) else np.nan,
            "DF": df_stat,
            "p-value": float(pval) if not np.isnan(pval) else np.nan,
            "method": method
        })

    out = pd.DataFrame(rows)

    # Multiple-testing correction across features. Keep the raw p-values and
    # expose the Benjamini-Hochberg FDR-adjusted value as the main decision metric.
    out["q-value"] = np.nan
    valid = out["p-value"].notna() & np.isfinite(out["p-value"].astype(float))
    if valid.any():
        from statsmodels.stats.multitest import multipletests
        out.loc[valid, "q-value"] = multipletests(
            out.loc[valid, "p-value"].astype(float).to_numpy(),
            alpha=alpha,
            method="fdr_bh",
        )[1]

    # User-facing binary result. A failed/unavailable test remains False rather
    # than being treated as evidence that no batch effect exists.
    out["BatchEffect"] = False
    out.loc[valid, "BatchEffect"] = out.loc[valid, "q-value"] < alpha

    out = out.sort_values(by="ChiSq", ascending=False, na_position="last").reset_index(drop=True)
    return out, model_defs_mul
