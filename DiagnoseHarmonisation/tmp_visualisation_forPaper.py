import re
import numpy as np
import pandas as pd
from DiagnoseHarmonisation import PlotDiagnosticResults
from DiagnoseHarmonisation import DiagnosticReport
from DiagnoseHarmonisation import DiagnosticFunctionsLong

import inspect
from DiagnoseHarmonisation import HarmonisationFunctions as HF
import matplotlib.pyplot as plt


# ============================================================
# LOAD MAIN IMAGING DATA
# ============================================================
data_path = "/Users/psyc1586_admin/12Sep_harmTool_forPaper/HarmonisationDiagnostics/DiagnoseHarmonisation/paper_tests/for_paper/00_only_between_harmonisation/onharmony_onlyBetweenScanner.csv"
covariate_path = "/Users/psyc1586_admin/12Sep_harmTool_forPaper/HarmonisationDiagnostics/DiagnoseHarmonisation/paper_tests/for_paper/00_only_between_harmonisation/onharmony_onlyBetweenScanner_demogs.csv"
feature_list_path = "/Users/psyc1586_admin/12Sep_harmTool_forPaper/HarmonisationDiagnostics/DiagnoseHarmonisation/paper_tests/for_paper/00_only_between_harmonisation/GM_volume_list.csv"

sheet_name = 0   # ignored for CSV

print("=" * 70)
print("LOADING IMAGING DATA")
print("=" * 70)

if data_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(
        data_path,
        sheet_name=sheet_name
    )
else:
    df = pd.read_csv(data_path)

print("Original imaging dataframe shape:", df.shape)


# ============================================================
# LOAD FEATURE LIST
# ============================================================

feature_df = pd.read_csv(feature_list_path)

features = (
    feature_df["featurelist"]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

# Remove duplicate feature names while preserving order
features = list(dict.fromkeys(features))

print("Number of requested features:", len(features))
print("First 10 requested features:")
print(features[:10])


# ============================================================
# CHECK THAT REQUIRED COLUMNS EXIST
# ============================================================

if "subject" not in df.columns:
    raise KeyError(
        "The imaging dataframe does not contain a 'subject' column."
    )

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:
    print("\nMissing requested FreeSurfer features:")
    for feature in missing_features:
        print("  ", feature)

    raise KeyError(
        f"{len(missing_features)} requested FreeSurfer features "
        "are not present in the imaging dataframe."
    )

# ============================================================
# KEEP SUBJECT + REQUESTED FEATURES
# ============================================================

df = df[
    ["subject"] + features
].copy()

# Make subject IDs consistent
df["subject"] = df["subject"].astype(str).str.strip()

# Convert imaging features to numeric
df[features] = df[features].apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nImaging dataframe after selecting features:", df.shape)


# ============================================================
# LOAD COVARIATE / BATCH DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING COVARIATE / BATCH DATA")
print("=" * 70)

df2 = pd.read_csv(covariate_path)

print("Original covariate dataframe shape:", df2.shape)


# ============================================================
# CHECK REQUIRED METADATA COLUMNS
# ============================================================

required_metadata = [
    "subject",
    "age",
    "sex",
    "timepoint",
    "scan_session"
]

missing_metadata = [
    col
    for col in required_metadata
    if col not in df2.columns
]

if missing_metadata:
    raise KeyError(
        "The covariate dataframe is missing the following columns: "
        + ", ".join(missing_metadata)
    )

# ============================================================
# PREPARE METADATA
# ============================================================

metadata = df2[
    [
        "subject",
        "age",
        "sex",
        "timepoint",
        "scan_session"
    ]
].copy()

# Consistent subject ID
metadata["subject"] = (
    metadata["subject"]
    .astype(str)
    .str.strip()
)

# Numeric variables
metadata["age"] = pd.to_numeric(
    metadata["age"],
    errors="coerce"
)

metadata["timepoint"] = pd.to_numeric(
    metadata["timepoint"]
    .astype(str)
    .str.strip()
    .str.replace("TP_", "", regex=False),
    errors="coerce",
)

# Batch as string
metadata["scan_session"] = (
    metadata["scan_session"]
    .astype(str)
    .str.strip()
)

# Sex:
# Keep categorical for now
metadata["sex"] = metadata["sex"].astype("category")


# ============================================================
# ALIGN MULTIPLE VISITS / SCANS
# ============================================================
#
# Important:
# The imaging dataframe does not currently contain scan_session
# or timepoint. Therefore, if a subject appears more than once,
# subject alone is not enough to uniquely identify the row.
#
# We create a within-subject occurrence number in both datasets:
#
# subject A, first row  -> occurrence 0
# subject A, second row -> occurrence 1
# subject B, first row  -> occurrence 0
#
# This assumes the repeated rows are in the same visit order
# in both CSV files.
# ============================================================

df["_visit_index"] = (
    df.groupby("subject", sort=False)
      .cumcount()
)

metadata["_visit_index"] = (
    metadata.groupby("subject", sort=False)
            .cumcount()
)


# ============================================================
# CHECK SUBJECT / VISIT COUNTS
# ============================================================

print("\nChecking subject/visit alignment...")

imaging_counts = (
    df.groupby("subject", sort=False)
      .size()
      .rename("imaging_n")
)

metadata_counts = (
    metadata.groupby("subject", sort=False)
            .size()
            .rename("metadata_n")
)

alignment_counts = pd.concat(
    [
        imaging_counts,
        metadata_counts
    ],
    axis=1
)

alignment_counts["imaging_n"] = (
    alignment_counts["imaging_n"]
    .fillna(0)
    .astype(int)
)

alignment_counts["metadata_n"] = (
    alignment_counts["metadata_n"]
    .fillna(0)
    .astype(int)
)

subjects_count_mismatch = alignment_counts[
    alignment_counts["imaging_n"] != alignment_counts["metadata_n"]
]

print(
    "Number of subjects with matching visit counts:",
    len(alignment_counts) - len(subjects_count_mismatch)
)

print(
    "Number of subjects with mismatched visit counts:",
    len(subjects_count_mismatch)
)

if len(subjects_count_mismatch) > 0:

    print("\nSubjects with mismatched numbers of rows:")

    print(
        subjects_count_mismatch.head(20)
    )

    raise ValueError(
        "\nThe imaging and metadata files contain different numbers "
        "of rows for one or more subjects. Because the imaging file "
        "does not contain scan_session/timepoint, the visits cannot "
        "be safely aligned automatically."
    )


# ============================================================
# MERGE METADATA ONTO IMAGING DATA
# ============================================================

df = df.merge(
    metadata,
    on=["subject", "_visit_index"],
    how="inner",
    validate="one_to_one",
    sort=False
)


# ============================================================
# CHECK MERGE RESULT
# ============================================================

print("\nAfter merging imaging + metadata:")
print("Shape:", df.shape)

if len(df) == 0:
    raise ValueError(
        "The imaging and metadata dataframes have no matching subjects."
    )

print(
    "\nFirst 20 aligned rows:"
)

print(
    df[
        [
            "subject",
            "_visit_index",
            "age",
            "sex",
            "timepoint",
            "scan_session"
        ]
    ].head(20)
)


# ============================================================
# CHECK DUPLICATE SUBJECT / VISIT PAIRS
# ============================================================

duplicate_rows = df.duplicated(
    subset=["subject", "_visit_index"],
    keep=False
)

if duplicate_rows.any():
    print("\nERROR: Duplicate subject/visit rows detected:")
    print(
        df.loc[
            duplicate_rows,
            [
                "subject",
                "_visit_index",
                "timepoint",
                "scan_session"
            ]
        ]
    )

    raise ValueError(
        "Duplicate subject/visit combinations found."
    )


# ============================================================
# CHECK MISSING METADATA
# ============================================================

print("\n" + "=" * 70)
print("MISSING METADATA")
print("=" * 70)

metadata_columns = [
    "age",
    "sex",
    "timepoint",
    "scan_session"
]

print(
    df[metadata_columns].isna().sum()
)


# ============================================================
# REMOVE ROWS WITH MISSING REQUIRED METADATA
# ============================================================
#
# Adjust this if your diagnostic package supports missing
# covariates, but generally harmonisation should not receive
# missing covariate/batch values.
# ============================================================

n_before = len(df)

df = df.dropna(
    subset=[
        "age",
        "sex",
        "timepoint",
        "scan_session"
    ]
).copy()

n_after = len(df)

print(
    f"\nRemoved {n_before - n_after} rows "
    "because of missing metadata."
)

print(
    "Remaining rows:",
    len(df)
)


# ============================================================
# REMOVE FEATURES CONTAINING ANY NaN
# ============================================================

print("\n" + "=" * 70)
print("CHECKING IMAGING FEATURES FOR NaNs")
print("=" * 70)

cols_with_nan = [
    col
    for col in features
    if df[col].isna().any()
]

print(
    f"Features containing NaN: {len(cols_with_nan)}"
)

if cols_with_nan:

    for col in cols_with_nan:
        n_missing = df[col].isna().sum()

        print(
            f"{col} -> {n_missing} NaN values"
        )

    # Remove features containing any NaN
    df = df.drop(
        columns=cols_with_nan
    )

    # Update feature list
    features = [
        f
        for f in features
        if f not in cols_with_nan
    ]


print(
    f"\nRemoved features: {len(cols_with_nan)}"
)

print(
    f"Remaining features: {len(features)}"
)

print(
    f"Final dataframe shape before renaming: {df.shape}"
)


# ============================================================
# CHECK THAT FEATURES REMAIN
# ============================================================

if len(features) == 0:
    raise ValueError(
        "No features remain after removing NaN features."
    )


# ============================================================
# CLEAN FEATURE NAMES: if you are using freesurfer features
# ============================================================

print("\n" + "=" * 70)
print("RENAMING FEATURES")
print("=" * 70)


def clean_feature_name(name):
    """
    Convert feature names into simpler Python-friendly
    column names.
    """

    name = str(name)

    # Remove redundant terms
    name = name.replace(
        "freesurfer_",
        ""
    )

    name = name.replace(
        "aparc.",
        ""
    )

    name = name.replace(
        "ThickAvg_",
        ""
    )

    # Replace special characters with underscore
    name = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        name
    )

    # Collapse repeated underscores
    name = re.sub(
        r"_+",
        "_",
        name
    )

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


# ============================================================
# CREATE RENAMING MAP
# ============================================================

rename_map = {
    feature: clean_feature_name(feature)
    for feature in features
}


# ============================================================
# CHECK FOR DUPLICATE CLEANED NAMES
# ============================================================

cleaned_names = list(
    rename_map.values()
)

if len(cleaned_names) != len(
    set(cleaned_names)
):

    duplicates = pd.Series(
        cleaned_names
    )

    duplicates = duplicates[
        duplicates.duplicated(
            keep=False
        )
    ]

    print(
        "\nERROR: Duplicate feature names after cleaning:"
    )

    for name in duplicates.unique():
        print(
            f"  {name}"
        )

    raise ValueError(
        "Renaming created duplicate feature names."
    )


# ============================================================
# PRINT NAME MAPPING
# ============================================================

for old_name, new_name in rename_map.items():

    print(
        f"{old_name} -> {new_name}"
    )


# ============================================================
# RENAME COLUMNS
# ============================================================

df = df.rename(
    columns=rename_map
)

# Update feature list
features = [
    rename_map[f]
    for f in features
]

print(
    "\nRenaming complete."
)

print(
    f"Number of retained features: {len(features)}"
)


# ============================================================
# CHECK FINAL DATAFRAME
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATAFRAME")
print("=" * 70)

print(
    "Shape:",
    df.shape
)

print(
    "Number of imaging features:",
    len(features)
)

print(
    "Number of subjects:",
    df["subject"].nunique()
)

print(
    "Number of rows/scans:",
    len(df)
)

print(
    "\nFirst 10 dataframe columns:"
)

print(
    df.columns[:10].tolist()
)


# ============================================================
# CREATE ARRAYS FOR HARMONISATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING HARMONISATION ARRAYS")
print("=" * 70)


# Imaging data
data_array = (
    df[features]
    .to_numpy()
)


# Batch / scanner / session
batch_array = (
    df["scan_session"]
    .astype(str)
    .tolist()
)


# Subject IDs
subject_array = (
    df["subject"]
    .astype(str)
    .tolist()
)


# ============================================================
# COVARIATES
# ============================================================

covariates = [
    "age",
    "sex"
]


# Keep covariates in a separate dataframe if your downstream
# code expects this object
df_covariates = df[
    [
        "subject",
        "age",
        "sex"
    ]
].copy()


# Numeric age
df_covariates["age"] = pd.to_numeric(
    df_covariates["age"],
    errors="coerce"
)


# Keep sex categorical
df_covariates["sex"] = (
    df_covariates["sex"]
    .astype("category")
)


# Covariate array
covariate_array = (
    df_covariates[covariates]
    .to_numpy()
)


# Covariate names
covariate_names = covariates.copy()


# Dictionary version
covariate_dict = {
    name: df_covariates[name].to_numpy()
    for name in covariates
}


# ============================================================
# TIMEPOINT
# ============================================================

timepoint = pd.to_numeric(
    df["timepoint"],
    errors="coerce"
)


# Dataframe containing covariates + timepoint
df_covariates_with_timepoint = df[
    [
        "subject",
        "age",
        "sex",
        "timepoint"
    ]
].copy()


df_covariates_with_timepoint["age"] = (
    pd.to_numeric(
        df_covariates_with_timepoint["age"],
        errors="coerce"
    )
)


df_covariates_with_timepoint["timepoint"] = (
    pd.to_numeric(
        df_covariates_with_timepoint["timepoint"],
        errors="coerce"
    )
)


df_covariates_with_timepoint["sex"] = (
    df_covariates_with_timepoint["sex"]
    .astype("category")
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


print(
    "data_array shape:",
    data_array.shape
)

print(
    "Number of batches:",
    len(batch_array)
)

print(
    "Number of subjects:",
    len(subject_array)
)

print(
    "covariate_array shape:",
    covariate_array.shape
)

print(
    "Number of timepoints:",
    len(timepoint)
)


# ------------------------------------------------------------
# All arrays must have the same number of rows
# ------------------------------------------------------------

n = len(df)

assert data_array.shape[0] == n, (
    "data_array has incorrect number of rows."
)

assert len(batch_array) == n, (
    "batch_array has incorrect number of rows."
)

assert len(subject_array) == n, (
    "subject_array has incorrect number of rows."
)

assert covariate_array.shape[0] == n, (
    "covariate_array has incorrect number of rows."
)

assert len(timepoint) == n, (
    "timepoint has incorrect number of rows."
)


# ------------------------------------------------------------
# Check no missing values in final imaging matrix
# ------------------------------------------------------------

if np.isnan(data_array).any():

    nan_count = np.isnan(
        data_array
    ).sum()

    raise ValueError(
        f"Final data_array contains {nan_count} NaN values."
    )


# ------------------------------------------------------------
# Check metadata missingness
# ------------------------------------------------------------

if df[
    [
        "age",
        "sex",
        "timepoint",
        "scan_session"
    ]
].isna().any().any():

    print(
        "\nWARNING: Missing values remain in metadata:"
    )

    print(
        df[
            [
                "age",
                "sex",
                "timepoint",
                "scan_session"
            ]
        ].isna().sum()
    )

    raise ValueError(
        "Missing metadata remain after filtering."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HARMONISATION INPUT SUMMARY")
print("=" * 70)

print(
    f"Subjects:              {df['subject'].nunique()}"
)

print(
    f"Rows / scans:          {len(df)}"
)

print(
    f"Features:              {len(features)}"
)

print(
    f"Batch values:          {df['scan_session'].nunique()}"
)

print(
    f"Covariates:            {covariate_names}"
)

print(
    f"Timepoint values:      {df['timepoint'].nunique()}"
)

print(
    "\nBatch distribution:"
)

print(
    df["scan_session"]
    .value_counts()
)


print(
    "\nSubject visit distribution:"
)

print(
    df.groupby("subject")
      .size()
      .value_counts()
      .sort_index()
)


# ============================================================
# OPTIONAL: SHOW THE FIRST 20 ALIGNED ROWS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 20 ALIGNED OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "subject",
            "_visit_index",
            "age",
            "sex",
            "timepoint",
            "scan_session"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# OPTIONAL: REMOVE INTERNAL ALIGNMENT COLUMN
# ============================================================

# It is no longer needed once all arrays have been created.
df = df.drop(
    columns=["_visit_index"]
)


# ============================================================
# AT THIS POINT YOUR VARIABLES ARE READY FOR THE DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("READY FOR DIAGNOSTIC FUNCTIONS")
print("=" * 70)

print("data_array:", data_array.shape)
print("batch_array:", len(batch_array))
print("subject_array:", len(subject_array))
print("covariate_array:", covariate_array.shape)
print("covariate_names:", covariate_names)
print("timepoint:", timepoint.shape)
print(
    "df_covariates_with_timepoint:",
    df_covariates_with_timepoint.shape
)
print("Number of features:", len(features))
print("=" * 70)

# ============================================================
# YOUR DIAGNOSTIC CODE GOES BELOW THIS LINE
# ============================================================
covariate_dict = {
    name: covariate_array[:, i]
    for i, name in enumerate(covariate_names)
}


df_covariates_with_timepoint = df_covariates.copy()
df_covariates_with_timepoint["timepoint"] = timepoint

# Ensure all covariate columns are numeric where applicable
df_covariates = df2[["subject", "age", "sex"]].copy()
df_covariates_no_subject = df_covariates.drop(columns=["subject"])


##################################
scanner_array = (
    df["scan_session"]
    .astype(str)
    .str.strip()
    .str.replace(r"\d+$", "", regex=True)
    .tolist()
)

df_covariates_with_timepoint = df_covariates.copy()
df_covariates_with_timepoint["timepoint"] = timepoint

# Ensure all covariate columns are numeric where applicable
df_covariates = df2[["subject", "age", "sex"]].copy()
df_covariates_no_subject = df_covariates.drop(columns=["subject"])

 #############################################################

######## RUN HARMONISATION METHODS ########

#############################################################

# Combat harmonisation
harmonised_data1 = HF.combat_modular(df[features],batch_array,mod=df_covariates_no_subject)
combat_IDP      = harmonised_data1["bayesdata"]

# CovBat harmonisation
harmonised_data2 = HF.combat_modular(df[features],batch_array,mod=df_covariates_no_subject,covbat_mode=True)
covbat_IDP      = harmonised_data2["bayesdata"]

# Longitudinal combat with no timepoint effect and only subject random effect: (1|subject)
harmonised_data3 = HF.long_combat(df[features], batch_array, model_inputs=df_covariates_with_timepoint, subject_col="subject", timepoint_as_fixed_effect=False, timepoint_as_random_slope=False)
Lcombat_noTP_IDP = harmonised_data3["Bayesdata"] 



""" #############################################################

######## RUN INDIVIDUAL REPORT GENERATION ########

#############################################################

####### You can either run this OR comparison report below ###########


DiagnosticReport.LongitudinalReport(data=data_array, 
                                    subject_ids=subject_array, 
                                    batch=batch_array, 
                                    timepoints=timepoint, 
                                    features=features, 
                                    covariates=covariate_dict,
                                    SaveArtifacts=True,
                                    save_dir="Raw_report") 

DiagnosticReport.LongitudinalReport(data=combat_IDP, 
                                    subject_ids=subject_array, 
                                    batch=batch_array, 
                                    timepoints=timepoint, 
                                    features=features, 
                                    covariates=covariate_dict,
                                    SaveArtifacts=True,
                                    save_dir="combat_report")

DiagnosticReport.LongitudinalReport(data=covbat_IDP, 
                                    subject_ids=subject_array, 
                                    batch=batch_array, 
                                    timepoints=timepoint, 
                                    features=features, 
                                    covariates=covariate_dict,
                                    SaveArtifacts=True,
                                    save_dir="covbat_report") 

DiagnosticReport.LongitudinalReport(data=Lcombat_noTP_IDP, 
                                    subject_ids=subject_array, 
                                    batch=batch_array, 
                                    timepoints=timepoint, 
                                    features=features, 
                                    covariates=covariate_dict,
                                    SaveArtifacts=True,
                                    save_dir="Lcombat_report")


#############################################################

######## RUN COMPARISON REPORT ########

#############################################################

datasets = {
    "Raw": data_array,
    "ComBat": combat_IDP,
    "CovBat": covbat_IDP,
    "LongComBat": Lcombat_noTP_IDP
}


comparison_report = (
    DiagnosticReport.LongitudinalComparisonReport(
        datasets=datasets,
        batch=batch_array,
        subject_ids=subject_array,
        timepoints=timepoint,
        covariates=covariate_dict,
        covariate_names=list(
            covariate_dict.keys()
        ),
        features=features,
        save_data=True,
        save_dir="ONHarmony_OnlyBetweenScanner_Comparison_report",
        report_name="ONHarmony_OnlyBetweenScanner_Comparison_report",
        include_raw=True,
        raw_name="Raw",
        SaveArtifacts=True,
        show=False,
    )
)  """

def plot_compare_longitudinal_zscore_histograms_forPaper_2rows(
    datasets: dict,
    batch,
    subjects,
    covariates=None,
    idp_names=None,
    probability_distribution: bool = True,
    draw_pdf: bool = True,
    bins: int = 80,
    xlim: tuple = (-8, 8),
    batch_order=None,
):
    """
    Create a paper-style 2 x 4 comparison of robust feature
    distributions across batches for four harmonisation methods.

    Layout
    ------
    Columns:
        One column per harmonisation method.

    Row 1:
        Unadjusted feature-wise robust z-scores.

    Row 2:
        Covariate- and subject-adjusted feature-wise robust z-scores.

    Adjustment model for each IDP:
        IDP ~ Batch + covariates + (1 | Subject)

    Adjustment removes:
        - covariate fixed-effect contributions
        - subject-specific random intercepts

    Batch effects are deliberately retained.

    Robust standardisation:
        z = (x - median) / (1.4826 * MAD)

    Notes
    -----
    - Exactly four datasets/methods are expected.
    - All datasets must have identical shape:
          (n_observations, n_IDPs)
    - Robust z-scoring is performed separately for each IDP.
    - Within each panel, z-scores are pooled across IDPs for
      each batch before plotting.
    """

    # ============================================================
    # Imports
    # ============================================================

    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    from scipy.stats import norm


    # ============================================================
    # Robust z-score
    # ============================================================

    def _robust_zscore(values):

        values = np.asarray(values, dtype=float)

        out = np.full(
            values.shape,
            np.nan,
            dtype=float,
        )

        valid = np.isfinite(values)

        if not valid.any():
            return out

        median = np.median(values[valid])

        mad = np.median(
            np.abs(values[valid] - median)
        )

        mad_scaled = 1.4826 * mad

        if (
            not np.isfinite(mad_scaled)
            or mad_scaled == 0
        ):
            out[valid] = 0.0
            return out

        out[valid] = (
            values[valid] - median
        ) / mad_scaled

        return out


    # ============================================================
    # Validate datasets
    # ============================================================

    if not isinstance(datasets, dict) or len(datasets) == 0:
        raise ValueError(
            "datasets must be a non-empty dictionary."
        )

    if len(datasets) != 4:
        raise ValueError(
            "This figure is designed for exactly four methods."
        )

    method_names = list(datasets.keys())

    first_data = np.asarray(
        datasets[method_names[0]],
        dtype=float,
    )

    if first_data.ndim != 2:
        raise ValueError(
            "Each dataset must be a 2D array."
        )

    n_samples, n_idps = first_data.shape

    expected_shape = (
        n_samples,
        n_idps,
    )

    normalized_datasets = {}

    for method, data in datasets.items():

        array = np.asarray(
            data,
            dtype=float,
        )

        if array.ndim != 2:
            raise ValueError(
                f"{method} dataset must be 2D."
            )

        if array.shape != expected_shape:
            raise ValueError(
                f"{method} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

        normalized_datasets[method] = array


    # ============================================================
    # Batch and subject information
    # ============================================================

    if len(batch) != n_samples:
        raise ValueError(
            "Length of batch must match number of observations."
        )

    if len(subjects) != n_samples:
        raise ValueError(
            "Length of subjects must match number of observations."
        )

    batch_arr = np.asarray(
        batch
    ).astype(str)

    subject_arr = np.asarray(
        subjects
    ).astype(str)


    # ============================================================
    # IDP names
    # ============================================================

    if idp_names is None:

        idp_names = [
            f"idp_{i + 1}"
            for i in range(n_idps)
        ]

    else:

        idp_names = list(
            idp_names
        )

        if len(idp_names) != n_idps:
            raise ValueError(
                "idp_names length must match "
                "the number of IDPs."
            )


    # ============================================================
    # Batch order
    # ============================================================

    if batch_order is None:

        batch_order = (
            pd.Series(batch_arr)
            .dropna()
            .unique()
            .tolist()
        )

    else:

        batch_order = [
            str(value)
            for value in batch_order
        ]


    # ============================================================
    # Prepare covariates
    # ============================================================

    covariate_df = pd.DataFrame(
        index=np.arange(n_samples)
    )

    covariate_terms = []

    if covariates:

        if not isinstance(
            covariates,
            dict,
        ):
            raise TypeError(
                "covariates must be a dictionary or None."
            )

        for name, values in covariates.items():

            values = np.asarray(
                values
            )

            if len(values) != n_samples:
                raise ValueError(
                    f"Covariate '{name}' has "
                    f"{len(values)} observations; "
                    f"expected {n_samples}."
                )

            safe_name = str(name)

            series = pd.Series(
                values,
                index=covariate_df.index,
            )

            if series.isna().any():
                raise ValueError(
                    f"Covariate '{name}' contains "
                    "missing values."
                )


            # ----------------------------------------------------
            # Numeric covariate
            # ----------------------------------------------------

            if pd.api.types.is_numeric_dtype(
                series
            ):

                covariate_df[safe_name] = (
                    pd.to_numeric(
                        series,
                        errors="raise",
                    ).astype(float)
                )

                covariate_terms.append(
                    safe_name
                )


            # ----------------------------------------------------
            # Categorical covariate
            # ----------------------------------------------------

            else:

                covariate_df[safe_name] = (
                    series.astype(str)
                )

                covariate_terms.append(
                    f"C(Q('{safe_name}'))"
                )


    # ============================================================
    # Fit and adjust one IDP
    # ============================================================

    def _fit_and_adjust_feature(
        df,
        feature,
    ):

        work = pd.DataFrame(
            index=df.index
        )

        work["_y"] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        work["_batch"] = (
            df["_batch"].astype(str)
        )

        work["_subject"] = (
            df["_subject"].astype(str)
        )


        # --------------------------------------------------------
        # Add covariates
        # --------------------------------------------------------

        for col in covariate_df.columns:

            work[col] = (
                covariate_df[col]
            )


        # --------------------------------------------------------
        # Complete cases
        # --------------------------------------------------------

        valid = np.isfinite(
            work["_y"].to_numpy(
                dtype=float
            )
        )

        for col in covariate_df.columns:

            if pd.api.types.is_numeric_dtype(
                work[col]
            ):

                valid &= np.isfinite(
                    pd.to_numeric(
                        work[col],
                        errors="coerce",
                    ).to_numpy(
                        dtype=float
                    )
                )

            else:

                valid &= (
                    work[col]
                    .notna()
                    .to_numpy()
                )


        work = (
            work.loc[valid]
            .copy()
        )

        if len(work) < 3:

            return np.full(
                len(df),
                np.nan,
                dtype=float,
            )


        # --------------------------------------------------------
        # Mixed-effects model
        #
        # IDP ~ Batch + covariates + (1 | Subject)
        # --------------------------------------------------------

        formula_terms = [
            "C(_batch)"
        ]

        formula_terms.extend(
            covariate_terms
        )

        formula = (
            "_y ~ "
            + " + ".join(
                formula_terms
            )
        )


        try:

            model = smf.mixedlm(
                formula=formula,
                data=work,
                groups=work["_subject"],
                re_formula="1",
            )

            fitted = model.fit(
                reml=True,
                method="lbfgs",
                disp=False,
            )

        except Exception as exc:

            warnings.warn(
                f"Mixed model failed for "
                f"{feature}: {exc}"
            )

            return np.full(
                len(df),
                np.nan,
                dtype=float,
            )


        # --------------------------------------------------------
        # Fixed effects
        # --------------------------------------------------------

        params = (
            fitted.fe_params
        )

        exog = (
            fitted.model.exog
        )

        exog_names = (
            fitted.model.exog_names
        )


        # --------------------------------------------------------
        # Calculate covariate contribution
        #
        # Intercept and batch contributions are NOT removed.
        # --------------------------------------------------------

        covariate_contribution = np.zeros(
            len(work),
            dtype=float,
        )

        for j, name in enumerate(
            exog_names
        ):

            if name == "Intercept":
                continue

            if "_batch" in name:
                continue

            covariate_contribution += (
                exog[:, j]
                * params[name]
            )


        # --------------------------------------------------------
        # Subject random intercept
        # --------------------------------------------------------

        random_effects = (
            fitted.random_effects
        )

        subject_random_intercept = np.zeros(
            len(work),
            dtype=float,
        )

        for pos, subject_id in enumerate(
            work["_subject"]
        ):

            effect = (
                random_effects.get(
                    subject_id
                )
            )

            if effect is None:
                continue

            try:

                subject_random_intercept[pos] = (
                    float(
                        np.asarray(
                            effect
                        ).reshape(-1)[0]
                    )
                )

            except Exception:

                subject_random_intercept[pos] = 0.0


        # --------------------------------------------------------
        # Adjusted values
        #
        # Remove:
        #   covariate contribution
        #   subject random intercept
        #
        # Retain:
        #   intercept
        #   batch effect
        # --------------------------------------------------------

        adjusted_values = (
            work["_y"].to_numpy(
                dtype=float
            )
            - covariate_contribution
            - subject_random_intercept
        )


        # --------------------------------------------------------
        # Restore original observation positions
        # --------------------------------------------------------

        output = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        output[
            work.index.to_numpy()
        ] = adjusted_values

        return output


    # ============================================================
    # Calculate raw + adjusted robust z-scores
    # ============================================================

    transformed_data = {}

    for method, data in normalized_datasets.items():

        # --------------------------------------------------------
        # RAW / UNADJUSTED robust z-scores
        #
        # Robust-standardise each IDP independently.
        # --------------------------------------------------------

        raw_z = np.full(
            expected_shape,
            np.nan,
            dtype=float,
        )

        for j in range(n_idps):

            raw_z[:, j] = (
                _robust_zscore(
                    data[:, j]
                )
            )


        # --------------------------------------------------------
        # Build working dataframe
        # --------------------------------------------------------

        work_df = pd.DataFrame(
            data,
            columns=idp_names,
        )

        work_df["_batch"] = (
            batch_arr
        )

        work_df["_subject"] = (
            subject_arr
        )


        # --------------------------------------------------------
        # Adjust every IDP
        # --------------------------------------------------------

        adjusted_values = np.full(
            expected_shape,
            np.nan,
            dtype=float,
        )

        for j, feature in enumerate(
            idp_names
        ):

            adjusted_values[:, j] = (
                _fit_and_adjust_feature(
                    work_df,
                    feature,
                )
            )


        # --------------------------------------------------------
        # Robust-standardise adjusted IDPs
        # --------------------------------------------------------

        adjusted_z = np.full(
            expected_shape,
            np.nan,
            dtype=float,
        )

        for j in range(n_idps):

            adjusted_z[:, j] = (
                _robust_zscore(
                    adjusted_values[:, j]
                )
            )


        transformed_data[method] = {
            "Unadjusted": pd.DataFrame(
                raw_z,
                columns=idp_names,
            ),
            "Adjusted": pd.DataFrame(
                adjusted_z,
                columns=idp_names,
            ),
        }


    # ============================================================
    # Figure
    #
    # 2 rows x 4 methods
    # ============================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )


    # ============================================================
    # Typography
    # ============================================================

    FONT_SIZE = 12
    MAIN_TITLE_SIZE = 14
    LEGEND_SIZE = 10


    # ============================================================
    # Plot one panel
    # ============================================================

    def _plot_panel(
        ax,
        z_df,
    ):

        for batch_name in batch_order:

            mask = (
                batch_arr == batch_name
            )


            # ----------------------------------------------------
            # Pool all feature-wise z-scores within batch
            # ----------------------------------------------------

            batch_data = (
                z_df.loc[mask]
                .to_numpy(
                    dtype=float
                )
                .ravel()
            )

            batch_data = (
                batch_data[
                    np.isfinite(
                        batch_data
                    )
                ]
            )

            if batch_data.size == 0:
                continue


            # ----------------------------------------------------
            # Histogram
            # ----------------------------------------------------

            hist_result = ax.hist(
                batch_data,
                bins=bins,
                density=probability_distribution,
                alpha=0.28,
                label=str(batch_name),
            )


            # ----------------------------------------------------
            # Fitted Gaussian distribution
            # ----------------------------------------------------

            if draw_pdf:

                mu = float(
                    np.mean(
                        batch_data
                    )
                )

                std = float(
                    np.std(
                        batch_data
                    )
                )

                if (
                    np.isfinite(std)
                    and std > 0
                ):

                    x = np.linspace(
                        xlim[0],
                        xlim[1],
                        300,
                    )


                    # --------------------------------------------
                    # Density
                    # --------------------------------------------

                    if probability_distribution:

                        y = norm.pdf(
                            x,
                            mu,
                            std,
                        )


                    # --------------------------------------------
                    # Frequency
                    # --------------------------------------------

                    else:

                        bin_edges = (
                            hist_result[1]
                        )

                        if len(
                            bin_edges
                        ) > 1:

                            bin_width = (
                                bin_edges[1]
                                - bin_edges[0]
                            )

                        else:

                            bin_width = 1.0

                        y = (
                            norm.pdf(
                                x,
                                mu,
                                std,
                            )
                            * len(
                                batch_data
                            )
                            * bin_width
                        )


                    # --------------------------------------------
                    # Match histogram colour but darken line
                    # --------------------------------------------

                    patches = (
                        hist_result[2]
                    )

                    if len(patches) > 0:

                        facecolor = (
                            patches[0]
                            .get_facecolor()
                        )

                        line_color = tuple(
                            max(
                                0.0,
                                channel * 0.65,
                            )
                            for channel
                            in facecolor[:3]
                        )

                        ax.plot(
                            x,
                            y,
                            color=line_color,
                            linewidth=2.2,
                            alpha=1.0,
                        )


        # --------------------------------------------------------
        # Panel formatting
        # --------------------------------------------------------

        ax.set_xlim(
            xlim
        )

        ax.tick_params(
            axis="both",
            labelsize=FONT_SIZE,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.18,
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )


        # --------------------------------------------------------
        # Legend inside every panel
        # --------------------------------------------------------

        ax.legend(
            title="Batch",
            loc="upper right",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_SIZE,
            frameon=False,
        )


    # ============================================================
    # Plot methods
    # ============================================================

    for col, method in enumerate(
        method_names
    ):

        # --------------------------------------------------------
        # TOP ROW: unadjusted
        # --------------------------------------------------------

        ax_top = axes[
            0,
            col,
        ]

        _plot_panel(
            ax_top,
            transformed_data[
                method
            ]["Unadjusted"],
        )

        ax_top.set_title(
            method,
            fontsize=FONT_SIZE,
            fontweight="bold",
            pad=8,
        )


        # --------------------------------------------------------
        # BOTTOM ROW: adjusted
        # --------------------------------------------------------

        ax_bottom = axes[
            1,
            col,
        ]

        _plot_panel(
            ax_bottom,
            transformed_data[
                method
            ]["Adjusted"],
        )

        ax_bottom.set_xlabel(
            "Robust z-score",
            fontsize=FONT_SIZE,
        )


    # ============================================================
    # Row labels / y-axis
    # ============================================================

    y_measure = (
        "Density"
        if probability_distribution
        else "Frequency"
    )


    # Top-left y label
    axes[
        0,
        0,
    ].set_ylabel(
        f"Unadjusted\n\n{y_measure}",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )


    # Bottom-left y label
    if covariates:

        adjusted_label = (
            "Covariate- and\n"
            "subject-adjusted"
        )

    else:

        adjusted_label = (
            "Subject-adjusted"
        )


    axes[
        1,
        0,
    ].set_ylabel(
        f"{adjusted_label}\n\n{y_measure}",
        fontsize=FONT_SIZE,
        fontweight="bold",
    )


    # ============================================================
    # Main title
    # ============================================================

    fig.suptitle(
        "Feature distributions across batches",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.985,
    )


    # ============================================================
    # Layout
    # ============================================================

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.92,
        bottom=0.09,
        wspace=0.16,
        hspace=0.22,
    )


    # ============================================================
    # Return
    # ============================================================

    figs = [
        (
            "Longitudinal comparison: "
            "unadjusted and adjusted robust "
            "feature distributions across batches",
            fig,
        )
    ]

    return figs

def plot_compare_longitudinal_zscore_histograms_forPaper(
    datasets: dict,
    batch,
    subjects,
    covariates=None,
    idp_names=None,
    probability_distribution: bool = True,
    draw_pdf: bool = True,
    bins: int = 80,
    xlim: tuple = (-8, 8),
    batch_order=None,
):
    """
    Independent paper-oriented comparison of adjusted robust IDP
    distributions across batches for exactly four methods.

    For each IDP and each method, fits:

        IDP ~ Batch + covariates + (1 | Subject)

    Adjustment removes:
        - covariate fixed-effect contributions
        - subject-specific random intercept

    Batch effects are deliberately retained.

    The adjusted values are then robust-standardised using:

        z = (x - median) / (1.4826 * MAD)

    Output
    ------
    A 2x2 figure, one panel per method, showing pooled adjusted
    robust z-score distributions for each batch.
    """

    # ============================================================
    # Imports
    # ============================================================

    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    from scipy.stats import norm


    # ============================================================
    # Internal helper: robust z-score
    # ============================================================

    def _robust_zscore(values):

        values = np.asarray(values, dtype=float)

        out = np.full(
            values.shape,
            np.nan,
            dtype=float,
        )

        valid = np.isfinite(values)

        if not valid.any():
            return out

        median = np.median(values[valid])

        mad = np.median(
            np.abs(values[valid] - median)
        )

        mad_scaled = 1.4826 * mad

        if (
            not np.isfinite(mad_scaled)
            or mad_scaled == 0
        ):
            out[valid] = 0.0
            return out

        out[valid] = (
            values[valid] - median
        ) / mad_scaled

        return out


    # ============================================================
    # Validate datasets
    # ============================================================

    if not isinstance(datasets, dict) or len(datasets) == 0:
        raise ValueError(
            "datasets must be a non-empty dictionary."
        )

    if len(datasets) != 4:
        raise ValueError(
            "This paper figure requires exactly four methods."
        )

    method_names = list(datasets.keys())

    first_data = np.asarray(
        datasets[method_names[0]],
        dtype=float,
    )

    if first_data.ndim != 2:
        raise ValueError(
            "Each dataset must be a 2D array."
        )

    n_samples, n_idps = first_data.shape

    expected_shape = (
        n_samples,
        n_idps,
    )

    normalized_datasets = {}

    for method, data in datasets.items():

        array = np.asarray(
            data,
            dtype=float,
        )

        if array.ndim != 2:
            raise ValueError(
                f"{method} dataset must be 2D."
            )

        if array.shape != expected_shape:
            raise ValueError(
                f"{method} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

        normalized_datasets[method] = array


    # ============================================================
    # Batch / subject information
    # ============================================================

    if len(batch) != n_samples:
        raise ValueError(
            "Length of batch must match number of observations."
        )

    if len(subjects) != n_samples:
        raise ValueError(
            "Length of subjects must match number of observations."
        )

    batch_arr = np.asarray(batch).astype(str)

    subject_arr = np.asarray(subjects).astype(str)


    # ============================================================
    # IDP names
    # ============================================================

    if idp_names is None:

        idp_names = [
            f"idp_{i + 1}"
            for i in range(n_idps)
        ]

    else:

        idp_names = list(idp_names)

        if len(idp_names) != n_idps:
            raise ValueError(
                "idp_names length must match "
                "the number of IDPs."
            )


    # ============================================================
    # Batch order
    # ============================================================

    if batch_order is None:

        batch_order = (
            pd.Series(batch_arr)
            .dropna()
            .unique()
            .tolist()
        )

    else:

        batch_order = [
            str(value)
            for value in batch_order
        ]


    # ============================================================
    # Prepare covariates
    # ============================================================

    covariate_df = pd.DataFrame(
        index=np.arange(n_samples)
    )

    covariate_terms = []

    if covariates:

        if not isinstance(covariates, dict):
            raise TypeError(
                "covariates must be a dictionary or None."
            )

        for name, values in covariates.items():

            values = np.asarray(values)

            if len(values) != n_samples:
                raise ValueError(
                    f"Covariate '{name}' has "
                    f"{len(values)} observations; "
                    f"expected {n_samples}."
                )

            safe_name = str(name)

            series = pd.Series(
                values,
                index=covariate_df.index,
            )

            if series.isna().any():
                raise ValueError(
                    f"Covariate '{name}' contains "
                    "missing values."
                )

            # Numeric covariate
            if pd.api.types.is_numeric_dtype(series):

                covariate_df[safe_name] = (
                    pd.to_numeric(
                        series,
                        errors="raise",
                    ).astype(float)
                )

                covariate_terms.append(
                    safe_name
                )

            # Categorical covariate
            else:

                covariate_df[safe_name] = (
                    series.astype(str)
                )

                covariate_terms.append(
                    f"C(Q('{safe_name}'))"
                )


    # ============================================================
    # Internal helper: fit one IDP
    # ============================================================

    def _fit_and_adjust_feature(
        df,
        feature,
    ):

        work = pd.DataFrame(
            index=df.index
        )

        work["_y"] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        work["_batch"] = (
            df["_batch"].astype(str)
        )

        work["_subject"] = (
            df["_subject"].astype(str)
        )

        # Add covariates
        for col in covariate_df.columns:

            work[col] = covariate_df[col]


        # --------------------------------------------------------
        # Complete cases
        # --------------------------------------------------------

        valid = np.isfinite(
            work["_y"].to_numpy(dtype=float)
        )

        for col in covariate_df.columns:

            if pd.api.types.is_numeric_dtype(
                work[col]
            ):

                valid &= np.isfinite(
                    pd.to_numeric(
                        work[col],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                )

            else:

                valid &= (
                    work[col]
                    .notna()
                    .to_numpy()
                )

        work = work.loc[valid].copy()

        if len(work) < 3:

            return np.full(
                len(df),
                np.nan,
                dtype=float,
            )


        # --------------------------------------------------------
        # Mixed model
        #
        # IDP ~ Batch + covariates + (1 | Subject)
        # --------------------------------------------------------

        formula_terms = [
            "C(_batch)"
        ]

        formula_terms.extend(
            covariate_terms
        )

        formula = (
            "_y ~ "
            + " + ".join(formula_terms)
        )


        try:

            model = smf.mixedlm(
                formula=formula,
                data=work,
                groups=work["_subject"],
                re_formula="1",
            )

            fitted = model.fit(
                reml=True,
                method="lbfgs",
                disp=False,
            )

        except Exception as exc:

            warnings.warn(
                f"Mixed model failed for "
                f"{feature}: {exc}"
            )

            return np.full(
                len(df),
                np.nan,
                dtype=float,
            )


        # --------------------------------------------------------
        # Fixed-effect design matrix
        # --------------------------------------------------------

        params = fitted.fe_params

        exog = fitted.model.exog

        exog_names = (
            fitted.model.exog_names
        )


        # --------------------------------------------------------
        # Covariate contribution
        #
        # IMPORTANT:
        # Do not remove:
        #     Intercept
        #     Batch
        # --------------------------------------------------------

        covariate_contribution = np.zeros(
            len(work),
            dtype=float,
        )

        for j, name in enumerate(
            exog_names
        ):

            if name == "Intercept":
                continue

            if "_batch" in name:
                continue

            covariate_contribution += (
                exog[:, j]
                * params[name]
            )


        # --------------------------------------------------------
        # Subject-specific random intercept
        # --------------------------------------------------------

        random_effects = (
            fitted.random_effects
        )

        subject_random_intercept = np.zeros(
            len(work),
            dtype=float,
        )

        for pos, subject_id in enumerate(
            work["_subject"]
        ):

            effect = random_effects.get(
                subject_id
            )

            if effect is None:
                continue

            try:

                subject_random_intercept[pos] = (
                    float(
                        np.asarray(effect)
                        .reshape(-1)[0]
                    )
                )

            except Exception:

                subject_random_intercept[pos] = 0.0


        # --------------------------------------------------------
        # Adjust values
        #
        # Remove:
        #     covariate effects
        #     subject random intercept
        #
        # Retain:
        #     intercept
        #     batch effects
        # --------------------------------------------------------

        adjusted_values = (
            work["_y"].to_numpy(dtype=float)
            - covariate_contribution
            - subject_random_intercept
        )


        # Return to original observation positions
        output = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        output[
            work.index.to_numpy()
        ] = adjusted_values

        return output


    # ============================================================
    # Calculate adjusted robust z-scores for each method
    # ============================================================

    adjusted_data = {}

    for method, data in normalized_datasets.items():

        work_df = pd.DataFrame(
            data,
            columns=idp_names,
        )

        work_df["_batch"] = batch_arr
        work_df["_subject"] = subject_arr


        # --------------------------------------------------------
        # Fit each IDP independently
        # --------------------------------------------------------

        adjusted_values = np.full(
            expected_shape,
            np.nan,
            dtype=float,
        )

        for j, feature in enumerate(
            idp_names
        ):

            adjusted_values[:, j] = (
                _fit_and_adjust_feature(
                    work_df,
                    feature,
                )
            )


        # --------------------------------------------------------
        # Robust z-score AFTER adjustment
        # --------------------------------------------------------

        adjusted_z = np.full(
            expected_shape,
            np.nan,
            dtype=float,
        )

        for j in range(n_idps):

            adjusted_z[:, j] = (
                _robust_zscore(
                    adjusted_values[:, j]
                )
            )


        adjusted_data[method] = (
            pd.DataFrame(
                adjusted_z,
                columns=idp_names,
            )
        )


    # ============================================================
    # Figure: 2 x 2
    # ============================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
        facecolor="white",
    )

    axes = axes.ravel()


    # ============================================================
    # Typography
    # ============================================================

    FONT_SIZE = 12
    MAIN_TITLE_SIZE = 14


    # ============================================================
    # Plot one method
    # ============================================================

    def _plot_panel(
        ax,
        z_df,
    ):

        for batch_name in batch_order:

            mask = (
                batch_arr == batch_name
            )

            # Pool all IDPs within this batch
            batch_data = (
                z_df.loc[mask]
                .to_numpy(dtype=float)
                .ravel()
            )

            batch_data = batch_data[
                np.isfinite(batch_data)
            ]

            if batch_data.size == 0:
                continue


            # ----------------------------------------------------
            # Histogram
            # ----------------------------------------------------

            hist_result = ax.hist(
                batch_data,
                bins=bins,
                density=probability_distribution,
                alpha=0.28,
                label=str(batch_name),
            )


            # ----------------------------------------------------
            # Gaussian distribution line
            # ----------------------------------------------------

            if draw_pdf:

                mu = float(
                    np.mean(batch_data)
                )

                std = float(
                    np.std(batch_data)
                )

                if (
                    np.isfinite(std)
                    and std > 0
                ):

                    x = np.linspace(
                        xlim[0],
                        xlim[1],
                        300,
                    )

                    if probability_distribution:

                        y = norm.pdf(
                            x,
                            mu,
                            std,
                        )

                    else:

                        bin_edges = (
                            hist_result[1]
                        )

                        if len(bin_edges) > 1:

                            bin_width = (
                                bin_edges[1]
                                - bin_edges[0]
                            )

                        else:

                            bin_width = 1.0

                        y = (
                            norm.pdf(
                                x,
                                mu,
                                std,
                            )
                            * len(batch_data)
                            * bin_width
                        )


                    # --------------------------------------------
                    # Darker version of histogram colour
                    # --------------------------------------------

                    patches = hist_result[2]

                    if len(patches) > 0:

                        facecolor = (
                            patches[0]
                            .get_facecolor()
                        )

                        line_color = tuple(
                            max(
                                0.0,
                                channel * 0.65,
                            )
                            for channel
                            in facecolor[:3]
                        )

                        ax.plot(
                            x,
                            y,
                            color=line_color,
                            linewidth=2.2,
                            alpha=1.0,
                        )


        # --------------------------------------------------------
        # Axes
        # --------------------------------------------------------

        ax.set_xlim(xlim)

        ax.set_xlabel(
            "Robust z-score",
            fontsize=FONT_SIZE,
        )

        y_measure = (
            "Density"
            if probability_distribution
            else "Frequency"
        )

        ax.set_ylabel(
            y_measure,
            fontsize=FONT_SIZE,
        )

        ax.tick_params(
            axis="both",
            labelsize=FONT_SIZE,
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.18,
        )

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)


        # --------------------------------------------------------
        # Legend INSIDE each subplot
        # --------------------------------------------------------

        ax.legend(
            title="Batch",
            loc="upper right",
            fontsize=10,
            title_fontsize=10,
            frameon=True,
        )


    # ============================================================
    # Plot four methods
    # ============================================================

    for ax, method in zip(
        axes,
        method_names,
    ):

        _plot_panel(
            ax,
            adjusted_data[method],
        )

        ax.set_title(
            f"{method} (Cov_Subj_adjusted)",
            fontsize=FONT_SIZE,
            fontweight="bold",
            pad=8,
        )


    # ============================================================
    # Main title
    # ============================================================

    fig.suptitle(
        "Feature distributions across batches",
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.98,
    )


    # ============================================================
    # Layout
    # ============================================================

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.09,
        wspace=0.20,
        hspace=0.28,
    )


    # ============================================================
    # Return in same style as comparison function
    # ============================================================

    figs = [
        (
            "Paper figure: adjusted robust feature "
            "distributions across batches",
            fig,
        )
    ]

    return figs

datasets = {
    "Raw": data_array,
    "ComBat": combat_IDP,
    "CovBat": covbat_IDP,
    "LongComBat": Lcombat_noTP_IDP
}

""" figs = plot_compare_longitudinal_zscore_histograms_forPaper(
    datasets=datasets,
    batch=batch_array,
    subjects=subject_array,
    covariates=covariate_dict,
)

plt.show()

fig = figs[0][1]

fig.savefig(
    "adjusted_distributions_forPaper.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    "adjusted_distributions_forPaper.pdf",
    bbox_inches="tight",
    facecolor="white"
) """


datasets = {
    "Raw": data_array,
    "ComBat": combat_IDP,
    "CovBat": covbat_IDP,
    "LongComBat": Lcombat_noTP_IDP
}

figs = plot_compare_longitudinal_zscore_histograms_forPaper_2rows(
    datasets=datasets,
    batch=batch_array,
    subjects=subject_array,
    covariates=covariate_dict,
)

plt.show()

fig = figs[0][1]

fig.savefig(
    "feature_distributions_unadjusted_adjusted.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    "feature_distributions_unadjusted_adjusted.pdf",
    bbox_inches="tight",
    facecolor="white",
)