import numpy as np
import pandas as pd

from DiagnoseHarmonisation import DiagnosticFunctions
from DiagnoseHarmonisation import PlotDiagnosticResults
from DiagnoseHarmonisation import DiagnosticReport

# Load CSV
df = pd.read_csv("tests/onharmony.csv")

# ---- REQUIRED STRUCTURAL VARIABLES ----
subjects   = df["subject"].astype(str).tolist()
timepoints = df["timepoint"].astype(str).tolist()
batches    = df["scan_session"].astype(str).tolist()
age = df["age"].astype(float).tolist()
sex = df["sex"].astype(str).tolist()

# ---- COVARIATES (dict) ----
covariates = {
    "age": age,
    "sex": sex
}
covariates_array = np.column_stack([age, sex])
#print(covariates_array)

# ---- IDPs ----
idp_names = ["T1_SIENAX_peripheral_GM_norm_vol", "T1_SIENAX_WM_norm_vol"]   # or infer automatically (see below)
idp_matrix = df[idp_names].to_numpy(dtype=float)

# ---- SANITY CHECKS ----
n_samples = len(df)
print("n_samples:", n_samples)
print("shape idp_matrix:", idp_matrix.shape)
print("subjects (unique):", sorted(set(subjects)))
print("batches (unique):", sorted(set(batches)))

#
DiagnosticReport.LongitudinalReport(data=idp_matrix, subject_ids=subjects, batch=batches, timepoints=timepoints, features=idp_names, covariates=covariates)