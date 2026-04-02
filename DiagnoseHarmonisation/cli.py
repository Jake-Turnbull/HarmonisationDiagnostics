#!/usr/bin/env python3
"""
Command-line wrapper for Harmonisation Diagnostics.

Usage examples:
  # minimal: specify data and covariates files and batch column index (1-based)
  harmdiag run --data data.csv --covariates cov.csv --batch-col 3

  # allow auto-detection of batch column
  harmdiag run --data data.csv --covariates cov.csv

  # show verbose info and write outputs to a dir
  harmdiag run -v --data data.csv --covariates cov.csv --outdir ./reports
"""

from __future__ import annotations
import argparse
import sys
import os
import textwrap
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd

# --- Helper utilities -------------------------------------------------------
BATCH_HEADER_CANDIDATES = [
    "batch", "site", "center", "centre", "scanner", "cohort", "study", "batch_id", "site_id"
]

def fuzzy_find_batch_column(headers: Sequence[str]) -> Optional[int]:
    """Return zero-based index of a header that looks like batch, or None."""
    low = [h.lower() for h in headers]
    for cand in BATCH_HEADER_CANDIDATES:
        if cand in low:
            return low.index(cand)
    # try partial matches
    for i, h in enumerate(low):
        for cand in BATCH_HEADER_CANDIDATES:
            if cand in h:
                return i
    return None

def validate_subject_ids(data_df: pd.DataFrame, cov_df: pd.DataFrame, data_id_col: Optional[str], cov_id_col: Optional[str]):
    """Ensure subject IDs are compatible. Raises ValueError on severe mismatch, else returns merged df."""
    if data_id_col is None or cov_id_col is None:
        raise ValueError("Both data_id_col and cov_id_col must be provided (or found automatically).")

    if data_id_col not in data_df.columns:
        raise ValueError(f"Subject ID column '{data_id_col}' not found in data file.")
    if cov_id_col not in cov_df.columns:
        raise ValueError(f"Subject ID column '{cov_id_col}' not found in covariates file.")

    data_ids = set(data_df[data_id_col].astype(str))
    cov_ids = set(cov_df[cov_id_col].astype(str))
    common = data_ids & cov_ids
    if len(common) == 0:
        raise ValueError("No matching subject IDs found between data and covariates files.")
    if len(common) < max(len(data_ids), len(cov_ids)) * 0.5:
        # warn but still proceed
        print(f"Warning: only {len(common)} subjects overlap between data ({len(data_ids)}) and covariates ({len(cov_ids)}). Proceeding with intersection.", file=sys.stderr)

    # subset both to intersection and preserve index alignment
    common_sorted = sorted(common)
    data_sub = data_df[data_df[data_id_col].astype(str).isin(common_sorted)].copy()
    cov_sub = cov_df[cov_df[cov_id_col].astype(str).isin(common_sorted)].copy()
    # align by subject ID
    data_sub.set_index(data_id_col, inplace=True)
    cov_sub.set_index(cov_id_col, inplace=True)
    data_sub = data_sub.loc[cov_sub.index.intersection(data_sub.index)]
    cov_sub = cov_sub.loc[data_sub.index]
    return data_sub, cov_sub

# --- Main CLI behaviour ----------------------------------------------------
def run_pipeline_from_cli(data_path: str,
                          cov_path: str,
                          batch_col_index: Optional[int],
                          data_id_col: str = None,
                          cov_id_col: str = None,
                          outdir: Optional[str] = None,
                          report_name: Optional[str] = None,
                          verbose: bool = False,
                          save_data: bool = True,
                          save_data_name: str = None) -> Optional[Dict[str, Any]]:
    # 1) read CSVs
    """
    if verbose:
        print(f"Reading data from: {data_path}")
    # Check file type by extension (basic check)
    data_df = pd.read_csv(data_path, header=0)
    if verbose:
        print(f"Reading covariates from: {cov_path}")
    cov_df = pd.read_csv(cov_path, header=0)"""
    # 1) # Check data type by extension, use appropriate pandas reader
    if verbose:
        print(f"Reading data from: {data_path}")
    if data_path.lower().endswith(".csv"):
        data_df = pd.read_csv(data_path, header=0)
    elif data_path.lower().endswith((".xls", ".xlsx")):
        data_df = pd.read_excel(data_path, header=0)
    else:
        raise ValueError(f"Unsupported data file format for '{data_path}'. Supported: .csv, .xls, .xlsx")
    if verbose:
        print(f"Reading covariates from: {cov_path}")
    if cov_path.lower().endswith(".csv"):
        cov_df = pd.read_csv(cov_path, header=0)
    elif cov_path.lower().endswith((".xls", ".xlsx")):
        cov_df = pd.read_excel(cov_path, header=0)
    else:
        raise ValueError(f"Unsupported covariates file format for '{cov_path}'. Supported: .csv, .xls, .xlsx")

    # Add in support for other file formats (e.g. Excel) if needed in the future, by checking file extension and using pd.read_excel etc.


    
    # 1a) basic checks
    if data_df.shape[0] == 0 or data_df.shape[1] == 0:
        raise ValueError("Data file appears empty or malformed.")
    if cov_df.shape[0] == 0 or cov_df.shape[1] == 0:
        raise ValueError("Covariates file appears empty or malformed.")

    # 2) assume subject ID column: try to infer if not provided
    # default: if first column name is something like 'subject' or 'id', use it
    if data_id_col is None:
        first_col = data_df.columns[0]
        data_id_col = first_col
        if verbose:
            print(f"Assuming subject ID column in data is '{first_col}'.")
    if cov_id_col is None:
        first_col = cov_df.columns[0]
        cov_id_col = first_col
        if verbose:
            print(f"Assuming subject ID column in covariates is '{first_col}'.")

    # 3) find batch column (zero-based)
    detected_batch_idx = None
    if batch_col_index is not None:
        # user supplied 1-based column index: convert to 0-based
        detected_batch_idx = int(batch_col_index) - 1
        if detected_batch_idx < 0 or detected_batch_idx >= cov_df.shape[1]:
            raise ValueError(f"batch-col {batch_col_index} out of range for covariates with {cov_df.shape[1]} columns.")
        batch_col_name = cov_df.columns[detected_batch_idx]
        if verbose:
            print(f"Using user-specified batch column: index {batch_col_index} -> '{batch_col_name}'")
    else:
        idx = fuzzy_find_batch_column(list(cov_df.columns))
        if idx is not None:
            detected_batch_idx = idx
            batch_col_name = cov_df.columns[detected_batch_idx]
            print(f"Detected batch column automatically as '{batch_col_name}'.")
            
        else:
            # not found — inform user and continue in no-batch / single-batch mode
            print("No batch-like column found in covariates headers. Running in single-batch (no batch) mode.", file=sys.stderr)
            batch_col_name = None

    # 4) validate/align subject ids and produce aligned dataframes
    data_sub, cov_sub = validate_subject_ids(data_df, cov_df, data_id_col, cov_id_col)

    # 5) extract feature matrix: data_sub columns are assumed to be IDPs (excluding subject id col which is index)
    # Check for bad formatting (eg e+ notation that can't be converted to float) and try and convert it if needed
    data_sub = data_sub.apply(pd.to_numeric, errors='coerce')

    X = data_sub.astype(float).values  # shape: (n_subjects, n_features)
    feature_names = list(data_sub.columns)

    # 6) covariate matrix
    # keep all covariates columns except subject id and batch
        # --- robust covariate / batch handling --------------------------------
        # --- robust covariate / batch handling (remove batch column entirely) ---
    # normalise column names (trim whitespace)
    cov_sub.columns = [c.strip() for c in cov_sub.columns]
    covariates_df = cov_sub.copy()

    # If a batch column name was detected / supplied, pop it out so it is removed from covariates_df
    if batch_col_name is not None:
        batch_col_name = batch_col_name.strip()
        if batch_col_name not in covariates_df.columns:
            raise ValueError(f"Detected batch column '{batch_col_name}' not present in covariates after indexing.")
        # pop removes column from covariates_df and returns the Series
        batch_series = covariates_df.pop(batch_col_name).astype(str)
        unique_batches = sorted(batch_series.unique())
        batch_codes = pd.Categorical(batch_series, categories=unique_batches).codes
    else:
        # fallback: single batch (no batch column in file)
        batch_series = pd.Series(["single_batch"] * covariates_df.shape[0], index=covariates_df.index)
        batch_codes = np.zeros(covariates_df.shape[0], dtype=int)

    # Defensive: remove any other covariate columns that are identical to batch (string-equality)
    # (this covers cases where the same info is duplicated under a different header)
    cols_to_drop = []
    for col in list(covariates_df.columns):
        try:
            if covariates_df[col].astype(str).equals(batch_series.astype(str)):
                cols_to_drop.append(col)
        except Exception:
            # if comparison fails for any reason, skip that column
            continue
    if cols_to_drop:
        if verbose:
            print(f"Removing covariate columns identical to batch: {cols_to_drop}", file=sys.stderr)
        covariates_df.drop(columns=cols_to_drop, inplace=True)

    # Remove any subject ID columns if they remain (safe drop)
    covariates_df.drop(columns=[data_id_col, cov_id_col], errors='ignore', inplace=True)

    # sanitize column names to safe python identifiers (optional)
    cleaned = []
    for name in covariates_df.columns:
        new = name.replace(" ", "_").replace(".", "_").replace(",", "_").replace("(", "_").replace(")", "_")
        cleaned.append(new)
        if new != name:
            print(f"Warning: renamed covariate column '{name}' -> '{new}'", file=sys.stderr)
    covariates_df.columns = cleaned

    # debug: show remaining covariate columns so you can confirm batch is NOT present
    if verbose:
        print("Final covariates columns (should NOT include batch):", covariates_df.columns.tolist())

    # Now call your report. Pass batch_codes (or batch_series) separately.
    from DiagnoseHarmonisation import DiagnosticReport
    try:
        DiagnosticReport.CrossSectionalReport(
            X,
            batch=batch_codes,              # numeric batch vector separate from covariates
            covariates=covariates_df,       # covariates WITHOUT any batch column
            covariate_names=list(covariates_df.columns),
            feature_names=None,
            save_dir=outdir,
            save_data=save_data,
            report_name=report_name,
            save_data_name=save_data_name,
        )

            
    except Exception as e:
        raise RuntimeError(f"Error running pipeline: {e}")

    
    return None

# --- CLI parser -------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(prog="harmdiag",
                                description="Harmonisation Diagnostics CLI — run harmonisation/reporting from the terminal.")
    sub = p.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="Run the diagnostics pipeline from data and covariates CSVs")
    runp.add_argument("--data", "-d", required=True, help="Path to data CSV (subjects x IDPs). First row must be feature names.")
    runp.add_argument("--covariates", "-c", required=True, help="Path to covariates CSV (first column subject ID).")
    runp.add_argument("--batch-col", type=int, default=None, help="1-based column number in covariates CSV where batch is located. If omitted, tries to auto-detect by header.")
    runp.add_argument("--data-id-col", default=None, help="Data subject ID column name (defaults to first column).")
    runp.add_argument("--cov-id-col", default=None, help="Covariates subject ID column name (defaults to first column).")
    runp.add_argument("--outdir", default=None, help="Directory to write summary / report files.")
    runp.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    runp.add_argument("--report-name", default=None, help="Optional name for the report (used in filenames).")
    runp.add_argument("--save-data", default = True, help="Whether to save the aligned data and covariates used for the report (for debugging).")
    runp.add_argument("--save-data-name", default=None, help="Optional name for the saved data files (used in filenames).")
    

    args = p.parse_args(argv)
    if args.command == "run":
        return run_pipeline_from_cli(
            data_path=args.data,
            cov_path=args.covariates,
            batch_col_index=args.batch_col,
            data_id_col=args.data_id_col,
            cov_id_col=args.cov_id_col,
            outdir=args.outdir,
            report_name=args.report_name,
            save_data=args.save_data,
            save_data_name=args.save_data_name,
        )
    else:
        p.print_help()
        return 0

if __name__ == "__main__":
    main()