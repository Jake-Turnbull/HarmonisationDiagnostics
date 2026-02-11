#!/usr/bin/env python3
"""
Command-line wrapper for Harmonization Diagnostics.

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
                          save_data: bool = False) -> Optional[Dict[str, Any]]:
    # 1) read CSVs
    if verbose:
        print(f"Reading data from: {data_path}")
    data_df = pd.read_csv(data_path, header=0)
    if verbose:
        print(f"Reading covariates from: {cov_path}")
    cov_df = pd.read_csv(cov_path, header=0)

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
    X = data_sub.astype(float).values  # shape: (n_subjects, n_features)
    feature_names = list(data_sub.columns)

    # 6) covariate matrix
    # keep all covariates columns except subject id (already indexed)
    covariates_df = cov_sub.copy()
    if batch_col_name is not None:
        if batch_col_name not in covariates_df.columns:
            raise ValueError(f"Detected batch column '{batch_col_name}' not present after indexing.")
        batch_series = covariates_df[batch_col_name].astype(str)
        # create numeric batch codes if needed
        unique_batches = sorted(batch_series.unique())
        batch_codes = pd.Categorical(batch_series, categories=unique_batches).codes
        covariates_df["_batch_code_"] = batch_codes
    else:
        batch_series = pd.Series(["single_batch"] * covariates_df.shape[0], index=covariates_df.index)
        covariates_df["_batch_code_"] = np.zeros(covariates_df.shape[0], dtype=int)

    # 7) convert to simple structs for downstream code if desired
    data_struct = {
        "X": X,  # numpy array (n_subjects x n_features)
        "feature_names": feature_names,
        "subject_ids": list(data_sub.index)
    }
    cov_struct = {
        "covariates_df": covariates_df,  # pandas DataFrame with covariates and _batch_code_
        "batch_series": batch_series,
        "batch_col_name": batch_col_name
    }

    if verbose:
        print("Data converted:")
        print(f"  - subjects: {X.shape[0]}, features: {X.shape[1]}")
        print(f"  - batch column: {batch_col_name} (unique={covariates_df['_batch_code_'].nunique()})")

    # 8) call the project's harmonisation / reporting function
    # --- IMPORTANT: replace the import & call below with your project's API. ---
    # Example placeholder:
    try:
        # try importing the expected function from your package
        from DiagnoseHarmonization import DiagnosticReport

        DiagnosticReport.CrossSectionalReport(X,
                             batch=cov_struct["batch_series"],
                             covariates=cov_struct["covariates_df"],
                             covariate_names=list(cov_struct["covariates_df"].columns),
                             feature_names=data_struct["feature_names"],
                             subject_ids=data_struct["subject_ids"],
                             save_dir=outdir,
                             save_data=save_data,
                             report_name=report_name)
        
        # For now, we call a simple placeholder that returns a summary dict:
        
        # TODO: when integrating, replace above with your function and pass data_struct, cov_struct.
    except Exception as e:
        raise RuntimeError(f"Error running pipeline: {e}")

    
    return None

# --- CLI parser -------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(prog="harmdiag",
                                description="Harmonization Diagnostics CLI — run harmonisation/reporting from the terminal.")
    sub = p.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="Run the diagnostics pipeline from data and covariates CSVs")
    runp.add_argument("--data", "-d", required=True, help="Path to data CSV (subjects x IDPs). First row must be feature names.")
    runp.add_argument("--covariates", "-c", required=True, help="Path to covariates CSV (first column subject ID).")
    runp.add_argument("--batch-col", type=int, default=None,
                      help="1-based column number in covariates CSV where batch is located. If omitted, tries to auto-detect by header.")
    runp.add_argument("--data-id-col", default=None, help="Data subject ID column name (defaults to first column).")
    runp.add_argument("--cov-id-col", default=None, help="Covariates subject ID column name (defaults to first column).")
    runp.add_argument("--outdir", default=None, help="Directory to write summary / report files.")
    runp.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    runp.add_argument("--report-name", default=None, help="Optional name for the report (used in filenames).")
    runp.add_argument("--save-data", action="store_true", help="Whether to save the aligned data and covariates used for the report (for debugging).")
    

    args = p.parse_args(argv)
    if args.command == "run":
        return run_pipeline_from_cli(
            data_path=args.data,
            cov_path=args.covariates,
            batch_col_index=args.batch_col,
            data_id_col=args.data_id_col,
            cov_id_col=args.cov_id_col,
            outdir=args.outdir,
            verbose=args.verbose
        )
    else:
        p.print_help()
        return 0

if __name__ == "__main__":
    main()

    import io
import pandas as pd

# Optional: if you have charset-normalizer or chardet installed use them
try:
    import chardet
except Exception:
    chardet = None

def robust_read_csv(path, pandas_kwargs=None, try_encodings=None):
    """
    Read CSV with a fallback strategy for encodings.
    Returns dataframe and used encoding.
    """
    if pandas_kwargs is None:
        pandas_kwargs = {}
    if try_encodings is None:
        try_encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

    # Fast attempt loop
    for enc in try_encodings:
        try:
            return pd.read_csv(path, encoding=enc, **pandas_kwargs), enc
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            # try with engine='python' as a fallback for weird separators
            try:
                return pd.read_csv(path, encoding=enc, engine="python", **pandas_kwargs), enc
            except Exception:
                continue

    # Attempt to detect encoding (best-effort)
    if chardet is not None:
        with open(path, "rb") as fh:
            raw = fh.read(100000)  # sample
            guess = chardet.detect(raw)
            enc = guess.get("encoding")
            if enc:
                try:
                    return pd.read_csv(path, encoding=enc, **pandas_kwargs), enc
                except Exception:
                    pass

    # Last resort: open with 'latin-1' but warn the user
    try:
        df = pd.read_csv(path, encoding="latin-1", **pandas_kwargs)
        return df, "latin-1 (fallback, may mangle non-latin text)"
    except Exception as e:
        # raise a helpful error
        raise UnicodeDecodeError(
            f"Failed to read CSV '{path}'. Tried encodings {try_encodings} "
            f"and detection (chardet={bool(chardet)}). Last error: {e}"
        )

