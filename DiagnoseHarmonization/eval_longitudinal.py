#!/usr/bin/env python3
"""
eval_longitudinal.py

Same as before, but now supports:
  --feature_pattern  (shell-style glob, e.g. 'T*', 'T1_*hipp*')
  --feature_regex    (regular expression)

Precedence:
  --feature_cols > --feature_pattern > --feature_regex > autodetect
"""

import argparse
import sys
import textwrap
import re
import fnmatch
import numpy as np
import pandas as pd
from DiagnoseHarmonization import DiagnosticReport

def parse_args():
    parser = argparse.ArgumentParser(
        prog="eval_longitudinal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            Run longitudinal harmonization diagnostics.
            Provide either:
              --single_file COMBINED.csv
            OR:
              --covariate_file COV.csv --features FEATURES.csv

            Feature selection precedence:
              --feature_cols (explicit list)
              --feature_pattern (glob-style, e.g. 'T*', 'T1_*hipp*')
              --feature_regex (regular expression)
              otherwise autodetect numeric columns.
        """)
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single_file", help="Single combined CSV (subject,timepoint,batch,covariates,features).")
    group.add_argument("--covariate_file", help="Covariate CSV (subject,timepoint,batch,...).")

    parser.add_argument("--features", help="Features CSV (required with --covariate_file).")
    parser.add_argument("--subject_col", default="subject", help="Subject ID column name (default: subject).")
    parser.add_argument("--timepoint_col", default="timepoint", help="Timepoint column name (default: timepoint).")
    parser.add_argument("--batch_col", default="scan_session", help="Batch column name (default: scan_session).")

    # Feature selection options
    parser.add_argument("--feature_cols", nargs="+", default=None,
                        help="Explicit list of feature/IDP columns (space-separated).")
    parser.add_argument("--feature_pattern", default=None,
                        help="Shell-style glob pattern to select feature columns (e.g. 'T*', quote it!).")
    parser.add_argument("--feature_regex", default=None,
                        help="Regex to select feature columns (e.g. '^T1_.*hippocampus$').")

    parser.add_argument("--covariate_cols", nargs="+", default=None,
                        help="Explicit list of covariate columns (e.g., age sex disorder).")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip subject/timepoint 1:1 matching check.")
    parser.add_argument("--prefer_covariate_cols", action="store_true",
                        help="When duplicate columns exist after merge, prefer covariate-side columns (default behavior).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print detected columns and exit without running diagnostics (useful for demos).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")
    return parser.parse_args()


def load_csv_or_exit(path, required_cols=None):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV '{path}': {e}", file=sys.stderr)
        sys.exit(2)
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"ERROR: CSV '{path}' missing required columns: {missing}", file=sys.stderr)
            sys.exit(2)
    return df


def resolve_merged_column(columns, colname, prefer_cov=True):
    if colname in columns:
        return colname
    cov = f"{colname}_cov"
    feat = f"{colname}_feat"
    if prefer_cov and cov in columns:
        return cov
    if not prefer_cov and feat in columns:
        return feat
    if cov in columns:
        return cov
    if feat in columns:
        return feat
    raise KeyError(colname)


def select_features_from_columns(all_columns, args, key_cols):
    """
    Determine idp_names from args in precedence order:
      1) feature_cols explicit
      2) feature_pattern (glob)
      3) feature_regex (re)
      4) autodetect numeric columns (excluding key columns)
    Returns a list of feature column names (must exist in all_columns).
    """
    # 1) explicit list
    if args.feature_cols:
        missing = [c for c in args.feature_cols if c not in all_columns]
        if missing:
            print(f"ERROR: Some --feature_cols not found in columns: {missing}", file=sys.stderr)
            sys.exit(2)
        return args.feature_cols

    # 2) glob pattern
    if args.feature_pattern:
        # use fnmatch to filter column names (pattern should be quoted on command line)
        matched = fnmatch.filter(all_columns, args.feature_pattern)
        if len(matched) == 0:
            print(f"ERROR: --feature_pattern matched zero columns (pattern: {args.feature_pattern}).", file=sys.stderr)
            print("Available columns:", all_columns, file=sys.stderr)
            sys.exit(2)
        return matched

    # 3) regex
    if args.feature_regex:
        prog = re.compile(args.feature_regex)
        matched = [c for c in all_columns if prog.search(c)]
        if len(matched) == 0:
            print(f"ERROR: --feature_regex matched zero columns (regex: {args.feature_regex}).", file=sys.stderr)
            print("Available columns:", all_columns, file=sys.stderr)
            sys.exit(2)
        return matched

    # 4) autodetect numeric columns
    # key_cols is a set of columns to exclude (subject/timepoint/batch)
    # actual autodetect will be done with a DataFrame in context; here we just return None to signal autodetect
    return None


def main():
    args = parse_args()

    # ---------- Load / detect data ----------
    if args.single_file:
        if args.verbose:
            print(f"Loading single combined file: {args.single_file}")
        combined = load_csv_or_exit(args.single_file, required_cols=[args.subject_col, args.timepoint_col])
        key_cols = {args.subject_col, args.timepoint_col, args.batch_col}

        # Determine feature columns using helper (may return None -> autodetect numeric)
        idp_names = select_features_from_columns(list(combined.columns), args, key_cols)

        if idp_names is None:
            numeric_cols = combined.select_dtypes(include=["number"]).columns.tolist()
            idp_names = [c for c in numeric_cols if c not in key_cols]
            if args.verbose:
                print("Autodetected numeric columns as features:", idp_names)

        # Covariates detection
        if args.covariate_cols:
            covariate_cols = args.covariate_cols
            missing = [c for c in covariate_cols if c not in combined.columns]
            if missing:
                print(f"ERROR: Covariate columns not found in single file: {missing}", file=sys.stderr)
                sys.exit(2)
        else:
            covariate_cols = [c for c in combined.columns if c not in key_cols and c not in idp_names]
            if len(covariate_cols) == 0:
                covariate_cols = [c for c in combined.select_dtypes(exclude=["number"]).columns if c not in key_cols]
            if args.verbose:
                print("Autodetected covariate columns:", covariate_cols)

        merged = combined.copy()
        resolved_covariate_cols = covariate_cols
        covariates = {c: merged[c].tolist() for c in resolved_covariate_cols}

    else:
        # two-file mode
        if not args.features:
            print("ERROR: --features is required with --covariate_file", file=sys.stderr)
            sys.exit(2)

        cov_df = load_csv_or_exit(args.covariate_file, required_cols=[args.subject_col, args.timepoint_col])
        feat_df = load_csv_or_exit(args.features, required_cols=[args.subject_col, args.timepoint_col])

        # Feature selection: precedence
        idp_names = select_features_from_columns(list(feat_df.columns), args, {args.subject_col, args.timepoint_col})
        if idp_names is None:
            # autodetect: all non-key columns in features file
            idp_names = [c for c in feat_df.columns if c not in {args.subject_col, args.timepoint_col}]
            if args.verbose:
                print("Autodetected feature columns from features file:", idp_names)
            if len(idp_names) == 0:
                print("ERROR: No feature columns detected in features file. Use --feature_cols or --feature_pattern.", file=sys.stderr)
                sys.exit(2)

        # Merge
        if args.verbose:
            print("Merging covariates and features on", args.subject_col, "+", args.timepoint_col)
        merged = pd.merge(cov_df, feat_df, on=[args.subject_col, args.timepoint_col], how="inner", suffixes=("_cov", "_feat"))

        # Covariates: explicit or detect from cov_df
        if args.covariate_cols:
            covariate_cols = args.covariate_cols
        else:
            covariate_cols = [c for c in cov_df.columns if c not in {args.subject_col, args.timepoint_col}]
            if args.verbose:
                print("Detected covariate columns from covariate file:", covariate_cols)

        # Resolve covariate names in merged columns
        resolved_covariate_cols = []
        for c in covariate_cols:
            if c in {args.subject_col, args.timepoint_col}:
                continue
            try:
                resolved = resolve_merged_column(merged.columns, c, prefer_cov=True)
                resolved_covariate_cols.append(resolved)
            except KeyError:
                print(f"ERROR: Covariate column '{c}' not found in merged data (nor with _cov/_feat suffix).", file=sys.stderr)
                print("Merged columns:", list(merged.columns), file=sys.stderr)
                sys.exit(2)

        covariates = {}
        for c in resolved_covariate_cols:
            key_name = c[:-4] if c.endswith("_cov") else c
            covariates[key_name] = merged[c].tolist()

    # ---------- Basic correspondence checks ----------
    if not args.no_check:
        try:
            key_pairs = merged[[args.subject_col, args.timepoint_col]].apply(lambda r: (str(r[args.subject_col]), str(r[args.timepoint_col])), axis=1)
        except Exception as e:
            print("ERROR while extracting subject/timepoint keys from merged data:", e, file=sys.stderr)
            sys.exit(3)
        if key_pairs.duplicated().any():
            print("WARNING: Duplicate (subject, timepoint) rows found in merged data.", file=sys.stderr)
        if args.verbose:
            print("Merged rows (after inner join):", merged.shape[0])
    else:
        if args.verbose:
            print("Skipping 1:1 correspondence check (--no-check).")

    # ---------- Resolve batch column ----------
    try:
        batch_col_resolved = resolve_merged_column(merged.columns, args.batch_col, prefer_cov=args.prefer_covariate_cols)
    except KeyError:
        print(f"ERROR: Batch column '{args.batch_col}' not found in merged data (nor with _cov/_feat).", file=sys.stderr)
        print("Columns available:", list(merged.columns), file=sys.stderr)
        sys.exit(2)

    # ---------- Resolve features to merged names ----------
    resolved_idp_cols = []
    for name in idp_names:
        if name in merged.columns:
            resolved_idp_cols.append(name)
        elif f"{name}_feat" in merged.columns:
            resolved_idp_cols.append(f"{name}_feat")
        elif f"{name}_cov" in merged.columns:
            resolved_idp_cols.append(f"{name}_cov")
        else:
            print(f"ERROR: Feature column '{name}' not found in merged data (nor with _feat/_cov suffix).", file=sys.stderr)
            print("Merged columns:", list(merged.columns), file=sys.stderr)
            sys.exit(2)

    # ---------- Build arrays ----------
    try:
        idp_matrix = merged[resolved_idp_cols].to_numpy(dtype=float)
    except Exception as e:
        print("ERROR converting IDP columns to numeric matrix:", e, file=sys.stderr)
        sys.exit(5)

    subjects = merged[args.subject_col].astype(str).tolist()
    timepoints = merged[args.timepoint_col].astype(str).tolist()
    batches = merged[batch_col_resolved].astype(str).tolist()

    # ---------- Dry-run ----------
    if args.dry_run:
        print("\n--- DRY RUN: Detected configuration ---")
        print("Rows (merged):", merged.shape[0])
        print("Subject column:", args.subject_col)
        print("Timepoint column:", args.timepoint_col)
        print("Batch column resolved to:", batch_col_resolved)
        print("Features (resolved):", resolved_idp_cols)
        print("Covariates (resolved):", list(covariates.keys()))
        print("First 5 subjects:", subjects[:5])
        print("IDP matrix shape:", idp_matrix.shape)
        print("\nExiting due to --dry_run.")
        sys.exit(0)

    # ---------- Sanity prints ----------
    print("\n--- Sanity checks ---")
    print("Merged rows:", merged.shape[0])
    print("Number of features (IDPs):", len(resolved_idp_cols))
    print("IDP matrix shape:", idp_matrix.shape)
    print("Batch column resolved to:", batch_col_resolved)
    print("Covariates included:", list(covariates.keys()))
    print("Example subjects (first 5):", subjects[:5])
    print("Example batches (first 5):", batches[:5])

    # ---------- Run DiagnosticReport ----------
    print("\nRunning DiagnosticReport.LongitudinalReport ...")
    try:
        feature_names_clean = [c.replace("_feat", "").replace("_cov", "") for c in resolved_idp_cols]
        DiagnosticReport.LongitudinalReport(
            data=idp_matrix,
            subject_ids=subjects,
            batch=batches,
            timepoints=timepoints,
            features=feature_names_clean,
            covariates=covariates
        )
        print("Diagnostic report completed successfully.")
    except Exception as e:
        print("ERROR while running the diagnostic report:", e, file=sys.stderr)
        sys.exit(6)


if __name__ == "__main__":
    main()
