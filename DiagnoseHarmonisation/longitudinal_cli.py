#!/usr/bin/env python3
"""
Strict command-line wrapper for longitudinal Harmonisation Diagnostics.

This CLI does not auto-detect columns during `run`.
Users must explicitly provide subject ID, timepoint, batch, and feature columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from DiagnoseHarmonisation.gui import launch_cross_sectional_gui
from DiagnoseHarmonisation.longitudinal_workflow import (
    LongitudinalRunConfig,
    inspect_longitudinal_inputs,
    run_longitudinal_report,
)

def launch_gui() -> None:
    from DiagnoseHarmonisation.gui_longitudinal import launch_longitudinal_gui
    launch_longitudinal_gui()


def run_pipeline_from_cli(
    data_path: str,
    subject_id_col: str,
    timepoint_col: str,
    batch_col: str,
    feature_cols: str | None = None,
    features_file: str | None = None,
    cov_path: str | None = None,
    cov_subject_id_col: str | None = None,
    cov_timepoint_col: str | None = None,
    covariate_cols: str | None = None,
    covariates_file: str | None = None,
    covariate_names: str | None = None,
    outdir: str | None = None,
    report_name: str | None = None,
    verbose: bool = False,
    save_data: bool = False,
    save_data_name: str | None = None,
):
    selected_features = _parse_columns_or_file(feature_cols, features_file)

    if selected_features is None or len(selected_features) == 0:
        raise ValueError("Feature columns are required. Use --feature-cols or --features-file.")

    selected_covariates = _parse_columns_or_file(covariate_cols, covariates_file)
    selected_covariate_names = _parse_comma_separated(covariate_names)

    if selected_covariate_names is not None and selected_covariates is None:
        raise ValueError("--covariate-names requires --covariate-cols or --covariates-file.")

    config = LongitudinalRunConfig(
        data_path=data_path,
        subject_id_column=subject_id_col,
        timepoint_column=timepoint_col,
        batch_column=batch_col,
        selected_features=selected_features,
        covariates_path=cov_path,
        covariates_subject_id_column=cov_subject_id_col,
        covariates_timepoint_column=cov_timepoint_col,
        selected_covariates=selected_covariates,
        covariate_names=selected_covariate_names,
        output_dir=outdir,
        report_name=report_name,
        save_data=_coerce_bool(save_data),
        save_data_name=save_data_name,
        timestamped_reports=True,
    )

    return run_longitudinal_report(
        config,
        verbose=verbose,
        status_callback=print if verbose else None,
    )


def inspect_inputs_from_cli(
    data_path: str,
    cov_path: str | None = None,
):
    summary = inspect_longitudinal_inputs(data_path, cov_path)

    print("Data columns:")
    for index, column in enumerate(summary.data_columns, start=1):
        print(f"  {index}. {column}")

    if summary.covariates_columns:
        print("\nCovariates columns:")
        for index, column in enumerate(summary.covariates_columns, start=1):
            print(f"  {index}. {column}")

    return 0


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        prog="DHarm-longitudinal",
        description="Strict Harmonisation Diagnostics CLI for longitudinal reports.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the longitudinal diagnostics pipeline with explicit column choices.",
    )

    run_parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to longitudinal data CSV/XLS/XLSX.",
    )

    run_parser.add_argument(
        "--subject-id-col",
        required=True,
        help="Subject ID column name in the data file.",
    )

    run_parser.add_argument(
        "--timepoint-col",
        required=True,
        help="Timepoint/visit column name in the data file.",
    )

    run_parser.add_argument(
        "--batch-col",
        required=True,
        help="Batch/site/scanner column name in the data file.",
    )

    feature_group = run_parser.add_mutually_exclusive_group(required=True)

    feature_group.add_argument(
        "--feature-cols",
        default=None,
        help="Comma-separated feature/IDP column names.",
    )

    feature_group.add_argument(
        "--features-file",
        default=None,
        help="Text file containing one feature/IDP column name per line.",
    )

    run_parser.add_argument(
        "--covariates",
        "-c",
        default=None,
        help=(
            "Optional path to separate longitudinal covariates CSV/XLS/XLSX. "
            "If omitted, covariates are read from --data."
        ),
    )

    run_parser.add_argument(
        "--cov-subject-id-col",
        default=None,
        help="Subject ID column name in the separate covariates file.",
    )

    run_parser.add_argument(
        "--cov-timepoint-col",
        default=None,
        help="Timepoint/visit column name in the separate covariates file.",
    )

    covariate_group = run_parser.add_mutually_exclusive_group(required=False)

    covariate_group.add_argument(
        "--covariate-cols",
        default=None,
        help="Comma-separated covariate column names.",
    )

    covariate_group.add_argument(
        "--covariates-file",
        default=None,
        help="Text file containing one covariate column name per line.",
    )

    run_parser.add_argument(
        "--covariate-names",
        default=None,
        help=(
            "Optional comma-separated covariate display names. "
            "Must match the order and number of selected covariates."
        ),
    )

    run_parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write report files if LongitudinalReport supports save_dir.",
    )

    run_parser.add_argument(
        "--report-name",
        default=None,
        help="Optional name for the report.",
    )

    run_parser.add_argument(
        "--save-data",
        action="store_true",
        help=(
            "Attempt to save aligned input data. "
            "Warning: current LongitudinalReport may fail with covariates because "
            "its save_data block expects array-like covariates, while modelling expects dict covariates."
        ),
    )

    run_parser.add_argument(
        "--save-data-name",
        default=None,
        help="Optional name for saved input data files if supported.",
    )

    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Print input file columns. Does not infer or select columns.",
    )

    inspect_parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to longitudinal data CSV/XLS/XLSX.",
    )

    inspect_parser.add_argument(
        "--covariates",
        "-c",
        default=None,
        help="Optional path to separate covariates CSV/XLS/XLSX.",
    )

    subparsers.add_parser(
        "gui",
        help="Open the desktop GUI for generating a longitudinal report.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        return run_pipeline_from_cli(
            data_path=args.data,
            subject_id_col=args.subject_id_col,
            timepoint_col=args.timepoint_col,
            batch_col=args.batch_col,
            feature_cols=args.feature_cols,
            features_file=args.features_file,
            cov_path=args.covariates,
            cov_subject_id_col=args.cov_subject_id_col,
            cov_timepoint_col=args.cov_timepoint_col,
            covariate_cols=args.covariate_cols,
            covariates_file=args.covariates_file,
            covariate_names=args.covariate_names,
            outdir=args.outdir,
            report_name=args.report_name,
            verbose=args.verbose,
            save_data=args.save_data,
            save_data_name=args.save_data_name,
        )

    if args.command == "inspect":
        return inspect_inputs_from_cli(
            data_path=args.data,
            cov_path=args.covariates,
        )
    
    if args.command == "gui":
                launch_gui()
    return 0

    parser.print_help()
    return 0



def _parse_columns_or_file(
    columns: str | None,
    columns_file: str | None,
) -> list[str] | None:
    if columns and columns_file:
        raise ValueError("Provide either comma-separated columns or a columns file, not both.")

    if columns:
        return [column.strip() for column in columns.split(",") if column.strip()]

    if columns_file:
        path = Path(columns_file)
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    return None


def _parse_comma_separated(value: str | None) -> list[str] | None:
    if value is None:
        return None

    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}

    return bool(value)


if __name__ == "__main__":
    main()