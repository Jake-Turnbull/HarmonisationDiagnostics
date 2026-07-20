#!/usr/bin/env python3
"""
Command-line wrapper for Harmonisation Diagnostics.

Usage examples:
    DHarm run --data data.csv --covariates cov.csv --batch-col 3
    DHarm run --data data.csv --covariates cov.csv --outdir ./reports
    DHarm gui
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from DiagnoseHarmonisation.cross_sectional_workflow import (
    CrossSectionalRunConfig,
    inspect_cross_sectional_inputs,
    run_cross_sectional_report,
)


def run_pipeline_from_cli(
    data_path: str,
    cov_path: str,
    batch_col_index: Optional[int],
    data_id_col: str | None = None,
    cov_id_col: str | None = None,
    outdir: str | None = None,
    report_name: str | None = None,
    verbose: bool = False,
    save_data: bool = True,
    save_data_name: str | None = None,
):
    batch_mode = "auto"
    batch_column = None

    if batch_col_index is not None:
        summary = inspect_cross_sectional_inputs(data_path, cov_path)
        zero_based_index = int(batch_col_index) - 1
        if zero_based_index < 0 or zero_based_index >= len(summary.covariates_columns):
            raise ValueError(
                f"batch-col {batch_col_index} out of range for covariates with {len(summary.covariates_columns)} columns."
            )
        batch_mode = "selected"
        batch_column = summary.covariates_columns[zero_based_index]

    config = CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=cov_path,
        data_id_column=data_id_col,
        covariates_id_column=cov_id_col,
        batch_mode=batch_mode,
        batch_column=batch_column,
        output_dir=outdir,
        report_name=report_name,
        save_data=_coerce_bool(save_data),
        save_data_name=save_data_name,
        timestamped_reports=True,
        allow_missing_batch_in_auto=True,
    )
    return run_cross_sectional_report(
        config,
        verbose=verbose,
        status_callback=print if verbose else None,
    )


def launch_gui() -> None:
    from DiagnoseHarmonisation.gui import launch_cross_sectional_gui

    launch_cross_sectional_gui()


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        prog="DHarm",
        description="Harmonisation Diagnostics CLI for scripted runs and the desktop cross-sectional GUI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the cross-sectional diagnostics pipeline from data and covariates files.",
    )
    run_parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to data CSV/XLS/XLSX (subjects x features).",
    )
    run_parser.add_argument(
        "--covariates",
        "-c",
        required=True,
        help="Path to covariates CSV/XLS/XLSX (one row per subject).",
    )
    run_parser.add_argument(
        "--batch-col",
        type=int,
        default=None,
        help="1-based covariates column number where batch is located. If omitted, tries to auto-detect by header.",
    )
    run_parser.add_argument(
        "--subject-id-col",
        default=None,
        help="Data subject ID column name (defaults to first column).",
    )
    run_parser.add_argument(
        "--cov-id-col",
        default=None,
        help="Covariates subject ID column name (defaults to first column).",
    )
    run_parser.add_argument(
        "--outdir",
        default=None,
        help="Directory to write summary and report files.",
    )
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    run_parser.add_argument(
        "--report-name",
        default=None,
        help="Optional name for the report (used in filenames).",
    )
    run_parser.add_argument(
        "--save-data",
        default=True,
        help="Whether to save the aligned data and covariates used for the report.",
    )
    run_parser.add_argument(
        "--save-data-name",
        default=None,
        help="Optional name for the saved data files (used in filenames).",
    )

    subparsers.add_parser(
        "gui",
        help="Open the desktop GUI for generating a cross-sectional report.",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_pipeline_from_cli(
            data_path=args.data,
            cov_path=args.covariates,
            batch_col_index=args.batch_col,
            data_id_col=args.data_id_col,
            cov_id_col=args.cov_id_col,
            outdir=args.outdir,
            report_name=args.report_name,
            verbose=args.verbose,
            save_data=args.save_data,
            save_data_name=args.save_data_name,
        )
    if args.command == "gui":
        launch_gui()
        return 0

    parser.print_help()
    return 0


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


if __name__ == "__main__":
    main()
