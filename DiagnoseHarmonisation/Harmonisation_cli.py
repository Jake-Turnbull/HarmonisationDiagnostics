"""
Command line entry point for running harmonisation methods using the DiagnoseHarmonisation package.
This CLI is designed for cross-sectional harmonisation.

The recommended usage is to run any harmonisation method in python using the DiagnoseHarmonisation package.
This command line interface is provided for convenience but lacks the full functionality and flexibility
of the python package (nested/advanced options such as prior_weight_opts, interactions, and
covariate_types mappings are not exposed here and require the python API).

Usage examples:
    DHarm harmonise combat --data data.csv --covariates cov.csv --report single
    DHarm harmonise covbat --data data.tsv --covariates cov.tsv --reference-batch site_1
    DHarm harmonise combat_gam --data data.xlsx --covariates cov.xlsx --n-splines 8
    DHarm harmonise combat_modular --data data.csv --covariates cov.csv --mean-model gam --prior-mode local
    DHarm harmonise linear_model --data data.csv --covariates cov.csv --model-type mixedlm --subject-col subjectID

Args:
    method: The harmonisation method to be used.

        combat:
                Implementation of the classic ComBat harmonisation method (via the
                `combat_modular` fast-path with `mean_model='ols'`, `prior_mode='global'`).
        covbat:
                CovBat harmonisation (ComBat plus a covariance-correction step via PCA).
        combat_gam:
                ComBat with a GAM/spline-basis mean model instead of plain OLS.
        combat_modular:
                Full modular ComBat entry point, exposing the mean model (ols/gam) and
                prior mode (global/local, with configurable local-prior weighting methods).
        linear_model:
                Linear (or linear mixed-effects) model based harmonisation via
                `linearmodelling_harmonisation`.

    data: Path to the input data file (CSV, TSV, or Excel format), subjects x features.
    covariates: Path to the covariates file (CSV, TSV, or Excel format), one row per subject.
    report: What report (if any) to generate from the harmonised data:

        none (default):
                Just write the harmonised data to a CSV file.
        single:
                Pass the harmonised data through `DiagnosticReport.CrossSectionalReport`.
        comparison:
                Pass both the raw and harmonised data through
                `DiagnosticReport.CrossSectionalComparisonReport`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

BATCH_HEADER_CANDIDATES = [
    "batch",
    "site",
    "center",
    "centre",
    "scanner",
    "cohort",
    "study",
    "batch_id",
    "site_id",
]

PRIOR_WEIGHT_METHOD_CHOICES = [
    "correlation_similarity",
    "covariance_similarity",
    "variance_similarity",
    "magnitude_similarity",
    "directional_bias",
]


# --------------------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------------------
def read_tabular_file(path: str | Path) -> pd.DataFrame:
    """Read a CSV, TSV, or Excel file into a DataFrame based on its extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, header=0)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(file_path, header=0, sep="\t")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path, header=0)
    raise ValueError(
        f"Unsupported file format for '{file_path}'. Supported: .csv, .tsv, .txt, .xls, .xlsx"
    )


def fuzzy_find_batch_column(headers: Sequence[str]) -> Optional[int]:
    """Return the zero-based index of a batch-like header, or None."""
    lowered = [str(header).lower().strip() for header in headers]
    for candidate in BATCH_HEADER_CANDIDATES:
        if candidate in lowered:
            return lowered.index(candidate)
    for index, header in enumerate(lowered):
        for candidate in BATCH_HEADER_CANDIDATES:
            if candidate in header:
                return index
    return None


@dataclass
class HarmonisationInputs:
    data: np.ndarray  # (n_samples, n_features)
    feature_names: list[str]
    subject_ids: pd.Index
    batch: pd.Series
    mod: pd.DataFrame  # covariates excluding id/batch columns
    save_dir: Path


def load_harmonisation_inputs(
    data_path: str | Path,
    covariates_path: str | Path,
    data_id_col: str | None = None,
    cov_id_col: str | None = None,
    batch_col: str | None = None,
    outdir: str | Path | None = None,
) -> HarmonisationInputs:
    """Read and align the data/covariates files for a harmonisation run."""
    data_df = read_tabular_file(data_path)
    cov_df = read_tabular_file(covariates_path)
    if data_df.shape[0] == 0 or data_df.shape[1] == 0:
        raise ValueError("Data file appears empty or malformed.")
    if cov_df.shape[0] == 0 or cov_df.shape[1] == 0:
        raise ValueError("Covariates file appears empty or malformed.")

    data_id_col = data_id_col or str(data_df.columns[0])
    cov_id_col = cov_id_col or str(cov_df.columns[0])
    if data_id_col not in data_df.columns:
        raise ValueError(f"Subject ID column '{data_id_col}' not found in data file.")
    if cov_id_col not in cov_df.columns:
        raise ValueError(f"Subject ID column '{cov_id_col}' not found in covariates file.")

    data_df = data_df.set_index(data_id_col)
    cov_df = cov_df.set_index(cov_id_col)

    common_ids = data_df.index.astype(str).intersection(cov_df.index.astype(str))
    if len(common_ids) == 0:
        raise ValueError("No matching subject IDs found between data and covariates files.")

    data_df.index = data_df.index.astype(str)
    cov_df.index = cov_df.index.astype(str)
    data_df = data_df.loc[common_ids]
    cov_df = cov_df.loc[common_ids]

    if batch_col is None:
        batch_idx = fuzzy_find_batch_column(list(cov_df.columns))
        if batch_idx is None:
            raise ValueError(
                "No batch-like column was detected automatically. Pass --batch-col explicitly."
            )
        batch_col = str(cov_df.columns[batch_idx])
    elif batch_col not in cov_df.columns:
        raise ValueError(f"Batch column '{batch_col}' not found in covariates file.")

    batch = cov_df[batch_col].astype(str)
    mod = cov_df.drop(columns=[batch_col])

    data_df = data_df.apply(pd.to_numeric, errors="coerce")
    feature_names = [str(col) for col in data_df.columns]

    save_dir = Path(outdir) if outdir is not None else Path.cwd()
    save_dir.mkdir(parents=True, exist_ok=True)

    return HarmonisationInputs(
        data=data_df.to_numpy(dtype=float),
        feature_names=feature_names,
        subject_ids=data_df.index,
        batch=batch,
        mod=mod,
        save_dir=save_dir,
    )


# --------------------------------------------------------------------------------------
# Harmonisation dispatch
# --------------------------------------------------------------------------------------
def _extract_harmonised_array(output, method: str) -> np.ndarray:
    """Pull the harmonised (n_samples, n_features) array out of a method's output."""
    if method == "linear_model":
        return np.asarray(output["Residuals"].to_numpy(dtype=float))
    return np.asarray(output["bayesdata"])


def _run_combat_family(args, inputs: HarmonisationInputs):
    from DiagnoseHarmonisation import HarmonisationFunctions as hf

    gam_opts = None
    if args.method in {"combat_gam", "combat_modular"} and getattr(args, "mean_model", "gam") == "gam":
        gam_opts = {"n_splines": args.n_splines, "degree": args.degree}

    return hf.combat_modular(
        data=inputs.data,
        batch=inputs.batch,
        mod=inputs.mod if inputs.mod.shape[1] > 0 else None,
        mean_model=("gam" if args.method == "combat_gam" else getattr(args, "mean_model", "ols")),
        gam_opts=gam_opts,
        prior_mode=getattr(args, "prior_mode", "global"),
        prior_weight_methods=getattr(args, "prior_weight_methods", None),
        parametric=not args.no_parametric,
        DeltaCorrection=not args.no_delta_correction,
        UseEB=not args.no_use_eb,
        ReferenceBatch=args.reference_batch,
        RegressCovariates=args.regress_covariates,
        GammaCorrection=not args.no_gamma_correction,
        covbat_mode=(args.method == "covbat"),
        return_priors=True,
    )


def _run_linear_model(args, inputs: HarmonisationInputs):
    from DiagnoseHarmonisation import HarmonisationFunctions as hf

    return hf.linearmodelling_harmonisation(
        data=inputs.data,
        batch=inputs.batch,
        covariates=inputs.mod if inputs.mod.shape[1] > 0 else None,
        feature_names=inputs.feature_names,
        model_type=args.model_type,
        batch_as_random=args.batch_as_random,
        subject_col=args.subject_col,
        residuals=args.residuals,
        standardize_continuous=not args.no_standardize_continuous,
        unique_fraction_threshold=args.unique_fraction_threshold,
        missing=args.missing,
        reml=args.reml,
        optimizers=args.optimizers,
        maxiter=args.maxiter,
        min_group_n=args.min_group_n,
        return_models=False,
        verbose=args.verbose,
    )


def run_method(args, inputs: HarmonisationInputs):
    """Dispatch to the requested harmonisation function and return its raw output."""
    if args.method == "linear_model":
        if args.subject_col is not None and args.subject_col not in inputs.mod.columns:
            raise ValueError(
                f"--subject-col '{args.subject_col}' not found among covariate columns."
            )
        return _run_linear_model(args, inputs)
    return _run_combat_family(args, inputs)


# --------------------------------------------------------------------------------------
# Report integration
# --------------------------------------------------------------------------------------
def _write_harmonised_csv(harmonised: np.ndarray, inputs: HarmonisationInputs, args) -> Path:
    out_name = (args.save_data_name or args.method) + "_harmonised.csv"
    out_path = inputs.save_dir / out_name
    df = pd.DataFrame(harmonised, columns=inputs.feature_names, index=inputs.subject_ids)
    df.to_csv(out_path)
    return out_path


def _run_single_report(harmonised: np.ndarray, inputs: HarmonisationInputs, args):
    from DiagnoseHarmonisation import DiagnosticReport
    from DiagnoseHarmonisation.LoggingTool import StatsReporter

    covariates = inputs.mod if inputs.mod.shape[1] > 0 else None
    # CrossSectionalReport returns a data dict (not the report) when save_data=True,
    # so pass in our own reporter to read back the final report_path.
    rep = StatsReporter(save_artifacts=False, save_dir=None)
    DiagnosticReport.CrossSectionalReport(
        harmonised,
        batch=inputs.batch,
        covariates=covariates,
        covariate_names=list(inputs.mod.columns) if covariates is not None else None,
        feature_names=inputs.feature_names,
        save_dir=inputs.save_dir,
        save_data=True,
        save_data_name=args.save_data_name,
        report_name=args.report_name,
        SaveArtifacts=False,
        show=False,
        timestamped_reports=True,
        rep=rep,
    )
    report_path = rep.report_path
    rep.__exit__(None, None, None)
    return report_path


def _run_comparison_report(harmonised: np.ndarray, inputs: HarmonisationInputs, args):
    from DiagnoseHarmonisation import DiagnosticReport

    covariates = inputs.mod if inputs.mod.shape[1] > 0 else None
    datasets = {"Raw": inputs.data, args.method: harmonised}
    report = DiagnosticReport.CrossSectionalComparisonReport(
        datasets,
        batch=inputs.batch,
        covariates=covariates,
        covariate_names=list(inputs.mod.columns) if covariates is not None else None,
        feature_names=inputs.feature_names,
        save_dir=inputs.save_dir,
        save_data=True,
        save_data_name=args.save_data_name,
        report_name=args.report_name,
        SaveArtifacts=False,
        show=False,
        timestamped_reports=True,
    )
    return getattr(report, "report_path", None)


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------
def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", "-d", required=True, help="Path to data CSV/TSV/XLS/XLSX (subjects x features).")
    parser.add_argument("--covariates", "-c", required=True, help="Path to covariates CSV/TSV/XLS/XLSX (one row per subject).")
    parser.add_argument("--data-id-col", default=None, help="Data subject ID column name (defaults to first column).")
    parser.add_argument("--cov-id-col", default=None, help="Covariates subject ID column name (defaults to first column).")
    parser.add_argument("--batch-col", default=None, help="Covariates batch column name (auto-detected if omitted).")
    parser.add_argument(
        "--report",
        choices=["none", "single", "comparison"],
        default="none",
        help="Report to generate from the harmonised data. Default: none (just save a CSV).",
    )
    parser.add_argument("--outdir", default=None, help="Directory to write CSV/report outputs.")
    parser.add_argument("--report-name", default=None, help="Optional name for the generated report.")
    parser.add_argument("--save-data-name", default=None, help="Optional prefix for saved output filenames.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")


def _add_combat_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-parametric", action="store_true", help="Disable parametric empirical Bayes adjustments.")
    parser.add_argument("--no-delta-correction", action="store_true", help="Disable delta (scale) correction.")
    parser.add_argument("--no-use-eb", action="store_true", help="Disable empirical Bayes shrinkage (use raw estimates).")
    parser.add_argument("--reference-batch", default=None, help="Name/index of a reference batch to leave unchanged.")
    parser.add_argument("--regress-covariates", action="store_true", help="Do not re-add covariate effects after harmonisation.")
    parser.add_argument("--no-gamma-correction", action="store_true", help="Disable gamma (mean) correction.")


def _add_gam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-splines", type=int, default=6, help="Number of spline basis functions for GAM mean model. Default: 6.")
    parser.add_argument("--degree", type=int, default=3, help="Spline degree for GAM mean model. Default: 3.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DHarm harmonise",
        description="Run a harmonisation method and optionally generate a diagnostic report.",
    )
    subparsers = parser.add_subparsers(dest="method", required=True)

    combat_p = subparsers.add_parser("combat", help="Classic ComBat harmonisation.")
    _add_common_args(combat_p)
    _add_combat_shared_args(combat_p)

    covbat_p = subparsers.add_parser("covbat", help="CovBat harmonisation (ComBat + covariance correction).")
    _add_common_args(covbat_p)
    _add_combat_shared_args(covbat_p)

    gam_p = subparsers.add_parser("combat_gam", help="ComBat with a GAM/spline mean model.")
    _add_common_args(gam_p)
    _add_combat_shared_args(gam_p)
    _add_gam_args(gam_p)

    modular_p = subparsers.add_parser("combat_modular", help="Full modular ComBat (mean model + prior mode).")
    _add_common_args(modular_p)
    _add_combat_shared_args(modular_p)
    _add_gam_args(modular_p)
    modular_p.add_argument("--mean-model", choices=["ols", "gam"], default="ols", help="Mean model used to estimate/remove covariate effects. Default: ols.")
    modular_p.add_argument("--prior-mode", choices=["global", "local"], default="global", help="Empirical Bayes prior pooling mode. Default: global.")
    modular_p.add_argument(
        "--prior-weight-methods",
        nargs="+",
        choices=PRIOR_WEIGHT_METHOD_CHOICES,
        default=None,
        help="Similarity method(s) used to build local-prior pooling weights (only used with --prior-mode local).",
    )

    lm_p = subparsers.add_parser("linear_model", help="Linear (or linear mixed-effects) model harmonisation.")
    _add_common_args(lm_p)
    lm_p.add_argument("--model-type", choices=["auto", "ols", "mixedlm"], default="auto", help="Model family to fit. Default: auto.")
    lm_p.add_argument("--batch-as-random", action="store_true", help="Treat batch as a random effect (mixedlm only).")
    lm_p.add_argument("--subject-col", default=None, help="Covariate column with subject labels (random intercept for mixedlm).")
    lm_p.add_argument("--residuals", choices=["Batch_only", "Full"], default="Batch_only", help="Which residuals to return. Default: Batch_only.")
    lm_p.add_argument("--no-standardize-continuous", action="store_true", help="Do not standardize continuous covariates before fitting.")
    lm_p.add_argument("--unique-fraction-threshold", type=float, default=0.30, help="Uniqueness fraction above which a numeric covariate is treated as continuous. Default: 0.30.")
    lm_p.add_argument("--missing", choices=["drop", "raise"], default="drop", help="How to handle missing values per feature. Default: drop.")
    lm_p.add_argument("--reml", action="store_true", help="Use REML instead of ML for mixedlm fitting.")
    lm_p.add_argument("--optimizers", nargs="+", default=["lbfgs", "bfgs", "powell"], help="MixedLM optimizers to try in order. Default: lbfgs bfgs powell.")
    lm_p.add_argument("--maxiter", type=int, default=400, help="Maximum optimizer iterations. Default: 400.")
    lm_p.add_argument("--min-group-n", type=int, default=3, help="Minimum samples required per feature to attempt a fit. Default: 3.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inputs = load_harmonisation_inputs(
        data_path=args.data,
        covariates_path=args.covariates,
        data_id_col=args.data_id_col,
        cov_id_col=args.cov_id_col,
        batch_col=args.batch_col,
        outdir=args.outdir,
    )
    if args.verbose:
        print(f"Loaded {inputs.data.shape[0]} subjects x {inputs.data.shape[1]} features.")
        print(f"Using batch column with {inputs.batch.nunique()} unique batches.")

    output = run_method(args, inputs)
    harmonised = _extract_harmonised_array(output, args.method)

    if args.report == "none":
        out_path = _write_harmonised_csv(harmonised, inputs, args)
        print(f"Harmonised data saved to: {out_path}")
    elif args.report == "single":
        report_path = _run_single_report(harmonised, inputs, args)
        print(f"Report saved to: {report_path}")
    elif args.report == "comparison":
        report_path = _run_comparison_report(harmonised, inputs, args)
        print(f"Comparison report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

