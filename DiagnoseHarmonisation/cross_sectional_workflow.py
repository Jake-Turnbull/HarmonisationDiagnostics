from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Optional, Sequence

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

StatusCallback = Optional[Callable[[str], None]]


@dataclass(slots=True)
class CrossSectionalRunConfig:
    data_path: str | Path
    covariates_path: str | Path
    data_id_column: str | None = None
    covariates_id_column: str | None = None
    batch_mode: str = "auto"
    batch_column: str | None = None
    selected_covariates: list[str] | None = None
    output_dir: str | Path | None = None
    report_name: str | None = None
    save_data: bool = True
    save_data_name: str | None = None
    timestamped_reports: bool = True
    allow_missing_batch_in_auto: bool = True


@dataclass(slots=True)
class CrossSectionalInputSummary:
    data_columns: list[str]
    covariates_columns: list[str]
    default_data_id_column: str | None
    default_covariates_id_column: str | None
    auto_batch_column: str | None
    default_covariate_columns: list[str]


@dataclass(slots=True)
class PreparedCrossSectionalInputs:
    data: np.ndarray
    batch: np.ndarray
    covariates: pd.DataFrame
    covariate_names: list[str]
    feature_names: list[str]
    save_dir: Path
    warnings: list[str]
    batch_column_used: str | None


@dataclass(slots=True)
class CrossSectionalRunResult:
    report_path: Path | None
    save_dir: Path
    warnings: list[str]
    batch_column_used: str | None
    covariate_names: list[str]


def fuzzy_find_batch_column(headers: Sequence[str]) -> Optional[int]:
    """Return the zero-based index of a batch-like header, or None."""
    lowered = [header.lower().strip() for header in headers]
    for candidate in BATCH_HEADER_CANDIDATES:
        if candidate in lowered:
            return lowered.index(candidate)
    for index, header in enumerate(lowered):
        for candidate in BATCH_HEADER_CANDIDATES:
            if candidate in header:
                return index
    return None


def read_tabular_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, header=0)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path, header=0)
    raise ValueError(
        f"Unsupported file format for '{file_path}'. Supported: .csv, .xls, .xlsx"
    )


def inspect_cross_sectional_inputs(
    data_path: str | Path,
    covariates_path: str | Path,
) -> CrossSectionalInputSummary:
    data_df = read_tabular_file(data_path)
    cov_df = read_tabular_file(covariates_path)
    _validate_non_empty_table(data_df, "Data")
    _validate_non_empty_table(cov_df, "Covariates")

    default_data_id = _default_id_column(data_df)
    default_cov_id = _default_id_column(cov_df)
    auto_batch_idx = fuzzy_find_batch_column(list(cov_df.columns))
    auto_batch_column = cov_df.columns[auto_batch_idx] if auto_batch_idx is not None else None

    default_covariates = [
        str(column)
        for column in cov_df.columns
        if column not in {default_cov_id, auto_batch_column}
    ]

    return CrossSectionalInputSummary(
        data_columns=[str(column) for column in data_df.columns],
        covariates_columns=[str(column) for column in cov_df.columns],
        default_data_id_column=default_data_id,
        default_covariates_id_column=default_cov_id,
        auto_batch_column=str(auto_batch_column) if auto_batch_column is not None else None,
        default_covariate_columns=default_covariates,
    )


def prepare_cross_sectional_inputs(
    config: CrossSectionalRunConfig,
    *,
    verbose: bool = False,
    status_callback: StatusCallback = None,
) -> PreparedCrossSectionalInputs:
    warnings: list[str] = []
    data_path = Path(config.data_path)
    covariates_path = Path(config.covariates_path)

    _emit_status(status_callback, f"Reading data from: {data_path}")
    data_df = read_tabular_file(data_path)
    _emit_status(status_callback, f"Reading covariates from: {covariates_path}")
    cov_df = read_tabular_file(covariates_path)

    _validate_non_empty_table(data_df, "Data")
    _validate_non_empty_table(cov_df, "Covariates")

    data_id_column = config.data_id_column or _default_id_column(data_df)
    covariates_id_column = config.covariates_id_column or _default_id_column(cov_df)

    if verbose and config.data_id_column is None and data_id_column is not None:
        _emit_status(status_callback, f"Assuming data subject ID column is '{data_id_column}'.")
    if verbose and config.covariates_id_column is None and covariates_id_column is not None:
        _emit_status(
            status_callback,
            f"Assuming covariates subject ID column is '{covariates_id_column}'.",
        )

    data_sub, cov_sub, id_warnings = validate_subject_ids(
        data_df,
        cov_df,
        data_id_column,
        covariates_id_column,
    )
    warnings.extend(id_warnings)
    for message in id_warnings:
        _emit_status(status_callback, f"Warning: {message}")

    batch_column_used = resolve_batch_column(
        cov_sub.columns,
        batch_mode=config.batch_mode,
        batch_column=config.batch_column,
        allow_missing_batch_in_auto=config.allow_missing_batch_in_auto,
    )
    if batch_column_used is None:
        _emit_status(
            status_callback,
            "No batch column selected. Running in single-batch mode.",
        )
    else:
        _emit_status(status_callback, f"Using batch column: '{batch_column_used}'")

    data_sub = data_sub.apply(pd.to_numeric, errors="coerce")
    feature_names = [str(column) for column in data_sub.columns]
    data_matrix = data_sub.astype(float).to_numpy()

    covariates_df, batch_labels, covariate_warnings = build_covariates_dataframe(
        cov_sub,
        data_id_column=data_id_column,
        covariates_id_column=covariates_id_column,
        batch_column=batch_column_used,
        selected_covariates=config.selected_covariates,
    )
    warnings.extend(covariate_warnings)
    for message in covariate_warnings:
        _emit_status(status_callback, f"Warning: {message}")

    if covariates_df.empty:
        raise ValueError(
            "At least one covariate column must be selected for the cross-sectional report."
        )

    save_dir = Path(config.output_dir) if config.output_dir is not None else Path.cwd()
    save_dir.mkdir(parents=True, exist_ok=True)

    return PreparedCrossSectionalInputs(
        data=data_matrix,
        batch=batch_labels.to_numpy(),
        covariates=covariates_df,
        covariate_names=list(covariates_df.columns),
        feature_names=feature_names,
        save_dir=save_dir,
        warnings=warnings,
        batch_column_used=batch_column_used,
    )


def run_cross_sectional_report(
    config: CrossSectionalRunConfig,
    *,
    verbose: bool = False,
    status_callback: StatusCallback = None,
) -> CrossSectionalRunResult:
    prepared = prepare_cross_sectional_inputs(
        config,
        verbose=verbose,
        status_callback=status_callback,
    )

    from DiagnoseHarmonisation import DiagnosticReport

    _emit_status(status_callback, "Running cross-sectional report...")
    report = DiagnosticReport.CrossSectionalReport(
        prepared.data,
        batch=prepared.batch,
        covariates=prepared.covariates,
        covariate_names=prepared.covariate_names,
        feature_names=prepared.feature_names,
        save_dir=prepared.save_dir,
        save_data=config.save_data,
        report_name=config.report_name,
        save_data_name=config.save_data_name,
        SaveArtifacts=False,
        show=False,
        timestamped_reports=config.timestamped_reports,
        ratio_type="rest",
        UMAP_embedding=True,
        UMAP_tuning="auto",
    )
    report_path = getattr(report, "report_path", None)
    if report_path is not None:
        report_path = Path(report_path)
        _emit_status(status_callback, f"Report saved to: {report_path}")

    return CrossSectionalRunResult(
        report_path=report_path,
        save_dir=prepared.save_dir,
        warnings=prepared.warnings,
        batch_column_used=prepared.batch_column_used,
        covariate_names=prepared.covariate_names,
    )


def validate_subject_ids(
    data_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    data_id_col: Optional[str],
    cov_id_col: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Ensure subject IDs overlap and return aligned tables plus warnings."""
    if data_id_col is None or cov_id_col is None:
        raise ValueError("Both data_id_col and cov_id_col must be provided or inferred.")
    if data_id_col not in data_df.columns:
        raise ValueError(f"Subject ID column '{data_id_col}' not found in data file.")
    if cov_id_col not in cov_df.columns:
        raise ValueError(f"Subject ID column '{cov_id_col}' not found in covariates file.")

    data_ids = set(data_df[data_id_col].astype(str))
    cov_ids = set(cov_df[cov_id_col].astype(str))
    common = data_ids & cov_ids
    if not common:
        raise ValueError("No matching subject IDs found between data and covariates files.")

    warnings: list[str] = []
    overlap_threshold = max(len(data_ids), len(cov_ids)) * 0.5
    if len(common) < overlap_threshold:
        warnings.append(
            f"only {len(common)} subjects overlap between data ({len(data_ids)}) and covariates ({len(cov_ids)}); proceeding with the shared intersection"
        )

    common_sorted = sorted(common)
    data_sub = data_df[data_df[data_id_col].astype(str).isin(common_sorted)].copy()
    cov_sub = cov_df[cov_df[cov_id_col].astype(str).isin(common_sorted)].copy()

    data_sub.set_index(data_id_col, inplace=True)
    cov_sub.set_index(cov_id_col, inplace=True)
    data_sub = data_sub.loc[cov_sub.index.intersection(data_sub.index)]
    cov_sub = cov_sub.loc[data_sub.index]
    return data_sub, cov_sub, warnings


def resolve_batch_column(
    headers: Sequence[str],
    *,
    batch_mode: str,
    batch_column: str | None,
    allow_missing_batch_in_auto: bool,
) -> str | None:
    header_names = [str(header).strip() for header in headers]
    if batch_mode == "selected":
        if batch_column is None:
            raise ValueError("A batch column must be selected when batch_mode='selected'.")
        if batch_column not in header_names:
            raise ValueError(f"Batch column '{batch_column}' not found in covariates file.")
        return batch_column

    if batch_mode == "none":
        return None

    if batch_mode != "auto":
        raise ValueError(f"Unsupported batch mode '{batch_mode}'.")

    batch_idx = fuzzy_find_batch_column(header_names)
    if batch_idx is None:
        if allow_missing_batch_in_auto:
            return None
        raise ValueError(
            "No batch-like column was detected automatically. Please choose a batch column or explicitly select no batch column."
        )
    return header_names[batch_idx]


def build_covariates_dataframe(
    cov_sub: pd.DataFrame,
    *,
    data_id_column: str,
    covariates_id_column: str,
    batch_column: str | None,
    selected_covariates: Sequence[str] | None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    warnings: list[str] = []
    covariates_df = cov_sub.copy()
    covariates_df.columns = [str(column).strip() for column in covariates_df.columns]

    if batch_column is not None:
        batch_column = batch_column.strip()
        if batch_column not in covariates_df.columns:
            raise ValueError(
                f"Batch column '{batch_column}' not present in covariates after subject alignment."
            )
        batch_series = covariates_df.pop(batch_column).astype(str)
    else:
        batch_series = pd.Series(
            ["single_batch"] * covariates_df.shape[0],
            index=covariates_df.index,
            dtype="string",
        )

    covariates_df.drop(
        columns=[data_id_column, covariates_id_column],
        errors="ignore",
        inplace=True,
    )

    if selected_covariates is None:
        selected = list(covariates_df.columns)
    else:
        selected = [str(column).strip() for column in selected_covariates]
        missing = [column for column in selected if column not in covariates_df.columns]
        if missing:
            raise ValueError(
                f"Selected covariate columns were not found in the covariates file: {missing}"
            )
        covariates_df = covariates_df.loc[:, selected]

    cols_to_drop: list[str] = []
    for column in list(covariates_df.columns):
        try:
            if covariates_df[column].astype(str).equals(batch_series.astype(str)):
                cols_to_drop.append(column)
        except Exception:
            continue
    if cols_to_drop:
        covariates_df.drop(columns=cols_to_drop, inplace=True)
        warnings.append(
            f"removed covariate columns identical to batch labels: {', '.join(cols_to_drop)}"
        )

    covariates_df.columns = _sanitize_column_names(list(covariates_df.columns))
    return covariates_df, batch_series, warnings


def _default_id_column(dataframe: pd.DataFrame) -> str | None:
    return str(dataframe.columns[0]) if len(dataframe.columns) > 0 else None


def _sanitize_column_names(columns: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    counts: dict[str, int] = {}
    for column in columns:
        base = re.sub(r"[\s\.,()]+", "_", str(column))
        base = base.strip("_") or "covariate"
        count = counts.get(base, 0)
        counts[base] = count + 1
        sanitized.append(base if count == 0 else f"{base}_{count + 1}")
    return sanitized


def _validate_non_empty_table(dataframe: pd.DataFrame, label: str) -> None:
    if dataframe.shape[0] == 0 or dataframe.shape[1] == 0:
        raise ValueError(f"{label} file appears empty or malformed.")


def _emit_status(status_callback: StatusCallback, message: str) -> None:
    if status_callback is not None:
        status_callback(message)
