from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


StatusCallback = Optional[Callable[[str], None]]


@dataclass(slots=True)
class LongitudinalRunConfig:
    data_path: str | Path

    subject_id_column: str
    timepoint_column: str
    batch_column: str
    selected_features: list[str]

    covariates_path: str | Path | None = None
    covariates_subject_id_column: str | None = None
    covariates_timepoint_column: str | None = None
    selected_covariates: list[str] | None = None
    covariate_names: list[str] | None = None

    output_dir: str | Path | None = None
    report_name: str | None = None
    save_data: bool = False
    save_data_name: str | None = None
    timestamped_reports: bool = True


@dataclass(slots=True)
class LongitudinalInputSummary:
    data_columns: list[str]
    covariates_columns: list[str]


@dataclass(slots=True)
class PreparedLongitudinalInputs:
    data: np.ndarray
    batch: list[str]
    subject_ids: list[str]
    timepoints: list[str]
    covariates: dict[str, list] | None
    covariate_names: list[str]
    feature_names: list[str]
    save_dir: Path
    warnings: list[str]
    subject_id_column_used: str
    timepoint_column_used: str
    batch_column_used: str


@dataclass(slots=True)
class LongitudinalRunResult:
    report_path: Path | None
    save_dir: Path
    warnings: list[str]
    subject_id_column_used: str
    timepoint_column_used: str
    batch_column_used: str
    covariate_names: list[str]
    feature_names: list[str]


def read_tabular_file(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, header=0)

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path, header=0)

    raise ValueError(
        f"Unsupported file format for '{file_path}'. Supported formats: .csv, .xls, .xlsx"
    )


def inspect_longitudinal_inputs(
    data_path: str | Path,
    covariates_path: str | Path | None = None,
) -> LongitudinalInputSummary:
    data_df = read_tabular_file(data_path)
    _validate_non_empty_table(data_df, "Data")
    data_df.columns = [str(column).strip() for column in data_df.columns]

    if covariates_path is not None:
        cov_df = read_tabular_file(covariates_path)
        _validate_non_empty_table(cov_df, "Covariates")
        cov_df.columns = [str(column).strip() for column in cov_df.columns]
    else:
        cov_df = pd.DataFrame()

    return LongitudinalInputSummary(
        data_columns=[str(column) for column in data_df.columns],
        covariates_columns=[str(column) for column in cov_df.columns],
    )


def prepare_longitudinal_inputs(
    config: LongitudinalRunConfig,
    *,
    verbose: bool = False,
    status_callback: StatusCallback = None,
) -> PreparedLongitudinalInputs:
    warnings: list[str] = []

    data_path = Path(config.data_path)
    _emit_status(status_callback, f"Reading longitudinal data from: {data_path}")

    data_df = read_tabular_file(data_path)
    _validate_non_empty_table(data_df, "Data")
    data_df.columns = [str(column).strip() for column in data_df.columns]

    feature_names = [str(column).strip() for column in config.selected_features]

    _require_columns(
        dataframe=data_df,
        columns=[
            config.subject_id_column,
            config.timepoint_column,
            config.batch_column,
            *feature_names,
        ],
        label="data",
    )

    _validate_unique_longitudinal_keys(
        data_df,
        id_column=config.subject_id_column,
        timepoint_column=config.timepoint_column,
        label="data",
    )

    data_matrix = (
        data_df.loc[:, feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    subject_ids = data_df[config.subject_id_column].astype(str).tolist()
    timepoints = _series_to_clean_str_list(data_df[config.timepoint_column])
    batch = data_df[config.batch_column].astype(str).tolist()

    covariates_payload: dict[str, list] | None = None
    covariate_names: list[str] = []

    if config.selected_covariates is not None and len(config.selected_covariates) > 0:
        if config.covariates_path is None:
            covariates_df = _build_covariates_from_data(
                data_df=data_df,
                selected_covariates=config.selected_covariates,
            )
        else:
            covariates_df = _build_covariates_from_separate_file(
                data_df=data_df,
                covariates_path=config.covariates_path,
                data_subject_id_column=config.subject_id_column,
                data_timepoint_column=config.timepoint_column,
                covariates_subject_id_column=config.covariates_subject_id_column,
                covariates_timepoint_column=config.covariates_timepoint_column,
                selected_covariates=config.selected_covariates,
                status_callback=status_callback,
            )

        covariates_payload, covariate_names = _covariates_dataframe_to_report_dict(
            covariates_df,
            requested_names=config.covariate_names,
        )

    save_dir = Path(config.output_dir) if config.output_dir is not None else Path.cwd()
    save_dir.mkdir(parents=True, exist_ok=True)

    _emit_status(status_callback, f"Using subject ID column: '{config.subject_id_column}'")
    _emit_status(status_callback, f"Using timepoint column: '{config.timepoint_column}'")
    _emit_status(status_callback, f"Using batch column: '{config.batch_column}'")
    _emit_status(status_callback, f"Using {len(feature_names)} feature columns.")

    if covariates_payload is None:
        _emit_status(status_callback, "No covariates selected.")
    else:
        _emit_status(
            status_callback,
            f"Using covariates: {', '.join(covariate_names)}",
        )

    return PreparedLongitudinalInputs(
        data=data_matrix,
        batch=batch,
        subject_ids=subject_ids,
        timepoints=timepoints,
        covariates=covariates_payload,
        covariate_names=covariate_names,
        feature_names=feature_names,
        save_dir=save_dir,
        warnings=warnings,
        subject_id_column_used=config.subject_id_column,
        timepoint_column_used=config.timepoint_column,
        batch_column_used=config.batch_column,
    )


def run_longitudinal_report(
    config: LongitudinalRunConfig,
    *,
    verbose: bool = False,
    status_callback: StatusCallback = None,
) -> LongitudinalRunResult:
    prepared = prepare_longitudinal_inputs(
        config,
        verbose=verbose,
        status_callback=status_callback,
    )

    from DiagnoseHarmonisation import DiagnosticReport

    _emit_status(status_callback, "Running longitudinal report...")

    if config.save_data and prepared.covariates is not None:
        raise ValueError(
            "save_data=True is currently not supported with longitudinal covariates "
            "because DiagnosticReport.LongitudinalReport expects dict covariates for modelling "
            "but uses covariates.shape when saving input data. Run without --save-data, "
            "or patch DiagnosticReport.LongitudinalReport to handle dict covariates in its save_data block."
        )

    report_callable = DiagnosticReport.LongitudinalReport
    accepted_parameters = set(inspect.signature(report_callable).parameters)

    report_kwargs = {
        "covariates": prepared.covariates,
        "features": prepared.feature_names,
        "report_name": config.report_name,
        "timestamped_reports": config.timestamped_reports,
    }

    optional_kwargs = {
        "save_dir": prepared.save_dir,
        "save_data": config.save_data,
        "save_data_name": config.save_data_name,
        "show": False,
    }

    for key, value in optional_kwargs.items():
        if key in accepted_parameters:
            report_kwargs[key] = value

    report = report_callable(
        prepared.data,
        prepared.batch,
        prepared.subject_ids,
        prepared.timepoints,
        **report_kwargs,
    )

    report_path = getattr(report, "report_path", None)

    if report_path is not None:
        report_path = Path(report_path)
        _emit_status(status_callback, f"Report saved to: {report_path}")

    return LongitudinalRunResult(
        report_path=report_path,
        save_dir=prepared.save_dir,
        warnings=prepared.warnings,
        subject_id_column_used=prepared.subject_id_column_used,
        timepoint_column_used=prepared.timepoint_column_used,
        batch_column_used=prepared.batch_column_used,
        covariate_names=prepared.covariate_names,
        feature_names=prepared.feature_names,
    )


def _build_covariates_from_data(
    *,
    data_df: pd.DataFrame,
    selected_covariates: Sequence[str],
) -> pd.DataFrame:
    selected = [str(column).strip() for column in selected_covariates]

    _require_columns(
        dataframe=data_df,
        columns=selected,
        label="data",
    )

    return data_df.loc[:, selected].copy()


def _build_covariates_from_separate_file(
    *,
    data_df: pd.DataFrame,
    covariates_path: str | Path,
    data_subject_id_column: str,
    data_timepoint_column: str,
    covariates_subject_id_column: str | None,
    covariates_timepoint_column: str | None,
    selected_covariates: Sequence[str],
    status_callback: StatusCallback = None,
) -> pd.DataFrame:
    if covariates_subject_id_column is None:
        raise ValueError(
            "When --covariates is provided, --cov-subject-id-col is required."
        )

    if covariates_timepoint_column is None:
        raise ValueError(
            "When --covariates is provided, --cov-timepoint-col is required."
        )

    covariates_path = Path(covariates_path)
    _emit_status(status_callback, f"Reading covariates from: {covariates_path}")

    cov_df = read_tabular_file(covariates_path)
    _validate_non_empty_table(cov_df, "Covariates")
    cov_df.columns = [str(column).strip() for column in cov_df.columns]

    selected = [str(column).strip() for column in selected_covariates]

    _require_columns(
        dataframe=cov_df,
        columns=[
            covariates_subject_id_column,
            covariates_timepoint_column,
            *selected,
        ],
        label="covariates",
    )

    _validate_unique_longitudinal_keys(
        cov_df,
        id_column=covariates_subject_id_column,
        timepoint_column=covariates_timepoint_column,
        label="covariates",
    )

    left = data_df[[data_subject_id_column, data_timepoint_column]].copy()
    left["_original_row_order"] = np.arange(len(left))

    right = cov_df[
        [
            covariates_subject_id_column,
            covariates_timepoint_column,
            *selected,
        ]
    ].copy()

    merged = left.merge(
        right,
        left_on=[data_subject_id_column, data_timepoint_column],
        right_on=[covariates_subject_id_column, covariates_timepoint_column],
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(data_df):
        raise ValueError(
            "Covariate alignment changed the number of rows. "
            "Check subject/timepoint uniqueness."
        )

    unmatched = merged[covariates_subject_id_column].isna().sum()
    if unmatched > 0:
        raise ValueError(
            f"{int(unmatched)} rows had no matching covariate row after "
            "subject/timepoint alignment."
        )

    merged = merged.sort_values("_original_row_order")

    return merged.loc[:, selected].copy()


def _covariates_dataframe_to_report_dict(
    covariates_df: pd.DataFrame | None,
    *,
    requested_names: Sequence[str] | None = None,
) -> tuple[dict[str, list] | None, list[str]]:
    if covariates_df is None or covariates_df.shape[1] == 0:
        return None, []

    source_columns = [str(column) for column in covariates_df.columns]

    if requested_names is not None:
        output_names = [str(name).strip() for name in requested_names]

        if len(output_names) != len(source_columns):
            raise ValueError(
                "Number of covariate display names must match number of selected covariates. "
                f"Got {len(output_names)} names for {len(source_columns)} covariates."
            )
    else:
        output_names = source_columns

    covariates: dict[str, list] = {}

    for source_column, output_name in zip(source_columns, output_names):
        series = covariates_df[source_column]

        numeric_series = pd.to_numeric(series, errors="coerce")
        non_missing_original = series.notna()
        converted_all_non_missing = numeric_series[non_missing_original].notna().all()

        if converted_all_non_missing:
            covariates[output_name] = numeric_series.astype(float).tolist()
        else:
            covariates[output_name] = series.astype(str).tolist()

    return covariates, output_names


def _series_to_clean_str_list(series: pd.Series) -> list[str]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing_original = series.notna()
    converted_all_non_missing = numeric[non_missing_original].notna().all()

    if converted_all_non_missing:
        non_missing_numeric = numeric.dropna().to_numpy(dtype=float)

        if len(non_missing_numeric) > 0:
            integer_like = np.isclose(
                non_missing_numeric,
                non_missing_numeric.astype(int),
            ).all()

            if integer_like:
                return numeric.astype("Int64").astype(str).tolist()

    return series.astype(str).tolist()


def _validate_unique_longitudinal_keys(
    dataframe: pd.DataFrame,
    *,
    id_column: str,
    timepoint_column: str,
    label: str,
) -> None:
    duplicated = dataframe.duplicated(subset=[id_column, timepoint_column])

    if duplicated.any():
        n_duplicates = int(duplicated.sum())
        raise ValueError(
            f"{label} contains {n_duplicates} duplicate subject/timepoint rows."
        )


def _require_columns(
    *,
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [str(column) for column in columns if str(column) not in dataframe.columns]

    if missing:
        raise ValueError(f"Missing required columns in {label} file: {missing}")


def _validate_non_empty_table(dataframe: pd.DataFrame, label: str) -> None:
    if dataframe.shape[0] == 0 or dataframe.shape[1] == 0:
        raise ValueError(f"{label} file appears empty or malformed.")


def _emit_status(status_callback: StatusCallback, message: str) -> None:
    if status_callback is not None:
        status_callback(message)