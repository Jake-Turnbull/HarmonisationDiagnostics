from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from DiagnoseHarmonisation import cli
from DiagnoseHarmonisation.cross_sectional_workflow import (
    CrossSectionalRunConfig,
    inspect_cross_sectional_inputs,
    prepare_cross_sectional_inputs,
    read_tabular_file,
    run_cross_sectional_report,
)
from DiagnoseHarmonisation.gui import (
    AUTO_DETECT_BATCH,
    NO_BATCH_COLUMN,
    build_gui_run_config,
)


TEST_RESULTS_DIR = Path(__file__).resolve().parents[1] / "TestResults"


def _write_example_inputs(base_path: Path) -> tuple[Path, Path]:
    base_path.mkdir(parents=True, exist_ok=True)
    data_path = base_path / "data.csv"
    covariates_path = base_path / "covariates.csv"

    pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3"],
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [4.0, 5.0, 6.0],
        }
    ).to_csv(data_path, index=False)

    pd.DataFrame(
        {
            "subject_id": ["s1", "s2", "s3"],
            "batch": ["A", "B", "A"],
            "age": [60, 61, 62],
            "sex": [0, 1, 1],
        }
    ).to_csv(covariates_path, index=False)
    return data_path, covariates_path


def test_inspect_cross_sectional_inputs_populates_defaults(test_results_dir):
    data_path, covariates_path = _write_example_inputs(test_results_dir / "cross_sectional_inputs")

    summary = inspect_cross_sectional_inputs(data_path, covariates_path)

    assert summary.default_data_id_column == "subject_id"
    assert summary.default_covariates_id_column == "subject_id"
    assert summary.auto_batch_column == "batch"
    assert summary.default_covariate_columns == ["age", "sex"]


def test_prepare_cross_sectional_inputs_removes_batch_duplicate_covariates(test_results_dir):
    data_path = test_results_dir / "cross_sectional_inputs" / "data.csv"
    covariates_path = test_results_dir / "cross_sectional_inputs" / "covariates.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "feature_1": [1.0, 2.0, 3.0],
        }
    ).to_csv(data_path, index=False)

    pd.DataFrame(
        {
            "subject": ["s1", "s2", "s3"],
            "batch": ["site_a", "site_b", "site_a"],
            "site_copy": ["site_a", "site_b", "site_a"],
            "Age (years)": [50, 55, 60],
        }
    ).to_csv(covariates_path, index=False)

    config = CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=covariates_path,
        batch_mode="selected",
        batch_column="batch",
        selected_covariates=["site_copy", "Age (years)"],
    )

    prepared = prepare_cross_sectional_inputs(config)

    assert prepared.covariate_names == ["Age_years"]
    assert prepared.batch_column_used == "batch"
    assert any("site_copy" in warning for warning in prepared.warnings)


def test_prepare_cross_sectional_inputs_raises_on_missing_overlap(test_results_dir):
    data_path = test_results_dir / "cross_sectional_inputs_missing" / "data.csv"
    covariates_path = test_results_dir / "cross_sectional_inputs_missing" / "covariates.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "subject_id": ["s1", "s2"],
            "feature_1": [1.0, 2.0],
        }
    ).to_csv(data_path, index=False)

    pd.DataFrame(
        {
            "subject_id": ["x1", "x2"],
            "batch": ["A", "B"],
            "age": [40, 41],
        }
    ).to_csv(covariates_path, index=False)

    config = CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=covariates_path,
    )

    with pytest.raises(ValueError, match="No matching subject IDs"):
        prepare_cross_sectional_inputs(config)


def test_read_tabular_file_dispatches_excel_reader(monkeypatch, test_results_dir):
    called = {}

    def fake_read_excel(path, header=0):
        called["path"] = Path(path)
        called["header"] = header
        return pd.DataFrame({"subject_id": ["s1"], "batch": ["A"], "age": [50]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = read_tabular_file(test_results_dir / "cross_sectional_inputs" / "covariates.xlsx")

    assert list(result.columns) == ["subject_id", "batch", "age"]
    assert called["path"].name == "covariates.xlsx"
    assert called["header"] == 0


def test_read_tabular_file_rejects_unsupported_extension(test_results_dir):
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tabular_file(test_results_dir / "cross_sectional_inputs" / "data.tsv")


def test_run_cross_sectional_report_invokes_report_backend(monkeypatch, test_results_dir):
    data_path, covariates_path = _write_example_inputs(test_results_dir / "cross_sectional_report_inputs")
    captured = {}

    class DummyReport:
        report_path = test_results_dir / "example_report.html"

    def fake_cross_sectional_report(data, **kwargs):
        captured["shape"] = data.shape
        captured["kwargs"] = kwargs
        return DummyReport()

    from DiagnoseHarmonisation import DiagnosticReport

    monkeypatch.setattr(DiagnosticReport, "CrossSectionalReport", fake_cross_sectional_report)

    config = CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=covariates_path,
        output_dir=test_results_dir,
        report_name="example_report",
        timestamped_reports=False,
    )

    result = run_cross_sectional_report(config)

    assert captured["shape"] == (3, 2)
    assert captured["kwargs"]["covariate_names"] == ["age", "sex"]
    assert captured["kwargs"]["timestamped_reports"] is False
    assert result.report_path == test_results_dir / "example_report.html"


def test_run_cross_sectional_report_allows_empty_covariates(monkeypatch, test_results_dir):
    data_path, covariates_path = _write_example_inputs(test_results_dir / "cross_sectional_report_no_covariates")
    captured = {}

    class DummyReport:
        report_path = test_results_dir / "example_report_no_covariates.html"

    def fake_cross_sectional_report(data, **kwargs):
        captured["kwargs"] = kwargs
        return DummyReport()

    from DiagnoseHarmonisation import DiagnosticReport

    monkeypatch.setattr(DiagnosticReport, "CrossSectionalReport", fake_cross_sectional_report)

    config = CrossSectionalRunConfig(
        data_path=data_path,
        covariates_path=covariates_path,
        output_dir=test_results_dir,
        report_name="example_report_no_covariates",
        timestamped_reports=False,
        selected_covariates=[],
    )

    result = run_cross_sectional_report(config)

    assert captured["kwargs"]["covariates"] is None
    assert captured["kwargs"]["covariate_names"] is None
    assert result.report_path == test_results_dir / "example_report_no_covariates.html"
    assert any("LMM diagnostics will be skipped" in warning for warning in result.warnings)


def test_run_pipeline_from_cli_uses_selected_batch_column(monkeypatch, test_results_dir):
    data_path, covariates_path = _write_example_inputs(test_results_dir / "cross_sectional_cli_inputs")
    captured = {}

    def fake_run_cross_sectional_report(config, *, verbose=False, status_callback=None):
        captured["config"] = config
        captured["verbose"] = verbose
        return "ok"

    monkeypatch.setattr(cli, "run_cross_sectional_report", fake_run_cross_sectional_report)

    result = cli.run_pipeline_from_cli(
        str(data_path),
        str(covariates_path),
        batch_col_index=2,
        verbose=True,
    )

    assert result == "ok"
    assert captured["config"].batch_mode == "selected"
    assert captured["config"].batch_column == "batch"
    assert captured["verbose"] is True


def test_cli_gui_command_dispatches_launcher(monkeypatch):
    called = {"count": 0}

    def fake_launch_gui():
        called["count"] += 1

    monkeypatch.setattr(cli, "launch_gui", fake_launch_gui)

    result = cli.main(["gui"])

    assert result == 0
    assert called["count"] == 1


def test_build_gui_run_config_smoke():
    config = build_gui_run_config(
        data_path=str(TEST_RESULTS_DIR / "data.csv"),
        covariates_path=str(TEST_RESULTS_DIR / "covariates.csv"),
        output_dir=str(TEST_RESULTS_DIR),
        data_id_column="subject_id",
        covariates_id_column="subject_id",
        batch_selection=AUTO_DETECT_BATCH,
        selected_covariates=["age", "sex"],
        report_name="My Report",
        save_data=True,
        timestamped_reports=False,
    )

    assert config.batch_mode == "auto"
    assert config.batch_column is None
    assert config.allow_missing_batch_in_auto is False
    assert config.report_name == "My Report"


def test_build_gui_run_config_supports_explicit_no_batch():
    config = build_gui_run_config(
        data_path=str(TEST_RESULTS_DIR / "data.csv"),
        covariates_path=str(TEST_RESULTS_DIR / "covariates.csv"),
        output_dir=str(TEST_RESULTS_DIR),
        data_id_column="subject_id",
        covariates_id_column="subject_id",
        batch_selection=NO_BATCH_COLUMN,
        selected_covariates=["age"],
        report_name="",
        save_data=False,
        timestamped_reports=True,
    )

    assert config.batch_mode == "none"
    assert config.batch_column is None
    assert config.report_name is None
    assert config.save_data is False
