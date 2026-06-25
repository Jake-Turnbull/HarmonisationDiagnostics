from __future__ import annotations

from pathlib import Path

import pandas as pd

from DiagnoseHarmonisation.gui_longitudinal import build_gui_run_config
from DiagnoseHarmonisation.longitudinal_workflow import LongitudinalRunConfig, prepare_longitudinal_inputs


TEST_RESULTS_DIR = Path(__file__).resolve().parents[1] / "TestResults"


def test_build_gui_run_config_allows_empty_covariates():
    config = build_gui_run_config(
        data_path=str(TEST_RESULTS_DIR / "data.csv"),
        covariates_path="",
        output_dir=str(TEST_RESULTS_DIR),
        subject_id_column="subject",
        timepoint_column="timepoint",
        batch_column="batch",
        selected_features=["feature1", "feature2"],
        selected_covariates=[],
        report_name="test_report",
        covariates_subject_id_column="",
        covariates_timepoint_column="",
    )

    assert config.covariates_path is None
    assert config.selected_covariates == []
    assert config.covariate_names is None
    assert config.report_name == "test_report"


def test_prepare_longitudinal_inputs_without_covariates(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "timepoint": [1, 2, 1, 2],
            "batch": ["A", "A", "B", "B"],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [1.1, 1.2, 1.3, 1.4],
        }
    ).to_csv(data_path, index=False)

    config = LongitudinalRunConfig(
        data_path=str(data_path),
        subject_id_column="subject",
        timepoint_column="timepoint",
        batch_column="batch",
        selected_features=["feature1", "feature2"],
        covariates_path=None,
        covariates_subject_id_column=None,
        covariates_timepoint_column=None,
        selected_covariates=None,
        covariate_names=None,
        output_dir=str(TEST_RESULTS_DIR / "longitudinal_output"),
        report_name="test_report",
        save_data=False,
        save_data_name=None,
        timestamped_reports=False,
    )

    prepared = prepare_longitudinal_inputs(config)

    assert prepared.data.shape == (4, 2)
    assert prepared.subject_ids == ["s1", "s1", "s2", "s2"]
    assert prepared.timepoints == ["1", "2", "1", "2"]
    assert prepared.batch == ["A", "A", "B", "B"]
    assert prepared.covariates is None
    assert prepared.covariate_names == []
