from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPORT_OUTPUT_DIR = Path(__file__).resolve().parent / "iqm_harmonisation_outputs"


def _configure_headless_reporting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))

    import matplotlib

    matplotlib.use("Agg", force=True)


def _simulate_longitudinal_iqm_data(
    n_subjects: int = 30,
    n_timepoints: int = 3,
    n_idps: int = 3,
    n_qcs: int = 4,
    seed: int = 27,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(seed)

    batches = np.array(["Siemens", "Philips", "GE", "Magnetom"])
    batch_effects = {
        "Siemens": 1.4,
        "Philips": 0.5,
        "GE": -0.6,
        "Magnetom": -1.2,
    }

    idp_list = [f"IDP_{i + 1}" for i in range(n_idps)]
    qc_list = [f"QC_{i + 1}" for i in range(n_qcs)]
    rows: list[dict[str, object]] = []

    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        baseline_age = rng.uniform(25, 70)
        sex = int(rng.integers(0, 2))
        subject_intercept = rng.normal(0.0, 1.0)
        dominant_batch = rng.choice(batches)

        for tp in range(n_timepoints):
            timepoint = f"tp-{tp + 1}"
            age = baseline_age + tp * rng.uniform(0.9, 1.1)
            batch = dominant_batch if rng.random() < 0.85 else rng.choice(batches)
            batch_shift = batch_effects[batch]

            row: dict[str, object] = {
                "subjectID": subject_id,
                "batch": batch,
                "timepoint": timepoint,
                "age": age,
                "sex": str(sex),
            }

            qc_values: dict[str, float] = {}
            for qc_idx, qc_name in enumerate(qc_list):
                qc_value = (
                    0.7 * batch_shift
                    + 0.018 * (age - 50)
                    + 0.2 * sex
                    + rng.normal(0.0, 0.35)
                )
                qc_value += rng.normal(0.0, 0.12) * (qc_idx + 1)
                row[qc_name] = qc_value
                qc_values[qc_name] = qc_value

            for idp_idx, idp_name in enumerate(idp_list):
                qc_component = (
                    0.9 * qc_values["QC_1"]
                    - 0.55 * qc_values["QC_2"]
                    + 0.25 * qc_values["QC_3"]
                )
                idp_value = (
                    100.0
                    + subject_intercept
                    - 0.25 * (age - 50)
                    + 1.2 * sex
                    + 2.2 * batch_shift
                    + 0.2 * tp
                    + qc_component
                    + rng.normal(0.0, 0.8)
                )
                idp_value += idp_idx * 2.5 + rng.normal(0.0, 0.15)
                row[idp_name] = idp_value

            rows.append(row)

    return pd.DataFrame(rows), idp_list, qc_list


def _run_iqm_harmonisation(
    frame: pd.DataFrame,
    idp_list: list[str],
    qc_list: list[str],
    output_dir: Path,
):
    from DiagnoseHarmonisation import HarmonisationFunctions

    output_dir.mkdir(parents=True, exist_ok=True)

    return HarmonisationFunctions.lme_iqm_harmonise(
        data=frame.copy(),
        idp_list=idp_list,
        qc_list=qc_list,
        preserve_covars=("age",),
        adjust_covars=("timepoint",),
        categorical_covars=("timepoint", "batch", "sex"),
        batch_col="batch",
        subject_col="subjectID",
        age_col="age",
        apply_pca=False,
        outfilename=str(output_dir / "iqm_harmonised.csv"),
        summary_csv=str(output_dir / "iqm_summary.csv"),
        additive_detail_csv=str(output_dir / "iqm_additive_details.csv"),
        multiplicative_detail_csv=str(output_dir / "iqm_multiplicative_details.csv"),
        selected_additive_qcs_csv=str(output_dir / "iqm_selected_additive_qcs.csv"),
        selected_multiplicative_qcs_csv=str(output_dir / "iqm_selected_multiplicative_qcs.csv"),
        allow_ols_fallback=True,
        optimizer_order=("lbfgs",),
        maxiter=500,
        verbose_model_fits=False,
    )


def _run_longitudinal_report(
    frame: pd.DataFrame,
    feature_cols: list[str],
    report_name: str,
    save_dir: Path,
) -> Path:
    from DiagnoseHarmonisation import DiagnosticReport

    save_dir.mkdir(parents=True, exist_ok=True)

    DiagnosticReport.LongitudinalReport(
        data=frame.loc[:, feature_cols].to_numpy(dtype=float),
        batch=frame["batch"].to_numpy(),
        subject_ids=frame["subjectID"].tolist(),
        timepoints=frame["timepoint"].tolist(),
        covariates={
            "age": frame["age"].tolist(),
            "sex": frame["sex"].tolist(),
        },
        features=feature_cols,
        save_dir=str(save_dir),
        report_name=report_name,
        show=False,
        timestamped_reports=False,
    )

    report_path = save_dir / f"{report_name}.html"
    assert report_path.exists(), f"Expected report was not generated: {report_path}"
    assert report_path.stat().st_size > 100, f"Report was generated but is unexpectedly small: {report_path}"
    return report_path


def _batch_mean_spread(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    spreads = []
    for col in feature_cols:
        batch_means = frame.groupby("batch", observed=False)[col].mean()
        spreads.append(float(batch_means.max() - batch_means.min()))
    return np.asarray(spreads, dtype=float)


def test_lme_iqm_harmonise_produces_sensible_output(tmp_path, monkeypatch):
    _configure_headless_reporting(tmp_path, monkeypatch)

    df, idp_list, qc_list = _simulate_longitudinal_iqm_data()
    output_dir = tmp_path / "iqm_outputs"

    harmonised_df, qc_selection, add_detail_df, mult_detail_df, summary_df = _run_iqm_harmonisation(
        frame=df,
        idp_list=idp_list,
        qc_list=qc_list,
        output_dir=output_dir,
    )

    harmonised_cols = [f"harmonised_{idp}" for idp in idp_list]
    pre_spread = _batch_mean_spread(df, idp_list)
    post_spread = _batch_mean_spread(harmonised_df, harmonised_cols)
    harmonisation_shift = np.abs(
        harmonised_df.loc[:, harmonised_cols].to_numpy(dtype=float)
        - harmonised_df.loc[:, idp_list].to_numpy(dtype=float)
    )

    assert harmonised_df.shape[0] == df.shape[0]
    assert summary_df["volume"].tolist() == idp_list
    assert qc_selection["volumes"] == idp_list
    assert all(col in harmonised_df.columns for col in harmonised_cols)
    assert np.isfinite(harmonised_df.loc[:, harmonised_cols].to_numpy(dtype=float)).all()
    assert add_detail_df["stage"].eq("additive").all()
    assert mult_detail_df["stage"].eq("multiplicative").all()
    assert summary_df["additive_count"].ge(0).all()
    assert summary_df["multiplicative_count"].ge(0).all()
    assert summary_df["additive_count"].sum() + summary_df["multiplicative_count"].sum() > 0
    assert np.nanmax(harmonisation_shift) > 1e-6
    assert np.median(post_spread) < np.median(pre_spread)

    expected_csvs = [
        "iqm_harmonised.csv",
        "iqm_summary.csv",
        "iqm_additive_details.csv",
        "iqm_multiplicative_details.csv",
        "iqm_selected_additive_qcs.csv",
        "iqm_selected_multiplicative_qcs.csv",
    ]
    for filename in expected_csvs:
        path = output_dir / filename
        assert path.exists(), f"Expected output file was not generated: {path}"
        assert path.stat().st_size > 0, f"Generated output file is empty: {path}"


def test_longitudinal_iqm_pipeline_generates_pre_and_post_reports(tmp_path, monkeypatch):
    _configure_headless_reporting(tmp_path, monkeypatch)

    df, idp_list, qc_list = _simulate_longitudinal_iqm_data()
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_report_path = _run_longitudinal_report(
        frame=df,
        feature_cols=idp_list,
        report_name="Longitudinal_Pre_IQM",
        save_dir=REPORT_OUTPUT_DIR,
    )

    harmonised_df, _, _, _, _ = _run_iqm_harmonisation(
        frame=df,
        idp_list=idp_list,
        qc_list=qc_list,
        output_dir=REPORT_OUTPUT_DIR,
    )

    post_idp_list = [f"harmonised_{idp}" for idp in idp_list]
    post_report_path = _run_longitudinal_report(
        frame=harmonised_df,
        feature_cols=post_idp_list,
        report_name="Longitudinal_Post_IQM",
        save_dir=REPORT_OUTPUT_DIR,
    )

    assert pre_report_path.name == "Longitudinal_Pre_IQM.html"
    assert post_report_path.name == "Longitudinal_Post_IQM.html"
