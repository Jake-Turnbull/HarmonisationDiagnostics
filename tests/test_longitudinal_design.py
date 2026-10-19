import numpy as np
import pandas as pd
import pytest

from DiagnoseHarmonisation.DiagnosticFunctionsLong import (
    AdditiveEffect_long,
    MixedEffects_long,
    MultiVariateBatchDifference_long,
    MultiplicativeEffect_long,
    SubjectOrder_long,
    WithinSubjVar_long,
    _classify_longitudinal_design,
    _resolve_time_adjustment,
)


def test_classifies_missing_and_single_timepoint_designs():
    batch = ["A", "A", "B", "B"]
    subjects = ["s1", "s1", "s2", "s2"]

    missing = _classify_longitudinal_design(batch, subjects)
    single = _classify_longitudinal_design(batch, subjects, ["t1"] * 4)

    assert missing["has_timepoint"] is False
    assert missing["n_timepoints"] == 0
    assert missing["recommended_mode"] == "cross_sectional"
    assert single["has_timepoint"] is True
    assert single["n_timepoints"] == 1
    assert single["recommended_mode"] == "cross_sectional"
    no_subject = _classify_longitudinal_design(batch, timepoint=["t1", "t2", "t1", "t2"])
    assert no_subject["has_subject"] is False
    assert no_subject["recommended_mode"] == "cross_sectional"


def test_classifies_repeated_measures_and_crossed_batch_time_designs():
    subjects = ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]
    timepoints = ["t1", "t2"] * 4
    constant_batch = ["A", "A", "A", "A", "B", "B", "B", "B"]
    varying_batch = ["A", "B", "B", "A", "A", "B", "B", "A"]

    constant = _classify_longitudinal_design(constant_batch, subjects, timepoints)
    varying = _classify_longitudinal_design(varying_batch, subjects, timepoints)

    assert constant["is_repeated_measures"] is True
    assert constant["batch_effect_identifiable"] is True
    assert constant["recommended_mode"] == "longitudinal"
    assert varying["is_repeated_measures"] is True
    assert varying["batch_effect_identifiable"] is True
    assert varying["batch_timepoint_perfectly_confounded"] is False


def test_classifies_perfect_and_partial_batch_time_confounding():
    crossed = _classify_longitudinal_design(
        ["A", "A", "B", "B"], ["s1", "s2", "s3", "s4"], ["t1", "t2", "t1", "t2"]
    )
    perfect = _classify_longitudinal_design(
        ["A", "A", "B", "B"], ["s1", "s2", "s3", "s4"], ["t1", "t1", "t2", "t2"]
    )
    partial = _classify_longitudinal_design(
        ["A", "A", "B"], ["s1", "s2", "s3"], ["t1", "t2", "t2"]
    )

    assert crossed["batch_timepoint_perfectly_confounded"] is False
    assert crossed["batch_timepoint_partially_confounded"] is False
    assert perfect["batch_timepoint_perfectly_confounded"] is True
    assert perfect["batch_effect_identifiable"] is False
    assert perfect["recommended_mode"] == "confounded"
    assert partial["batch_timepoint_perfectly_confounded"] is False
    assert partial["batch_timepoint_partially_confounded"] is True


def test_time_adjustment_policy_respects_identifiability():
    design = _classify_longitudinal_design(
        ["A", "A", "B", "B"], ["s1", "s2", "s3", "s4"], ["t1", "t2", "t1", "t2"]
    )
    confounded = _classify_longitudinal_design(
        ["A", "A", "B", "B"], ["s1", "s2", "s3", "s4"], ["t1", "t1", "t2", "t2"]
    )

    assert _resolve_time_adjustment(design)["include_time"] is True
    assert _resolve_time_adjustment(design, False)["include_time"] is False
    assert _resolve_time_adjustment(confounded)["include_time"] is False
    with pytest.raises(ValueError, match="not separately identifiable"):
        _resolve_time_adjustment(confounded, True)


def test_within_subject_variability_does_not_require_timepoint_labels():
    matrix = np.array([[1.0], [2.0], [3.0], [5.0]])
    subjects = ["s1", "s1", "s2", "s2"]

    without_time = WithinSubjVar_long(matrix, subjects, idp_names=["x"])
    with_time = WithinSubjVar_long(matrix, subjects, ["t1", "t2", "t1", "t2"], idp_names=["x"])

    pd.testing.assert_frame_equal(without_time, with_time)
    with pytest.raises(ValueError, match="Length of timepoints"):
        WithinSubjVar_long(matrix, subjects, ["t1"], idp_names=["x"])


def test_within_subject_variability_returns_median_absolute_rpd_only():
    matrix = np.array([[-1.0], [-2.0], [-10.0], [np.inf]])

    result = WithinSubjVar_long(matrix, ["s1"] * 4, idp_names=["x"])

    assert result.loc[0, "x"] == pytest.approx(400.0 / 3.0)
    assert np.isscalar(result.loc[0, "x"])
    assert "std" not in result.columns


def test_subject_order_aggregates_duplicate_keys_and_skips_non_finite_pairs():
    matrix = np.array(
        [
            [1.0, 10.0],
            [3.0, 12.0],
            [4.0, np.inf],
            [6.0, 30.0],
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
        ]
    )
    subjects = ["s1", "s1", "s2", "s3", "s1", "s2", "s3"]
    timepoints = ["t1", "t1", "t1", "t1", "t2", "t2", "t2"]

    result = SubjectOrder_long(
        matrix,
        subjects,
        timepoints,
        idp_names=["finite", "has_inf"],
        nPerm=10,
        seed=1,
    ).set_index("IDP")

    assert result.loc["finite", "nPairs"] == 3
    assert result.loc["finite", "SpearmanRho"] == pytest.approx(1.0)
    assert result.loc["has_inf", "nPairs"] == 2
    assert np.isnan(result.loc["has_inf", "SpearmanRho"])


def test_additive_effect_uses_stable_chi2_tail_and_applies_bh_fdr(monkeypatch):
    from types import SimpleNamespace

    from DiagnoseHarmonisation import DiagnosticFunctionsLong

    log_likelihoods = iter([10.0, 0.0, 50.0, 0.0])
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "_safe_fit_mixedlm",
        lambda *args, **kwargs: SimpleNamespace(llf=next(log_likelihoods)),
    )
    data = np.arange(16, dtype=float).reshape(8, 2)
    subjects = [f"s{i}" for i in range(8)]
    batches = ["A", "A", "B", "B"] * 2
    timepoints = ["t1", "t2"] * 4

    effects, *_ = AdditiveEffect_long(
        idp_matrix=data,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batches,
        idp_names=["lower_lr", "higher_lr"],
        verbose=False,
    )

    assert effects["Feature"].tolist() == ["lower_lr", "higher_lr"]
    effects = effects.set_index("Feature")
    assert effects.loc["lower_lr", "p-value"] > effects.loc["higher_lr", "p-value"]
    assert effects["p-value_fdr"].notna().all()
    assert effects.loc["lower_lr", "p-value_fdr"] == pytest.approx(
        effects.loc["lower_lr", "p-value"]
    )
    assert effects.loc["higher_lr", "p-value_fdr"] == pytest.approx(
        effects.loc["higher_lr", "p-value"] * 2
    )


def test_multiplicative_effect_preserves_feature_order(monkeypatch):
    from types import SimpleNamespace

    from DiagnoseHarmonisation import DiagnosticFunctionsLong

    fligner_results = iter([(1.0, 0.3), (9.0, 0.01)])
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "_safe_fit_mixedlm",
        lambda *args, **kwargs: SimpleNamespace(resid=np.arange(8, dtype=float)),
    )
    monkeypatch.setattr(
        DiagnosticFunctionsLong,
        "fligner",
        lambda *args, **kwargs: next(fligner_results),
    )

    effects, _ = MultiplicativeEffect_long(
        idp_matrix=np.arange(16, dtype=float).reshape(8, 2),
        subjects=[f"s{i}" for i in range(8)],
        batch_name=["A", "A", "B", "B"] * 2,
        idp_names=["lower_chi2", "higher_chi2"],
        verbose=False,
    )

    assert effects["Feature"].tolist() == ["lower_chi2", "higher_chi2"]
    assert effects["ChiSq"].tolist() == [1.0, 9.0]


def test_confounded_batch_models_do_not_report_pure_batch_effects():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(12, 2))
    batches = np.array(["A"] * 6 + ["B"] * 6)
    timepoints = np.array(["t1"] * 6 + ["t2"] * 6)
    subjects = np.array([f"s{i}" for i in range(12)])

    additive, *_ = AdditiveEffect_long(
        idp_matrix=data,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batches,
        idp_names=["x", "y"],
        verbose=False,
    )
    multiplicative, _ = MultiplicativeEffect_long(
        idp_matrix=data,
        subjects=subjects,
        timepoints=timepoints,
        batch_name=batches,
        idp_names=["x", "y"],
        verbose=False,
    )
    mixed, _ = MixedEffects_long(
        idp_matrix=data,
        subjects=subjects,
        timepoints=timepoints,
        batches=batches,
        idp_names=["x", "y"],
        reml=False,
    )

    assert additive["p-value"].isna().all()
    assert set(additive["status"]) == {"non_identifiable"}
    assert multiplicative["p-value"].isna().all()
    assert set(multiplicative["status"]) == {"non_identifiable"}
    assert mixed["n_is_batchSig"].isna().all()
    assert mixed["anova_batches"].isna().all()
    assert set(mixed["status"]) == {"non_identifiable"}


def test_multivariate_reports_confounded_distances_as_combined():
    rng = np.random.default_rng(23)
    data = rng.normal(size=(12, 3))
    batches = np.array(["A"] * 6 + ["B"] * 6)
    timepoints = np.array(["t1"] * 6 + ["t2"] * 6)
    subjects = np.tile([f"s{i}" for i in range(6)], 2)

    distances, info = MultiVariateBatchDifference_long(
        data,
        batches,
        subject=subjects,
        timepoint=timepoints,
        return_info=True,
    )

    assert info["batch_timepoint_perfectly_confounded"] is True
    assert info["batch_effect_identifiable"] is False
    assert info["interpretation"] == "combined_batch_timepoint_difference"
    assert distances["pairwise_raw"]