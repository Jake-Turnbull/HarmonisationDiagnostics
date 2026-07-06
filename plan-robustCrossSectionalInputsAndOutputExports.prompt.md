## Plan: Robust Cross-Sectional Inputs and Output Exports

Update the cross-sectional and comparison paths so they accept pandas-native inputs, tolerate missing covariates with warnings, standardize high-feature plotting behavior, and persist the extra UMAP/PCA/Frobenius/scree outputs through the existing save pipeline.  
The implementation keeps report return types stable and focuses on backward-compatible additions.

**Steps**
1. Phase 1: Input normalization and covariate-optional execution.
2. Add shared input normalization in [DiagnoseHarmonisation/DiagnosticReport.py](DiagnoseHarmonisation/DiagnosticReport.py) for:
- data as DataFrame or array-like (coerce numeric; if DataFrame, default feature names from columns).
- batch as Series, one-column DataFrame, or array-like (normalize to 1D array; raise clear error for multi-column DataFrame).
3. Apply normalization at entry points in [DiagnoseHarmonisation/DiagnosticReport.py](DiagnoseHarmonisation/DiagnosticReport.py): CrossSectionalReport, CrossSectionalReportMin, CrossSectionalComparisonReport.
4. For missing covariates, change behavior to warning + continue, but skip only LMM-related parts and clearly log skipped status in report text and summary fields. Depends on step 2.
5. Update workflow layer in [DiagnoseHarmonisation/cross_sectional_workflow.py](DiagnoseHarmonisation/cross_sectional_workflow.py) to allow empty covariate selection without raising and pass through warning-driven behavior. Depends on step 4.

6. Phase 2: Feature plotting policy and Levene visualization.
7. Introduce a shared feature-label policy in plotting modules: if features > 20, keep plots but hide feature tick labels; if features <= 20, show diagonal labels (45°).
8. Apply this policy consistently in [DiagnoseHarmonisation/PlotDiagnosticResults.py](DiagnoseHarmonisation/PlotDiagnosticResults.py) and [DiagnoseHarmonisation/PlotComparisonResults.py](DiagnoseHarmonisation/PlotComparisonResults.py). Parallel with step 9.
9. Make Levene plotting windows square for easier visual inspection in [DiagnoseHarmonisation/PlotDiagnosticResults.py](DiagnoseHarmonisation/PlotDiagnosticResults.py), covering both single and residual variants.
10. Ensure Levene outputs consistently expose per-feature p-values under p_value keys end-to-end (compute, plotting, and saved summaries), reusing existing Levene schema in [DiagnoseHarmonisation/DiagnosticFunctions.py](DiagnoseHarmonisation/DiagnosticFunctions.py).

11. Phase 3: Persist/return enriched outputs and interpretation guidance.
12. Extend save paths in [DiagnoseHarmonisation/DiagnosticReport.py](DiagnoseHarmonisation/DiagnosticReport.py) using [DiagnoseHarmonisation/SaveDiagnosticResults.py](DiagnoseHarmonisation/SaveDiagnosticResults.py) so save_data mode writes:
- UMAP embeddings.
- PCA explained variance and correlation tables.
- Frobenius raw and normalized pairwise matrices.
- Per-batch scree and cumulative variance tables.
13. Ensure comparison result objects attached to report keep these outputs consistently accessible in-memory (without changing primary return type). Depends on step 12.
14. Expand interpretation guidance in report narrative sections in [DiagnoseHarmonisation/DiagnosticReport.py](DiagnoseHarmonisation/DiagnosticReport.py), then mirror in docs pages:
- [docs/api/CrossSectionalComparisonReport.md](docs/api/CrossSectionalComparisonReport.md)
- [docs/api/CrossSectionalWorkflow.md](docs/api/CrossSectionalWorkflow.md)
- [docs/terminal.md](docs/terminal.md)

15. Phase 4: Regression and behavior verification.
16. Update/add tests for workflow/report input typing and no-covariate behavior in:
- [tests/test_cross_sectional_workflow.py](tests/test_cross_sectional_workflow.py)
- [tests/test_DiagnosticReport.py](tests/test_DiagnosticReport.py)
- [tests/test_cross_sectional_comparison_report.py](tests/test_cross_sectional_comparison_report.py)
17. Add plotting tests for >20 feature label suppression + <=20 diagonal labels and Levene square figure expectations in [tests/test_levene_plot.py](tests/test_levene_plot.py).
18. Add save-artifact tests verifying new CSV outputs are present and shaped as expected. Depends on step 12.

**Verification**
1. Run targeted tests:
- pytest tests/test_cross_sectional_workflow.py -q
- pytest tests/test_cross_sectional_comparison_report.py -q
- pytest tests/test_levene_plot.py -q
- pytest tests/test_DiagnosticReport.py -q
2. Manual smoke run: comparison report with >20 features to confirm label suppression behavior.
3. Manual save_data run to confirm new UMAP/PCA/Frobenius/scree CSV artifacts.
4. Confirm report narrative and docs include expanded interpretation and skipped-LMM messaging.

**Decisions captured**
- Missing covariates: skip only LMM, keep other analyses, and warn.
- >20 features: keep plots, hide feature labels.
- Extra outputs: persist via existing save_test_results pattern.
- Batch as DataFrame: require exactly one column, otherwise raise clear error.
- Scope: includes both workflow helpers and report APIs.
- Interpretation guidance: expand in both report text and docs.

Plan is saved in session memory at /memories/session/plan.md and ready for handoff approval or edits.