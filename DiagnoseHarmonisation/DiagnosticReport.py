# Diagnostic report generation using DiagnosticFunctions 
from ensurepip import version
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from DiagnoseHarmonisation import DiagnosticFunctions
from DiagnoseHarmonisation import PlotDiagnosticResults
from DiagnoseHarmonisation.LoggingTool import StatsReporter

# Helper function 
def covariate_to_numeric(covariates):
    """
    Convert categorical covariates to numeric codes for corresponding functions that require numeric input (e.g. PCA correlation function).
    This is a helper function that is included here for user clarity:

    Args:
        covariates (np.ndarray or pd.DataFrame): Covariate matrix with categorical variables.
    
    Returns:
        np.ndarray: Covariate matrix with categorical variables converted to numeric codes.
    
    Note:
        - If covariates is a DataFrame, it will be factorized column-wise.
        - If covariates is a numpy array, it will be factorized column-wise.
        - The function assumes that categorical variables are of string or object type. Numeric columns will be left unchanged.
    """
    # Check covariate columns independently and factorize if they are categorical (string/object), otherwise keep as is. This allows for mixed covariate types.
    if covariates is None:
        return None
    for i in range(covariates.shape[1]):
        if covariates[:, i].dtype.kind in {"U", "S", "O"}:  # string/object categorical
            covariates[:, i], unique = pd.factorize(covariates[:, i])
        elif covariates[:, i].dtype.kind in {"i", "f"}:  # numeric, keep as is
            pass
    covariate_numeric = covariates.astype(float)  # ensure all numeric for functions that require numeric input
    
    return covariate_numeric


def _generate_harmonisation_advice(
    cohens_d_results,
    mahalanobis_results,
    lmm_results_df,
    variance_summary_df,
    covariance_results,
    batch_sizes,
):
    """
    Turn the existing diagnostic outputs into short harmonisation advice.

    The thresholds here are intentionally heuristic so the report can give
    practical guidance without introducing new statistical tests.
    """
    abs_d = np.abs(np.asarray(cohens_d_results, dtype=float))
    if abs_d.size == 0:
        max_large_effect_fraction = 0.0
        median_abs_d = 0.0
    else:
        max_large_effect_fraction = float(np.nanmax(np.mean(abs_d >= 0.5, axis=1)))
        median_abs_d = float(np.nanmedian(abs_d))

    pairwise_mahal = np.asarray(
        list((mahalanobis_results or {}).get("pairwise_raw", {}).values()),
        dtype=float,
    )
    centroid_mahal = np.asarray(
        list(
            ((mahalanobis_results or {}).get("centroid_resid") or
             (mahalanobis_results or {}).get("centroid_raw") or {}).values()
        ),
        dtype=float,
    )
    max_pairwise_mahal = float(np.nanmax(pairwise_mahal)) if pairwise_mahal.size else 0.0
    max_centroid_mahal = float(np.nanmax(centroid_mahal)) if centroid_mahal.size else 0.0

    icc_values = np.array([], dtype=float)
    if lmm_results_df is not None and "ICC" in lmm_results_df:
        icc_values = pd.to_numeric(lmm_results_df["ICC"], errors="coerce").dropna().to_numpy(dtype=float)
    median_icc = float(np.nanmedian(icc_values)) if icc_values.size else 0.0
    high_icc_fraction = float(np.mean(icc_values >= 0.1)) if icc_values.size else 0.0

    mean_signal_details = {
        "cohens_d": (max_large_effect_fraction >= 0.2) or (median_abs_d >= 0.35),
        "mahalanobis": (max_pairwise_mahal >= 1.0) or (max_centroid_mahal >= 1.0),
        "lmm": (median_icc >= 0.1) or (high_icc_fraction >= 0.2),
    }
    has_mean_differences = any(mean_signal_details.values())

    has_scale_differences = False
    if variance_summary_df is not None and not variance_summary_df.empty:
        median_logs = pd.to_numeric(
            variance_summary_df["Median log ratio"],
            errors="coerce",
        ).to_numpy(dtype=float)
        has_scale_differences = bool(
            np.any(np.abs(median_logs) >= np.log(1.25))
        )

    normalized_covariance = (covariance_results or {}).get("pairwise_frobenius_normalized")
    covariance_strength = 0.0
    if normalized_covariance is not None:
        if isinstance(normalized_covariance, pd.DataFrame):
            covariance_array = normalized_covariance.to_numpy(dtype=float)
        else:
            covariance_array = np.asarray(normalized_covariance, dtype=float)

        if covariance_array.ndim == 2 and covariance_array.size > 0:
            upper_idx = np.triu_indices_from(covariance_array, k=1)
            upper_values = covariance_array[upper_idx]
        else:
            upper_values = covariance_array.ravel()

        if upper_values.size:
            covariance_strength = float(np.nanmax(upper_values))
    has_covariance_differences = covariance_strength >= 0.3

    largest_batch = max(batch_sizes, key=batch_sizes.get)
    smallest_batch = min(batch_sizes, key=batch_sizes.get)
    largest_batch_n = batch_sizes[largest_batch]
    smallest_batch_n = batch_sizes[smallest_batch]
    has_large_batch_imbalance = smallest_batch_n > 0 and (largest_batch_n > (2 * smallest_batch_n))

    mean_signal_labels = [
        label.replace("_", " ")
        for label, present in mean_signal_details.items()
        if present
    ]
    advice_lines = []

    if has_mean_differences:
        if mean_signal_labels:
            advice_lines.append(
                "Strong mean-shift signals were detected from \n "
                + ", ".join(mean_signal_labels)
                + ".\n"
            )
    else:
        advice_lines.append(
            "Mean-shift diagnostics were not especially strong, so any harmonisation choice should be made cautiously.\n"
        )

    if has_large_batch_imbalance and has_mean_differences and has_scale_differences and has_covariance_differences:
        advice_lines.append(
            f"{largest_batch} is much larger than the other batches (n={largest_batch_n} vs smallest n={smallest_batch_n}), \n"
            f"and the diagnostics suggest differences in mean, scale, and covariance structure. \n"
            f"CovBat with {largest_batch} as the reference batch looks like the strongest candidate.\n"
        )
    else:
        if has_mean_differences and not has_scale_differences and not has_covariance_differences:
            advice_lines.append(
                "The residual batch effects look mainly additive, so a regression-based harmonisation approach or ComBat would be a sensible first choice.\n"
            )

        if has_scale_differences:
            advice_lines.append(
                "Scale differences were also detected, so ComBat is likely a better fit than a mean-only regression adjustment.\n"
            )

        if has_covariance_differences:
            advice_lines.append(
                "Covariance structure differences were detected between batches, so CovBat could be a good alternative when multivariate structure needs to be aligned.\n"
            )

        if has_large_batch_imbalance:
            advice_lines.append(
                f"Batch sizes are imbalanced and {largest_batch} is the largest batch (n={largest_batch_n}), "
                f"so using {largest_batch} as the ComBat reference batch may help avoid over-correcting that cohort.\n"
            )

    if not any(
        [
            has_mean_differences,
            has_scale_differences,
            has_covariance_differences,
            has_large_batch_imbalance,
        ]
    ):
        advice_lines.append(
            "The diagnostics do not indicate a strong harmonisation target pattern, so a lighter-touch adjustment or no harmonisation may be reasonable depending on the study goal.\n"
        )

    return {
        "advice_lines": advice_lines,
        "has_mean_differences": has_mean_differences,
        "has_scale_differences": has_scale_differences,
        "has_covariance_differences": has_covariance_differences,
        "has_large_batch_imbalance": has_large_batch_imbalance,
        "largest_batch": largest_batch,
        "largest_batch_n": largest_batch_n,
        "smallest_batch": smallest_batch,
        "smallest_batch_n": smallest_batch_n,
        "mean_signal_details": mean_signal_details,
        "covariance_strength": covariance_strength,
    }

# Min report (for quick surface level checks;)
def CrossSectionalReportMin(data,
                             batch,
                             covariates=None,
                             covariate_names=None,
                             save_data: bool = True,
                             save_data_name: str | None = None,
                             save_dir: str | os.PathLike | None = None,
                             feature_names: list | None = None,
                             report_name: str | None = None,
                             SaveArtifacts: bool = False,
                             rep= None,
                             show: bool = False,
                             timestamped_reports: bool = True,
                             covariate_types: list | None = None,
                             ratio_type: str = "rest"
                             ):
    """A minimal cross-sectional diagnostic report with a limited set of tests and visualizations for quick checks.
        This function is designed for users who want a faster, surface-level diagnostic report that includes only key tests and visualizations. 
        
        For a more comprehensive analysis, please use the full CrossSectionalReport function.
    
    Args:
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        feature_names (list, optional): Names of features.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
        timestamped_reports (bool, optional): Whether to append a timestamp to the report filename.
        covariate_types (list, optional): Types of covariates (e.g., 'categorical', 'numeric').

    Returns:
        HTML report saved to specified directory (or cd by default). If save_data is True, also returns a dictionary of saved data arrays.
        If SaveArtifacts is True, intermediate artifacts will be saved to the same directory with appropriate naming.
        """

    if save_dir is None:
        save_dir = Path.cwd()
        # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "CrossSectionalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialized HTML report at: {rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()
        report = ctx
    else:
        report = report_ctx
    # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger
        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 125
        report.text_simple("This is a minimal diagnostic report that includes only Z-score visualization, Cohen's D test for mean differences, Variance ratio test for variance differences between batches,\n\n "
        "and ICC/R² from LMM diagnostics. For a more comprehensive report with additional tests and visualizations, please use the full CrossSectionalReport function.")


        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        # Print version info from _version.py
        from DiagnoseHarmonisation._version import version
        report.log_text(f"DiagnoseHarmonisation version: {version}")
            
        # Get todays date for saving results
        report_date = datetime.now().date().isoformat()

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")

        # Samples per batch
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

        # Check for missing data (NaNs) in the dataset, if high proportion log a warning:
        # Create array of size data, if NaN, mark as 1, else 0. Then sum per feature and per batch to get proportion of missing data.
        nan_mask = np.isnan(data)
        # Check proportion of missing data per batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_nans = nan_mask[batch_mask, :]
            prop_nans = np.mean(batch_nans)
            if prop_nans > 0.1:  # arbitrary threshold for logging warning
                logger.warning(f"Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
                report.text_simple(f"Warning: Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
            else:
                report.text_simple(f"Batch {b} has a proportion of missing data: {prop_nans:.2%}")

        # Replace NaNs with structured noise of mean and variance of each batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_data = data[batch_mask, :]
            batch_mean = np.nanmean(batch_data, axis=0)
            batch_std = np.nanstd(batch_data, axis=0)
            # For each Nan, fill with random normal noise with batch mean and std:
            for i in range(batch_data.shape[0]):
                for j in range(batch_data.shape[1]):
                    if np.isnan(batch_data[i, j]):
                        batch_data[i, j] = np.random.normal(loc=batch_mean[j], scale=batch_std[j])
            
            # Replace in original data
            data[batch_mask, :] = batch_data


        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple(" The order of tests is as follows: Multivariate distribution comparisson, Additive tests, Multiplicative tests, Model fit")
        report.text_simple(line_break_in_text)

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        report.text_simple("Z-score normalization (median-centred) visualization across batches,\n" \
        "Here, we convert each feature to a median absolute deviation (MAD) and express each observation as a histogram.\n " \
        "As the normalisation is done globally, batchwise histograms that appear differently (width or location) indicate batch differences in mean and/or variance across features. ")
        
        zscored_data = DiagnosticFunctions.robust_z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)

        covariates_numeric = covariates
        # if dataframe or dictionary, convert to numeric array:
        if covariates is not None:
            if isinstance(covariates, pd.DataFrame):
                covariates_numeric = covariate_to_numeric(covariates.values)
            elif isinstance(covariates, dict):
                covariates_numeric = covariate_to_numeric(np.column_stack(list(covariates.values())))
            elif isinstance(covariates, np.ndarray):
                covariates_numeric = covariate_to_numeric(covariates)
            else:
                raise ValueError("Covariates must be a numpy array, pandas DataFrame, or dictionary of arrays")
        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch, covariates=covariates,covariate_names=covariate_names, covariate_types=covariate_types)
        report.text_simple("Cohen's D test for mean differences completed")

        # Plot (PlotDiagnosticResults should call rep.log_plot internally; our report.log_section ensures plots are attached)
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results, pair_labels=pairlabels, rep=report)
        report.log_text("Cohen's D plot added to report")

        # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict = {}
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results
        if save_data:
            save_test_results(data_dict,
            test_name="Cohens_D",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date, 
            report_name=report_name,
            )

        # Use the same code from CrossSectionalReport, report LMM diagnostics, Variance ratio
        # run LMM diagnostics
        lmm_results_df, lmm_summary = DiagnosticFunctions.Run_LMM_cross_sectional(data, batch, covariates=covariates,
                                                feature_names=feature_names,
                                                covariate_names=covariate_names,
                                                min_group_n=2)

        report.text_simple("LMM diagnostics completed.")
        report.log_text("LMM results table added to report")

        # add summary text
        report.text_simple(
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
        )

        # list common notes
        note_lines = []
        for tag, count in sorted(lmm_summary.items(), key=lambda x: -x[1])[:10]:
            if tag == 'n_features':
                continue
            note_lines.append(f"{tag}: {count}")
        report.text_simple("LMM diagnostics notes (top):\n" + "\n".join(note_lines))
        data_dict = {}
        # Save DF if needed
        if save_data:
            data_dict['LMM_results_df'] = lmm_results_df
            data_dict['LMM_summary'] = lmm_summary
        
        # Save LMM results as csv
        save_test_results(data_dict,
        test_name="LMM_Results",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date,
        report_name=report_name
        )

        report.text_simple("Histogram of ICC (proportion of variance explained by batch):")
        # How to interpret ICC:
        report.text_simple("Intraclass Correlation Coefficient (ICC) is the ratio of variance due to batch effects to the total variance (batch + residual). \n" 
        "It quantifies the extent to which batch membership explains variability in the data.")
        report.text_simple(
            "Interpretation of ICC values:\n"
            "- ICC close to 0: Little to no variance explained by batch; suggests minimal batch effect.\n"
            "- ICC around 0.1-0.3: Small batch effect; may be acceptable depending on context.\n"
            "- ICC around 0.3-0.5: Moderate batch effect; consider further investigation or correction.\n"
            "- ICC above 0.5: Strong batch effect; likely requires correction to avoid confounding.\n"
        )
        try:
            icc_nonan = lmm_results_df['ICC'].dropna()
            if len(icc_nonan) > 0:
                plt.figure(figsize=(10, 4))
                plt.bar(range(len(icc_nonan)), icc_nonan)
                plt.xlabel("Feature index")
                plt.ylabel("ICC")
                plt.title("ICC values per feature")
                report.log_plot(plt, caption="ICC values per feature")
                plt.close()

        except Exception:
            logger.exception("Could not produce ICC histogram")

        
        # Plot conditional and marginal R^2 per feature, indicate what each means for interpretation
        report.text_simple("Marginal R² represents the variance explained by fixed effects (covariates)\n"
                           "while Conditional R² represents the variance explained by both fixed and random effects (batch + covariates).")
        lmm_r = lmm_results_df[['R2_marginal', 'R2_conditional']].dropna()
        if len(lmm_r) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(lmm_r['R2_marginal'].values, label='Marginal R²', alpha=0.7)
            plt.plot(lmm_r['R2_conditional'].values, label='Conditional R²', alpha=0.7)
            plt.xlabel("Feature index")
            plt.ylabel("R² value")
            plt.title("Marginal and Conditional R² values per feature")
            plt.legend()
            report.log_plot(plt, caption="Marginal and Conditional R² values per feature")
            plt.close()

        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # Variance ratio
        mode = ratio_type
        report.log_section("variance_ratio", "Variance ratio test (F-test) for variance differences between batches")
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratios, pair_labels = DiagnosticFunctions.Variance_Ratios(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
            mode=mode
        )

        report.log_text("Variance ratio test completed")

        # save variance ratios raw:
        if save_data:
            save_test_results(
                variance_ratios,
                test_name="Variance_Ratios_Raw",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        # Summarise variance ratio results
        data_dict = {}
        summary_rows = []

        # variance_ratios is (num_pairs x num_features)
        n_pairs = variance_ratios.shape[0]

        for i in range(n_pairs):
            ratios = np.array(variance_ratios[i], dtype=float)

            # Safe log: treat non-positive values as NaN for log stats
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)

            mean_log = np.nanmean(log_ratios)
            median_log = np.nanmedian(log_ratios)
            iqr_log = np.nanpercentile(log_ratios, [25, 75])
            # Proportion > 0: treat NaNs as False
            prop_higher = np.nanmean(np.where(np.isnan(log_ratios), False, log_ratios > 0))

            # exponentiate summary stats where meaningful
            median_ratio = np.exp(median_log) if not np.isnan(median_log) else np.nan
            mean_ratio = np.exp(mean_log) if not np.isnan(mean_log) else np.nan

            label = pair_labels[i]
            # Try to split label into two parts like "A / B" -> b1, b2. Otherwise keep label as-is.
            if isinstance(label, str) and " / " in label:
                b1, b2 = [s.strip() for s in label.split(" / ", 1)]
            else:
                # fallback: present full label in Batch 1, leave Batch 2 empty
                b1 = label
                b2 = ""

            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })

            # sanitize label for keys (replace spaces and parentheses)
            safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_vs_")

            data_dict[f"VarianceRatio_{safe_label}"] = ratios
            data_dict[f"MedianLogVarianceRatio_{safe_label}"] = median_log
            data_dict[f"MeanLogVarianceRatio_{safe_label}"] = mean_log
            data_dict[f"IQRLowerLogVarianceRatio_{safe_label}"] = iqr_log[0]
            data_dict[f"IQRUpperLogVarianceRatio_{safe_label}"] = iqr_log[1]
            data_dict[f"PropHigherLogVarianceRatio_{safe_label}"] = prop_higher
            data_dict[f"MedianVarianceRatioExp_{safe_label}"] = median_ratio
            data_dict[f"MeanVarianceRatioExp_{safe_label}"] = mean_ratio

            # human-readable report line
            report.text_simple(
                f"Variance ratio {label}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in {b1}"
            )

        # Save summary as well
        if save_data:
            save_test_results(
                data_dict,
                test_name="Variance_Ratio_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        summary_df = pd.DataFrame(summary_rows)
        report.text_simple("Variance ratio test summaries (per batch pair):")
        # Plot using your plot function
        PlotDiagnosticResults.variance_ratio_plot(variance_ratios, pair_labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")
    
        report.text_simple(line_break_in_text)
    
        report.text_simple(line_break_in_text)
        # add summary text

        report.log_section("report_conclusion", "Minimal Cross-Sectional Report")
        report.text_simple("This concludes the minimal cross-sectional diagnostic report. For a more comprehensive analysis with additional tests and visualizations, please use the full CrossSectionalReport function.")

        report.text_simple("Summary:")
                # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict = {}
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        # Report LMM diagnostics summary
        report.text_simple(
            f"LMM diagnostics summary:\n"
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
        )
        # Report variance ratio summary
        report.text_simple("Variance ratio test summaries (per batch pair):")
        for _, row in summary_df.iterrows():
            report.text_simple(
                f"Batch {row['Batch 1']} vs Batch {row['Batch 2']}: median log ratio={row['Median log ratio']:.3f} "
                f"(IQR {row['IQR lower']:.3f}–{row['IQR upper']:.3f}), "
                f"{row['Prop > 0']*100:.1f}% of features higher in batch {row['Batch 1']}"
            )
        # Finalise report:
    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report
            report_ctx.__exit__(None, None, None)

# Full cross-sectional report with all tests and visualizations:
def CrossSectionalReport(
    data,
    batch,
    covariates=None,
    covariate_names=None,
    save_data: bool = True,
    save_data_name: str | None = None,
    save_dir: str | os.PathLike | None = None,
    feature_names: list | None = None,
    report_name: str | None = None,
    SaveArtifacts: bool = False,
    rep= None,
    show: bool = False,
    timestamped_reports: bool = True,
    covariate_types: list | None = None,
    ratio_type: str = "rest"
):
    """
    Create a diagnostic report for dataset differences across batches;
    Uses the following tests and visualisations in this order:
    - Z-score visualization
    - Cohen's D test for mean differences
    - Mahalanobis distance between batches
    - ICC and R^2 from LMM diagnostics
    - Variance ratio test for variance differences between batches
    - PCA correlation with batch and covariates (if covariates provided)
    - PCA clustering by batch and covariates (if covariates provided)
    - UMAP visualization colored by batch and covariates (if covariates provided)
    - Two sample Kolmogorov-Smirnov test for distribution differences between batches

    Args:
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
    
    Returns:
        HTML report saved to specified directory (or cd by default).
        dict or None: If save_data is True, returns a dictionary of saved data arrays.
    
    Note:
        This function takes an additional argument `covariate_types` which is a list of the same length as `covariate_names` indicating the type of each covariate (e.g., 'categorical', 'numeric'). This allows the function to handle covariates appropriately in different tests and visualizations based on their type. For example, categorical covariates can be factorized for numeric tests, while numeric covariates can be used directly. The function will log the types of covariates detected and how they are being handled in the report.
        If this is not given, the code will use an arbitary number (10) to decide between categorical and numeric covariates. If this isn't desired, please either provide the covariate types or ensure that covariates are in a format that can be correctly inferred (e.g., categorical covariates as strings/objects, numeric covariates as numeric types).
        Covariate types: 0 - binary, 1 - categorical, 2 - numeric. If covariate_types is provided, it should be a list of these values corresponding to each covariate in `covariate_names`. This will allow the function to handle each covariate appropriately based on its type in different tests and visualizations. For example, binary and categorical covariates can be factorized for numeric tests, while numeric covariates can be used directly. The function will log the types of covariates detected and how they are being handled in the report.




    """


    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "CrossSectionalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialized HTML report at: {rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()
        report = ctx
    else:
        report = report_ctx
    # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 125
        report.text_simple("This is the full diagnostic cross-sectional report that includes a comprehensive set of tests and visualizations to assess batch effects in the dataset. \n\n")

        report.text_simple("For full documentation and interpretation of each test, please refer to the online documentation at https://jake-turnbull.github.io/HarmonisationDiagnostics/")

        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of subjects: {data.shape[0]}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        # Print version info from _version.py
        from DiagnoseHarmonisation._version import version
        report.log_text(f"DiagnoseHarmonisation version: {version}")

            
        # Get todays date for saving results
        report_date = datetime.now().date().isoformat()

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                report.text_simple(f"Batch categories detected: {list(set(batch))}. Numeric codes will be used for tests, but string labels will be kept for plots and summaries where possible.")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")

        # Samples per batch
        unique_batches, counts = np.unique(batch, return_counts=True)
        report.text_simple("Number of samples per batch:")
        for b, c in zip(unique_batches, counts):
            report.text_simple(f"Batch {b}: {c} samples")
        report.text_simple(line_break_in_text)

                # Check for missing data (NaNs) in the dataset, if high proportion log a warning:
        # Create array of size data, if NaN, mark as 1, else 0. Then sum per feature and per batch to get proportion of missing data.
        nan_mask = np.isnan(data)
        # Check proportion of missing data per batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_nans = nan_mask[batch_mask, :]
            prop_nans = np.mean(batch_nans)
            if prop_nans > 0.001:  # arbitrary threshold for logging warning
                logger.warning(f"Batch {b} has a high proportion of missing data: {prop_nans:.2%}")
                report.text_simple("Missing data will be replaced with batch-specific structured noise (random normal)")
                logger.warning("If this is unwanted, or if batch size is too small to reliably estimate batch-specific noise, consider removing these samples prior to analysis.")
            else:
                report.text_simple(f"Batch {b} has a proportion of missing data: {prop_nans:.2%}")

        

        # Replace NaNs with structured noise of mean and variance of each batch:
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_data = data[batch_mask, :]
            batch_mean = np.nanmean(batch_data, axis=0)
            batch_std = np.nanstd(batch_data, axis=0)
            # For each Nan, fill with random normal noise with batch mean and std:
            for i in range(batch_data.shape[0]):
                for j in range(batch_data.shape[1]):
                    if np.isnan(batch_data[i, j]):
                        batch_data[i, j] = np.random.normal(loc=batch_mean[j], scale=batch_std[j])
            
            # Replace in original data
            data[batch_mask, :] = batch_data

        # Check if any columns still have NaNs, if this has happened, its because all data had missing data for that batch, so we fill with global mean and std:
        nan_mask_after = np.isnan(data)
        if np.any(nan_mask_after):
            global_mean = np.nanmean(data, axis=0)
            global_std = np.nanstd(data, axis=0)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isnan(data[i, j]):
                        data[i, j] = np.random.normal(loc=global_mean[j], scale=global_std[j])
                        # Warn user that this has happened in the report and that this feature will not be reliable for diagnostics:
                        # Check if covariate names given to give index and name of feature with all NaNs:
                        if feature_names is not None and j < len(feature_names):
                            feature_name = feature_names[j]
                        else:                           
                            feature_name = f"index {j}"
                        logger.warning(f"Feature {feature_name} has all missing data for at least one batch. NaNs in this feature have been replaced with global mean and std, which may not be reliable for diagnostics.")

        # Final check; if there are still NaNs it is because whole column for all batches is NaN, so we set all values to 1 and log a warning that this feature will not be reliable for diagnostics:
        nan_mask_final = np.isnan(data)
        if np.any(nan_mask_final):
            for j in range(data.shape[1]):
                if np.any(nan_mask_final[:, j]):
                    data[:, j] = 1.0
                    if feature_names is not None and j < len(feature_names):
                        feature_name = feature_names[j]
                    else:                           
                        feature_name = f"index {j}"
                    logger.warning(f"Feature {feature_name} has all missing data across all batches. NaNs in this feature have been replaced with 1. This feature will not be reliable for diagnostics.")

        report.text_simple("Missing data (NaNs) have been replaced with batch-specific structured noise (random normal with batch mean and std) for the purposes of diagnostics. \n")

        report.text_simple(line_break_in_text)
        report.text_simple("\n\n")

        # Repeat data replacement process for covariates if needed:
        if covariates is not None:
            covariates_numeric = covariates
            # if dataframe or dictionary, convert to numeric array:
            if isinstance(covariates, pd.DataFrame):
                covariates_numeric = covariate_to_numeric(covariates.values)
            elif isinstance(covariates, dict):
                covariates_numeric = covariate_to_numeric(np.column_stack(list(covariates.values())))
            elif isinstance(covariates, np.ndarray):
                covariates_numeric = covariate_to_numeric(covariates)
            else:
                raise ValueError("Covariates must be a numpy array, pandas DataFrame, or dictionary of arrays")
            
        #  Check for NaNs and replace with batch mean (for numeric) or mode (for categorical) as appropriate:
        if covariates is not None:
            n_total = covariates_numeric.shape[0]
            cat_threshold = int(np.ceil(0.05 * n_total))  # 5% of total dataset size

            for col_idx in range(covariates_numeric.shape[1]):
                col = covariates_numeric[:, col_idx]
                nan_mask_cov = np.isnan(col)
                if not np.any(nan_mask_cov):
                    report.text_simple(f"Covariate column {col_idx} has no missing data.")
                    continue

                # Determine if column is categorical:
                # categorical if values are integers and number of unique non-NaN values <= 5% of n_total
                col_nonan = col[~nan_mask_cov]
                unique_vals = np.unique(col_nonan)
                is_integer_valued = np.all(col_nonan == np.floor(col_nonan))
                is_categorical = is_integer_valued and (len(unique_vals) <= cat_threshold)

                for b in unique_batches:
                    batch_mask_cov = (batch == b) & nan_mask_cov
                    if not np.any(batch_mask_cov):
                        continue

                    batch_col_nonan = col[(batch == b) & ~nan_mask_cov]

                    if is_categorical:
                        # Replace with batch-specific mode
                        if len(batch_col_nonan) > 0:
                            vals, counts_cov = np.unique(batch_col_nonan, return_counts=True)
                            mode_val = vals[np.argmax(counts_cov)]
                        else:
                            # fallback to global mode if no non-NaN values in this batch
                            vals, counts_cov = np.unique(col_nonan, return_counts=True)
                            mode_val = vals[np.argmax(counts_cov)]
                        covariates_numeric[batch_mask_cov, col_idx] = mode_val
                        report.text_simple(
                            f"Covariate column {col_idx} (categorical): replaced {np.sum(batch_mask_cov)} "
                            f"NaNs in batch '{b}' with mode={mode_val}"
                        )
                    else:
                        # Replace with batch-specific Gaussian noise
                        if len(batch_col_nonan) > 1:
                            b_mean = np.nanmean(batch_col_nonan)
                            b_std = np.nanstd(batch_col_nonan)
                        elif len(batch_col_nonan) == 1:
                            b_mean = batch_col_nonan[0]
                            b_std = np.nanstd(col_nonan) if len(col_nonan) > 1 else 0.0
                        else:
                            # fallback to global stats
                            b_mean = np.nanmean(col_nonan)
                            b_std = np.nanstd(col_nonan)
                        n_missing = np.sum(batch_mask_cov)
                        fill_vals = np.random.normal(loc=b_mean, scale=max(b_std, 1e-8), size=n_missing)
                        covariates_numeric[batch_mask_cov, col_idx] = fill_vals
                        report.text_simple(
                            f"Covariate column {col_idx} (continuous): replaced {n_missing} "
                            f"NaNs in batch '{b}' with Gaussian noise (mean={b_mean:.4f}, std={b_std:.4f})"
                        )

        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple("This pipeline breaks down batch analysis into the following tests:\n" \
        "1. Multivariate visualisation of batch differences (MAD histograms)," \
        "2. Univariate additive tests (Cohen's D for mean differences) " \
        "3. Multivariate additive tests (Mahalanobis distance) \n "
        "4. LMM diagnostics, unique variance explained by batch and model fit comparisson\n"\
        "5. Multiplicative tests (Variance ratio test for variance differences between batches) \n" \
        "6. Correlation between batch, covariates and principal components\n" \
        "7. PCA eigenvalues by batch and overall difference in covariance structure\n"\
        "8. UMAP visualization of batch and covariate clusters\n"
        "9. Population similarity between batches using univariate two sample Kolmogorov-Smirnov test and multivariate MMD test\n" 
        )
        report.text_simple(line_break_in_text)

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        logger.info("Generating Z-score normalization visualization")

        report.text_simple("Z-score normalization (median-centred) visualization across batches,\n" \
        "Here, we convert each feature to a median absolute deviation (MAD) and express each observation as a histogram.\n " \
        "As the normalisation is done globally, batchwise histograms that appear differently (width or location) indicate batch differences in mean and/or variance across features. ")


        zscored_data = DiagnosticFunctions.robust_z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)

        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch, covariates=covariates_numeric,covariate_names=covariate_names, covariate_types=covariate_types)
        report.text_simple("Cohen's D test for mean differences completed")

        # Plot (PlotDiagnosticResults should call rep.log_plot internally; our report.log_section ensures plots are attached)
        PlotDiagnosticResults.Cohens_D_plot(cohens_d_results, pair_labels=pairlabels, rep=report)
        report.log_text("Cohen's D plot added to report")
        data_dict = {}
        # Summaries per pair
        for i, (b1, b2) in enumerate(pairlabels):
            report.text_simple(f"Summary of Cohen's D results for batch comparison: {b1} vs {b2}")
            cohens_d_pair = cohens_d_results[i, :]
            if save_data:
                data_dict[f"CohensD_{b1}_vs_{b2}"] = cohens_d_pair
                
            small_effect = (np.abs(cohens_d_pair) < 0.2).sum()
            medium_effect = ((np.abs(cohens_d_pair) >= 0.2) & (np.abs(cohens_d_pair) < 0.5)).sum()
            large_effect = (np.abs(cohens_d_pair) >= 0.5).sum()
            report.text_simple(
                f"Number of features with small effect size (|d| < 0.2): {small_effect}\n"
                f"Number of features with medium effect size (0.2 <= |d| < 0.6): {medium_effect}\n"
                f"Number of features with large effect size (|d| >= 0.6): {large_effect}\n"
            )
        from DiagnoseHarmonisation.SaveDiagnosticResults import save_test_results
        
        if save_data:
            save_test_results(data_dict,
            test_name="Cohens_D",
            save_root=save_dir,
            feature_names=feature_names,
            report_date=report_date, 
            report_name=report_name,
            )
        report.text_simple("Cohen's D test summaries added to report and saved as csv if requested")
        report.text_simple(line_break_in_text)

        # Mahalanobis
        report.log_section("mahalanobis", "Mahalanobis distance test")
        logger.info("Doing Mahalanobis distance test for multivariate mean differences")
        mahalanobis_results = DiagnosticFunctions.Mahalanobis_Distance(data, batch, covariates=covariates_numeric)
        report.log_text("Mahalanobis distance test for multivariate mean differences completed")
        PlotDiagnosticResults.mahalanobis_distance_plot(mahalanobis_results, rep=report)
        report.log_text("Mahalanobis distance plot added to report")

        # Summaries from mahalanobis_results
        pairwise_distances = mahalanobis_results.get("pairwise_raw", {})
        for (b1, b2), dist in pairwise_distances.items():
            report.text_simple(f"Mahalanobis distance between {b1} and {b2}: {dist:.4f}")

        centroid_distances = mahalanobis_results.get("centroid_raw", {})
        for b, dist in centroid_distances.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid: {dist:.4f}")

        centroid_resid_distance = mahalanobis_results.get("centroid_resid", {})
        for b, dist in centroid_resid_distance.items():
            report.text_simple(f"Mahalanobis distance of {b} to overall centroid after residualising by covariates: {dist:.4f}")
        data_dict = {}
        if save_data:
            for b, dist in centroid_distances.items():
                data_dict[f"Mahonalobis_Centroid_Batch{b}"] = dist
            for b, dist in centroid_resid_distance.items():
                data_dict[f"Mahonalobis_Centroid_Resid_Batch{b}"] = dist
        
        save_test_results(data_dict,
        test_name="Mahalanobis_Distance",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date, 
        report_name=report_name,
        )
        report.text_simple("Mahalanobis distance test summaries added to report and saved as csv if requested")
        report.text_simple(line_break_in_text)
        # ---------------------
        # Mixed model tests
        # ---------------------
        logger.info("Beginning LMM diagnostics")
        report.log_section("lmm_diagnostics", "Linear mixed effects diagnostics (batch + covariates)")
        report.text_simple("Fitting per-feature LMMs (random intercept for batch). Where LMM fails or batch variance is zero we fallback to OLS fixed-effects.")

        from DiagnoseHarmonisation import temp
        # run LMM diagnostics
        lmm_results_df, lmm_summary = DiagnosticFunctions.Run_LMM_cross_sectional(
        data,
        batch,
        covariates=covariates,
        feature_names=feature_names,
        covariate_names=covariate_names,
        min_group_n=2
    )

        report.text_simple("LMM diagnostics completed.")
        report.log_text("LMM results table added to report")

        report.text_simple(
            f"Number of features analyzed: {lmm_summary.get('n_features', 0)}\n"
            f"Features where LMM succeeded: {lmm_summary.get('succeeded_LMM', 0)}\n"
            f"Features using fallback (OLS or skipped): {lmm_summary.get('used_fallback', 0)}"
    )

        # list common notes
        note_lines = []
        for tag, count in sorted(lmm_summary.items(), key=lambda x: -x[1])[:10]:
            if tag == 'n_features':
                continue
            note_lines.append(f"{tag}: {count}")
        report.text_simple("LMM diagnostics notes (top):\n" + "\n".join(note_lines))
        data_dict = {}
        # Save DF if needed
        if save_data:
            data_dict = lmm_results_df
        
        # Save LMM results as csv
        save_test_results(data_dict,
        test_name="LMM_fitting_results",
        save_root=save_dir,
        feature_names=feature_names,
        report_date=report_date,
        report_name=report_name
        )

        report.text_simple("Histogram of ICC (proportion of variance explained by batch):")
        # How to interpret ICC:
        report.text_simple("Intraclass Correlation Coefficient (ICC) is the ratio of variance due to batch effects to the total variance (batch + residual). \n" 
        "It quantifies the extent to which batch membership explains variability in the data.")
        report.text_simple(
            "Interpretation of ICC values:\n"
            "- ICC close to 0: Little to no variance explained by batch; suggests minimal batch effect.\n"
            "- ICC around 0.1-0.3: Small batch effect; may be acceptable depending on context.\n"
            "- ICC around 0.3-0.5: Moderate batch effect; consider further investigation or correction.\n"
            "- ICC above 0.5: Strong batch effect; likely requires correction to avoid confounding.\n"
        )
        
        # Plot conditional and marginal R^2 per feature, indicate what each means for interpretation
        report.text_simple("Marginal R² represents the variance explained by fixed effects (covariates)\n"
                           "while Conditional R² represents the variance explained by both fixed and random effects (batch + covariates).")
        lmm_r = lmm_results_df[['R2_marginal', 'R2_conditional']].dropna()
        lmm_figs = PlotDiagnosticResults.LMM_Diagnostics_Plot(
        lmm_results_df,
        feature_order="original",
        include_delta_r2=True,
        include_status_summary=True,
    )

        for caption, fig in lmm_figs:
            report.log_plot(fig, caption=caption)
            plt.close(fig)

        # ---------------------
        # Multiplicative tests
        # ---------------------
    
        # Variance ratio
        report.log_section("variance_ratio", "Variance ratio test (F-test) for variance differences between batches")
        logger.info("Variance ratio test between each unique batch pair")
        mode = ratio_type
        variance_ratios, pair_labels = DiagnosticFunctions.Variance_Ratios(
            data,
            batch,
            covariates=covariates_numeric,
            covariate_names=covariate_names,
            covariate_types=covariate_types,
            mode = mode
        )


        # Summarise variance ratio results
        data_dict = {}
        Ratios={}
        summary_rows = []

        # variance_ratios is (num_pairs x num_features)
        n_pairs = variance_ratios.shape[0]

        for i in range(n_pairs):
            ratios = np.array(variance_ratios[i], dtype=float)

            # Safe log: treat non-positive values as NaN for log stats
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratios = np.where(ratios > 0, np.log(ratios), np.nan)

            mean_log = np.nanmean(log_ratios)
            median_log = np.nanmedian(log_ratios)
            iqr_log = np.nanpercentile(log_ratios, [25, 75])
            # Proportion > 0: treat NaNs as False
            prop_higher = np.nanmean(np.where(np.isnan(log_ratios), False, log_ratios > 0))

            # exponentiate summary stats where meaningful
            median_ratio = np.exp(median_log) if not np.isnan(median_log) else np.nan
            mean_ratio = np.exp(mean_log) if not np.isnan(mean_log) else np.nan

            label = pair_labels[i]
            # Try to split label into two parts like "A / B" -> b1, b2. Otherwise keep label as-is.
            if isinstance(label, str) and " / " in label:
                b1, b2 = [s.strip() for s in label.split(" / ", 1)]
            else:
                # fallback: present full label in Batch 1, leave Batch 2 empty
                b1 = label
                b2 = ""

            summary_rows.append({
                "Batch 1": b1,
                "Batch 2": b2,
                "Median log ratio": median_log,
                "Mean log ratio": mean_log,
                "IQR lower": iqr_log[0],
                "IQR upper": iqr_log[1],
                "Prop > 0": prop_higher,
                "Median ratio (exp)": median_ratio,
                "Mean ratio (exp)": mean_ratio,
            })

            # sanitize label for keys (replace spaces and parentheses)
            safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_vs_")

            Ratios[f"VarianceRatio_{safe_label}"] = ratios
            data_dict[f"MedianLogVarianceRatio_{safe_label}"] = median_log
            data_dict[f"MeanLogVarianceRatio_{safe_label}"] = mean_log
            data_dict[f"IQRLowerLogVarianceRatio_{safe_label}"] = iqr_log[0]
            data_dict[f"IQRUpperLogVarianceRatio_{safe_label}"] = iqr_log[1]
            data_dict[f"PropHigherLogVarianceRatio_{safe_label}"] = prop_higher
            data_dict[f"MedianVarianceRatioExp_{safe_label}"] = median_ratio
            data_dict[f"MeanVarianceRatioExp_{safe_label}"] = mean_ratio

            # human-readable report line
            report.text_simple(
                f"Variance ratio {label}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in {b1}"
            )
            report.log_text("Variance ratio test completed")

        # save variance ratios raw:
        if save_data:
            save_test_results(
                Ratios,
                test_name="Variance_Ratios_Raw",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        # Save summary as well
        if save_data:
            save_test_results(
                data_dict,
                test_name="Variance_Ratio_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        summary_df = pd.DataFrame(summary_rows)
        report.text_simple("Variance ratio test summaries (per batch pair):")
        # Plot using your plot function
        PlotDiagnosticResults.variance_ratio_plot(variance_ratios, pair_labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")
        # ---------------------
        # PCA and clustering
        # ---------------------
        report.log_section("pca", "PCA & covariate correlations")
        logger.info("Running PCA")
        if covariates is not None:
            if covariate_names is None or len(covariate_names) != covariates.shape[1]:
                logger.warning("Variable names not provided or do not match number of covariates. Using defaults.")
                covariate_names = ["batch"] + [f"covariate_{i+1}" for i in range(covariates.shape[1])]
            else:
                logger.info(f"Using provided variable names: {covariate_names}")
        else:
            covariate_names = ["batch"]

        variable_names = ["batch"] + covariate_names
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch, covariates=covariates_numeric, variable_names=variable_names
        )

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")
    
        PlotDiagnosticResults.PC_corr_plot(
            score, batch, covariates=covariates_numeric, variable_names=covariate_names,
            PC_correlations=True, rep=report, show=False
        )
        report.log_text("PCA correlation plot added to report")

        # Demean the data before PCA to avoid mean differences dominating first PC (i.e don't force PC'S > 1 to be orthogonal to mean)
        #data_demeaned = data - np.mean(data, axis=0)
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch,N_components=20, covariates=covariates_numeric, variable_names=variable_names
        )

        data_dict = {}
        # save just the scores, not the full PCA object
        # Give each PC a name like PC1, PC2, ... as is more intuitive
        if save_data:
            n_pcs = score.shape[1]
            for pc_idx in range(n_pcs):
                pc_name = f"PC{pc_idx + 1}"    
                data_dict[pc_name] = score[:, pc_idx]
            # Create dummy index to replace feature names as in the dictionary, each PC is length of subjects:
            feature_names = [f"Feature_{idx+1}" for idx in range(n_pcs)]

            save_test_results(data_dict,
                test_name="PCA_Scores_demeaned",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        report.log_section("Eigenvalue_Scree", "PCA Eigenvalues and Covariance Structure")  
        report.text_simple("Using the computed PCA from earlier, visualise the eigenvalues associated with each principal component (PC) \n")
        report.text_simple("Display as an Eigenvalue Spectrum comparisson across batches and display Fronenius norm of the covariance matrices across batches \n")
        report.text_simple(line_break_in_text)

        logger.info("Generating PCA Eigenvalue Scree Plot:")
        report.text_simple("The scree plot displays the eigenvalues associated with each principal component (PC). \n")
        report.text_simple("Using this test, we can see if the variance by batch is the same across all PCs \n")
        report.text_simple("In short, the steepness and shape of the batchwise plots can help to differ batchwise differences in the variance structure across features \n")
        spectra_res = PlotDiagnosticResults.plot_eigen_spectra_and_cumulative(score, batch, rep=report)
        logger.info("PCA Eigenvalue Scree Plot added to report")
        report.text_simple(line_break_in_text)


        # Change Frobenius norm plot to be based on the covariance matrices of the data for each batch, rather than the PCA scores, as this is more interpretable in terms of the original features and their covariance structure (which is what we want to compare across batches)
        logger.info("Generating Frobenius Norm Plot")
        report.text_simple("The Frobenius norm plot displays the pairwise Frobenius norms of the covariance matrices between batches. \n")
        report.text_simple("Using this test, we can see if the covariance structure by batch is the same across all batches \n")
        report.text_simple("In short, larger Frobenius norms between batches indicate greater differences in covariance structure across features \n")
        report.text_simple("We calculate the overall covariance matrix for each batch and then compute the pairwise Frobenius norms between these covariance matrices. \n")
        covres = PlotDiagnosticResults.plot_covariance_frobenius(data, batch, rep=report)

        logger.info("Frobenius norm plot between covariance matrices added to report")
        report.text_simple(line_break_in_text)

        report.log_section("Clustering", "Clustering and visualiation of batch and covariate clusters")
        logger.info("Beginning cluster visulisation of batch and covariate clusters")

        report.text_simple("Using UMAP and PCA to visualise clustering of samples by batch and covariates. \n" \
        "If samples cluster by batch more strongly than covariates, this indicates strong batch effects. \n" \
        "We include both PCA and UMAP visualisations to show both linear and non-linear clustering patterns. \n" \
        "If you see clear clustering by batch in either PCA or UMAP, this suggests strong batch effects that may require correction. \n" \
        "If you see clustering by covariates, this suggests that covariates are driving some of the variance in the data, which may be important to account for in harmonisation. \n" \
        "If you see no clear clustering by either batch or covariates, this suggests minimal batch effects and that the data may be relatively homogeneous across batches. \n")


        if len(data) > 1000:
            logger.info("Large dataset detected, this could make UMAP very slow, especially if not using GPU.")

        PlotDiagnosticResults.clustering_analysis_all(score,
        data,
        batch,
        covariates=covariates,
        rep=report,
        variable_names=covariate_names,
        UMAP_embedding=True)
        plt.close()
        logger.info("Clustering visualizations added to report")

        # ---------------------
        # Distribution tests (KS)
        # ---------------------
        report.log_section("ks", "Two-sample Kolmogorov-Smirnov tests")
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_Test(data, batch, feature_names=None, covariates=covariates_numeric, do_fdr=True,residualize_covariates=True,
                                                 covariate_names=covariate_names,covariate_types=covariate_types)
        
        report.log_text("Two-sample Kolmogorov-Smirnov test completed")

        for key, value in ks_results.items():
            if key != "params":
                logger.info(f"Key: {key}, Value type: {type(value)}")

        report.text_simple(
            "- each value is a dict with:\n"
            "    'statistic': np.array of D statistics (length n_features)\n"
            "    'p_value': np.array of p-values (nan where test not run)\n"
            "    'p_value_fdr': np.array of BH-corrected p-values (if do_fdr else None)\n"
            "    'n_group1': array of sample counts per feature for group1\n"
            "    'n_group2': array of counts for group2\n"
        )
        report.text_simple("The KS test compares the distribution of each feature between batches. \n" \
        "A significant KS test (low p-value) indicates that the distribution of that feature differs between the groups being compared (either batch vs overall, or batch vs batch) being compared. \n" \
        "The D statistic indicates the magnitude of the distribution difference, with higher values indicating greater differences. \n" \
        "By examining the KS test results across features, we can identify which features show the most significant distribution differences between batches, which can inform our choice of harmonisation method and whether to apply it globally or on specific features. \n")

        report.text_simple("Users should look both at the plot by P-value magnitude and in the distributions of D statistics and p-values across features. \n" \
                           "If specific clusters of features show significant KS differences, this may indicate that certain types of features are more affected by batch effects and may benefit from targeted harmonisation approaches. \n" \
                           "If KS differences are widespread across many features, this may indicate more global batch effects that could benefit from global harmonisation approaches. \n" \
                           "If KS differences are minimal, this may indicate minimal batch effects and that harmonisation may not be necessary. \n")
        data_dict = {}
        if save_data:
            for key, value in ks_results.items():
                if key != "params":
                    data_dict[f"KS_Stat_{key}"] = value["statistic"]
                    data_dict[f"KS_PValue_{key}"] = value["p_value"]
                    if value.get("p_value_fdr") is not None:
                        data_dict[f"KS_PValueFDR_{key}"] = value["p_value_fdr"]
                    
            save_test_results(data_dict,
                test_name="KS_Test",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )

        PlotDiagnosticResults.KS_plot(ks_results, rep=report)
        report.log_text("Two-sample Kolmogorov-Smirnov test plot added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")

        # Save data dictionary as csv if requested 
        report.log_section("Summary","Summary of Diagnostic Report and Advice")
        report.text_simple("Summary of diagnostic findings and advice for harmonisation:")
        report.text_simple("Based on the diagnostic tests performed, we can summarise the major findings regarding batch differences in the data. \n" \
                           "We can also provide advice on which harmonisation methods may be most appropriate given the observed batch effects. \n")

        batch_sizes = {b: np.sum(batch == b) for b in unique_batches}
        advice_summary = _generate_harmonisation_advice(
            cohens_d_results=cohens_d_results,
            mahalanobis_results=mahalanobis_results,
            lmm_results_df=lmm_results_df,
            variance_summary_df=summary_df,
            covariance_results=covres,
            batch_sizes=batch_sizes,
        )

        for advice_line in advice_summary["advice_lines"]:
            report.text_simple(advice_line)
        
        return data_dict if save_data else None

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report
            report_ctx.__exit__(None, None, None)
# Longitudinal testing:
from typing import Optional, Union
def LongitudinalReport(data, batch,
                          subject_ids,
                          timepoints,
                          covariates=None,
                          covariate_names=None,
                          features = None,
                          save_data: bool = False,
                          save_data_name: Optional[str]= None,
                          save_dir: Optional[Union[str,os.PathLike]] = None,
                          report_name: Optional[str] = None,
                          SaveArtifacts: bool = False,
                          rep= None,
                          show: bool = False,
                          timestamped_reports: bool = True):
    """
    Create a diagnostic report for dataset differences across batches in longitudinal data.

    Args: 
        data (np.ndarray): Data matrix (samples x features).
        batch (list or np.ndarray): Batch labels for each sample.
        subject_ids (list or np.ndarray): Subject IDs for each sample.
        covariates (np.ndarray, optional): Covariate matrix (samples x covariates).
        covariate_names (list of str, optional): Names of covariates.
        save_data (bool, optional): Whether to save input data and results.
        save_data_name (str, optional): Filename for saved data.
        save_dir (str or os.PathLike, optional): Directory to save report and data.
        report_name (str, optional): Name of the report file.
        SaveArtifacts (bool, optional): Whether to save intermediate artifacts.
        rep (StatsReporter, optional): Existing report object to use.
        show (bool, optional): Whether to display plots interactively.
    
    Outputs:
        Generates an HTML report with diagnostic plots and statistics for longitudinal data.
        If `save_data` is True, also returns a dictionary and csv with input data and results.
        If SaveArtifacts is True, saves intermediate plots to `save_dir`.
    Note:
        This function is designed for repeated data where we do not expect to see a longitudinal trent over time.
        If need arises, we will revise this to include an additional function where we would expect to see a longitudinal trend and want to test for that explicitly.
    
    """
    from pprint import pformat
    from DiagnoseHarmonisation import DiagnosticFunctionsLong

    # Check inputs and revert to defaults as needed 

    # Check inputs and revert to defaults as needed
    if save_dir is None:
        save_dir = Path.cwd()
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if report_name is None:
        base_name = "LongitudinalReport.html"
    else:
        base_name = report_name if report_name.endswith(".html") else report_name + ".html"

    if timestamped_reports:
        stem, ext = base_name.rsplit(".", 1)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{stem}_{timestamp_str}.html"

    # Helper to configure a report object
    def _configure_report(report_obj):
        report_obj.save_dir = save_dir
        report_obj.report_name = base_name
        # write an initial report (optional) and log the path
        rp = report_obj.write_report()  # writes to report_obj.report_path
        report_obj.log_text(f"Initialized HTML report at: {rp}")
        print(f"Report will be saved to: {rp}")
        return report_obj

    # If user passed a report object, use it (do not close it here).
    # Otherwise create one and use it as a context manager so it's closed on exit.
    created_local_report = False
    if rep is None:
        created_local_report = True
        report_ctx = StatsReporter(save_artifacts=SaveArtifacts, save_dir=None)
    else:
        report_ctx = rep

    # If we're using our own, enter the context manager
    if created_local_report:
        ctx = report_ctx.__enter__()  # type: ignore
        report = ctx
    else:
        report = report_ctx
        # Report begins here within try block: ***NOTE: may change in the future to run main code outside try/finally if needed***
    try:
        logger = report.logger

        # configure save dir/name and write initial stub report
        _configure_report(report)

        line_break_in_text = "-" * 125
        unique_subjects = set(subject_ids)
        # Basic dataset summary
        report.text_simple("Summary of dataset:")
        report.text_simple(line_break_in_text)
        report.log_text(
            f"Analysis started\n"
            f"Number of measures: {data.shape[0]}\n"
            f"Unique subjects: {len(set(subject_ids))}\n"
            f"Number of features: {data.shape[1]}\n"
            f"Unique batches: {set(batch)}\n"
            f"Unique Covariates: {set(covariate_names) if covariate_names is not None else set()}\n"
            f"HTML report: {report.report_path}\n"
        )
        report.text_simple(line_break_in_text)

        # Ensure batch is numeric array where needed
        logger.info("Checking data format")
        if isinstance(batch, (list, np.ndarray)):
            batch = np.array(batch)
            if batch.dtype.kind in {"U", "S", "O"}:  # string/object categorical
                logger.info(f"Original batch categories: {list(set(batch))}")
                logger.info("Creating numeric codes for batch categories")
                batch_numeric, unique = pd.factorize(batch)
                logger.info(f"Numeric batch codes: {list(set(batch_numeric))}")
                # keep string labels in `batch` if plotting expects them; numeric conversions can be used inside tests as needed
        else:
            raise ValueError("Batch must be a list or numpy array")
        
        
        # Check that covariates are an array if provided (.shape[1] throwing error with a list), convert to array if needed
        if covariates is not None:
            if isinstance(covariates, list):
                covariates = np.array(covariates)
            elif isinstance(covariates, dict):
                pass
            elif not isinstance(covariates, np.ndarray):
                raise ValueError(f"Covariates must be a numpy array or list if provided, covariates type: {type(covariates)}")

                    
        # Check if there is only one covariate and convert to 2D array if that is the case (avoid shape issue in next call):
        try:
            if covariates is not None and covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
        except AttributeError:
            pass

        # Prepare save-data dict if requested
        if save_data:
            data_dict = {}
            data_dict["batch"] = batch
            if covariates is not None:
                for i in range(covariates.shape[1]):
                    if covariate_names is not None and i < len(covariate_names):
                        cov_name = covariate_names[i]
                    else:
                        cov_name = f"covariate_{i+1}"
                    data_dict[cov_name] = covariates[:, i]
            if save_data_name is None:
                save_data_name = "DiagnosticReport_InputData.csv"
        else:
            data_dict = None
        # Check batch, subject_ids, and data dimensions
        #if not (len(batch) == len(subject_ids) == data.shape[0]):
        #    raise ValueError("Length of batch and subject_ids must match number of samples in data")
        #if len(covariates) is not None and len(covariates) != data.shape[0]:
        #    raise ValueError("Number of rows in covariates must match number of samples in data")
        #if len(covariate_names) is not None and len(covariate_names) != covariates.shape[1]:
        #    raise ValueError("Length of covariate_names must match number of columns in covariates")
        

        report.log_section("Introduction", "Longitudinal Data Diagnostic Report Introduction")
        report.text_simple(
    "This report provides diagnostic analyses for longitudinal data collected "
    "across multiple batches.\n\n"
    "Longitudinal data consist of repeated measurements from the same subjects "
    "over time. Such designs require evaluation of measurement stability, "
    "batch-related variability, and preservation of biologically meaningful signal.\n\n"
    "The following diagnostics are performed:\n\n"
    "1. Subject-level variability\n"
    "   • Subject order consistency (rank preservation across timepoints)\n"
    "   • Within-subject variability (Coefficient of Variation / Relative Percent Difference)\n\n"
    "2. Batch-level variability\n"
    "   • Additive batch effects (mean shifts; mixed-effects models)\n"
    "   • Pairwise batch mean differences (post-hoc comparisons)\n"
    "   • Multiplicative batch effects (variance differences; Fligner–Killeen test)\n"
    "   • Multivariate batch differences relative to a reference (Mahalanobis distance)\n\n"
    "3. Between-subject variability\n"
    "   • Intra-Class Correlation (ICC; variance decomposition via mixed models)\n\n"
    "4. Biological variability\n"
    "   • Statistical significance of biological covariates\n"
    "   • Effect sizes (beta coefficients)\n"
    "   • 95% confidence intervals\n\n"
    "Together, these diagnostics assess whether harmonisation reduces unwanted "
    "batch effects while preserving meaningful biological and between-subject variation."
)

        report.log_section(
            "subject_order_consistency",
            "Subject-level variability: Subject order consistency analysis"
        )

        report.log_text(
            "This metric evaluates whether subjects preserve their relative ranking "
            "between two timepoints."
        )

        report.log_text(
            "METHOD: We compute Spearman’s rank correlation (ρ) between subject-level "
            "values at the two timepoints."
        )

        report.log_text(
            "To assess statistical significance, we perform a permutation test by "
            "randomly shuffling subject labels at one timepoint and recomputing ρ "
            "across many iterations to generate a null distribution."
        )

        report.log_text(
            "The permutation p-value represents the proportion of shuffled correlations "
            "that are equal to or greater than the observed ρ."
        )

        report.log_text(
            "INTERPRETATION: Higher ρ values indicate stronger preservation of subject "
            "ranking across timepoints, reflecting greater within-subject consistency."
        )

        report.log_text(
            "A permutation p-value < 0.05 (*) suggests that the observed consistency "
            "is unlikely to arise by chance under random subject labeling."
        )

        report.log_text(
            "NOTE: This metric is most applicable in test–retest or traveling-subjects "
            "designs where the same individuals are measured repeatedly."
        )


        # Subject-level: Subject order consistency
        subjorder = DiagnosticFunctionsLong.SubjectOrder_long(idp_matrix=data,
                                                          subjects=subject_ids,
                                                          timepoints=timepoints,
                                                          idp_names=features,
                                                          nPerm=100)
        print("\nSUBJECT ORDER CONSISTENCY: RANK CORRELATIONS WITH PERMUTATION TESTS")
        print(subjorder)
        PlotDiagnosticResults.plot_SubjectOrder(subjorder,                 
                              ncols=2,
                              figsize_per_plot=(3.6,3.6),
                              limit_idps=None,
                              sample_method='random',
                              random_state=42,
                              rep=report) 
        report.log_text("Subject order consistency plots added to report")

        # Subject-level: Within Subject Consistency 
        report.log_section(
             "Within_subject_variability",
             "Subject-level variability: Within-subject variability analysis"
        )

        report.log_text(
            "This metric quantifies the magnitude of variation in each subject’s "
            "feature values across timepoints."
        )

        report.log_text(
            "METHOD: For datasets with more than two timepoints, we compute the "
            "Coefficient of Variation (CV = standard deviation / mean) within each subject."
        )

        report.log_text(
            "For datasets with exactly two timepoints, we compute the Relative Percent "
            "Difference (RPD), defined as the absolute difference between timepoints "
            "divided by their mean."
        )

        report.log_text(
            "All variability metrics are computed at the subject level and "
            "summarized across subjects and features. "
        )

        report.log_text(
            "INTERPRETATION: Lower CV or RPD values indicate lower within-subject "
            "variability across timepoints."
        )

        report.log_text(
            "Lower variability reflects greater measurement stability and reduced "
            "non-biological variation in the features."
        )

        report.log_text(
            "NOTE: Variability estimates should be interpreted alongside between-subject "
            "variability to ensure that reductions are not due to over-smoothing or "
            "loss of biologically meaningful signal."
        )

        report.log_text(
        "NOTE: This metric applies to any repeated-measures dataset. "
        "In short-term test–retest settings, variability primarily reflects "
        "measurement noise. In longitudinal studies, variability may reflect "
        "both biological change and technical variation and should be interpreted accordingly."
        )

        wsv = DiagnosticFunctionsLong.WithinSubjVar_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            idp_names=features,
                          )
        print("\nWITHIN SUBJECT VARIABILITY: BETWEEN TIMEPOINTS")
        print(wsv)
        PlotDiagnosticResults.plot_WithinSubjVar(
            wsv,
            subject_col='subject',
            limit_subjects=35,
            limit_idps_for_legend=10,
            rep=report
            )
        report.log_text("Within subject variability plots added to report")

         # Batch-level: Additive batch effects
        report.log_section(
             "Additive_batch_effects_mixed_models",
            "Batch variability (Univariate): Additive batch effect analysis (mean shift)"
        )

        report.log_text(
            "This analysis evaluates whether batch membership explains additional "
            "variance in feature values beyond subject-level variability."
        )

        report.log_text(
            "METHOD: For each feature, we fit a linear mixed-effects model including "
            "batch as a fixed effect and subject as a random effect."
        )

        report.log_text(
            "We compare this full model to a nested model without the batch term "
            "using a Kenward–Roger F-test to assess whether including batch "
            "significantly improves model fit."
        )

        report.log_text(
            "The resulting p-values are reported in the corresponding plots."
        )

        report.log_text(
            "INTERPRETATION: Non-significant p-values suggest no evidence of "
            "additive batch effects (i.e., no systematic mean shift across batches)."
        )

        report.log_text(
            "Significant p-values indicate that batch membership explains additional "
            "variance in feature means, consistent with residual batch-related "
            "mean differences."
        )

        report.log_text(
            "When comparing harmonisation strategies, a reduction in the number "
            "or magnitude of significant batch effects indicates improved removal "
            "of additive (mean shift) batch variability."
        )

        report.log_text(
            "NOTE: This analysis assesses mean differences only and does not "
            "evaluate variance differences or multivariate batch structure."
        )

        addeff,model_defs_add = DiagnosticFunctionsLong.AdditiveEffect_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batch_name=batch,
            idp_names=features,
            covariates=covariates,
            #fix_eff=["age", "sex"],   # fixed effects
            #ran_eff=["subjects"],            # random intercepts
            do_zscore=True,                  # z-score predictors AND response per feature
            reml=False,
            verbose=True)
        print("\nRESULTS: ADDITIVE EFFECTS")
        print(addeff)
        #report.log_text(pformat(model_defs_add, width=60, sort_dicts=False))
        PlotDiagnosticResults.plot_AddMultEffects(addeff,
                                     feature_col='Feature',
                                     p_col='p-value',
                                     labels=['Additive batch effect'],
                                     p_thr=0.05,
                                     annot_fmt="{:.3f}",
                                     value_scale='p',
                                     figsize=(6,8),
                                     rep=report)
        report.log_text("Additive batch effect plot added to report")
        
        # Batch-level: Pairwise batch comparison
        report.log_text(
           "We test the fixed effect of batch within the mixed-effects framework "
           "and perform post-hoc pairwise comparisons between batches to evaluate "
           "differences in feature means."
        )

        report.log_text(
                "For each feature, we compute the number of batch pairs showing "
                "statistically significant mean differences following multiple "
                "comparison correction.",
        )

        report.log_text(
            "INTERPRETATION: A higher number of significant batch pairs indicates "
            "stronger residual batch-related mean differences."
        )

        report.log_text(
            "A reduction in the number of significant batch pairs after harmonisation "
            "suggests improved mitigation of additive batch effects."
        )


        mf,model_defs = DiagnosticFunctionsLong.MixedEffects_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batches=batch,
            idp_names=features,
            covariates=covariates,  # optional
            p_corr=1
            #fix_eff=["age","sex"],   # batch is included automatically
            #ran_eff=["subjects"],
            #force_categorical=["sex"],
            #force_numeric=["age"],
            #zscore_var=["age"]
            ) 
        print("\nMIXED EFFECTS OUTPUTS:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(mf) 
        #report.log_text(pformat(model_defs, width=60, sort_dicts=False))

        n_batches = len(np.unique(batch))
        total_pairs = n_batches * (n_batches - 1) // 2
        report.log_text(f"Total number of pairs {total_pairs} for {len(np.unique(batch))} batches")

        if len(features) < 30:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['n_is_batchSig'],
                        plot_type='bar',
                        seed=123,
                        figsize=(16,4), rep=report)
        else:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                      idp_col='IDP',
                      metrics=['n_is_batchSig'],
                      plot_type='box',
                      limit_idps=2,
                      seed=123,
                      figsize=(16,4), rep=report)
        report.log_text("Pairwise batch variability plots added to report")

        # Multiplicative batch effects
        report.log_section(
             "Multiplicative_batch_effects_Fligner_Killeen",
             "Batch variability (Univariate): Multiplicative batch effect analysis (variance scaling)"
        )

        report.log_text(
            "This analysis evaluates whether feature variability differs across batches, "
            "indicating potential multiplicative (scaling) batch effects."
        )

        report.log_text(
            "METHOD: After adjusting for covariate effects, we apply the "
            "Fligner–Killeen test to compare feature variances across batches."
        )

        report.log_text(
                "The Fligner–Killeen test is a non-parametric, robust test for "
                "homogeneity of variances that is less sensitive to deviations "
                "from normality.",
        )

        report.log_text(
            "For each feature, the resulting p-value assesses whether variance "
            "differs significantly between batches."
        )

        report.log_text(
            "INTERPRETATION: Non-significant p-values suggest no evidence of "
            "multiplicative (scaling) batch effects."
        )

        report.log_text(
            "Significant p-values indicate residual variance differences across "
            "batches, consistent with multiplicative batch-related effects."
        )

        report.log_text(
            "When comparing harmonisation strategies, a reduction in the number "
            "of features with significant variance differences indicates improved "
            "mitigation of scaling-related batch variability."
        )

        report.log_text(
            "NOTE: This test evaluates variance differences only and does not "
            "assess mean shifts or multivariate batch structure."
        )

        muleff,model_defs_mul = DiagnosticFunctionsLong.MultiplicativeEffect_long(
            idp_matrix=data,
            subjects=subject_ids,
            timepoints=timepoints,
            batch_name=batch,
            idp_names=features,
            covariates=covariates,
            #fix_eff=["age", "sex"],   # fixed effects
            #ran_eff=["subjects"],            # random intercepts
            do_zscore=True,                  # z-score predictors AND response per feature
            verbose=True)
        print("\nRESULTS: MULTIPLICATIVE EFFECTS")
        print(muleff)
        #report.log_text(pformat(model_defs_mul, width=60, sort_dicts=False))
        PlotDiagnosticResults.plot_AddMultEffects(muleff,
                                     feature_col='Feature',
                                     p_col='p-value',
                                     labels=['Multiplicative batch effect'],
                                     p_thr=0.05,
                                     annot_fmt="{:.3f}",
                                     value_scale='p',
                                     figsize=(6,8),
                                     rep=report)
        report.log_text("Multiplicative batch effect plots added to report")
       
       # Multivariate site differences using Mahalanobis distances
        report.log_section(
           "Multivariate_batch_difference_reference",
           "Batch variability (Multivariate): Difference from reference distribution"
        )

        report.log_text(
            "This analysis evaluates multivariate batch differences relative to a "
            "reference distribution."
        )

        report.log_text(
            "METHOD: For each batch, we compute Mahalanobis distances between "
            "feature vectors and the reference distribution, accounting for the "
            "covariance structure of the data."
        )

        report.log_text(
            pformat(
                "Mahalanobis distance measures how far a batch distribution lies "
                "from the reference in multivariate space while adjusting for "
                "correlations between features.",
                width=90,
                sort_dicts=False
            )
        )

        report.log_text(
            "We report both batch-wise average distances and the overall average "
            "distance across batches."
        )

        report.log_text(
            "INTERPRETATION: Lower Mahalanobis distances indicate that batch "
            "distributions more closely resemble the reference distribution."
        )

        report.log_text(
            "Reductions in distance after harmonisation suggest improved alignment "
            "of batch distributions in multivariate feature space."
        )

        report.log_text(
            "NOTE: Distance values depend on the dimensionality and covariance "
            "structure of the data; comparisons should therefore be made within "
            "the same dataset and analysis framework."
        )
        md = DiagnosticFunctionsLong.MultiVariateBatchDifference_long(
           idp_matrix=data,
           batch=batch,
           idp_names=features)
        print("\nMULTIVARIATE PAIRWISE SITE DIFFERENCES:")
        print(md)
        PlotDiagnosticResults.plot_MultivariateBatchDifference(md, rep=report) 
        report.log_text("Multivariate batch variability plots added to report")


        # Across subjects-level: Intraclass correlation and within/between subject variability
        report.log_section(
            "Between_subject_variability_mixed_models",
            "Between-subject variability (Univariate): Cross-subject variability analysis"
        )

        report.log_text(
            "This analysis quantifies the proportion of total variance attributable "
            "to differences between subjects."
        )

        report.log_text(
            "METHOD: We fit a linear mixed-effects model with subject included as "
            "a random effect to decompose variance into between-subject and "
            "within-subject components."
        )

        report.log_text(
                "From the variance components, we compute the Intra-Class Correlation "
        )

        report.log_text(
            "The ICC represents the proportion of total variance explained by "
            "between-subject differences."
        )

        report.log_text(
            "INTERPRETATION: Higher ICC values (closer to 1) indicate stronger "
            "between-subject differentiation relative to residual variability."
        )

        report.log_text(
            "When evaluating harmonisation strategies, preservation or improvement "
            "of ICC suggests that biologically meaningful between-subject signal "
            "is retained while reducing unwanted variability."
        )

        report.log_text(
            "NOTE: ICC interpretation depends on study design and model specification; "
            "comparisons should be made within the same analytical framework."
        )

        #report.log_text(pformat(model_defs, width=60, sort_dicts=False))
        if len(features) < 30:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['ICC'],
                        plot_type='bar',
                        seed=123,
                        figsize=(16,4), rep=report)
        else:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                      idp_col='IDP',
                      metrics=['ICC'],
                      plot_type='box',
                      limit_idps=2,
                      seed=123,
                      figsize=(16,4))
        report.log_text("ICC plot added to report")  
        
        # Biological variability
        report.log_section(
            "Biological_variability_mixed_models",
            "Biological variability analysis"
        )

        report.log_text(
            "This analysis evaluates whether biologically meaningful variables "
            "explain variation in feature values after accounting for subject-level "
            "and batch-related effects."
        )

        report.log_text(
            "METHOD: We fit linear mixed-effects models including biological "
            "covariates of interest as fixed effects, with subject and/or batch "
            "modeled appropriately depending on study design."
        )

        report.log_text(
            "Outputs include:"
        )

        report.log_text(
            "1) Statistical significance of biological covariates"
        )

        report.log_text(
            "2) Estimated effect sizes (beta coefficients)"
        )

        report.log_text(
            "3) 95% confidence intervals for effect estimates"
        )

        report.log_text(
            "INTERPRETATION: Significant biological effects with stable or increased "
            "effect sizes after harmonisation suggest preservation of meaningful signal."
        )

        report.log_text(
            "Attenuation or loss of biological associations may indicate "
            "over-correction or removal of true biological variability."
        )

        report.log_text(
            "Conversely, strong residual batch effects can obscure or bias "
            "biological associations, reducing interpretability."
        )

        report.log_text(
            "NOTE: Interpretation should consider effect size magnitude, direction, "
            "and confidence intervals rather than p-values alone."
        )
        inferred_fix = list(covariates.keys())
        PlotDiagnosticResults.plot_MixedEffectsPart2(mf,
                      idp_col='IDP',
                      fix_eff=inferred_fix,
                      p_thr=0.05,
                      figsize=(8,4),
                      rep=report)
        report.log_text("Biological variability plots added to report")

        # Finalize
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")
        
        report.log_section(
            "REFERENCES",
            "REFERENCES"
        )
        report.log_text("Subject order consistency metric: https://doi.org/10.1162/imag_a_00042\n")
        report.log_text("Multivariate batch differences: https://10.1016/j.neuroimage.2022.119768\n")
        report.log_text("Application to real dataset: https://doi.org/10.1002/alz70856_097537\n")


    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore
