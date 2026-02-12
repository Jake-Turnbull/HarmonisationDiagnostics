# Diagnostic report generation using DiagnosticFunctions 
from ensurepip import version
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from DiagnoseHarmonization import DiagnosticFunctions
from DiagnoseHarmonization import PlotDiagnosticResults
from DiagnoseHarmonization.LoggingTool import StatsReporter

def covariate_to_numeric(covariates):
    """
    Convert categorical covariates to numeric codes for corresponding functions that require numeric input (e.g. PCA correlation function).


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




def CrossSectionalReport(
    data,
    batch,
    covariates=None,
    covariate_names=None,
    save_data: bool = False,
    save_data_name: str | None = None,
    save_dir: str | os.PathLike | None = None,
    feature_names: list | None = None,
    report_name: str | None = None,
    SaveArtifacts: bool = False,
    rep= None,
    power_analysis: bool = False,
    show: bool = False,
    timestamped_reports: bool = True,
):
    """
    Create a diagnostic report for dataset differences across batches.

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
        power_analysis (bool, optional): Whether to perform power analysis.
        show (bool, optional): Whether to display plots interactively.
    
    Returns:
        HTML report saved to specified directory (or cd by default).
        dict or None: If save_data is True, returns a dictionary of saved data arrays.

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
        from DiagnoseHarmonization._version import version
        report.log_text(f"DiagnoseHarmonization version: {version}")
            
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

        # Begin tests
        logger.info("Beginning diagnostic tests")
        report.text_simple(" The order of tests is as follows: Additive tests, Multiplicative tests, Tests of distribution")
        report.text_simple(line_break_in_text)

        report.log_section("Z-score visualization", "Z-score normalization visualization")

        logger.info("Generating Z-score normalization visualization")
        report.text_simple("Z-score normalization (median-centred) visualization across batches, " \
        "the further the histograms are apart, the larger the mean batch differences on average across features." )
        report.text_simple("We show also here a heatmap sorted by batch for further visualisation of batch effects, " \
        "larger blocks of similar colours that are far from zero indicate larger batch effects compared to average across batches ")
        zscored_data = DiagnosticFunctions.z_score(data)
        PlotDiagnosticResults.Z_Score_Plot(zscored_data, batch, rep=report)
        report.log_text("Z-score normalization visualization added to report")
        report.text_simple(line_break_in_text)

        # Create numeric versions of covariates for tests that require numeric input (e.g. PCA correlation function)
        if covariates is not None: # Check if dictionary:
            if isinstance(covariates, dict):
                covariates_numeric = covariates.copy()
                for col in covariates_numeric.columns:
                    if covariates_numeric[col].dtype.kind in {"U", "S", "O"}:  # string/object categorical
                        covariates_numeric[col], _ = pd.factorize(covariates_numeric[col])
                        covariates = covariates_numeric
        else:
            covariates_numeric = None

        # Temp fix to convert covariates to numeric if they are not already, for functions that require numeric input (e.g. PCA correlation function). This will be replaced with a more robust solution in the future that handles mixed covariate types and allows user control over which covariates are converted.
        covariates_numeric = covariate_to_numeric(covariates)
        covariates = covariates_numeric
   

        # ---------------------
        # Additive tests
        # ---------------------
        report.log_section("cohens_d", "Cohen's D test for mean differences")
        logger.info("Cohen's D test for mean differences")
        cohens_d_results, pairlabels = DiagnosticFunctions.Cohens_D(data, batch, covariates=covariates)
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
        from DiagnoseHarmonization.SaveDiagnosticResults import save_test_results
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
        mahalanobis_results = DiagnosticFunctions.Mahalanobis_Distance(data, batch, covariates=covariates)
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
    
        # Variance ratio
        report.log_section("variance_ratio", "Variance ratio test (F-test) for variance differences between batches")
        logger.info("Variance ratio test between each unique batch pair")
        variance_ratio = DiagnosticFunctions.Variance_Ratios(data, batch, covariates=covariates)
        report.log_text("Variance ratio test between each unique batch pair completed")

        labels = [f"Batch {b1} vs Batch {b2}" for (b1, b2) in variance_ratio.keys()]
        ratio_array = np.array(list(variance_ratio.values()))

        # save variance ratios raw:
        if save_data:
            save_test_results(variance_ratio,
                test_name="Variance_Ratios_Raw",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )
        # Summarise variance ratio results
        data_dict = {}
        summary_rows = []
        for (b1, b2), ratios in variance_ratio.items():
            ratios = np.array(ratios)
            log_ratios = np.log(ratios)
            mean_log = np.mean(log_ratios)
            median_log = np.median(log_ratios)
            iqr_log = np.percentile(log_ratios, [25, 75])
            prop_higher = np.mean(log_ratios > 0)
            median_ratio = np.exp(median_log)
            mean_ratio = np.exp(mean_log)
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
            data_dict[f"VarianceRatio_Batch{b1}_vs_Batch{b2}"] = ratios
            data_dict[f"MedianLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = median_log
            data_dict[f"MeanLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = mean_log
            data_dict[f"IQRLowerLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = iqr_log[0]
            data_dict[f"IQRUpperLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = iqr_log[1]
            data_dict[f"PropHigherLogVarianceRatio_Batch{b1}_vs_Batch{b2}"] = prop_higher
            data_dict[f"MedianVarianceRatioExp_Batch{b1}_vs_Batch{b2}"] = median_ratio
            data_dict[f"MeanVarianceRatioExp_Batch{b1}_vs_Batch{b2}"] = mean_ratio

            report.text_simple(
                f"Variance ratio {b1} vs {b2}: median log={median_log:.3f} "
                f"(IQR {iqr_log[0]:.3f}–{iqr_log[1]:.3f}), "
                f"{prop_higher*100:.1f}% of features higher in batch {b1}"
            )
        
        # Save summary as well
        if save_data:
            save_test_results(data_dict,
                test_name="Variance_Ratio_Summary",
                save_root=save_dir,
                feature_names=feature_names,
                report_date=report_date,
                report_name=report_name,
            )
        
        summary_df = pd.DataFrame(summary_rows)
        report.text_simple("Variance ratio test summaries (per batch pair):")
        PlotDiagnosticResults.variance_ratio_plot(ratio_array, labels, rep=report)
        report.log_text("Variance ratio plot(s) added to report")
    
        report.text_simple(line_break_in_text)
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
            data, batch, covariates=covariates, variable_names=variable_names
        )

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {covariate_names}")
    
        PlotDiagnosticResults.PC_corr_plot(
            score, batch, covariates=covariates, variable_names=covariate_names,
            PC_correlations=True, rep=report, show=False
        )
        report.log_text("PCA correlation plot added to report")

        # Demean the data before PCA to avoid mean differences dominating first PC (i.e don't force PC'S > 1 to be orthogonal to mean)
        #data_demeaned = data - np.mean(data, axis=0)
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data, batch,N_components=20, covariates=covariates, variable_names=variable_names
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

        logger.info("PCA Frobenius norm plot added to report")
        report.text_simple(line_break_in_text)
    
        # ---------------------
        # Distribution tests (KS)
        # ---------------------
        report.log_section("ks", "Two-sample Kolmogorov-Smirnov tests")
        logger.info("Two-sample Kolmogorov-Smirnov test for distribution differences between each unique batch pair")
        ks_results = DiagnosticFunctions.KS_Test(data, batch, feature_names=None, covariates=covariates, do_fdr=True,residualize_covariates=True)
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

        # Report the major findings in brief and provide advice on harmonisation method to apply in this context
        # Summarise the biggest additive differences between batches (Cohen's d, Mahalanobis)

        # Check size of each batch
        batch_sizes = {b: np.sum(batch == b) for b in unique_batches}
        min_batch_size = min(batch_sizes.values())
        max_batch_size = max(batch_sizes.values())

        # Check if large differences in batch sizes (if the difference between smallest and largest batch is more than double)
        if max_batch_size > 2 * min_batch_size:
            report.text_simple(
                f"Note: Large differences in batch sizes detected (smallest batch size: {min_batch_size}, largest batch size: {max_batch_size}). "
                "When applying harmonisation, be aware that most methods will apply a greater correction to smaller batches. " \
                "This isn't inherently problematic but may not be exactly what you want so proceed with caution or consider using the larger batch as a reference batch explicitly so that it remains unchanged in harmonisation"
            )
        
        return data_dict if save_data else None

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report
            report_ctx.__exit__(None, None, None)

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
    
    """
    from pprint import pformat

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
            "This report provides diagnostic analyses for longitudinal data collected across multiple batches.\n "
            "Longitudinal data involves repeated measurements from the same subjects over time, which introduces\n "
            "additional considerations for batch effects and variability. "
            "The following diagnostics will be performed:\n"
            " - Subject-level variability - subject order consistency, within subject variability,\n" \
            " - Batch-level variability - additive, pairwise, multivariate and multiplicative batch effects \n" \
            " - Across-subject level variability - ICC, within/between subject variability,\n" \
            " - Biological variability (e.g., age). "
    )
        report.log_section("subject_order_consistency", "Subject-level variability: Subject Order Consistency analysis")
        report.log_text("Computing Spearman rank correlation between the timepoints")

        # Subject-level: Subject order consistency
        subjorder = DiagnosticFunctions.SubjectOrder_long(idp_matrix=data,
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
        report.log_text("Subject order consistency plot added to report")

        # Subject-level: Within Subject Consistency 
        report.log_section("Within_subject_variability", "Subject-level variability: Within-subject variability analysis")
        report.log_text("Computing IDP variation within subject (Coefficient or variation OR relative difference if two timepoints)")
        wsv = DiagnosticFunctions.WithinSubjVar_long(
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
            limit_subjects=20,
            limit_idps_for_legend=10,
            rep=report
            )
        report.log_text("Within subject variability plot added to report")

         # Batch-level: Additive batch effects
        report.log_section("Additive batch effects using Mixed effect models", "Batch variability (Univariate): Batch effect analysis (mean comparison)")
        report.log_text(pformat("Outputs p-values from additive tests by testing whether the average value of a IDP differs across batches, after accounting for subject effects and other covariates",width=90, sort_dicts=False))
        addeff,model_defs_add = DiagnosticFunctions.AdditiveEffect_long(
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
        report.text_simple("The following model was fit for each feature/IDP")
        report.text_simple("Example using feature/IDP in position 0:")

        report.text_simple(pformat(model_defs_add[0], width=60, sort_dicts=False))

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
        report.log_text(pformat("Outputs no. of significant batch pairs by testing which if specific pair of batches differ from each other",width=90, sort_dicts=False))
        
        mf,model_defs = DiagnosticFunctions.MixedEffects_long(
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
        # Print generic model that is fit for each feature, not for each IDP as that is too much to show, but just the general formula and approach used in the model fitting:

        report.log_text("The following model was fit for each feature (with batch as fixed effect and subject as random effect):")
        report.log_text("Same attempted models for each IDP, though optimisers used may differ based on convergence and model fit")
        # Report model only on the first IDP as an example, as the same model is fit for each IDP
        report.log_text(pformat(model_defs[0], width=60, sort_dicts=False))

        print("\nMIXED EFFECTS OUTPUTS:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(mf) 
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
        report.log_section("Multiplicative batch effects using Fligner-Kileen test", "Batch variability (Univariate): Batch effect analysis (variance comparison)")
        report.log_text(pformat("Outputs p-values from multiplicative tests by comparing variance across batches, after accounting for subject effects and other covariates",width=90, sort_dicts=False))
        muleff,model_defs_mul = DiagnosticFunctions.MultiplicativeEffect_long(
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
        report.log_text(pformat(model_defs_mul, width=60, sort_dicts=False))
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
        report.log_section("Multivariate batch difference with reference", "Batch variability (Multivariate): Difference from overall reference distribution")
        report.log_text(pformat("Outputs average and batch-specific Mahalanobis distances from the reference",width=90, sort_dicts=False))
        md = DiagnosticFunctions.MultiVariateBatchDifference_long(
           idp_matrix=data,
           batch=batch,
           idp_names=features)
        print("\nMULTIVARIATE PAIRWISE SITE DIFFERENCES:")
        print(md)
        PlotDiagnosticResults.plot_MultivariateBatchDifference(md, rep=report) 
        report.log_text("Multivariate batch variability plots added to report")


        # Across subjects-level: Intraclass correlation and within/between subject variability
        report.log_section("Between subject variability using Mixed effect models", "Between subject variability (Univariate): Cross-subject variability analysis")
        report.log_text(pformat("Outputs intra-class correlation and within to between subject variability ratio",width=90, sort_dicts=False))
        #report.log_text(pformat(model_defs, width=60, sort_dicts=False))
        if len(features) < 30:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['ICC','WCV'],
                        plot_type='bar',
                        seed=123,
                        figsize=(16,4), rep=report)
        else:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                      idp_col='IDP',
                      metrics=['ICC', 'WCV'],
                      plot_type='box',
                      limit_idps=2,
                      seed=123,
                      figsize=(16,4))
        report.log_text("ICC and WCV plots added to report")  
        
        # Biological variability
        report.log_section("Biological variability analysis using Mixed effect models", "Biological variability analysis")
        report.log_text("Outputs:\n" 
                        "1) significance\n" 
                        "2) effect sizes (beta)\n"
                        "3) confidence intervals")
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


    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore


def longitudinal_report_experimental(data, batch,
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
    
    """
    from pprint import pformat

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
        # Check batch, subject_ids, and data dimensions
        #if not (len(batch) == len(subject_ids) == data.shape[0]):
        #    raise ValueError("Length of batch and subject_ids must match number of samples in data")
        #if len(covariates) is not None and len(covariates) != data.shape[0]:
        #    raise ValueError("Number of rows in covariates must match number of samples in data")
        #if len(covariate_names) is not None and len(covariate_names) != covariates.shape[1]:
        #    raise ValueError("Length of covariate_names must match number of columns in covariates")
        

        report.log_section("Introduction", "Longitudinal Data Diagnostic Report Introduction")
        report.text_simple(
            "This report provides diagnostic analyses for longitudinal data collected across multiple batches.\n "
            "Longitudinal data involves repeated measurements from the same subjects over time, which introduces\n "
            "additional considerations for batch effects and variability. "
            "The following diagnostics will be performed:\n"
            " - Subject-level variability - subject order consistency, within subject variability,\n" \
            " - Batch-level variability - additive, pairwise, multivariate and multiplicative batch effects \n" \
            " - Across-subject level variability - ICC, within/between subject variability,\n" \
            " - Biological variability (e.g., age). ")
    
        
        report.log_section("subject_level_variability", "Subject-level variability: Subject Order Consistency analysis")
        report.log_text("Starting subject level comparisons: Descriptions seen below:")
        report.text_simple("Within subject variability: Measures of how much each subject's IDPs vary across their timepoints:\n\n"
            "   i. If number of timepoints equal to 2: Relative percentage difference between the timepoints for each subject,\n"
            "   ii. If number of timepoints greater than 2: Coefficient of variation\n\n"
            "Subject order consistency: (ideal for test-retest/traveling heads-like datasets)\n\n"
            "   i.	Spearman correlation between the timepoints\n"
)
        
        report.log_text("Computing IDP variation within subject (Coefficient or variation OR relative difference if two timepoints)")
        wsv = DiagnosticFunctions.WithinSubjVar_long(
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
            limit_subjects=20,
            limit_idps_for_legend=10,
            rep=report
            )
        report.log_text("Within subject variability plot added to report")

        report.log_text("Computing Spearman rank correlation between the timepoints")

        # Subject-level: Subject order consistency
        subjorder = DiagnosticFunctions.SubjectOrder_long(idp_matrix=data,
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
        report.log_text("Subject order consistency plot added to report")

        report.log_section("Batch_effects", "Batch effect analysis (mean and variance comparison, Uni- and Multi-variate)")
        report.log_text("Starting batch level comparisons: Descriptions seen below:")

        report.text_simple(" Univariate batch comparissons\n\n"
            "i. Additive batch effects: \n"
            "   ia.	If batch explains variance in the overall data\n"
            "   ib.	Mean comparison between batches\n"
            "   ic.	Using linear contrasts pairwise comparison between batches \n"
            "ii. Multiplicative batch effects:\n" 
            "   iia. Variance comparison between batches\n\n"
            " Multivariate batch-wise difference with overall data\n\n"
            "i.	Using Mahalanobis distances to compare multivariate mean differences\n")
        
        report.log_text("Additive batch effects using Mixed effect models")
        report.text_simple(pformat("Outputs p-values from additive tests by testing whether the average value of a IDP differs across batches, after accounting for subject effects and other covariates",width=90, sort_dicts=False))
        addeff,model_defs_add = DiagnosticFunctions.AdditiveEffect_long(
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
        report.text_simple("The following model was fit for each feature/IDP")
        report.text_simple("Example using feature/IDP in position 0:")

        report.text_simple(pformat(model_defs_add[0], width=60, sort_dicts=False))

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
        report.log_text(pformat("Outputs no. of significant batch pairs by testing which if specific pair of batches differ from each other",width=90, sort_dicts=False))
        
        mf,model_defs = DiagnosticFunctions.MixedEffects_long(
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
        # Print generic model that is fit for each feature, not for each IDP as that is too much to show, but just the general formula and approach used in the model fitting:

        report.log_text("The following model was fit for each feature (with batch as fixed effect and subject as random effect):")
        report.log_text("Same attempted models for each IDP, though optimisers used may differ based on convergence and model fit")
        # Report model only on the first IDP as an example, as the same model is fit for each IDP
        report.log_text(pformat(model_defs[0], width=60, sort_dicts=False))

        print("\nMIXED EFFECTS OUTPUTS:")
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(mf) 
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
        report.log_text("Multiplicative batch effects using Fligner-Kileen test")    
        report.text_simple(pformat("Outputs p-values from multiplicative tests by comparing variance across batches, after accounting for subject effects and other covariates",width=90, sort_dicts=False))
        muleff,model_defs_mul = DiagnosticFunctions.MultiplicativeEffect_long(
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
        report.log_text(pformat(model_defs_mul, width=60, sort_dicts=False))
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
        report.log_text("Multivariate batch differences")
        report.text_simple(pformat("Average and batch-specific Mahalanobis distances from the reference",width=90, sort_dicts=False))
        md = DiagnosticFunctions.MultiVariateBatchDifference_long(
           idp_matrix=data,
           batch=batch,
           idp_names=features)
        print("\nMULTIVARIATE PAIRWISE SITE DIFFERENCES:")
        print(md)
        PlotDiagnosticResults.plot_MultivariateBatchDifference(md, rep=report) 
        report.log_text("Multivariate batch variability plots added to report")

        report.log_section("Cross-subject_variability", "Between subject variability analysis using mixed effect models")
        report.log_text("Starting across-subject level comparisons: Descriptions seen below:")
        report.text_simple("Across-subject level variability\n\n"
                           "i.	Intra-class correlation: ratio of between subject variability and total variability\n"
                           "ii.	Ratio of within and between-subject variability\n")
        
        report.log_text(pformat("Outputs intra-class correlation and within to between subject variability ratio",width=90, sort_dicts=False))
            #report.log_text(pformat(model_defs, width=60, sort_dicts=False))
        if len(features) < 30:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                        idp_col='IDP',
                        metrics=['ICC','WCV'],
                        plot_type='bar',
                        seed=123,
                        figsize=(16,4), rep=report)
        else:
            PlotDiagnosticResults.plot_MixedEffectsPart1(mf,
                    idp_col='IDP',
                    metrics=['ICC', 'WCV'],
                    plot_type='box',
                    limit_idps=2,
                    seed=123,
                    figsize=(16,4))
        report.log_text("ICC and WCV plots added to report")  
        
        # Biological variability
        

        # Finalize
        report.log_section("Biological_variability_analysis", "Biological variability analysis")
        report.log_text("Starting biological variability analysis: Descriptions seen below:")
        report.text_simple("Biological variability analysis\n\n"
                        "i.	Compute effect sizes, CIs and significance of each covariate and IDP\n"
                        "ii. Correlation of biological covariates with batch effects, subject-level variability and first four PC's\n"
                        "iii. KS-test of residuals (after regressing out covariates) to check non-biological population distributions\n")
        report.log_text("Outputs:\n" 
                        "1) significance\n" 
                        "2) effect sizes (beta)\n"
                        "3) confidence intervals")
        inferred_fix = list(covariates.keys())
        PlotDiagnosticResults.plot_MixedEffectsPart2(mf,
                    idp_col='IDP',
                    fix_eff=inferred_fix,
                    p_thr=0.05,
                    figsize=(8,4),
                    rep=report)
        
        report.log_text("Biological variability plots added to report")


        # Covariates often given as a dictionary here, extract so they work for PCA correlation function (which expects a matrix of covariates and a list of covariate names)
        if covariates is not None and isinstance(covariates, dict):
            covariate_names = list(covariates.keys())
            # Get covariate matrix as an array:
            covariate_matrix = np.column_stack(list(covariates.values()))
            assert type(covariate_matrix) == np.ndarray, f"Covariate matrix should be a numpy array, got {type(covariate_matrix)}"
            # print values of covariate matrix in report for debugging:

            report.log_text(f"Covariate matrix values:\n{covariate_matrix}")    
            
        
        # Add subject ID and timepoint to covariate matrix here for correlation plot:
        # Check subject_ids and timepoints are same length as data and covariates, if not try transpose or raise error


        covariate_matrix = np.column_stack([covariate_matrix, subject_ids, timepoints])
        variable_names = covariate_names + ["subject_id", "timepoint"] 
        variable_names_with_batch = ["batch"] + variable_names
        covariate_matrix = covariate_to_numeric(covariate_matrix)

        # Check age in covariate matrix, print all covariates in report
        report.log_text(f"Covariates in matrix: {variable_names_with_batch}")
        report.log_text(f"Covariate matrix shape: {covariate_matrix.shape}")
        
        report.text_simple(f"Covariate matrix:\n"
                           f"{print(covariate_matrix)}")
        report.text_simple(f"Covariate matrix non-numeric values:\n"
                            f"{print(covariates)}")
        
        assert max(covariate_matrix[:, 0]) > 40, f"Max age should be greater than 50 Max age in covariate matrix: {max(covariate_matrix[:, 0])}"

        
        data_demeaned = data - np.mean(data, axis=0)
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data_demeaned, batch, covariates=covariate_matrix, variable_names=variable_names_with_batch
        )

        report.text_simple("Returning correlations of covariates and batch with first four PC's")
        report.text_simple("Returning scatter plots of first two PC's, grouped/coloured by:")
        report.log_text(f"Variable names used in PCA correlation plots and PC1 vs PC2 plot: {variable_names_with_batch}")
        
        # Variable names here need batch:
        PlotDiagnosticResults.PC_corr_plot(
            score, batch, covariates=covariate_matrix, variable_names=variable_names_with_batch,
            PC_correlations=True, rep=report, show=False
        )
        report.log_text("PCA correlation plot added to report")

        # Demean the data before PCA to avoid mean differences dominating first PC (i.e don't force PC'S > 1 to be orthogonal to mean)
        #data_demeaned = data - np.mean(data, axis=0)
        explained_variance, score, batchPCcorr, pca = DiagnosticFunctions.PC_Correlations(
            data_demeaned, batch,N_components=20, covariates=covariate_matrix, variable_names=variable_names_with_batch
        )

        logger.info("Generating PCA Eigenvalue Scree Plot:")
        report.text_simple("The scree plot displays the eigenvalues associated with each principal component (PC). \n")
        report.text_simple("Using this test, we can see if the variance by batch is the same across all PCs \n")
        report.text_simple("In short, the steepness and shape of the batchwise plots can help to differ batchwise differences in the variance structure across features \n")
        spectra_res = PlotDiagnosticResults.plot_eigen_spectra_and_cumulative(score, batch, rep=report)
        logger.info("PCA Eigenvalue Scree Plot added to report")
        report.text_simple(line_break_in_text)
                           
        report.log_section("Placeholder for title","Lonitudinal batch difference report")
        logger.info("Diagnostic tests completed")
        logger.info(f"Report saved to: {report.report_path}")

    finally:
        # If we created the local report context, close it properly
        if created_local_report:
            # call __exit__ on the context-managed report (no exception info)
            report_ctx.__exit__(None, None, None)  # type: ignore