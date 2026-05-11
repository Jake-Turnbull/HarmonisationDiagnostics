import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple simulator with persistent data (use Run Simulation first)")

# UI controls
number_batches = st.slider("Number of Batches", min_value=2, max_value=10, value=3, step=1)
samples_per_batch = st.slider("Samples per Batch", min_value=10, max_value=500, value=40, step=10)
feature_dim = st.slider("Feature Dimension", min_value=10, max_value=100, value=50, step=10)
covariates = st.multiselect("Select Covariates to Simulate",
                            options=["Age", "Sex", "Diagnosis"],
                            default=["Age"])
batch_additive_severity = st.slider("Batch Mean Effect Severity", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
batch_multiplicative_severity = st.slider("Batch Variance Effect Severity", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# Run simulation and store results in session_state so they persist across reruns
if st.button("Run Simulation"):
    st.write("Generating Data...")
    total_samples = number_batches * samples_per_batch

    # ------------------------
    # Base data: treat columns as 'z-score-like' but centered around 1 (i.e. N(1, 1))
    # ------------------------
    # Draw base values from N(1, 1) so each feature ~ mean 1, sd 1 before batch effects
    data = np.random.randn(total_samples, feature_dim) + 1.0

    # ------------------------
    # Covariates
    # ------------------------
    covariate_data = pd.DataFrame(index=range(total_samples))
    if "Age" in covariates:
        covariate_data["Age"] = 20 + 60 * np.random.rand(total_samples)  # Age range 20-80
    if "Sex" in covariates:
        covariate_data["Sex"] = np.random.randint(0, 2, size=total_samples)
    if "Diagnosis" in covariates:
        covariate_data["Diagnosis"] = np.random.randint(0, 5, size=total_samples)

    # ------------------------
    # Batch labels
    # ------------------------
    batch_labels = np.array([f"Batch_{i+1}" for i in range(number_batches) for _ in range(samples_per_batch)])

    # ------------------------
    # Batch effect parameters (per-batch)
    # - multiplicative scale: controls spread (sd) of the distribution for that batch
    #   we allow some batches to be narrower (<1) and some wider (>1) by using a reciprocal range
    # - additive shift: controls mean shift (location) per batch
    # ------------------------
    # defensive: if user sets multiplicative severity <= 0, clamp to 1 (no multiplicative change)
    mult_sev = float(batch_multiplicative_severity) if batch_multiplicative_severity is not None else 1.0
    if mult_sev <= 0:
        mult_sev = 1.0

    # Create per-batch scale factors spanning [1/mult_sev, mult_sev] so some batches can be narrower (<1)
    if mult_sev == 1.0:
        scale_factors = np.ones(number_batches)
    else:
        scale_factors = np.linspace(1.0 / mult_sev, mult_sev, number_batches)

    # Per-batch additive mean shifts spanning [-batch_additive_severity, +batch_additive_severity]
    add_shifts = np.linspace(-batch_additive_severity, batch_additive_severity, number_batches)

    # ------------------------
    # Apply batch effects (vectorized inside loop per-batch)
    # - interpret data as deviations from 1.0 (since base ~N(1,1))
    # - scale those deviations to make distribution wider/narrower
    # - add a per-batch mean shift
    # - also add small per-feature batch offsets and small extra noise proportional to scale
    # ------------------------
    for i in range(number_batches):
        start = i * samples_per_batch
        end = start + samples_per_batch

        sf = float(scale_factors[i])           # multiplicative spread factor for this batch
        mu_shift = float(add_shifts[i])        # additive mean shift for this batch

        # Extract slice
        batch_slice = data[start:end, :]      # shape (samples_per_batch, feature_dim)

        # 1) Feature-specific batch offsets (small): some features shift more in some batches
        feature_offsets = np.random.normal(loc=0.0, scale=0.5, size=(feature_dim,))  # tiny per-feature shift

        # 2) Compute deviations from 1.0 (z-score-like origin) and scale them
        deviations = batch_slice - 1.0
        scaled = 1.0 + deviations * sf

        # 3) Add batch-level mean shift (same across features) and feature offsets
        scaled += mu_shift
        scaled += feature_offsets.reshape(1, -1)

        # 4) Add residual per-sample-per-feature noise whose sd grows with sf (so wider batches also noisier)
        noise_sd = 0.8 * max(1.0, sf)  # base small noise, scales up if sf>1
        extra_noise = np.random.normal(loc=0.0, scale=noise_sd, size=(samples_per_batch, feature_dim))

        # Write back
        data[start:end, :] = scaled + extra_noise

    # ------------------------
    # Covariate effects (vectorized) — more realistic: per-feature coefficients (small, heterogeneous)
    # ------------------------
    rng = np.random.RandomState(42)  # reproducible small heterogeneity in coefficients

    if "Age" in covariates:
        # Coefficients ~ N(-0.05, 0.01) per feature (older reduces signal slightly on average)
        age_coefs = rng.normal(loc=-0.05, scale=0.01, size=(feature_dim,))
        age_centered = covariate_data["Age"].values - covariate_data["Age"].mean()
        data += age_centered.reshape(-1, 1) * age_coefs.reshape(1, -1)

    if "Sex" in covariates:
        # Sex effect small and sparse: some features strongly influenced, most weakly
        sex_base = 0.25
        sex_coefs = rng.normal(loc=sex_base, scale=0.05, size=(feature_dim,))
        sex_centered = covariate_data["Sex"].values - covariate_data["Sex"].mean()
        data += sex_centered.reshape(-1, 1) * sex_coefs.reshape(1, -1)

    if "Diagnosis" in covariates:
        # Diagnosis is categorical 0-4: give an increasing additive trend across categories with per-feature variation
        diag_base = 0.5
        diag_coefs = rng.normal(loc=diag_base, scale=0.1, size=(feature_dim,))
        diag_centered = covariate_data["Diagnosis"].values - covariate_data["Diagnosis"].mean()
        data += diag_centered.reshape(-1, 1) * diag_coefs.reshape(1, -1)


    # Save to session_state
    st.session_state["data"] = data
    st.session_state["batch_labels"] = batch_labels
    st.session_state["covariate_data"] = covariate_data
    st.session_state["samples_per_batch"] = samples_per_batch
    st.session_state["number_batches"] = number_batches
    st.session_state["feature_dim"] = feature_dim

    st.write("Data Generation Complete.")

# Feature slider (use session_state feature_dim if available)
max_feature = st.session_state.get("feature_dim", feature_dim) - 1
feature_to_plot = st.slider("Select Feature to Plot Histogram and Boxplots", min_value=0, max_value=max_feature, value=0, step=1)


# Plot only if data exists
if "data" not in st.session_state:
    st.warning("No data yet — please click **Run Simulation** to generate data before plotting.")
else:
    data = st.session_state["data"]
    number_batches = st.session_state["number_batches"]
    samples_per_batch = st.session_state["samples_per_batch"]

    fig, ax = plt.subplots()
    for i in range(number_batches):
        start = i * samples_per_batch
        end = start + samples_per_batch
        batch_data = data[start:end, feature_to_plot]
        ax.hist(batch_data, bins=30, alpha=0.5, label=f"Batch_{i+1}")
    ax.set_title(f"Histogram of Feature {feature_to_plot} by Batch")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    fig2, ax   = plt.subplots()
    ax.boxplot([data[i*samples_per_batch:(i+1)*samples_per_batch, feature_to_plot] for i in range(number_batches)],
               labels=[f"Batch_{i+1}" for i in range(number_batches)])
    ax.set_title(f"Boxplot of Feature {feature_to_plot} by Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Feature Value")
    st.pyplot(fig2)

# New button that runs CrossSectionalReport on the data and batches in session_state, keep name fixed and save to current directory so only one report exists at a time

if st.button("Generate Cross-Sectional Report"):
    if "data" not in st.session_state:
        st.warning("No data yet — please click **Run Simulation** to generate data before generating report.")
    else:
        from DiagnoseHarmonisation import DiagnosticReport
        data = st.session_state["data"]
        batch_labels = st.session_state["batch_labels"]
        covariate_data = st.session_state["covariate_data"]
        st.write("Generating Cross-Sectional Report...")
        if np.unique(batch_labels).shape[0] > 3:
            SaveArtifacts = False
        else:
            SaveArtifacts = True

        report = DiagnosticReport.CrossSectionalReport(data,
                                                        batch_labels,
                                                          covariate_data,
                                                            covariate_names=covariate_data.columns.tolist(),
                                                                save_dir=".",
                                                                    save_data=True,
                                                                    report_name="Simulator_Report",
                                                                        SaveArtifacts=SaveArtifacts,
                                                                        rep=None,
                                                                            show=False
                                                          )
        report_filename = "Simulator_Report.html"
        st.write(f"Report generated and saved as {report_filename}. You can open this file in your web browser to view the report.")

# Add a final option to harmonise the data using ComBat, run the report again on the harmonised data, and provide download link for harmonised data and unharmonised data as a csv file

if st.button("Harmonize Data with ComBat and Generate Report"):
    if "data" not in st.session_state:
        st.warning("No data yet — please click **Run Simulation** to generate data before harmonizing.")
    else:
        from DiagnoseHarmonisation import DiagnosticReport
        from DiagnoseHarmonisation import HarmonisationFunctions
        data = st.session_state["data"]
        batch_labels = st.session_state["batch_labels"]
        covariate_data = st.session_state["covariate_data"]
        st.write("Harmonizing Data with ComBat...")
        [harmonized_data, _,_] = HarmonisationFunctions.combat(data.T, batch_labels, covariate_data, parametric=True)
        harmonized_data = harmonized_data.T  # Transpose back to original shape
        st.write("Generating Cross-Sectional Report on Harmonized Data...")
        SaveArtifacts = False

        report = DiagnosticReport.CrossSectionalReport(harmonized_data,
                                                        batch_labels,
                                                          covariate_data,
                                                            covariate_names=covariate_data.columns.tolist(),
                                                                save_dir=".",
                                                                    save_data=True,
                                                                    report_name="Simulator_Harmonized_Report",
                                                                        SaveArtifacts=SaveArtifacts,
                                                                        rep=None,
                                                                            show=False
                                                          )
        report_filename = "Simulator_Harmonized_Report.html"
        st.write(f"Harmonized report generated and saved as {report_filename}. You can open this file in your web browser to view the report.")

        # Provide download links for harmonized and unharmonized data
        unharmonized_df = pd.DataFrame(data)
        harmonized_df = pd.DataFrame(harmonized_data)

        unharmonized_csv = unharmonized_df.to_csv(index=False).encode('utf-8')
        harmonized_csv = harmonized_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Unharmonized Data as CSV",
            data=unharmonized_csv,
            file_name='unharmonized_data.csv',
            mime='text/csv',
        )

        st.download_button(
            label="Download Harmonized Data as CSV",
            data=harmonized_csv,
            file_name='harmonized_data.csv',
            mime='text/csv',
        )