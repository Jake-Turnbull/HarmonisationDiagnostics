# Simulator

Batch effect simulator that opens an interactive web-browser and allows the user to generate simulated datasets with varying numbers of unique batches, severity of batch effects (additive and multiplicative) and different covariate effects.

The user can then visualise the feature-wise difference in batches using histograms and box-plots, generate a cross-sectional diagnostic report to view the effects in more detail and apply harmonisation (using ComBat). This allows the user to get a direct comparisson of the before/after of applying harmonisation by comparing the reports in a semi-realistic scenario.

To run the simulator, run *streamlit run simulator.py* in the terminal (must have streamlit installed on python path).

This will open up an interactive web browsers where you can simulate different severities of batch effects (as well as number of batches and covariates) before running the tool and returning a report. The simulator also gives the option to apply ComBat for harmonisation, giving users the option to see a before vs after of what the batch differences are.

