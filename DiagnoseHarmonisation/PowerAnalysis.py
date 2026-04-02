# PowerAnalysis.py
# This module has a set of functions to perform power analysis for each of the functions in DiagnosticReport.py
# Where appropriate, we will use avaiable packages form scipy or statsmodels to perform power analysis, where this isn't possible we will
# Simulate data to estimate power for a given test.

# - Currently a placeholder file with only Cohen's d power analysis implemented. -

# ---- Cohen's d power analysis ----
from scipy import stats
import numpy as np
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
import matplotlib.pyplot as plt

# As this is always between two groups, we can use TTestIndPower from statsmodels, return
def Cohens_D_PowerAnalysis(data, batch, a,b):
    """
    Perform power analysis for Cohen's d between each pair of batches given in the batch variable.
    Each batch pair is tested seperately and we return the power vs sample size curve for three effect sizes:
    0.1, 0.5, 0.8 and 1.2 (small, medium, large, very large)
    The values for Cohen's d given in a are for each unique pair of batches in batch variable.
    Note: As Cohen's d is a relatively simple test, we can use analytical solutions using a 2 sample t-test power analysis.
    This assumption may not hold for highly non-normal data or for other tests
    """
    # Define effect sizes to test and plot relative power curves for this sample size:

    effect_sizes = [0.1, 0.5, 0.8, 1.2]  # small, medium, large, very large
    alpha = [0.05, 0.01, 0.001]  # significance levels

    # Unique batch pairs are given by b, the cohens d scores are in a and the batches in batch
    # Get batch sizes:
    unique_batches = np.unique(batch)
    batch_sizes = {ub: np.sum(batch == ub) for ub in unique_batches}
    power_analysis = TTestIndPower()
    power_results = {}
    for i, (batch_pair, cohens_d) in enumerate(zip(b, a)):
        n1 = batch_sizes[batch_pair[0]]
        n2 = batch_sizes[batch_pair[1]]
        n_total = n1 + n2
        power_results[batch_pair] = {}
        for es in effect_sizes:
            power_results[batch_pair][es] = {}
            for a_level in alpha:
                # Calculate power for given effect size and alpha level
                power = power_analysis.solve_power(effect_size=es, nobs1=n1, alpha=a_level, ratio=n2/n1, alternative='two-sided')
                power_results[batch_pair][es][a_level] = power
    figs = []
    # If Nan returned by power analysis (e.g greater than 1), set to 1
    # (shouldn't happen with current parameters, but just in case)
    # Clean power results
    for batch_pair in power_results.keys():
        for es in effect_sizes:
            for a_level in alpha:
                if np.isnan(power_results[batch_pair][es][a_level]) or power_results[batch_pair][es][a_level] > 1.0:
                    power_results[batch_pair][es][a_level] = 1.0 

    # Generate power curves and place each batch comparison on the curves 
    for batch_pair in power_results.keys():
        fig, ax = plt.subplots()
        for es in effect_sizes:
            powers = [power_results[batch_pair][es][a_level] for a_level in alpha]
            ax.plot(alpha, powers, marker='o', label=f'Effect Size: {es}')
        ax.set_title(f'Power Analysis for Cohen\'s d between {batch_pair[0]} and {batch_pair[1]}')
        ax.set_xlabel('Alpha Level')
        ax.set_ylabel('Power')
        ax.set_xticks(alpha)
        ax.set_ylim(0, 1)
        ax.legend()
        figs.append((f'Power Analysis for Cohen\'s d between {batch_pair[0]} and {batch_pair[1]}', fig))
    return power_results, figs

# Note: Additional power analysis functions for other tests can be added here as needed.
  
def Variance_Ratio_PowerAnalysis(data, batch, ratio_of_variance, unique_batches):
    """
    As variance ratio is tested using an F-test, we can use the FTestAnovaPower from statsmodels to perform power analysis.
    """
    # Variance ratio only gives out ratios as entries in dict, labelled by unique batch pairs of b1 and b2 in unique_batces

    effect_sizes = [0.1, 0.5, 0.8, 1.2]  # small, medium, large, very large
    alpha = [0.05, 0.01, 0.001]
    power_analysis = FTestAnovaPower()
    power_results = {}
    for i, (batch_pair, var_ratio) in enumerate(ratio_of_variance.items()):
        n1 = np.sum(batch == batch_pair[0])
        n2 = np.sum(batch == batch_pair[1])
        n_total = n1 + n2
        power_results[batch_pair] = {}
        for es in effect_sizes:
            power_results[batch_pair][es] = {}
            for a_level in alpha:
                # Calculate power for given effect size and alpha level
                power = power_analysis.solve_power(effect_size=es, nobs=n_total, alpha=a_level, k_groups=2)
                power_results[batch_pair][es][a_level] = power
    figs = []
    # Clean power results
    for batch_pair in power_results.keys():
        for es in effect_sizes:
            for a_level in alpha:
                if np.isnan(power_results[batch_pair][es][a_level]) or power_results[batch_pair][es][a_level] > 1.0:
                    power_results[batch_pair][es][a_level] = 1.0
    # Generate power curves and place each batch comparison on the curves 
    for batch_pair in power_results.keys():
        fig, ax = plt.subplots()
        for es in effect_sizes:
            powers = [power_results[batch_pair][es][a_level] for a_level in alpha]
            ax.plot(alpha, powers, marker='o', label=f'Effect Size: {es}')
        ax.set_title(f'Power Analysis for Variance Ratio between {batch_pair[0]} and {batch_pair[1]}')
        ax.set_xlabel('Alpha Level')
        ax.set_ylabel('Power')
        ax.set_xticks(alpha)
        ax.set_ylim(0, 1)
        ax.legend()
        figs.append((f'Power Analysis for Variance Ratio between {batch_pair[0]} and {batch_pair[1]}', fig))
    return power_results, figs



