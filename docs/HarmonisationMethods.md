# Harmonisation Methods

This document provides an overview of the harmonisation methods implemented and available in the DiagnoseHarmonisation package.

## Overview

Batch effects are systematic biases introduced by differences in data collection equipment, scanners, protocols, or sites. Harmonisation (or harmonization) aims to remove or reduce these batch effects while preserving biological or phenotypic signals of interest.

The methods below range from simple parametric approaches to advanced machine learning techniques.

## 1. Linear Modelling

The simplest harmonisation approach models batch effects as fixed or random effects in a linear regression framework.

**Method**: Fits a linear model of the form:
$$Y = X\beta + Z_b u_b + \epsilon$$

where $Y$ is the data, $X$ contains covariates of interest, $Z_b$ encodes batch membership, $u_b$ are batch random effects, and $\epsilon$ is residual error.

**Advantages**:
- Highly interpretable and transparent
- Computationally efficient
- No tuning hyperparameters
- Direct inference on covariate effects

**Limitations**:
- Assumes linear relationships
- May not capture complex batch interactions
- Assumes homogeneous variance across batches

**Use case**: Quick reference harmonisation or when interpretability is paramount.

---

## 2. ComBat

**Combat** (Correcting Batch Effects using Empirical Bayes) is a widely-used batch harmonisation method from bioinformatics (Johnson et al., 2007).

**Method**: ComBat uses empirical Bayes to estimate and correct batch-specific location (mean) and scale (variance) shifts:
1. Fits a parametric model with batch-specific parameters.
2. Uses empirical Bayes shrinkage to estimate these parameters.
3. Adjusts data by subtracting batch means and rescaling by batch variance.

**Key features**:
- Handles variable batch sizes
- Empirical Bayes shrinkage prevents overfitting
- Can incorporate biological covariates (mod variable) to avoid removing true signal
- Fast and widely benchmarked

**Advantages**:
- Well-established and validated in neuroimaging studies
- Preserves biological covariates
- Computationally efficient

**Limitations**:
- Assumes location-scale model (mean and variance only)
- Homogeneity assumption across features
- May over-correct in small-sample scenarios

**Use case**: Standard choice for multi-site neuroimaging studies.

---

## 3. ComBat with GAM

**ComBat-GAM** extends ComBat by replacing the parametric mean model with a Generalized Additive Model (GAM).

**Method**: 
1. Uses GAM to model covariate effects (more flexible than linear)
2. Combines flexible covariate adjustment with ComBat's batch correction
3. Estimates batch effects on the GAM residuals

**Key features**:
- Captures non-linear covariate relationships
- Spline basis functions for smooth covariate effects
- Falls back to linear model if spline fitting fails

**Advantages**:
- More flexible than ComBat for non-linear covariate patterns
- Maintains empirical Bayes batch correction
- Graceful degradation (falls back to linear)

**Limitations**:
- More complex and less interpretable than linear/ComBat
- Requires more data to accurately estimate splines
- Slightly slower than parametric ComBat

**Use case**: When covariate effects are expected to be non-linear (e.g., age, scanner field strength).

---

## 4. CovBat

**CovBat** (Covariate-adjusted ComBat) refines ComBat by jointly modelling covariate and batch effects without requiring a separate mod variable.

**Method**:
1. Jointly estimates covariate and batch effects in a unified model
2. Uses empirical Bayes for improved robustness
3. Solves for optimal ridge regression weights

**Key features**:
- Internally handles covariate adjustment
- No separate mod matrix required
- Provides confidence intervals for batch effect estimates
- Computationally stable ridge regression approach

**Advantages**:
- Simpler interface than ComBat (no mod matrix)
- Better calibrated confidence intervals
- Unified treatment of covariates and batch

**Limitations**:
- Slightly slower than standard ComBat
- Less widely used (newer method)

**Use case**: When you want integrated covariate and batch handling without manual mod matrix construction.

---

## 5. Reference-Based ComBat

**Reference-Based ComBat** extends ComBat to a multi-site setting by designating one or more sites as a "reference" against which other sites are harmonised.

**Method**:
1. Treats reference site(s) as gold standard
2. Estimates batch effects of non-reference sites relative to reference
3. Applies ComBat correction targeting reference distribution

**Key features**:
- Preserves reference site characteristics
- Useful when one acquisition protocol is known to be high-quality
- Can use multiple reference sites

**Advantages**:
- Meaningful reference frame (e.g., gold-standard scanner)
- Avoids blending all sites into an average
- Interpretable target distribution

**Limitations**:
- Requires prior designation of reference site(s)
- May not work well if reference is atypical
- Assumes reference effects are minimal

**Use case**: Multi-site studies where one site is known to have optimal data quality.

---

## 6. IQM-Based Harmonisation

**IQM-Based Harmonisation** (Intrinsic Quality Metric harmonisation) leverages data quality metrics to guide harmonisation.

**Method**:
1. Computes Intrinsic Quality Metrics (IQM) for each scan (e.g., contrast-to-noise, signal stability)
2. Identifies scanner/site effects using IQM
3. Adjusts data based on IQM-derived quality scores

**Key features**:
- Incorporates domain knowledge about quality metrics
- Can be combined with other methods
- Adaptive to acquisition variations
- Physically interpretable

**Advantages**:
- Links harmonisation to measurable quality indicators
- Handles non-linear quality-related batch effects
- Transparent and auditable

**Limitations**:
- Requires valid IQM computation
- Sensitive to IQM accuracy
- May not capture all batch factors

**Use case**: MRI, diffusion imaging, or other modalities with standard quality metrics.

---

## 7. SV-ComBat (In Development)

**SV-ComBat** (Similarity-guided ComBat) is an in-development method that uses similarity matrices to inform batch prior pooling in ComBat.

**Method**:
1. Computes pairwise similarity between batch effects (e.g., based on covariate distributions, effect correlation)
2. Uses similarity to inform Bayesian prior pooling
3. Dynamically pools batch parameters based on similarity structure
4. Applies adjusted ComBat correction

**Key features**:
- Adaptive prior pooling based on batch relationships
- Preserves structure in batch effects
- Can incorporate multiple similarity metrics
- Research-stage implementation

**Advantages**:
- Exploits relationships between batch effects
- Improved prior estimates when batches are similar
- Flexible similarity metric design
- Principled Bayesian approach

**Limitations**:
- Still in active development
- Not yet validated in large-scale studies
- Requires tuning similarity metric
- Increased computational cost

**Use case**: Studies with many similar batches (e.g., same scanner model at different sites, replicated protocols).

**Status**: ⚠️ **Experimental/In Development** — Use with caution and report findings carefully.

---

## Comparison Table

| Method | Complexity | Speed | Covariate Flexibility | Batch Effect Model | Status |
|--------|-----------|-------|---------------------|--------------------|--------|
| Linear Modelling | Low | Very fast | Linear | Random effect | Stable |
| ComBat | Medium | Fast | Linear (mod matrix) | Location-scale | Stable |
| ComBat-GAM | Medium-High | Fast | Non-linear (GAM) | Location-scale + GAM | Stable |
| CovBat | Medium | Moderate | Linear (integrated) | Location-scale (unified) | Stable |
| Reference-Based ComBat | Medium | Fast | Linear (mod matrix) | Location-scale (relative to reference) | Stable |
| IQM-Based | Medium-High | Moderate | Metric-driven | Quality-metric-informed | Stable |
| SV-ComBat | High | Moderate-Slow | Linear (mod matrix) | Similarity-pooled priors | **Experimental** |

---

## When to Use Each Method

### Quick/exploratory analysis
→ **Linear Modelling**

### Standard multi-site neuroimaging
→ **ComBat**

### Non-linear covariate effects expected
→ **ComBat-GAM**

### Integrated covariate handling preferred
→ **CovBat**

### One high-quality reference site available
→ **Reference-Based ComBat**

### Physical quality metrics are key
→ **IQM-Based Harmonisation**

### Many structurally similar batches
→ **SV-ComBat** (with caution — experimental)

---

## Implementation in DiagnoseHarmonisation

Most methods are implemented in the [`HarmonisationFunctions`](api/HarmonisationFunctions.md) module, particularly via the modular `combat_modular()` function which supports:
- Different mean models (`'ols'`, `'gam'`, etc.)
- Prior mode selection (`'global'`, `'local'`, etc.)
- Custom prior weighting strategies

For programmatic usage, see the [API documentation](api/index.md).

---

## References

- **ComBat**: Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*.

    Fortin et al., (2018). Harmonization of cortical thickness measurements across scanners and sites. NeuroImage, 167, 104–120. https://doi.org/10.1016/j.neuroimage.2017.11.024

    Fortin et al., Harmonization of multi-site diffusion tensor imaging data. NeuroImage, 161, 149–170. https://doi.org/10.1016/j.neuroimage.2017.08.047

- **ComBat-GAM**: Pomponio et al., (2020). Harmonization of large MRI datasets for the analysis of brain imaging patterns throughout the lifespan. NeuroImage, 208, 116450. https://doi.org/10.1016/j.neuroimage.2019.116450

- **Reference batch ComBat** Jacob Turnbull et al., (2026). bioRxiv 2026.05.22.726536; doi: https://doi.org/10.64898/2026.05.22.726536


- **CovBat**:
Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., Shou, H., & Initiative, T. A. D. N. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. Human Brain Mapping, 43(4), 1179–1195. https://doi.org/10.1002/hbm.25688

- **IQM-based**: Emma Prevot, Dieter A. Häring, Laura Gaetano, Russell T. Shinohara, Chris C. Holmes, Thomas E. Nichols, Habib Ganjgahi (2025). BARTharm: MRI Harmonization Using Image Quality Metrics and Bayesian Non-parametric bioRxiv 2025.06.04.657792; doi: https://doi.org/10.1101/2025.06.04.657792

    Gaurav Bhalerao et al., (2026). Harmonising Structural Brain MRI from Multiple Sites with Limited Sample Sizes. medRxiv 2026.04.21.26351106; doi: https://doi.org/10.64898/2026.04.21.26351106

---

## Citation

When using any harmonisation method from DiagnoseHarmonisation in your research, please cite the original method papers as well as this package.
