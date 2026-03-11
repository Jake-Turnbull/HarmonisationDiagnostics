# Theory

This page will briefly cover the theory behind each of the tests used in the diagnostic pipeline, as well as provide links to helpful resources:

This page will not go into the full mathematical explanation behind each of the tests but will instead explain why they have been chosen.

## Introduction

MRI harmonisation, or indeed any form of harmonisation in multisite/batch analysis must begin a thorough analysis of what the differences between datasets is. Often when choosing a method, this is somewhat overlooked, leading to non-optimal or even harmful harmonisation approaches being applied.

No clear guidelines on how to best approach this form of analysis are givej by the literature and metrics of success vary between methods. This problem was the motivation for this libraries development, serving as a set of guidelines and metrics from which researchers can base their choice in harmonisation methods on but also from which they can evaluate their efficacy.

We break down harmonisation efficacy assesment into two main categories: 1. Removal of site/batch differences and 2. Preservation of biological information. The tests we have chosen for this pipeline were collected from frequently cited metrics from harmonisation literature, but also chosen to be as interpretable as possible. For clarity we will described below the basic form of each test and how to best interpret it.

## Additive batch effects

The most easily detectable batch effects tend to be those that are additive in nature (shown by a difference in means). The easiest way to see this difference is often just by looking at the data as histograms. If there is a difference between the centre of the peaks, then there is a difference in means.

### Histograms of MAD scores

To show this, we use Median Absolute Deviations (MAD) scores. These are calculated for each feature (column) in the data and plotted per batch as a histogram, serving as the easiest way to visually compare batch differences across features.

### Cohen's d scores

To show the mean difference across features, we use [Cohen's d](https://en.wikipedia.org/wiki/Effect_size). as the metric of comparisson. Here, we go a level deeper than the MAD scores when showing the difference by modelling out covariate effects first using OLS. In this way, we isolate dataset differences to those purely caused by batch differences. This is especially important for cases where there may be strong confounding of a variable of interest with batch (i.e one batch may be systematically older).

Cohen's d is calculated per batch against the rest of the data, using the unweighted mean and variance in each case.
$d_i  = \frac{\mu_{batch=i}-\mu_{batch \not= i}}{(1/2)(\sigma_{batch = i} + \sigma_{batch \not= i})}$

Here, the mean (mu) is calculated from the residuals of the data after removing covariate effects predicted with OLS.

### Mahalanobis distance

The [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) is a multivariate extention to the Cohen's d score that uses a projection of all data to measure the difference in the multivariate mean between different groups. We show this metric in three ways; as a heatmap between the raw data between all batches; As a heatmap between the residuals of the data after removing covariate effects; And as a difference between each batch to the overall data centroid as both raw and residuals.

## Multiplicative effects

### Mixed effect modelling

The first test we show here uses [mixed effect model](https://en.wikipedia.org/wiki/Mixed_model) fitting, with the random term set to the batch term. The script builds up a patsy formula using all provided covariates, distringuising their type by the number of unique variables (less than 3 = binary, between 3 and 10 = categorical, greater than 10 = continuous). The best way to use the tool is to format the covariates properly yourself prior to running, but it does support some native handling of different types of covariates.

The tool then compares the model fit with and without batch as the random term, showing the results as R^2 values. If the model fit is higher with batch, then it explains a large portion of the variance.

### Ratio of variance (F-test)

The most simple way to assess difference in the scale between two datasets/samples is through an F-test (ratio of variance). Here, we use OLS similarly to as with Cohen's d to model out covariate effects. We then take the variance of a given batch of data and compare it to the variance of the whole dataset or the variance of the individual batch (as decided by one of four input modes).

By seeing which batches have larger ratios compared to the rest of the data, or individual batches, one can see how the scale is affected by batch. We would expect the ratio to be close to 1 for datasets that are similar.

## Correlations and covariance

In this section we show a batch effect approach that differes slightly from general methods but one that can be useful for showing preservation of covariate effects and reduction of batch effects.

### PCA correlations

[Principle Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) is a linear dimensionality technique projects a dataset to a set of N orthoganal components with each N+i component explaining less variance to the Nth component. PCA can be very useful for reducing the size of large datasets with many dimensions to only a few while retaining a large portion of the variance.

We use it here to show the correlations with the batch effect and covariates. By using PCA to reduce the whole dataset into just four components, we can see how well correlated the variables given are with the linear axes explaining the most variance in the data. We also show here the explained variance by principle component number by batch which can show overall if the multivariate variance structure differs between batches.

If batch explains a large proportion of the variance, it is likely to be well correlated with one or more of the first 4 PC's. The same is then true for covariates. This analysis can tell you how well correlated these variables are with the overall variance structure of the data, but also with eacother. After harmonisation, we would define success by a reduction of correlation of the PC's with batch and a preservation (or indeed increase) of the correlations with covariates (such as age)

### Covariance structure

Another measure of interest that can be affected by batch differences is covariance, which explains how different features change in relation to eachother. Take for example a dataset concisting of derived measures of corticial thickness and volume. We know these features aren't truly independant and would expect them to have certain covariance structure which changes relatively similarly between batches. If batches have a large covariance difference, this can be due to batch effects which aren't explicitely shown through other analysis.

We compare the difference in the overall covariance structure here by using the Eucledian/Frobenius norm between the covariance matrices of individual batches, scaled by the norm of the covariance of the whole dataset. This returns a sinle value expressing a scaled sum of squared difference between two matrices, with a larger value indicating a greater difference in covariance.

## Dimensionality reduction visualisation

One other way that is frequently use to assess batch effects is through visual methods involving dimensionality reduction. One such was is using linear dimensionality reduction using PCA (as described above), but non-linear techniques can often provide more global information.

### PCA

We compute and plot the first two principle components as a scatter plot, colouring the points by the variable of interest. Prior to harmonisation, one might expect distinct clusters of data from different batches. After harmonisation, these clusters should appear less visible. Additionally, we would expect values such as age and sex to have distinct directions of variation and seperation respectively.

### UMAP

For non-linear dimensionality reduction, we use [UMAP](https://umap-learn.readthedocs.io/en/latest/)
Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. When using UMAP, care must be taken to pick optimal parameters for specific datasets. These are described more thoroughly in the [original paper](https://joss.theoj.org/papers/10.21105/joss.00861) describing the theory and underlying assumptions of UMAP, but also in these tutorials courtesy of Andy Coenen and Adam Pearce: [Understanding UMAP](https://pair-code.github.io/understanding-umap/) and [A deeper dive into UMAP theory](https://pair-code.github.io/understanding-umap/supplement.html)

## Testing overall distribution similarity

### Two-Sample KS-tests (hypothesis testing)

The last test we include here is the two sample [KS test](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test). Once again we perform this test on the residuals after removing covariate effects in order to see deviations that are driven only by batch differences. The two sample KS test looks at the difference between the empirical distributions between batches using a step function. This test is a popular and robust technique that can assess both mean and scale differences.

Here, we plot the results as log-transformed P-values for each feature. These are plotted in ascending order and as a bar chart, allowing users to see the magnitude and distributions of P-values. Here, somewhat unintuitively (and perhaps mortifying for frequentist statisticians) we define the null-hypothesis as there being no difference between the empirical distributions. In this case we define reduction in significant differences as our success metric for harmonisation.
