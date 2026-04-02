"""
Docstring for tests.test_covbat

This module contains unit tests for the covbat package. 
CovBat was copied from the GitHub repository at: https://github.com/andy1764/CovBat_Harmonisation/tree/master

CovBat was originally proposed in the paper:
Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., Shou, H., & Initiative, T. A. D. N. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. Human Brain Mapping, 43(4), 1179–1195. https://doi.org/10.1002/hbm.25688
"""

def test_Covbat():
    """Correction of *Cov*ariance *Bat* effects

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical
    pct_var : numeric
        Numeric between 0 and 1 indicating the percent of variation that is
        explained by the adjusted PCs
    n_pc : numeric
        If >0, then this specifies the number of PCs to adjust. Overrides pct_var

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    # Prepare data for testing:
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonisation import HarmonisationFunctions

    # create data:
    np.random.seed(0)
    n_features = 100
    n_samples = 50
    n_batches = 3
    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"feature_{i}" for i in range(n_features)],
        columns=[f"sample_{i}" for i in range(n_samples)],
    )
    # create batch information:
    batch = pd.Series(
        np.random.choice([f"batch_{i}" for i in range(n_batches)], size=n_samples),
        index=data.columns,
    )
    # create model matrix:
    import patsy
    model_df = pd.DataFrame(
        {
            "age": np.random.randint(20, 60, size=n_samples),
            "sex": np.random.choice(["M", "F"], size=n_samples),
        },
        index=data.columns,
    )
    model = patsy.dmatrix("age + sex", model_df, return_type="dataframe")
    numerical_covariates = ["age"]  
    # Run CovBat:
    corrected_data = HarmonisationFunctions.covbat(
        data,
        batch,
        model=model,
        numerical_covariates=numerical_covariates,
        pct_var=0.95,
        n_pc=0,
    )
    # Check that the output has the same shape as the input:
    assert corrected_data.shape == data.shape
    # Check that the output is a DataFrame:
    assert isinstance(corrected_data, pd.DataFrame)
    # Check that the output does not contain NaN values:
    assert not corrected_data.isnull().values.any()
    # Check that the output values are different from the input values:
    assert not corrected_data.equals(data)
if __name__ == "__main__":
    test_Covbat()

def test_CovBat_efficacy():
    """
    Use CovBat to correct simulated data with known batch effects and verify
    """

    import numpy as np
    import pandas as pd
    from DiagnoseHarmonisation import HarmonisationFunctions
    from sklearn.decomposition import PCA

    # Simulate data with batch effects
    np.random.seed(42)
    n_features = 50
    n_samples_per_batch = 30
    n_batches = 3
    total_samples = n_samples_per_batch * n_batches

    # Create batch-specific means and covariances
    batch_means = [np.random.randn(n_features) * 5 for _ in range(n_batches)]
    batch_covs = [np.diag(np.random.rand(n_features) + 0.5) for _ in range(n_batches)]

    data_list = []
    batch_labels = []
    for i in range(n_batches):
        batch_data = np.random.multivariate_normal(
            mean=batch_means[i],
            cov=batch_covs[i],
            size=n_samples_per_batch
        )
        data_list.append(batch_data)
        batch_labels.extend([f"batch_{i}"] * n_samples_per_batch)

    data = pd.DataFrame(
        np.vstack(data_list),
        index=[f"sample_{i}" for i in range(total_samples)],
        columns=[f"feature_{i}" for i in range(n_features)],
    ).T  # Transpose to (n_features, n_samples) 

    batch = pd.Series(batch_labels, index=data.columns)
    model = None
    numerical_covariates = []
    # Apply CovBat
    corrected_data = HarmonisationFunctions.covbat(
        data,
        batch,
        model=model,
        numerical_covariates=numerical_covariates,
        pct_var=0.95,
        n_pc=0,
    )
    # Evaluate efficacy by comparing PCA before and after correction
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(data.T)
    corrected_pca = pca.fit_transform(corrected_data.T)
    # Check that batch effects are reduced in the corrected data
    from sklearn.metrics import silhouette_score
    original_silhouette = silhouette_score(original_pca, batch)
    corrected_silhouette = silhouette_score(corrected_pca, batch)
    assert corrected_silhouette < original_silhouette, "CovBat did not reduce batch effects"
    
    # Produce plots to visualize PCA before and after correction
    # Create colours for batch for c= parameter in scatter
    batch_colors = batch.astype('category').cat.codes

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_pca[:, 0], original_pca[:, 1], c=batch_colors)
    plt.title("PCA Before CovBat")
    plt.subplot(1, 2, 2)
    plt.scatter(corrected_pca[:, 0], corrected_pca[:, 1], c=batch_colors)
    plt.title("PCA After CovBat")
    plt.show()

    

if __name__ == "__main__":
    test_CovBat_efficacy()  


    