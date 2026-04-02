"""Test lme_harmonisation function"""

# Harmonisation using linear mixed model:

def test_lme_harmonisation():
    """
    Test lme_harmonisation function with simulated data.
    """
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonisation import HarmonisationFunctions

    # Create simulated data
    np.random.seed(0)
    n_features = 50
    n_samples = 30
    n_batches = 2

    # IMPORTANT: data must be shape (n_samples, n_features) -> rows = samples
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=[f"sample_{i}" for i in range(n_samples)],    # sample labels on rows
        columns=[f"feature_{j}" for j in range(n_features)],# feature labels on columns
    )

    # Create batch information aligned to data.index (samples)
    batch = pd.Series(
        np.random.choice([f"batch_{i}" for i in range(n_batches)], size=n_samples),
        index=data.index,
    )

    # Introduce batch effects (correct axis: select rows by idx)
    for b in batch.unique():
        idx = batch[batch == b].index                   # sample indices in this batch
        # add shape (n_rows_in_batch, n_features) to those rows
        data.loc[idx, :] += np.random.randn(len(idx), n_features) * 2

    # Run lme_harmonisation (function expects samples x features)
    corrected_data = HarmonisationFunctions.lme_harmonisation(
        data, batch, mod=None, variable_names=None
    )

    # Basic sanity checks
    assert isinstance(corrected_data, pd.DataFrame)
    assert corrected_data.shape == data.shape