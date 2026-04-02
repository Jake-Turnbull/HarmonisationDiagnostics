"""
Docstring for tests.test_combat

Test the combat and combat(covbat) functions here.


"""

def test_combat_with_covbat():
    """Test combat function used within CovBat

    This test ensures that the combat function works correctly when used
    within the CovBat harmonisation process.

    Steps:
    1. Prepare synthetic data and batch information.
    2. Call the CovBat function which internally uses combat.
    3. Verify that the output is as expected (e.g., shape, type).

    Note: This is a basic test and can be expanded with more specific checks
    based on the expected behavior of the CovBat function.
    """
    import numpy as np
    import pandas as pd
    from DiagnoseHarmonisation import HarmonisationFunctions

    # Create synthetic data
    np.random.seed(42)
    n_features = 50
    n_samples = 30
    n_batches = 2

    data = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=[f"feature_{i}" for i in range(n_features)],
        columns=[f"sample_{i}" for i in range(n_samples)],
    )

    # Create batch information
    batch = pd.Series(
        np.random.choice([f"batch_{i}" for i in range(n_batches)], size=n_samples),
        index=data.columns,
    )

    # Call CovBat function which uses combat internally
    corrected_data = HarmonisationFunctions.combat(data=data, batch=batch, mod=None, UseEB=True, parametric=True,covbat_mode=True)
    # Verify output, should be an array of same shape as input data
    assert isinstance(corrected_data, np.ndarray), "Output should be a numpy array"
    assert corrected_data.shape == data.shape, "Output shape should match input data shape"
    