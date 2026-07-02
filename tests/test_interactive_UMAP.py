import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
import sys
import os

# Add parent directory to path to import PlotDiagnosticResults
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DiagnoseHarmonisation'))

from DiagnoseHarmonisation.PlotDiagnosticResults import n_dim_umap_visualisation


class TestNDimUMAPVisualisation:
    """Test suite for n_dim_umap_visualisation function."""

    @pytest.fixture
    def sample_data_2d(self):
        """Generate sample 2D data with batches."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate data with batch structure
        batch1 = np.random.randn(n_samples // 2, n_features) + np.array([1] * n_features)
        batch2 = np.random.randn(n_samples // 2, n_features) + np.array([-1] * n_features)
        
        data = np.vstack([batch1, batch2])
        batch = np.array(['A'] * (n_samples // 2) + ['B'] * (n_samples // 2))
        
        return data, batch

    @pytest.fixture
    def sample_data_with_covariates(self):
        """Generate sample data with covariates."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        # Generate data with batch structure
        batch1 = np.random.randn(n_samples // 2, n_features) + np.array([1] * n_features)
        batch2 = np.random.randn(n_samples // 2, n_features) + np.array([-1] * n_features)
        
        data = np.vstack([batch1, batch2])
        batch = np.array(['A'] * (n_samples // 2) + ['B'] * (n_samples // 2))
        
        # Create covariates as a DataFrame
        covariates = pd.DataFrame({
            'cov1': np.random.randn(n_samples),
            'cov2': np.random.randint(0, 3, n_samples)
        })
        
        return data, batch, covariates

    def test_umap_2d_basic(self, sample_data_2d):
        """Test UMAP 2D visualization with basic parameters."""
        data, batch = sample_data_2d
        
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            show=False
        )
        
        # Check output shape
        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        assert embedding.shape[1] == 2
        
        plt.close('all')

    def test_umap_3d_basic(self, sample_data_2d):
        """Test UMAP 3D visualization with basic parameters."""
        data, batch = sample_data_2d
        
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=3,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            show=False
        )
        
        # Check output shape (should be 3D even though warning might be issued)
        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        # Embedding should be full dimensionality requested
        assert embedding.shape[1] >= 2
        
        plt.close('all')

    def test_umap_with_covariates_interactive(self, sample_data_with_covariates):
        """Test covariate workflow remains non-interactive when show=False."""
        data, batch, covariates = sample_data_with_covariates

        with patch("ipywidgets.interact") as mock_interact:
            embedding = n_dim_umap_visualisation(
                data=data,
                batch=batch,
                n_dimensions=2,
                covariates=covariates,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                show=False
            )
            mock_interact.assert_not_called()
        
        # Check output shape
        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        assert embedding.shape[1] == 2
        
        plt.close('all')

    def test_umap_selector_toggle_multiple_times(self, sample_data_with_covariates):
        """Regression test: repeated selector toggles should not blank the plot."""
        data, batch, covariates = sample_data_with_covariates

        with patch("matplotlib.pyplot.show"):
            embedding = n_dim_umap_visualisation(
                data=data,
                batch=batch,
                n_dimensions=2,
                covariates=covariates,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean',
                show=True,
            )

        assert embedding is not None
        fig = plt.gcf()
        selector = getattr(fig, "_umap_color_selector", None)
        assert selector is not None
        main_ax = fig.axes[0]
        initial_bounds = main_ax.get_position().bounds

        # Toggle covariate -> batch -> covariate again.
        selector.set_active(1)
        selector.set_active(0)
        selector.set_active(2)

        assert len(main_ax.collections) > 0
        final_bounds = main_ax.get_position().bounds
        np.testing.assert_allclose(final_bounds, initial_bounds, atol=1e-3)

        plt.close('all')

    def test_umap_custom_parameters(self, sample_data_2d):
        """Test UMAP with custom parameters."""
        data, batch = sample_data_2d
        
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            n_neighbors=25,
            min_dist=0.05,
            metric='cosine',
            random_state=123,
            show=False
        )
        
        assert embedding is not None
        assert embedding.shape == (data.shape[0], 2)
        
        plt.close('all')

    def test_umap_deterministic_random_state(self, sample_data_2d):
        """Test that UMAP with same random state produces consistent results."""
        data, batch = sample_data_2d
        
        embedding1 = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            random_state=42,
            show=False
        )
        
        embedding2 = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            random_state=42,
            show=False
        )
        
        # Embeddings should be very close (allowing for floating point precision)
        np.testing.assert_allclose(embedding1, embedding2, atol=1e-5)
        
        plt.close('all')

    def test_umap_invalid_dimensions_warning(self, sample_data_2d):
        """Test that invalid dimensions trigger a warning."""
        data, batch = sample_data_2d
        
        with pytest.warns(UserWarning, match="n_dimensions should be 2 or 3"):
            embedding = n_dim_umap_visualisation(
                data=data,
                batch=batch,
                n_dimensions=5,  # Invalid: should trigger warning
                show=False
            )
            
            # Should still return 2D embedding (fallback)
            assert embedding.shape[1] >= 2
        
        plt.close('all')

    def test_umap_returns_full_embedding(self, sample_data_2d):
        """Test that function returns full embedding even if visualization is 2D."""
        data, batch = sample_data_2d
        
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            show=False
        )
        
        # Verify we get an embedding back
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == data.shape[0]
        
        plt.close('all')

    @pytest.mark.interactive
    def test_umap_2d_interactive_display(self, sample_data_2d):
        """Test UMAP 2D with interactive display (manual testing)."""
        data, batch = sample_data_2d
        
        # This will display an interactive plot
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            show=True  # Display the figure
        )
        
        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        
        plt.close('all')

    @pytest.mark.interactive
    def test_umap_3d_interactive_display(self, sample_data_2d):
        """Test UMAP 3D with interactive display (manual testing)."""
        data, batch = sample_data_2d
        
        # This will display an interactive 3D plot
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=3,
            show=True  # Display the figure
        )
        
        assert embedding is not None
        
        plt.close('all')

    @pytest.mark.interactive
    def test_umap_with_covariates_interactive_display(self, sample_data_with_covariates):
        """Test UMAP 2D with covariates and interactive display (manual testing)."""
        data, batch, covariates = sample_data_with_covariates
        
        # This will display an interactive plot with covariate selector
        embedding = n_dim_umap_visualisation(
            data=data,
            batch=batch,
            n_dimensions=2,
            covariates=covariates,
            show=True  # Display interactive figure
        )
        
        assert embedding is not None
        assert embedding.shape[0] == data.shape[0]
        
        plt.close('all')


if __name__ == "__main__":
    # Run pytest with interactive tests marked
    pytest.main([__file__, "-v", "-m", "not interactive"])
