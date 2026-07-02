# Test function for the age chart visualization
import pytest
import pandas as pd
import numpy as np
from DiagnoseHarmonisation.PlotDiagnosticResults import plot_age_percentile_chart


def test_plot_age_percentile_chart():
    # Create a sample DataFrame with age data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'age': np.random.randint(20, 80, size=100),
        'batch': np.random.choice(['A', 'B'], size=100),
        'data': np.random.randn(100)
    })

    # Call the function to generate the age percentile chart
    fig, ax = plot_age_percentile_chart(sample_data, age_col='age', batch_col='batch', value_col='data')

    # Check if the returned object is a matplotlib figure
    assert fig is not None
    assert hasattr(fig, 'savefig')  # Check if it's a matplotlib figure

    # Optionally, you can check if the figure has axes and lines
    assert len(fig.axes) > 0
    assert len(fig.axes[0].lines) > 0


def test_plot_age_percentile_chart_custom_percentiles():
    np.random.seed(7)
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 85, size=120),
        'batch': np.random.choice(['A', 'B', 'C'], size=120),
        'data': np.random.randn(120)
    })

    fig, ax = plot_age_percentile_chart(
        sample_data,
        age_col='age',
        batch_col='batch',
        value_col='data',
        percentiles=(10, 50, 90),
    )

    assert fig is not None
    assert ax is not None
    # At least one curve per supplied percentile.
    assert len(ax.lines) >= 3


def test_plot_age_percentile_chart_invalid_percentiles_raises():
    np.random.seed(11)
    sample_data = pd.DataFrame({
        'age': np.random.randint(20, 80, size=100),
        'batch': np.random.choice(['A', 'B'], size=100),
        'data': np.random.randn(100)
    })

    with pytest.raises(ValueError, match="between 0 and 100"):
        plot_age_percentile_chart(
            sample_data,
            age_col='age',
            batch_col='batch',
            value_col='data',
            percentiles=(-5, 50, 105),
        )