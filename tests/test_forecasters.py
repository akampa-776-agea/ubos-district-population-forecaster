
import numpy as np
import pytest

from ubos_forecaster import (LinearTrendForecaster, CAGRForecaster, FibonacciForecaster, bootstrap_prediction_intervals,)


def test_linear_trend_forecast():
    """Linear trend should recover a perfectly linear series."""
    years = np.array([2020, 2021, 2022])
    values = np.array([100, 110, 120])

    model = LinearTrendForecaster().fit(years, values)
    predictions = model.predict(2)

    np.testing.assert_allclose(predictions, [130, 140])


def test_cagr_forecast():
    """CAGR should preserve a constant compounded growth rate."""
    years = np.array([2020, 2021, 2022])
    values = np.array([100, 110, 121])

    model = CAGRForecaster().fit(years, values)
    predictions = model.predict(2)

    np.testing.assert_allclose(predictions, [133.1, 146.41])


def test_cagr_rejects_zero_population():
    """CAGR must reject non-positive population values."""
    years = np.array([2020, 2021, 2022])
    values = np.array([100, 0, 121])

    with pytest.raises(ValueError):
        CAGRForecaster().fit(years, values)


def test_fibonacci_forecast_returns_requested_horizon():
    """Experimental Fibonacci model should return the requested number of forecasts."""
    years = np.array([2020, 2021, 2022])
    values = np.array([100, 110, 120])

    model = FibonacciForecaster().fit(years, values)
    predictions = model.predict(5)

    assert len(predictions) == 5
    assert np.all(np.isfinite(predictions))


def test_bootstrap_intervals_are_reproducible():
    """A fixed seed should reproduce the same bootstrap intervals."""
    years = np.arange(2015, 2022)
    values = np.array([100, 105, 111, 116, 122, 128, 135])

    model = CAGRForecaster().fit(years, values)
    result_1 = bootstrap_prediction_intervals(  model, years, values,     horizon=3, n_resamples=1000, seed=42 )
    result_2 = bootstrap_prediction_intervals(   model, years, values,   horizon=3, n_resamples=1000, seed=42 )

    for first, second in zip(result_1, result_2):
        np.testing.assert_allclose(first, second)


def test_bootstrap_requires_at_least_1000_resamples():
    """The extension requires at least 1,000 resamples."""
    years = np.array([2020, 2021, 2022])
    values = np.array([100, 110, 121])

    model = CAGRForecaster().fit(years, values)

    with pytest.raises(ValueError, match="at least 1000"):
        bootstrap_prediction_intervals(  model, years, values,      horizon=3, n_resamples=100, seed=42 )
