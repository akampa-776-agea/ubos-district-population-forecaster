
# This file will contain the abstract Forecaster base class
# and all three forecasting subclasses.

from abc import ABC, abstractmethod
import numpy as np


class Forecaster(ABC):
    """Abstract base class for population forecasting models."""

    @abstractmethod
    def fit(self, years, values):
        """Fit the model to historical data."""
        pass

    @abstractmethod
    def predict(self, horizon):
        """Predict a specified number of future periods."""
        pass


class LinearTrendForecaster(Forecaster):
    """Forecast population using a linear trend."""

    def fit(self, years, values):
        self.years = np.asarray(years, dtype=float)
        self.values = np.asarray(values, dtype=float)

        if len(self.years) != len(self.values):
            raise ValueError("Years and values must have equal lengths.")
        if len(self.years) < 2:
            raise ValueError("At least two observations are required.")

        self.slope, self.intercept = np.polyfit(self.years, self.values, 1)
        self.last_year = self.years[-1]
        return self

    def predict(self, horizon):
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")

        future_years = self.last_year + np.arange(1, horizon + 1)
        return self.slope * future_years + self.intercept


class CAGRForecaster(Forecaster):
    """Forecast population using a constant compounded growth rate."""

    def fit(self, years, values):
        self.years = np.asarray(years, dtype=float)
        self.values = np.asarray(values, dtype=float)

        if len(self.years) != len(self.values):
            raise ValueError("Years and values must have equal lengths.")
        if len(self.values) < 2:
            raise ValueError("At least two observations are required.")
        if np.any(self.values <= 0):
            raise ValueError("Population values must be positive.")

        intervals = self.years[-1] - self.years[0]
        if intervals <= 0:
            raise ValueError("Years must span a positive interval.")

        self.rate = (self.values[-1] / self.values[0]) ** (1 / intervals) - 1

        self.last_year = self.years[-1]
        self.last_value = self.values[-1]
        return self

    def predict(self, horizon):
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")

        steps = np.arange(1, horizon + 1)
        return self.last_value * (1 + self.rate) ** steps


class FibonacciForecaster(Forecaster):
    """
    Experimental Fibonacci-ratio scaling model.

    This is an assumed interpretation of the assignment's
    Fibonacci-ratio approach, not a verified reproduction
    of the previous cohort's implementation.
    """

    def fit(self, years, values):
        self.years = np.asarray(years, dtype=float)
        self.values = np.asarray(values, dtype=float)

        if len(self.years) != len(self.values):
            raise ValueError("Years and values must have equal lengths.")
        if len(self.values) == 0:
            raise ValueError("Training data cannot be empty.")
        if np.any(self.values <= 0):
            raise ValueError("Population values must be positive.")

        self.last_year = self.years[-1]
        self.last_value = self.values[-1]
        return self

    def predict(self, horizon): 
        if horizon < 1:
            raise ValueError("Horizon must be at least 1.")

        fib = [1, 1]
        while len(fib) < horizon + 1:
            fib.append(fib[-1] + fib[-2])

        ratios = [fib[i + 1] / fib[i] for i in range(horizon)]

        return self.last_value * np.asarray(ratios)

    
def bootstrap_prediction_intervals(model, years, values,horizon: int = 5, n_resamples: int = 1000,seed: int = 42,):
    """
    Estimate approximate prediction intervals by bootstrapping
    in-sample model residuals.

    Parameters
    ----------
    model : fitted LinearTrendForecaster or CAGRForecaster
        A model already fitted to historical data.
    years : array-like
        Historical years used to fit the model.
    values : array-like
        Historical population values, in thousands.
    horizon : int
        Number of future years to forecast.
    n_resamples : int
        Number of bootstrap samples; must be at least 1000.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Point forecasts, lower 95% limits, upper 95% limits.
    """
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(years) != len(values):
        raise ValueError("Years and values must have equal lengths.")
    if len(values) < 2:
        raise ValueError("At least two observations are required.")
    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")
    if n_resamples < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples.")

    # Calculate fitted historical values for the selected model.
    if isinstance(model, LinearTrendForecaster):
        fitted_values = model.slope * years + model.intercept
    elif isinstance(model, CAGRForecaster):
        fitted_values = (values[0] * (1 + model.rate) ** (years - years[0]))
    else:
        raise TypeError(
            "Bootstrap intervals currently support LinearTrendForecaster "
            "and CAGRForecaster.")

    # Residuals represent differences between actual and fitted values.
    residuals = values - fitted_values
    residuals = residuals - np.mean(residuals)

    if not np.all(np.isfinite(residuals)):
        raise ValueError("Residuals must be finite.")

    # Generate point forecasts and resample historical residuals.
    point_forecasts = model.predict(horizon)
    rng = np.random.default_rng(seed)

    sampled_residuals = rng.choice(residuals, size=(n_resamples, horizon),  replace=True,)

    simulated_forecasts = point_forecasts + sampled_residuals

    lower_limits = np.quantile(simulated_forecasts, 0.025, axis=0)
    upper_limits = np.quantile(simulated_forecasts, 0.975, axis=0)

    return point_forecasts, lower_limits, upper_limits

