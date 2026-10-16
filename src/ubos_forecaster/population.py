
# This file is responsible for storing and validating district data.

# Task 1: Create District Class
"""
Build a reusable class that can store and validate figures.

Object: DistrictPopulation
Attributes: name, years and population figures.
"""

# Import necessary libraries to store figures as NumPy arrays
import numpy as np

class DistrictPopulation:
    """
    This class stores a district's name, years and population data.
    """

    def __init__(self, name: str, years: np.ndarray, populations: np.ndarray):
        # Store the district name and convert data to NumPy arrays
        self.name = name
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)

        # Validate the data when a district object is created
        self._validate()

    def _validate(self):
        """
        Check that years and populations have equal lengths,
        population data is not empty, and populations are non-negative.
        """
        if len(self.years) != len(self.populations):
            raise ValueError(f"Years and populations must be of equal lengths for {self.name}")

        if len(self.years) == 0:
            raise ValueError(f"Population data for {self.name} cannot be empty")

        if np.any(self.populations < 0):
            raise ValueError(f"Population cannot be negative for {self.name}")

    def __repr__(self):
        """
        Provide a useful description of the object when displayed
        in a notebook or during debugging.
        """
        return (f"DistrictPopulation(name='{self.name}', " f"N={len(self.years)} records, " f"years={self.years[0]}-{self.years[-1]})")

    def __len__(self):
        """
        Return the number of years of population observations.
        """
        return len(self.years)

    def year_on_year_growth(self):
        return (self.populations[1:] / self.populations[:-1] - 1) * 100

    def cagr(self):
        intervals = self.years[-1] - self.years[0]
        return ((self.populations[-1] / self.populations[0]) ** (1 / intervals) - 1) * 100
