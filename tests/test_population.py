
import numpy as np
import pytest

from ubos_forecaster import DistrictPopulation


def test_district_population_creation():
    """A district should store its name, years and populations."""
    years = np.array([2020, 2021, 2022])
    populations = np.array([100, 110, 120])

    district = DistrictPopulation("Test District", years, populations)

    assert district.name == "Test District"
    assert len(district) == 3
    np.testing.assert_array_equal(district.years, years)
    np.testing.assert_array_equal(district.populations, populations)


def test_district_population_rejects_mismatched_lengths():
    """Years and population arrays must have equal lengths."""
    years = np.array([2020, 2021, 2022])
    populations = np.array([100, 110])

    with pytest.raises(ValueError, match="equal lengths"):
        DistrictPopulation("Test District", years, populations)


def test_district_population_rejects_negative_population():
    """Population values must not be negative."""
    years = np.array([2020, 2021])
    populations = np.array([100, -10])

    with pytest.raises(ValueError, match="negative"):
        DistrictPopulation("Test District", years, populations)
