# UBOS District Population Forecaster

## 1. Project Overview

The UBOS District Population Forecaster is a Python-based data science mini-project that analyses historical district population data and forecasts population for five future years (2025–2029). It demonstrates object-oriented programming, exploratory data analysis, forecasting, model evaluation, uncertainty estimation, and application of forecasts to planning.

**Data note:** The population figures used in this project are illustrative and synthetic. They are not official UBOS population estimates or forecasts.

## 2. Project Objectives

- Develop a reusable `DistrictPopulation` class using NumPy arrays.
- Calculate year-on-year population growth and compound annual growth rates (CAGR).
- Implement and compare linear trend, CAGR, and experimental Fibonacci-ratio forecasting methods.
- Evaluate forecasting performance using MAE, RMSE, and MAPE.
- Produce five-year population forecasts and approximate bootstrap prediction intervals.
- Demonstrate how forecasts can inform primary-school classroom planning under stated assumptions.
- Test the project's main functions and validation rules.

## 3. Project Structure

```text
Mini-Project 1 UBOS District Population Forecaster/
├── README.md
├── requirements.txt
├── pytest.ini
├── .vscode/
│   └── settings.json
├── notebooks/
│   └── population_forecaster.ipynb
├── src/
│   └── ubos_forecaster/
│       ├── __init__.py
│       ├── population.py
│       └── forecasters.py
└── tests/
    ├── test_population.py
    └── test_forecasters.py
```

## 4. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Pytest

## 5. Installation and Setup

Open a terminal in the project root directory and run:

```bash
python -m pip install -r requirements.txt
```

## 6. Running the Project

Open `notebooks/population_forecaster.ipynb` in VS Code with the Jupyter extension installed. Select the appropriate Python kernel and run the notebook cells from top to bottom.

To run the automated tests from the project root, execute:

```bash
python -m pytest -v
```

## 7. Forecasting Methods

**Linear Trend:** Fits a straight-line relationship between year and population.

**Compound Annual Growth Rate (CAGR):** Projects population using the average annual compound growth rate between the first and last observations.

**Fibonacci-Ratio Model:** Included as an experimental comparison. Its ratios do not have an established demographic justification in this project, and its validation performance was poor. It is therefore excluded from the final recommended forecasts.

The model with the lowest validation error was selected separately for each district.

## 8. Key Findings

- CAGR performed best among the tested methods for Kampala, Wakiso, Gulu, and Mbarara.
- Linear Trend performed best for Mbale.
- The classroom-planning scenario estimated approximately 4,687 additional classrooms across the five illustrative districts between 2024 and 2029, assuming 18% of the population is of primary-school age and 53 pupils per classroom.
- The classroom estimate represents potential additional demand associated with population growth, not a measured classroom shortage.

## 9. Limitations

- The population series are illustrative rather than official data.
- Only ten annual observations are available for each district.
- Forecast performance on a short historical holdout may not represent future performance.
- Bootstrap intervals provide approximate uncertainty estimates and do not capture every source of forecast uncertainty.
- Classroom planning results depend on assumptions about the school-age population and classroom capacity. Actual shortages require data on existing classrooms, enrolment, and utilisation.

The forecasts should therefore be treated as a demonstration of analytical methods, not as official demographic projections for policy decisions.

## 10. AI Assistance and Academic Integrity

An AI coding assistant was used to support debugging. The project author is responsible for reviewing the implementation, understanding the methods, verifying the results, and ensuring that the submitted work complies with the course's academic-integrity requirements.
