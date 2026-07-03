# Student Performance Analytics & Predictive Modeling

A comprehensive data engineering and exploratory data analysis (EDA) pipeline designed to inspect, clean, 
and predict student performance metrics utilizing the Student Performance Dataset.

## Project Structure

The project directory is structured as follows:

day3.py/
├── day3.py                  # Main Python pipeline script
├── student-mat.csv          # Student math course dataset (Semicolon delimited)
└── student-mat-enhanced.csv # Generated dataset featuring engineered columns

Features Covered
The pipeline dynamically orchestrates your data science functions categorized across four distinct structural phases:

Data Loading & Pre-Processing: Handles non-standard semicolon delimiters (sep=';') and engineers categorical flags like pass_fail benchmarks and multi-tiered age_group brackets.

Exploratory Data Analysis (EDA): Performs automated integrity checks mapping column dimensional shapes, validating data types, counting null fields, and scoring structural row duplicates.

Statistical Segmentation: Measures target attribute correlations (G3) alongside conditional filters to contrast high-attendance cohorts against chronic absentee trends.

Machine Learning: Builds an inline feature mapping matrix utilizing One-Hot Encoding dummy variables to train a scikit-learn LinearRegression predictive model.

Prerequisites
Ensure you have Python installed along with the required core analysis libraries:
pip install pandas scikit-learn
Getting Started

Ensure student-mat.csv is fully unzipped and residing in the exact same directory folder as your script execution file (day3.py).

Run the processing engine from your terminal:
python day3.py

Model Output MetricsUpon completion, the execution layout automatically provides:Mean Absolute Error (MAE):
Measures average absolute predictive distance variations from actual final marks.R-squared ($R^2$) Score: Evaluates total variance fit accuracy.Feature Weights Matrix: Breaks down model coefficient weights to determine the exact positive or negative performance impact value of individual student attributes.
