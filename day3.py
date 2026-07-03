from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically target the directory where this script is saved
script_dir = Path(__file__).resolve().parent
csv_path = script_dir / 'student-mat.csv'


# =====================================================================
# 1. DATA LOADING & PRE-PROCESSING
# =====================================================================

def load_data(path):
    """
    Loads the student dataset. 
    Note: The dataset uses semicolons instead of commas as delimiters.
    """
    df = pd.read_csv(path, sep=';')
    print(f"Dataset successfully loaded. Current shape: {df.shape}")
    return df


def enhance_features(df):
    """
    Creates new columns to give us better analytical angles.
    - pass_fail: Categorizes if a student passed (score of 10 or above).
    - age_group: Groups kids into distinct age bracket bins.
    """
    # Create the pass/fail flag based on the G3 score
    df['pass_fail'] = df['G3'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    print("\nPass/Fail breakdown:\n", df['pass_fail'].value_counts())
    
    # Bucket the ages into logical groups
    df['age_group'] = pd.cut(df['age'], bins=[14, 16, 18, 22], labels=['15-16', '17-18', '19+'])
    print("\nAverage final grade by age group:\n", df.groupby('age_group')['G3'].mean())
    
    return df


# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

def inspect_data(df):
    """
    A comprehensive health check on the structural composition of the dataset.
    """
    print("\n--- FIRST 5 ROWS (df.head) ---")
    print(df.head())
    
    print("\n--- LAST 5 ROWS (df.tail) ---")
    print(df.tail())
    
    print("\n--- DATA TYPES & STORAGE (df.info) ---")
    print(df.info())
    
    print("\n--- SHAPE ---", df.shape)
    print("\n--- ALL COLUMNS ---", list(df.columns))
    
    print("\n--- MISSING VALUE COUNTS ---")
    print(df.isnull().sum())
    
    print("\n--- DUPLICATED ROWS COUNT ---", df.duplicated().sum())


def describe_data(df):
    """
    Generates summary statistics for both numbers and text fields.
    """
    print("\n--- NUMERIC STATS SUMMARY (df.describe) ---")
    print(df.describe())
    print("\n--- CATEGORICAL STATS SUMMARY ---")
    print(df.describe(include='object'))


def analyze_categorical_columns(df, columns):
    """
    Counts entries falling into categories and converts them to percentages.
    """
    for col in columns:
        print(f"\n--- {col.upper()} COUNTS ---")
        print(df[col].value_counts())
        print(f"--- {col.upper()} PERCENTAGE DISTRIBUTION ---")
        print(df[col].value_counts(normalize=True).round(3) * 100)


def target_correlations(df, target_col='G3'):
    """
    Checks which numerical factors have the strongest linear relationship with final grades.
    """
    corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print(f"\n--- LINEAR CORRELATION WITH {target_col} ---")
    print(corr)
    return corr


# =====================================================================
# 3. FILTERS, SORTING & SEGMENTATIONS
# =====================================================================

def isolate_heavy_studiers(df, min_studytime=2):
    """
    Filters out students who study more than a given threshold to check their performance.
    """
    hardworkers = df[df['studytime'] > min_studytime]
    print(f"\nStudents studying more than {min_studytime} hours: {len(hardworkers)}")
    print("Their average final grade:", hardworkers['G3'].mean())
    return hardworkers


def extract_extreme_scorers(df, n=5):
    """
    Sorts and pulls out the top and bottom 'n' performers based on final marks.
    """
    # Sort descending to find top scorers
    top = df.sort_values(by='G3', ascending=False).head(n)[['age', 'studytime', 'G3']]
    # Sort ascending to find bottom scorers
    bottom = df.sort_values(by='G3', ascending=True).head(n)[['age', 'studytime', 'G3']]
    
    print(f"\n--- TOP {n} STUDENTS (via sort_values) ---\n", top)
    print(f"\n--- BOTTOM {n} STUDENTS (via sort_values) ---\n", bottom)
    return top, bottom


def run_groupby_metrics(df, group_col, target_col='G3'):
    """
    Groups data by a single column to look at aggregate performance metrics.
    """
    result = df.groupby(group_col)[target_col].agg(['mean', 'min', 'max', 'count'])
    print(f"\n--- AGGREGATING {group_col.upper()} AGAINST {target_col} ---")
    print(result)
    return result


def run_multi_groupby_metrics(df, group_cols, target_col='G3'):
    """
    Groups data across multiple layers to find deeper performance trends.
    """
    result = df.groupby(group_cols)[target_col].mean()
    print(f"\n--- AGGREGATING {group_cols} AGAINST {target_col} ---")
    print(result)
    return result


def compare_absentee_impact(df, low_limit=3, high_limit=10):
    """
    Measures the grade gap between students with low absences vs high absences.
    """
    low_abs_mean = df[df['absences'] <= low_limit]['G3'].mean()
    high_abs_mean = df[df['absences'] > high_limit]['G3'].mean()
    print(f"\nLow Absence Group (<= {low_limit} days) mean score: {low_abs_mean:.2f}")
    print(f"High Absence Group (> {high_limit} days) mean score: {high_abs_mean:.2f}")
    return low_abs_mean, high_abs_mean


# =====================================================================
# 4. DATA VISUALIZATION
# =====================================================================

def plot_studytime_vs_grade(df, save_path=None):
    """Generates a bar plot looking at average grades across different study buckets."""
    plt.figure(figsize=(6, 4))
    df.groupby('studytime')['G3'].mean().plot(kind='bar', color='steelblue')
    plt.title("Average Final Grade by Study Time")
    plt.xlabel("Study Time Level")
    plt.ylabel("Average G3 Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved bar plot to {save_path}")
    plt.close()


def plot_correlation_heatmap(df, save_path=None):
    """Generates a heatmap showing relationships between all numeric columns."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    plt.close()


def plot_grade_distribution(df, save_path=None):
    """Plots a histogram with a smooth distribution curve for the final marks."""
    plt.figure(figsize=(6, 4))
    sns.histplot(df['G3'], bins=20, kde=True, color='seagreen')
    plt.title("Overall Distribution of Final Grades (G3)")
    plt.xlabel("Final Grade (G3)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved distribution plot to {save_path}")
    plt.close()


# =====================================================================
# 5. MACHINE LEARNING & EXPORT
# =====================================================================

def train_predictive_model(df):
    """
    Prepares features, handles categorical dummy conversions inline, 
    and trains a simple Linear Regression model to predict final grades.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score

    # We will blend numeric variables and a few categorical variables
    numeric_features = ['studytime', 'absences', 'failures', 'G1', 'G2']
    categorical_features = ['sex', 'internet', 'school', 'address']
    
    # Process categorical variables using get_dummies directly inline here
    X = pd.get_dummies(df[numeric_features + categorical_features], columns=categorical_features, drop_first=True)
    y = df['G3']

    # Split the dataset into an 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train our linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Output how well the model performed
    print("\n--- PREDICTIVE MODEL PERFORMANCE ---")
    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
    print("R-squared (R2) Score:", r2_score(y_test, predictions))
    print("Model Feature Weights:", dict(zip(X.columns, model.coef_)))
    return model


def save_processed_data(df, filename='student-mat-enhanced.csv'):
    """Saves our newly transformed dataset into a clean CSV file."""
    output_path = script_dir / filename
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data successfully saved to: {output_path}")


# =====================================================================
# PIPELINE EXECUTION ENGINE
# =====================================================================

def main():
    # 1. Load data and add feature flags
    raw_data = load_data(csv_path)
    df = enhance_features(raw_data)
    
    # 2. Run structural and visual analysis inspections
    inspect_data(df)
    describe_data(df)
    analyze_categorical_columns(df, ['sex', 'internet', 'school', 'address'])
    target_correlations(df)

    # 3. Slice, sort, and segment the data
    isolate_heavy_studiers(df)
    extract_extreme_scorers(df)
    run_groupby_metrics(df, 'studytime')
    run_groupby_metrics(df, 'internet')
    run_multi_groupby_metrics(df, ['studytime', 'internet'])
    compare_absentee_impact(df)

    # 4. Save analytical charts to disk
    plot_studytime_vs_grade(df, script_dir / 'studytime_vs_grade.png')
    plot_correlation_heatmap(df, script_dir / 'correlation_heatmap.png')
    plot_grade_distribution(df, script_dir / 'grade_distribution.png')

    # 5. Train predictive model and write final data out
    train_predictive_model(df)
    save_processed_data(df, 'student-mat-enhanced.csv')

    return df


if __name__ == "__main__":
    final_processed_df = main()