import sys
import pandas as pd
import numpy as np

def load_data(csv_path="../../datasets/soil_test.csv"):
    # Allow a custom CSV file path via command-line argument; otherwise, use the default path.
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The dataset file was not found at '{csv_path}'. Please ensure it exists in the specified path.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was an error parsing the CSV file.")
        return None

    return df

def clean_data(df):
    """
    Clean the data by handling missing values and removing outliers.

    - Fills missing numeric values with the column mean.
    - Removes outliers from the 'soil_ph' column that are more than 3 standard deviations away from the mean.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        # Fill missing numeric values with the column mean
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Remove outliers in 'soil_ph' column (if it exists)
        if 'soil_ph' in df.columns:
            mean_val = df['soil_ph'].mean()
            std_val = df['soil_ph'].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            df = df[(df['soil_ph'] >= lower_bound) & (df['soil_ph'] <= upper_bound)]
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
    return df

def compute_statistics(df, column_name):
    """
    Compute descriptive statistics for a given numeric column.

    The statistics computed are:
      - Minimum
      - Maximum
      - Mean
      - Median
      - Standard Deviation

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numeric column.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
        
        col_data = df[column_name]
        stats = {
            "Minimum": col_data.min(),
            "Maximum": col_data.max(),
            "Mean": col_data.mean(),
            "Median": col_data.median(),
            "Standard Deviation": col_data.std()
        }
        return stats
    except Exception as e:
        print(f"An error occurred while computing statistics: {e}")
        return {}

def main():
    """
    Main function to execute the workflow:
      1. Load the dataset.
      2. Clean the data.
      3. Compute and print descriptive statistics for 'soil_ph'.
    """
    df = load_data()
    if df is None:
        return  # Exit if data loading fails
    
    df_clean = clean_data(df)
    stats = compute_statistics(df_clean, "soil_ph")
    
    if stats:
        print("Descriptive Statistics for 'soil_ph':")
        for stat, value in stats.items():
            print(f"{stat:20}: {value:.2f}")

if __name__ == "__main__":
    main()
