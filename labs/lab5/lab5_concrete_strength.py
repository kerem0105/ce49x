"""
Lab 5: Linear Regression with Scikit-Learn

This script loads Concrete Compressive Strength data from an Excel file,
performs exploratory data analysis (EDA) and preprocessing,
trains a linear regression model using Scikit-Learn, evaluates its performance,
and produces visualizations for predicted versus actual values and residuals.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Enable interactive mode so plots are non-blocking
plt.ion()
# Use a clean style for plots
sns.set(style="whitegrid")


def load_data(file_path):
    """
    Load the concrete dataset from an Excel file.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if an error occurs.
    """
    print(f"Loading file: {file_path}")
    try:
        df = pd.read_excel(file_path)
        print("Columns found:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except Exception as e:
        print("Error loading Excel file:", e)
    return None


def perform_eda(df):
    """
    Perform exploratory data analysis (EDA) on the dataset.

    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\n=== First 5 Rows ===")
    print(df.head())
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Plot histograms for all columns
    df.hist(bins=20, figsize=(14, 10))
    plt.suptitle('Histograms of Features and Target')
    plt.show(block=False)


def preprocess_data(df):
    """
    Preprocess the dataset by separating features and target, scaling features,
    and splitting into training and testing sets.

    Parameters:
        df (pd.DataFrame): The dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) after preprocessing.
    """
    # Assume the target is the last column (Concrete compressive strength)
    X = df.iloc[:, :-1]  # All columns except the last
    y = df.iloc[:, -1]   # Last column is the target
    
    # Optionally, scale features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def train_linear_model(X_train, y_train):
    """
    Train a linear regression model using the training data.

    Parameters:
        X_train (np.array): Training features.
        y_train (pd.Series or np.array): Training target values.

    Returns:
        LinearRegression: The trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error (MSE) and R-squared (R²).

    Parameters:
        model (LinearRegression): The trained model.
        X_test (np.array): Test features.
        y_test (pd.Series or np.array): True target values.

    Returns:
        np.array: Predicted target values.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n=== Model Evaluation Metrics ===")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")
    return y_pred


def plot_predictions(y_test, y_pred):
    """
    Plot predicted vs. actual target values.

    Parameters:
        y_test (pd.Series or np.array): True target values.
        y_pred (np.array): Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Actual Compressive Strength (MPa)')
    plt.ylabel('Predicted Compressive Strength (MPa)')
    plt.title('Predicted vs. Actual Concrete Compressive Strength')
    # Diagonal line for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show(block=False)


def plot_residuals(y_test, y_pred):
    """
    Plot residuals to analyze prediction errors.

    Parameters:
        y_test (pd.Series or np.array): True target values.
        y_pred (np.array): Predicted target values.
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.xlabel('Predicted Compressive Strength (MPa)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(0, color='red', linestyle='--')
    plt.show(block=False)


def main():
    """
    Main function that:
    - Loads the concrete dataset.
    - Performs exploratory data analysis.
    - Preprocesses the data.
    - Trains and evaluates a linear regression model.
    - Generates visualizations for predictions and residuals.
    """
    # Default file path; allow override via command-line argument
    default_file = "../../datasets/concrete_strength/Concrete_Data.xls"
    file_path = sys.argv[1] if len(sys.argv) > 1 else default_file

    # Load the dataset
    df = load_data(file_path)
    if df is None:
        print("Exiting due to error in loading data.")
        return

    # Perform exploratory data analysis
    perform_eda(df)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train the linear regression model
    model = train_linear_model(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)

    # Visualize the predictions and residuals
    plot_predictions(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # Keep the plots open until user input
    input("Press [Enter] to exit and close all plots...")

if __name__ == "__main__":
    main()
