"""
Lab 3: ERA5 Weather Data Analysis

This script loads ERA5 wind data for Berlin and Munich using relative paths,
assuming the CSV files are located two folders above the script.
It performs exploratory data analysis, computes temporal aggregations,
identifies extreme wind events, and produces visualizations for comparison.

"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use a clean style for plots
sns.set(style="whitegrid")

def load_data(csv_path):
    """
    Load CSV data with proper date parsing.

    Parameters:
        csv_path (str): Relative path to the CSV file.

    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if an error occurs.
    """

    print(f"Loading file: {csv_path}")  # Debugging output

    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        print("Columns found:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_path}'.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except pd.errors.ParserError as e:
        print("Error parsing CSV file:", e)
    
    return None

def calculate_wind_speed(u, v):
    """Calculate wind speed from u and v components."""
    return np.sqrt(u**2 + v**2)

def assign_season(month):
    """Assign a season based on the month."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return np.nan

def process_data(df):
    """
    Process the DataFrame by handling missing values, calculating wind speed,
    and extracting time components.
    """
    # Ensure required columns are present
    for col in ['u10m', 'v10m', 'timestamp']:
        if col not in df.columns:
            print(f"Error: Expected column '{col}' not found.")
            return None

    df = df.dropna(subset=['u10m', 'v10m'])
    df['wind_speed'] = calculate_wind_speed(df['u10m'], df['v10m'])
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['season'] = df['month'].apply(assign_season)
    return df

def compute_monthly_average(df, column):
    """Compute monthly average for a specified column."""
    return df.groupby('month')[column].mean()

def compute_seasonal_average(df, column):
    """Compute seasonal average for a specified column."""
    return df.groupby('season')[column].mean()

def plot_monthly_wind_speed(berlin_monthly, munich_monthly):
    """Plot monthly average wind speeds for Berlin and Munich."""
    plt.figure(figsize=(10, 6))
    plt.plot(berlin_monthly.index, berlin_monthly, marker='o', label='Berlin')
    plt.plot(munich_monthly.index, munich_monthly, marker='o', label='Munich')
    plt.xlabel('Month')
    plt.ylabel('Average Wind Speed')
    plt.title('Monthly Average Wind Speed Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_seasonal_comparison(berlin_seasonal, munich_seasonal):
    """Plot a bar chart comparing seasonal average wind speeds."""
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    berlin_values = [berlin_seasonal.get(season, np.nan) for season in seasons]
    munich_values = [munich_seasonal.get(season, np.nan) for season in seasons]
    x = np.arange(len(seasons))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, berlin_values, width, label='Berlin')
    plt.bar(x + width/2, munich_values, width, label='Munich')
    plt.xlabel('Season')
    plt.ylabel('Average Wind Speed')
    plt.title('Seasonal Average Wind Speed Comparison')
    plt.xticks(x, seasons)
    plt.legend()
    plt.show()

def plot_diurnal_cycle(df_berlin, df_munich):
    """Plot the diurnal cycle of wind speed for both cities."""
    berlin_diurnal = df_berlin.groupby('hour')['wind_speed'].mean()
    munich_diurnal = df_munich.groupby('hour')['wind_speed'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(berlin_diurnal.index, berlin_diurnal, marker='o', label='Berlin')
    plt.plot(munich_diurnal.index, munich_diurnal, marker='o', label='Munich')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Wind Speed')
    plt.title('Diurnal Wind Speed Patterns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_wind_direction_histogram(df_berlin, df_munich):
    """
    Compute wind direction and plot histograms for both cities.
    """
    df_berlin['wind_direction'] = (np.degrees(np.arctan2(df_berlin['v10m'], df_berlin['u10m'])) + 360) % 360
    df_munich['wind_direction'] = (np.degrees(np.arctan2(df_munich['v10m'], df_munich['u10m'])) + 360) % 360
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df_berlin['wind_direction'], bins=36, range=(0, 360), alpha=0.7)
    plt.xlabel('Wind Direction (degrees)')
    plt.ylabel('Frequency')
    plt.title('Berlin Wind Direction Histogram')
    plt.subplot(1, 2, 2)
    plt.hist(df_munich['wind_direction'], bins=36, range=(0, 360), alpha=0.7)
    plt.xlabel('Wind Direction (degrees)')
    plt.ylabel('Frequency')
    plt.title('Munich Wind Direction Histogram')
    plt.tight_layout()
    plt.show()

def identify_extreme_winds(df, top_n=5):
    """Identify the top_n records with the highest wind speeds."""
    return df.nlargest(top_n, 'wind_speed')

def main():
    """
    Main function that:
    - Loads Berlin and Munich wind data using default relative paths.
    - Allows custom paths via command-line arguments.
    - Processes the data.
    - Computes statistics.
    - Generates plots.
    """

    # Default relative paths
    berlin_file = "../../datasets/berlin_era5_wind_20241231_20241231.csv"
    munich_file = "../../datasets/munich_era5_wind_20241231_20241231.csv"

    # Allow overriding via command-line arguments
    if len(sys.argv) > 1:
        berlin_file = sys.argv[1]  # First argument is Berlin dataset
    if len(sys.argv) > 2:
        munich_file = sys.argv[2]  # Second argument is Munich dataset

    # Load the datasets
    berlin_df = load_data(berlin_file)
    munich_df = load_data(munich_file)

    if berlin_df is None or munich_df is None:
        print("Error loading one or both CSV files. Exiting.")
        return

    # Process the datasets
    berlin_df = process_data(berlin_df)
    munich_df = process_data(munich_df)
    if berlin_df is None or munich_df is None:
        print("Error processing the data. Exiting.")
        return

    # Print basic info
    print("=== Berlin Data Info ===")
    print(berlin_df.info())
    print("\n=== Munich Data Info ===")
    print(munich_df.info())

    # Compute averages
    berlin_monthly_avg = compute_monthly_average(berlin_df, 'wind_speed')
    munich_monthly_avg = compute_monthly_average(munich_df, 'wind_speed')
    berlin_seasonal_avg = compute_seasonal_average(berlin_df, 'wind_speed')
    munich_seasonal_avg = compute_seasonal_average(munich_df, 'wind_speed')

    print("\n=== Berlin Monthly Average Wind Speed ===")
    print(berlin_monthly_avg)
    print("\n=== Munich Monthly Average Wind Speed ===")
    print(munich_monthly_avg)

    print("\n=== Berlin Seasonal Average Wind Speed ===")
    print(berlin_seasonal_avg)
    print("\n=== Munich Seasonal Average Wind Speed ===")
    print(munich_seasonal_avg)

    # Identify extreme wind conditions
    print("\n=== Extreme Wind Conditions in Berlin ===")
    print(identify_extreme_winds(berlin_df))
    print("\n=== Extreme Wind Conditions in Munich ===")
    print(identify_extreme_winds(munich_df))

    # Create visualizations
    plot_monthly_wind_speed(berlin_monthly_avg, munich_monthly_avg)
    plot_seasonal_comparison(berlin_seasonal_avg, munich_seasonal_avg)
    plot_diurnal_cycle(berlin_df, munich_df)
    plot_wind_direction_histogram(berlin_df, munich_df)

# GitHub: Skyrim repository description
#
# Skyrim Repository:
# Skyrim is an open-source tool that streamlines the process of running advanced weather models 
# (including Graphcast, Pangu, and Fourcastnet) with minimal configuration. It harnesses initial 
# condition data from sources like NOAA GFS and ECMWF IFS to produce forecasts and visualizations. 
# In civil and environmental engineering, Skyrim can incorporate cutting-edge weather 
# predictions into planning, risk analysis, and infrastructure design, thereby enhancing resilience 
# and supporting informed decision-making.

if __name__ == "__main__":
    main()

