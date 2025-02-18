# lab1_traffic_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    # Allow a custom CSV file path via command-line argument; otherwise, use default path.
    csv_path = '../../datasets/traffic_data.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    # Load the traffic dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The dataset file was not found at '{csv_path}'. Please ensure it exists in the specified path.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return
    except pd.errors.ParserError:
        print("Error: There was an error parsing the CSV file.")
        return

    # Check if the required column exists
    if 'vehicle_count' not in df.columns:
        print("Error: 'vehicle_count' column not found in the dataset.")
        return

    # Compute basic descriptive statistics for 'vehicle_count'
    min_value = df['vehicle_count'].min()
    max_value = df['vehicle_count'].max()
    mean_value = df['vehicle_count'].mean()

    # Print the results
    print("Traffic Data Analysis:")
    print(f"Minimum vehicle count: {min_value}")
    print(f"Maximum vehicle count: {max_value}")
    print(f"Mean vehicle count: {mean_value:.2f}")

    # Optional: Plot a histogram of vehicle counts
    plt.figure(figsize=(10, 6))
    df['vehicle_count'].hist(bins=20, color='blue', edgecolor='black')
    plt.title('Vehicle Count Distribution')
    plt.xlabel('Vehicle Count')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()

if __name__ == '__main__':
    main()
