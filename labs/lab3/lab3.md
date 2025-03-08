# Lab 3: ERA5 Weather Data Analysis

**Objective:**  
In this lab, you will:
- Work with real-world ERA5 reanalysis weather data for Berlin and Munich.
- Practice data exploration, visualization, and statistical analysis using Python.
- Compute seasonal and monthly averages and other descriptive statistics.
- Create meaningful visualizations to represent weather patterns.
- Learn about accessing weather forecast data through an open-source project called Skyrim.
- Practice using Git to commit and push your work to your repository.

---

## 1. Overview

Building on Labs 1 and 2, this lab focuses on analyzing meteorological data from the ERA5 reanalysis dataset. ERA5 is a global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). You will work with wind data for Berlin and Munich, exploring seasonal patterns and statistical properties.

---

## 2. About ERA5

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate. Reanalysis combines model data with observations from across the world into a globally complete and consistent dataset. ERA5 provides hourly estimates of a large number of atmospheric, land, and oceanic climate variables, covering the Earth on a 30km grid and resolving the atmosphere using 137 levels from the surface up to a height of 80km.

Key features of ERA5:
- Covers the period from 1940 to present (with regular updates)
- Provides hourly data on many atmospheric, land-surface, and sea-state parameters
- Spatial resolution of approximately 31 km globally
- Used extensively in climate research, meteorology, renewable energy, agriculture, and many other fields

For this lab, we'll be working with wind data, which is particularly important for applications like renewable energy planning, aviation, and understanding local climate patterns.

---

## 3. Prerequisites

Before starting, ensure that you have:
- Completed Labs 1 and 2 and set up your Python environment.
- Python 3.10 or later installed.
- Required Python libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn
  ```
- A local Git repository for your course work.

---

## 4. The Dataset

For this lab, you will work with two CSV files containing ERA5 wind data:
- `berlin_era5_wind_20241231_20241231.csv`: Wind data for Berlin
- `munich_era5_wind_20241231_20241231.csv`: Wind data for Munich

These files contain hourly wind data including:
- Timestamps
- Wind speed components ([u10m](https://codes.ecmwf.int/grib/param-db/165) and [v10m](https://codes.ecmwf.int/grib/param-db/166)) 

The data covers most of the year 2024, allowing you to analyze seasonal/monthly patterns and compare weather conditions between the two cities.

---

## 5. Instructions

### 5.1. Create the Python Script

Create a new file named `lab3_era5_analysis.py` in the `/labs/lab3/` directory.

### 5.2. Script Requirements

Your script should include the following functionality:

1. **Load and Explore the Datasets:**  
   - Load both Berlin and Munich ERA5 data files. Be careful with the time objects (i.e. check `pd.Timestamp`).
   - Display basic information about the datasets (shape, columns, data types).
   - Handle any missing values appropriately.
   - Calculate and display summary statistics of the variables.

2. **Compute Temporal Aggregations:**  
   - Write a function to calculate the wind speed from `u10m` and `v10m`.
   - Calculate monthly averages for wind speed and temperature.
   - Calculate seasonal and monthly averages.
   - Compare the seasonal patterns between Berlin and Munich.

3. **Statistical Analysis:**  
   - Identify days/periods with extreme weather conditions (e.g., highest wind speeds). Compare your findings with the news sources found on the web.
   - Calculate diurnal (daily) patterns in wind speed.

4. **Visualization:**  
   - Create at least 3 meaningful visualizations, such as:
     - Time series plot of monthly average wind speeds for both cities
     - Seasonal comparison bar charts
     - Wind rose diagrams or directional analysis
   
6. **Modular Code:**  
   - Organize your code into functions with clear purposes.
   - Add appropriate comments and docstrings.
   - Include error handling for file operations and data processing.
  

7. **GitHub Task:**
   - Star and explore the Skyrim repository: [https://github.com/secondlaw-ai/skyrim](https://github.com/secondlaw-ai/skyrim)
   - In your script or a separate markdown file, write a brief (2-3 sentence) description of what this repository does and how it could be useful for weather data analysis.

---

## 6. Skyrim: Weather Forecasting Tool

As part of this lab, you'll explore the Skyrim repository, which provides a unified interface to run state-of-the-art large weather models. This tool allows users to:

- Run various weather models (like Graphcast, Pangu, Fourcastnet) with minimal setup
- Access initial conditions from sources like NOAA GFS and ECMWF IFS
- Generate forecasts for different time horizons
- Visualize and analyze weather predictions

Skyrim makes it possible to run complex weather models on consumer-grade hardware, democratizing access to advanced weather forecasting capabilities that were previously only available on supercomputers.

### Task:
1. Visit [https://github.com/secondlaw-ai/skyrim](https://github.com/secondlaw-ai/skyrim)
2. Star the repository (click the "Star" button in the upper right)
3. In your submission mail/report, briefly describe how you might use this for a civil/environmental engineering project

---

## 7. Running Your Script

From the `/labs/lab3/` directory, run your script by executing:
```bash
python lab3_era5_analysis.py
```

Ensure that the script runs without errors and displays the computed statistics and visualizations.

---

## 8. Version Control: Committing and Pushing Your Code

Once you have completed the lab, follow these steps to push your work:

1. **Check your repository status:**
   ```bash
   git status
   ```

2. **Stage your changes:**
   ```bash
   git add labs/lab3/lab3_era5_analysis.py
   git add labs/lab3/[any_other_files_you_created]
   ```

3. **Commit your changes with a descriptive message:**
   ```bash
   git commit -m "Lab 3: Add ERA5 weather data analysis"
   ```

4. **Push your changes to your repository:**
   ```bash
   git push origin main
   ```
   > **Note:** If you are working on a separate branch, push that branch instead.

---

## 9. Submission Instructions

When you have completed Lab 3, send an email to **eyuphan.koc@bogazici.edu.tr** with the subject line:
```
Name, LastName, Lab 3 Completed
```
Include the following in your email:
- A link to your GitHub repository
- A screenshot showing you've starred the Skyrim repository
- A brief summary (3-5 sentences) of your findings from the ERA5 data analysis, and 
- A brief paragraph on how you might use Skyrim (or any other similar tool that gives access to weather models) for a civil/environmental engineering project

---

## 10. Additional Resources

- **ERA5 Documentation:** [https://confluence.ecmwf.int/display/CKB/ERA5](https://confluence.ecmwf.int/display/CKB/ERA5)
- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **Matplotlib Documentation:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- **Seaborn Documentation:** [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- **Skyrim Repository:** [https://github.com/secondlaw-ai/skyrim](https://github.com/secondlaw-ai/skyrim)

Good luck with Lab 3! If you have any questions or encounter issues, please post on the class discussion board or reach out to the instructor. 