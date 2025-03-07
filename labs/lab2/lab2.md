# Lab 2: Soil Test Data Analysis

**Objective:**  
In this lab, you will:
- Reinforce Python fundamentals by working with a small soil test dataset.
- Practice reading, cleaning, and transforming data from a CSV file.
- Compute and display descriptive statistics (minimum, maximum, mean, median, and standard deviation) for a numeric column.
- Organize your code into functions and add appropriate error handling.
- Practice using Git to commit and push your work to your own repository.
- Notify the instructor upon completion.

---

## 1. Overview

Building on Lab 1, this lab focuses on data cleaning and transformation. You will work with a soil test dataset containing values like soil pH, nitrogen, phosphorus, and moisture. Your task is to load the data, handle missing values, optionally remove outliers, and compute basic descriptive statistics.

---

## 2. Prerequisites

Before starting, ensure that you have:
- Completed Lab 1 and set up your Python environment.
- Python 3.10 or later installed.
- Required Python libraries:
  ```bash
  pip install pandas numpy
  ```
- A local Git repository for your course work.

---

## 3. The Dataset

The file `soil_test.csv` is located in the `/datasets/` directory. A sample of its content is shown below:

```csv
sample_id,soil_ph,nitrogen,phosphorus,moisture
1,6.5,20,15,30
2,7.0,25,20,35
3,5.8,18,NaN,32
4,6.2,22,17,28
5,6.8,NaN,16,33
```

> **Note:** "NaN" indicates missing values.

---

## 4. Instructions

### 4.1. Create the Python Script

Create a new file named `lab2_soil_analysis.py` in the `/labs/lab2/` directory.

### 4.2. Script Requirements

Your script should include the following functionality:

1. **Load the Dataset:**  
   - Use Pandas to load `soil_test.csv`.
   - Gracefully handle the case when the file is not found.

2. **Data Cleaning:**  
   - Handle missing values (e.g., fill with the column mean or drop the rows).
   - *(Optional)* Remove outliers for a chosen column (e.g., soil_ph values more than 3 standard deviations away from the mean).

3. **Compute Descriptive Statistics:**  
   - For at least one numeric column (e.g., `soil_ph`), compute:
     - Minimum
     - Maximum
     - Mean
     - Median
     - Standard Deviation
   - Print these statistics in a clear, formatted manner.

4. **Modular Code:**  
   - Organize your code into functions (e.g., `load_data()`, `clean_data()`, `compute_statistics()`).
   - Add comments and docstrings for clarity.

5. **Error Handling:**  
   - Use try/except blocks to manage potential errors during file I/O and data processing.


---

## 4. Running Your Script

From the `/labs/lab2/` directory, run your script by executing:
```bash
python lab2_soil_analysis.py
```

Ensure that the script runs without errors and displays the computed statistics.

---

## 6. Version Control: Committing and Pushing Your Code

Once you have completed the lab, follow these steps to push your work:

1. **Check your repository status:**
   ```bash
   git status
   ```

2. **Stage your changes:**
   ```bash
   git add labs/lab2/lab2_soil_analysis.py
   ```

3. **Commit your changes with a descriptive message:**
   ```bash
   git commit -m "Lab 2: Add soil test data analysis script"
   ```

4. **Push your changes to your repository:**
   ```bash
   git push origin main
   ```
   > **Note:** If you are working on a separate branch, push that branch instead.

---

## 7. Submission Instructions

When you have completed Lab 2, send an email to **eyuphan.koc@bogazici.edu.tr** with the subject line:
```
Name, LastName, Lab 2 Completed
```
Include a link to your GitHub repository in the email.

---

## 8. Additional Resources

- **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **Numpy Documentation:** [https://numpy.org/doc/](https://numpy.org/doc/)
- **Git and GitHub Guides:** [https://guides.github.com/](https://guides.github.com/)

Good luck with Lab 2! If you have any questions or encounter issues, please post on the class discussion board or reach out to the instructor.
