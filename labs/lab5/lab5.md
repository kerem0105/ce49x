# Lab 5: Linear Regression with Scikit-Learn

**Team Work:**  
This lab should be completed in pairs. Collaboration is encouraged!

**Objective:**  
- Apply linear regression using Python’s Scikit-Learn.
- Perform exploratory data analysis (EDA) and preprocessing.
- Build, train, and evaluate a linear regression model.
- Visualize model predictions and assess performance.
- Prepare a short presentation summarizing your work.

---

## 1. Overview

In this lab, you will work with a real-world dataset to predict a continuous target variable using linear regression. We will use a dataset on **Concrete Compressive Strength**. The tasks include:

- **Loading and Exploring the Dataset:** Understand the features and target.
- **Data Preprocessing:** Handle any missing values and scale/normalize features if necessary.
- **Model Building:** Train a linear regression model using Scikit-Learn.
- **Model Evaluation:** Assess performance using metrics such as Mean Squared Error (MSE) and R².
- **Visualization:** Create plots to compare predicted vs. actual values and, optionally, a residual plot.
- **Presentation:** Prepare a short presentation (5 slides, maximum 5 minutes) summarizing your approach, results, and insights.

---

## 2. Prerequisites

- Familiarity with Python and basic operations using NumPy and Pandas.
- Basic understanding of machine learning concepts from our recent lecture.
- Ensure you have installed the required Python libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn

## 3. The Dataset

The dataset is located in the `/datasets/concrete_strength/concrete_strength.csv` file.  
**Dataset Description:**  
- **Features:** Measurements such as cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age.
- **Label:** Concrete compressive strength (MPa).

*(Refer to the dataset documentation in the repository for more details.)*

---

## 4. Instructions

### 4.1. Exploratory Data Analysis (EDA)
- **Load the Dataset:** Import the data using Pandas.
- **Summary Statistics:** Display the first few rows, summary statistics, and check for missing values.
- **Visualizations:** Create visualizations (e.g., histograms, pair plots) to understand the distributions of features and the target variable.

### 4.2. Data Preprocessing
- **Handling Missing Values:** Address any missing values in the dataset.
- **Scaling/Normalization:** Optionally apply scaling or normalization to the features.
- **Data Splitting:** Split the dataset into training and testing sets (e.g., 80% training, 20% testing).

### 4.3. Model Building and Evaluation
- **Train a Linear Regression Model:** Use Scikit-Learn’s `LinearRegression` to train your model on the training set.
- **Predict and Evaluate:** Use your model to predict the target on the test set. Evaluate the model using metrics such as Mean Squared Error (MSE) and R².
- **Visualizations:**  
  - Plot the predicted vs. actual values.
  - *(Optional)* Create a residual plot to analyze prediction errors.

### 4.4. Presentation
- **Prepare a Short Presentation:**  
  - Create **5 slides maximum** summarizing your work.
  - **Duration:** 5 minutes or less.
  - Include:
    - Overview of your approach (EDA, preprocessing, modeling).
    - Key results (evaluation metrics and visualizations).
    - Insights and any challenges encountered.
- **Presentation Delivery:** Be ready to present your slides in class.

---

## 5. Submission Instructions

1. **Repository Update:**  
   - Push your lab notebook/script and any supplementary files to your personal GitHub repository for the course by deadline, **17:00 on April 4, 2025**.

2. **Presentation Slides:**  
   - Prepare a 5-slide presentation summarizing your work.

3. **Email Submission:**  
   - Once completed, send an email with the subject:  
     ```
     [Your Name, Partner's Name] - Lab 5 Completed
     ```
   - Include a link to your GitHub repository.
   - Also attach or provide a link to your presentation slides.
   - Send the email to: **eyuphan.koc@bogazici.edu.tr**

---

## 6. Additional Resources

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
