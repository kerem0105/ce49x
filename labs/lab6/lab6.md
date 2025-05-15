# Lab 6: Feature Engineering and Document Classification in Construction Projects

**Team Work:**  
This lab should be completed in pairs. Collaboration is encouraged!

**Objective:**  
- Understand and apply feature engineering techniques for construction document analysis
- Implement text classification for construction project documents
- Work with categorical and text data in civil engineering context
- Handle missing data through imputation
- Prepare a short presentation summarizing your work

---

## 1. Overview

In this lab, you will work with construction project documents to build an automated document classification system. You will practice various feature engineering techniques while working with real-world civil engineering documentation. The tasks include:

- **Feature Engineering:** Work with construction document metadata and text content
- **Text Processing:** Convert construction documents into numerical features using techniques like TF-IDF
- **Model Building:** Implement Naive Bayes classification for document categorization
- **Model Evaluation:** Assess performance using metrics like accuracy and confusion matrix
- **Visualization:** Create plots to visualize results and key terms in different document types
- **Presentation:** Prepare a short presentation summarizing your approach, results, and insights

---

## 2. Prerequisites

- Familiarity with Python and basic operations using NumPy and Pandas
- Basic understanding of feature engineering and Naive Bayes from our recent lecture
- Ensure you have installed the required Python libraries:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn nltk

## 3. The Dataset

We will work with a synthetic dataset of construction project documents that includes five different types of documents commonly found in construction projects:

**Document Types:**
1. **Progress Reports:** Monthly updates on project status, completion percentages, and milestone achievements
2. **Requests for Information (RFIs):** Technical queries about specifications, drawings, or construction methods
3. **Change Orders:** Formal requests for modifications to project scope, timeline, or budget
4. **Safety Incident Reports:** Documentation of safety events, near-misses, and corrective actions
5. **Quality Inspection Reports:** Findings from quality control inspections and material testing

**Dataset Structure:**  
- Each document contains:
  - Text content
  - Metadata (date, project phase, author role)
  - Document type (label)
- The dataset includes 1000 documents (200 of each type)
- Documents contain typical construction industry terminology and formats

---

## 4. Instructions

### 4.1. Feature Engineering Practice
- **Categorical Data:**
  - Process document metadata (project phase, author role)
  - Practice one-hot encoding using `DictVectorizer` for categorical features
  - Visualize the distribution of document types across project phases

- **Text Processing:**
  - Implement construction-specific text preprocessing:
    - Remove common construction abbreviations
    - Handle technical measurements and units
    - Process industry-specific terminology
  - Convert document text to numerical features using:
    - Count Vectorization (`CountVectorizer`)
    - TF-IDF Vectorization (`TfidfVectorizer`)

- **Missing Data:**
  - Handle missing metadata in documents
  - Practice different imputation strategies using `SimpleImputer`
  - Compare the effects of different imputation methods on classification

### 4.2. Document Classification
- **Data Preparation:**
  - Load and preprocess the construction document dataset
  - Split into training and testing sets
  - Create features from both text content and metadata

- **Model Building:**
  - Train a Multinomial Naive Bayes classifier
  - Create a pipeline combining:
    - Metadata preprocessing
    - Text vectorization
    - Naive Bayes classification
  - Experiment with different feature combinations

- **Evaluation:**
  - Calculate accuracy, precision, recall, and F1-score
  - Create and visualize the confusion matrix
  - Analyze common misclassifications between document types
  - Identify key terms that distinguish different document types

### 4.3. Advanced Tasks (Optional)
- Implement cross-validation
- Create a hybrid model combining text and metadata features
- Compare performance with other classifiers
- Build a simple interface for classifying new documents
- Analyze temporal patterns in document types

### 4.4. Presentation
- **Prepare a Short Presentation:**  
  - Create **5 min walkthrough** summarizing your work
  - **Duration:** 5 minutes or less
  - Include:
    - Overview of document classification approach
    - Key features that distinguish document types
    - Model performance and insights
    - Potential applications in construction projects
- **Presentation Delivery:** Be ready to present your walkthrough in class

---

## 5. Submission Instructions

1. **Repository Update:**  
   - Push your lab notebook/script and any supplementary files to your personal GitHub repository for the course by deadline, **13:00 on May 16, 2025**

2. **Email Submission:**  
   - Once completed, send an email with the subject:  
     ```
     [Your Name, Partner's Name] - Lab 6 Completed
     ```
   - Include a link to your GitHub repository
   - Send the email to: **eyuphan.koc@bogazici.edu.tr**

---

## 6. Additional Resources

- [Scikit-Learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Scikit-Learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Construction Project Documentation Best Practices](https://www.constructionexec.com/article/best-practices-for-construction-project-documentation) 