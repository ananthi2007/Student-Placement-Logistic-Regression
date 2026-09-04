# Student Placement Eligibility Prediction Using Logistic Regression

## 📌 Project Overview

This project implements a **Logistic Regression** machine learning model to predict whether a student is likely to meet a placement eligibility criterion.

The model uses academic and experience-related features such as CGPA, attendance, coding score, projects completed, internship experience, and backlogs.

This project was developed as part of the **LearnDepth Academy Logistic Regression practice tasks**.

> **Note:** The dataset is synthetic and intended for educational practice. The results should not be used for real student placement decisions.

---

## 🎯 Objective

The main objective of this project is to build a binary classification model that predicts:

- `1` → Eligible
- `0` → Not Eligible

The project also evaluates the model using multiple classification metrics.

---

## 📂 Dataset

The dataset contains **1,000 records** and **6 predictor variables** along with one binary target variable.

### Features

| Feature | Description |
|---|---|
| `cgpa` | Student's CGPA |
| `attendance_pct` | Attendance percentage |
| `coding_score` | Coding score |
| `projects_completed` | Number of completed projects |
| `internship_months` | Internship experience in months |
| `backlogs` | Number of academic backlogs |
| `target` | Placement eligibility class |

### Dataset Information

- **Total records:** 1,000
- **Input features:** 6
- **Target classes:** 2
- **Missing values:** 0
- **Duplicate rows:** 0
- **Class 0:** 500
- **Class 1:** 500

The target classes are therefore evenly balanced.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook
- GitHub

---

## 🔍 Project Workflow

The project follows the following machine learning workflow:

1. Load the dataset
2. Inspect the dataset
3. Check missing values
4. Check duplicate records
5. Analyze target distribution
6. Perform Exploratory Data Analysis (EDA)
7. Analyze feature correlations
8. Separate features and target
9. Split data into training and testing sets
10. Train Logistic Regression model
11. Generate predictions
12. Generate probability predictions
13. Evaluate the model
14. Analyze the confusion matrix
15. Analyze Logistic Regression coefficients
16. Interpret the results
17. Identify limitations

---

## 📊 Exploratory Data Analysis

The following visualizations were created during the analysis:

- Target distribution
- Feature-wise boxplots
- Correlation heatmap
- Logistic Regression coefficient plot

EDA was performed to understand the distribution of the features and identify patterns between the input variables and the target classes.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected because this is a **binary classification problem**.

The model was trained using:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
