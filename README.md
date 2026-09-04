# 🎓 Student Placement Eligibility Prediction Using Logistic Regression

## 📌 Project Overview

This project implements a **Logistic Regression machine learning model** to predict whether a student is likely to meet a placement eligibility criterion.

The project uses the following input features:

- CGPA
- Attendance percentage
- Coding score
- Projects completed
- Internship experience
- Backlogs

This project was developed as part of the **LearnDepth Academy Logistic Regression practice tasks**.

> ⚠️ **Note:** The dataset is synthetic and intended for educational practice. It does not represent real students or real placement decisions.

---

## 🎯 Objective

The objective of this project is to build a **binary classification model** for the student placement eligibility use case.

The model predicts:

- `1` → Eligible
- `0` → Not Eligible

The model is evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix.

---

## 📂 Dataset

The dataset used in this project is:

`dataset_07_student_placement_eligibility.csv`

The dataset contains **1,000 rows**, **6 predictor variables**, and **1 binary target variable**.

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

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook
- GitHub

---

## 🔄 Project Workflow

The project follows these steps:

1. Load the dataset
2. Inspect the dataset
3. Check missing values
4. Check duplicate records
5. Check class balance
6. Perform Exploratory Data Analysis (EDA)
7. Analyze feature correlations
8. Separate features and target
9. Split the data into training and testing sets
10. Train the Logistic Regression model
11. Generate predictions
12. Generate probability predictions
13. Evaluate the model
14. Analyze the confusion matrix
15. Analyze Logistic Regression coefficients
16. Interpret the results
17. Identify limitations and possible improvements

---

## 📊 Exploratory Data Analysis

EDA was performed to understand the dataset and the relationship between the input features and the target.

The following visualizations were created:

### Target Distribution

The target distribution was analyzed to check the balance between the two classes.

The dataset contains:

- Class `0`: 500 records
- Class `1`: 500 records

### Feature-wise Boxplots

Boxplots were created to compare the distributions of the input features across the two target classes.

### Correlation Heatmap

A correlation heatmap was created to examine the relationships between the numerical features and the target.

### Logistic Regression Coefficient Plot

A coefficient plot was created to visualize the coefficients learned by the Logistic Regression model.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected because the problem is a **binary classification problem**.

The model was trained using:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

The `max_iter=1000` configuration was used to allow sufficient iterations for model optimization.

---

## ✂️ Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

The split was performed using `train_test_split`:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Split Results

- **Training data:** 800 records
- **Testing data:** 200 records
- **Number of features:** 6

`stratify=y` was used to preserve the class distribution in both the training and testing datasets.

---

## 🔮 Predictions

Predictions were generated using:

```python
y_pred = model.predict(X_test)
```

Probability predictions were generated using:

```python
y_probability = model.predict_proba(X_test)
```

The probability output contains:

- Probability of class `0`
- Probability of class `1`

---

## 📈 Model Evaluation

The Logistic Regression model was evaluated using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Results

| Metric | Score |
|---|---:|
| Accuracy | **70.00%** |
| Precision | **68.87%** |
| Recall | **73.00%** |
| F1-score | **70.87%** |
| ROC-AUC | **78.73%** |

---

## 🧮 Confusion Matrix

The confusion matrix obtained on the test data was:

```text
[[67, 33],
 [27, 73]]
```

The values represent:

- **True Negative (TN):** 67
- **False Positive (FP):** 33
- **False Negative (FN):** 27
- **True Positive (TP):** 73

### Prediction Summary

- Correct predictions = `67 + 73 = 140`
- Incorrect predictions = `33 + 27 = 60`
- Total test samples = `200`

The model correctly classified **140 out of 200 test samples**.

---

## 📊 Feature Coefficient Analysis

The Logistic Regression model learned the following coefficients:

| Feature | Coefficient |
|---|---:|
| `backlogs` | **0.368014** |
| `projects_completed` | **0.318701** |
| `attendance_pct` | **0.028213** |
| `internship_months` | **-0.023373** |
| `coding_score` | **-0.303601** |
| `cgpa` | **-0.371978** |

### Interpretation

A positive coefficient means that, **holding the other features constant**, an increase in that feature is associated with an increase in the model's estimated probability of `target = 1`.

A negative coefficient means that an increase in that feature is associated with a decrease in the model's estimated probability of `target = 1`.

> **Important:** The features have different numerical scales. Therefore, raw coefficient magnitudes should not be treated as a direct ranking of feature importance.

The coefficients represent patterns learned by the model from this synthetic dataset and should not be interpreted as real-world causal relationships.

---

## 🔍 Key Findings

- The dataset contains **1,000 records**.
- There are **6 input features**.
- The target contains two balanced classes.
- No missing values were found.
- No duplicate records were found.
- The Logistic Regression model achieved **70.00% accuracy**.
- The model achieved **78.73% ROC-AUC**.
- The model correctly classified **140 out of 200 test samples**.
- The coefficient analysis provides insight into how the model uses the input features.

---

## ⚠️ Limitations

- The dataset is **synthetic** and intended for educational practice.
- The model has not been validated on real student placement data.
- The results should not be used for real placement or employment decisions.
- The input features have different scales, limiting direct comparison of raw coefficient magnitudes.
- Model performance may change when evaluated on a different dataset.
- Correlation and model coefficients do not imply causation.

---

## 🚀 Future Improvements

Possible improvements include:

- Apply feature scaling.
- Compare Logistic Regression with other classification algorithms.
- Perform hyperparameter tuning.
- Use cross-validation.
- Analyze feature importance using additional techniques.
- Evaluate the model on appropriately collected real-world data.
- Tune the classification threshold depending on application requirements.

---

## 📁 Repository Contents

```text
Student-Placement-Logistic-Regression/
│
├── README.md
├── Student_Placement_Logistic_Regression.ipynb
└── dataset_07_student_placement_eligibility.csv
```

---

## 🔗 Google Colab

[[Open Project in Google Colab](PASTE_YOUR_COLAB_LINK_HERE)](https://colab.research.google.com/drive/1odfHKKIfHC5PvxhclhfUbTOU6Qh5Q827?usp=sharing)

---

## 📓 Notebook

The complete implementation is available in:

**`Student_Placement_Logistic_Regression.ipynb`**

The notebook contains:

- Dataset loading
- Dataset inspection
- Data quality checks
- Exploratory Data Analysis
- Correlation analysis
- Train-test split
- Logistic Regression training
- Predictions
- Probability predictions
- Model evaluation
- Confusion matrix
- Coefficient analysis

---

## 📌 Conclusion

This project demonstrates the complete workflow of applying **Logistic Regression to a binary classification problem**.

The model achieved:

- **70.00% Accuracy**
- **68.87% Precision**
- **73.00% Recall**
- **70.87% F1-score**
- **78.73% ROC-AUC**

The project provides practical experience in dataset inspection, exploratory data analysis, train-test splitting, Logistic Regression model training, prediction, evaluation, confusion-matrix analysis, and coefficient interpretation.

---

## 👩‍💻 Author

**Ananthi S**

B.E. Computer Science and Engineering
