# 🧠 Heart Disease Classification

This project applies various machine learning classifiers to predict the presence of heart disease based on a set of patient features. The goal is to build and evaluate several models to determine which classifier provides the most accurate results.

The dataset used for this project is from the UCI Machine Learning Repository, which contains information about patients' medical records, such as age, sex, blood pressure, cholesterol levels, and other health metrics, to predict whether or not they have heart disease.

---

## 📁 Dataset

- **File**: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Source**: UCI Machine Learning Repository
- **Description**: The dataset contains 303 instances of patient data with 14 attributes, including:
  - `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (serum cholesterol), `fbs` (fasting blood sugar), `restecg` (resting electrocardiographic results), and more.

---

## 🔍 Project Overview

### ✅ Objectives

- Clean and preprocess the dataset
- Train and evaluate multiple classifiers
- Compare model performance and select the best classifier

### 🧪 Steps:

1. **Data Cleaning**:

   - Handled missing values and outliers
   - Encoded categorical variables (e.g., `sex`, `cp`) using label encoding
   - Normalized numerical features

2. **Exploratory Data Analysis (EDA)**:

   - Statistical summaries
   - Visualized the distribution of key features

3. **Model Training & Evaluation**:

   - Split the dataset into training and testing sets (80/20)
   - Trained classifiers: Decision Tree, Kernel SVM, KNN, Random Forest, Logistic Regression, Naive Bayes, and SVM
   - Used accuracy as the evaluation metric

4. **Model Comparison**:

   - Compared model performance based on accuracy scores

5. **Visualization**:
   - Displayed bar plot of classifier accuracy for comparison

---

## 📊 Results & Insights

The following table shows the accuracy for each classifier:

| Classifier          | Accuracy |
| ------------------- | -------- |
| Decision Tree       | 80.26%   |
| Kernel SVM          | 85.53%   |
| KNN                 | 81.58%   |
| Random Forest       | 85.53%   |
| Logistic Regression | 85.25%   |
| Naive Bayes         | 82.89%   |
| SVM                 | 81.58%   |

### Key Insights:

- **Kernel SVM** and **Random Forest** achieved the highest accuracy at **85.53%**.
- **Decision Tree** performed decently with **80.26%**, but other models outperformed it.
- **Logistic Regression** also performed well with **85.25%** accuracy.
- **Naive Bayes** showed good results with an accuracy of **82.89%**, making it a reliable model as well.

### Suggestions:

- **Best Models**: Kernel SVM and Random Forest should be considered for deployment due to their top performance.
- **Model Tuning**: Further hyperparameter tuning could potentially improve performance for all models.

---

## 🧠 Strategic Takeaways

- Use **Kernel SVM** or **Random Forest** for high accuracy predictions in clinical applications.
- Consider the trade-off between model complexity and performance, as some models (e.g., Naive Bayes) perform well with less computational expense.
- Explore feature engineering to improve model performance further.

---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook**

### Libraries Used:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/AnandSreekumar03/HeartDisease.git
   cd HeartDisease
   ```
