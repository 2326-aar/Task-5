# Task-5
# Decision Trees and Random Forests 

This repository contains the complete implementation of **Decision Tree** and **Random Forest** models using the **Heart Disease Dataset**. 
---

## 📌 Objectives

The main goal of this task is to:
1. Understand how **Decision Trees** and **Random Forests** work for classification.
2. Learn how to avoid **overfitting** using pruning techniques and hyperparameter tuning.
3. Train, evaluate, and compare models using appropriate metrics.
4. Visualize decision trees and interpret **feature importance**.
5. Practice **cross-validation** for robust model validation.

---

## 📁 Dataset Description

The dataset used is the **Heart Disease Dataset**, which contains medical attributes of patients and a target value indicating the presence of heart disease.

- **Filename**: `heart.csv`
- **Target column**: `target`  
  - 1 → Presence of heart disease  
  - 0 → Absence of heart disease
## ✅ Steps Followed

### Step 1: Data Loading & Preprocessing
- Loaded `heart.csv` using `pandas`.
- Split data into **features (X)** and **target (y)**.
- Performed an 80/20 train-test split.

### Step 2: Decision Tree Classifier
- Trained a default `DecisionTreeClassifier`.
- Evaluated performance on test set using accuracy score.
- Visualized the decision tree using `plot_tree()`.

### Step 3: Overfitting Control
- Limited tree depth (`max_depth=4`) to reduce overfitting.
- Compared pruned tree accuracy vs. full tree accuracy.

### Step 4: Random Forest Classifier
- Trained a `RandomForestClassifier` with 100 estimators.
- Evaluated performance using test accuracy and classification report.

### Step 5: Feature Importance
- Visualized importance of each feature using bar chart.

### Step 6: Cross-Validation
- Used `cross_val_score` to compute 5-fold cross-validation accuracy.
- Compared CV scores of Decision Tree and Random Forest.

**Attributes include**:
- age
- sex
- cp (chest pain type)
- trestbps (resting blood pressure)
- chol (cholesterol)
- fbs (fasting blood sugar)
- restecg (resting ECG)
- thalach (maximum heart rate achieved)
- exang (exercise-induced angina)
- oldpeak (ST depression)
- slope, ca, thal, etc.

---

## 🧰 Libraries Used

- **Libraries**:
  - `pandas` – data manipulation
  - `matplotlib` – plotting graphs
  - `scikit-learn` – ML algorithms and evaluation
  - `graphviz` – for tree visualization (optional)

---

