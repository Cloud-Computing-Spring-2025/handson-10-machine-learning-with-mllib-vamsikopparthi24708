# Hands-on #10: Machine Learning with MLlib – Customer Churn Prediction

## Overview

This project demonstrates a complete machine learning workflow for predicting customer churn using Apache Spark's MLlib. The implementation includes data preprocessing, feature engineering, model training and evaluation, feature selection, and hyperparameter tuning with cross-validation.

The workflow is divided into four main tasks:

Data Preprocessing and Feature Engineering

Logistic Regression Model Training and Evaluation

Feature Selection using Chi-Square Test

Hyperparameter Tuning and Model Comparison

## Requirements

Apache Spark (version 3.0 or higher)
Python (version 3.6 or higher)
PySpark
Dataset: customer_churn.csv

## Installation

Install Apache Spark following the official documentation for your system
### Install PySpark:
```bash
pip install pyspark
```


## Task-by-Task Explanation

### Task 1: Data Preprocessing and Feature Engineering

#### Code Explanation:

Handles missing values in TotalCharges by filling with 0

Identifies categorical and numerical columns

Uses StringIndexer to convert categorical columns to numeric indices

Applies OneHotEncoder to the indexed categorical columns

Combines all features using VectorAssembler

Converts the target variable "Churn" to numeric label


### Task 2: Train and Evaluate a Logistic Regression Model

#### Code Explanation:

Splits data into 80% training and 20% testing sets

Initializes and trains a Logistic Regression model

Makes predictions on test data

Evaluates model performance using AUC (Area Under ROC Curve)

### Task 3: Feature Selection using Chi-Square Test

#### Code Explanation:

Uses ChiSqSelector to select top 5 most relevant features

Displays the indices of selected features


### Task 4: Hyperparameter Tuning and Model Comparison
#### Code Explanation:

Defines four models: Logistic Regression, Decision Tree, Random Forest, and GBT

Sets up hyperparameter grids for each model

Performs 5-fold cross-validation for each model

Compares performance and identifies the best model based on AUC

### Results

[Output](model_outputs.txt)

 ```bash
Customer Churn Modeling Report
==============================

=== Logistic Regression ===
AUC: 0.7946

=== Feature Selection (Chi-Square) ===
Top 5 selected features (first 5 rows):
Row(selectedFeatures=DenseVector([0.0, 53.0, 1.0, 0.0, 0.0]), label=0.0)
Row(selectedFeatures=SparseVector(5, {1: 8.0, 3: 1.0}), label=0.0)
Row(selectedFeatures=DenseVector([1.0, 10.0, 0.0, 0.0, 1.0]), label=0.0)
Row(selectedFeatures=DenseVector([1.0, 60.0, 1.0, 0.0, 1.0]), label=1.0)
Row(selectedFeatures=DenseVector([0.0, 12.0, 1.0, 0.0, 1.0]), label=0.0)

=== Model Tuning and Comparison ===
LogisticRegression AUC: 0.7944
DecisionTree AUC: 0.6406
RandomForest AUC: 0.8204
GBTClassifier AUC: 0.7709
Best model: RandomForest with AUC = 0.8204
```


### Commands for execution

Generate input dataset

```python
python dataset_generator.py
```

Run the code
```python
python customer-churn-analysis.py
```
