
# 🏠 Boston House Price Prediction – ShadowFox AIML Internship

## 📌 Project Overview

This project is part of the *Beginner Level Task* for the *ShadowFox AIML Internship*.
The goal is to build a *regression model* to predict *Boston house prices* using features such as number of rooms, crime rate, property tax, and more.

By applying data preprocessing, exploratory data analysis, model building, and evaluation, we aim to develop a robust solution for real estate price prediction.

---

## 🎯 Objectives

* Preprocess and clean the Boston Housing dataset.
* Handle missing values and outliers.
* Train different regression models (Linear, Decision Tree, Random Forest, Gradient Boosting).
* Evaluate models using metrics like RMSE and R² Score.
* Visualize predictions vs. actual values.

---


## ⚙ Tech Stack

* *Language*: Python 🐍
* *Libraries*:

  * numpy, pandas (data preprocessing)
  * scikit-learn (ML models, evaluation)
  * matplotlib, seaborn, plotly (visualizations)

---

## 🔑 Steps Followed

### 1. Data Preprocessing

* Loaded dataset (Boston Housing).
* Checked for missing values and outliers (IQR method).
* Scaled features using *StandardScaler*.
* Split data into *80% training* and *20% testing*.

### 2. Model Training

* Tried multiple regression models:

  * Linear Regression
  * Ridge & Lasso Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Gradient Boosting Regressor

### 3. Model Evaluation

* Metrics used:

  * *RMSE (Root Mean Squared Error)*
  * *MAE (Mean Absolute Error)*
  * *R² Score*

---

## 📊 Visualizations

* *Predicted vs Actual Prices*
* *Residual Plot*
* *Feature Importance (for tree models)*

📌 Example :
RMSE: 4.93, R²: 0.67

---

## 🚀 How to Run

1. Clone the repo:

   bash
   git clone https://github.com/<your-username>/ShadowFox.git
   cd ShadowFox/Beginner_BostonHousePrice
   
2. Install dependencies:

   bash
   pip install -r requirements.txt
   
3. Run the script:

   bash
   python boston.py
   

---

## 👨‍💻 Author
Parth Dasharathbhai Prajapati 
ShadowFox AIML Intern

---
