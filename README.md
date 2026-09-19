# Customer Churn Prediction

A machine learning project predicting telecom customer churn using classical ML models (Logistic Regression, Random Forest), including a Flask API for serving predictions.

## Project Overview

Built end-to-end: data cleaning, EDA, feature engineering, model training/comparison, hyperparameter tuning, and deployment as a REST API.

## Key Findings

- Tenure and referral count are the strongest churn predictors
- Caught and fixed a data leakage bug (Churn Category/Reason columns) that was inflating model accuracy to a false 100%
- Tuned Random Forest (class-weighted) achieves 85% recall on churned customers, prioritized over raw accuracy since missing a real churner is costlier than a false alarm

## Tech Stack

Python, pandas, scikit-learn, Flask, joblib

## Files

- `churn_analysis.ipynb` — full analysis and model training pipeline
- `app.py` — Flask API serving the trained model

## Running the API
