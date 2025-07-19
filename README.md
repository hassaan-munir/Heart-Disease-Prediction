# Heart Disease Prediction using Machine Learning

This is a personal portfolio project that uses a machine learning model to predict the likelihood of heart disease in individuals based on key health metrics. It demonstrates the complete workflow from data processing to model deployment.

---

## Project Overview

The goal of this project is to apply machine learning to help detect the presence of heart disease using clinical features. It involves data cleaning, encoding, scaling, model training, evaluation, and an optional user-friendly web interface using Streamlit.

---

## Key Features

* Clean and preprocess clinical dataset
* One-hot encoding and feature scaling
* Train and evaluate Logistic Regression and Random Forest classifiers
* Hyperparameter tuning with GridSearchCV
* Save trained model and scaler with joblib
* Build a Streamlit web app to take user inputs and predict results interactively

---

## Dataset Information

* **Source**: Publicly available dataset
* **Rows**: 302 (after removing duplicates)
* **Features**: 13 input columns and 1 target column
* **Target**: Binary (0 = No heart disease, 1 = Presence of heart disease)

---

## Technologies Used

* Python 3.11
* pandas, numpy
* scikit-learn
* matplotlib, seaborn
* Streamlit
* joblib

---

## Project Structure

```
Heart-Disease-Prediction/
├── heart.csv                      # Dataset
├── heart_disease_prediction.ipynb  # Jupyter notebook (ML workflow)
├── rf_model.pkl                   # Trained Random Forest model
├── scaler.pkl                     # Fitted StandardScaler object
├── app.py                         # Streamlit web application
├── report.pdf                     # Project report (optional)
```

---

## How to Run the Project

### Run the Notebook

```bash
jupyter notebook heart_disease_prediction.ipynb
```

### Run the Web App (Optional)

```bash
streamlit run app.py
```

Make sure to have the following dependencies installed:

```bash
pip install -r requirements.txt
```

---

## Purpose

This project is built as part of my personal learning and portfolio to showcase my understanding of machine learning workflows, model deployment, and basic frontend integration using Streamlit.

