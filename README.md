# Heart Disease Prediction using Machine Learning

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
* matplotlib, seaborn (for EDA in notebook)
* Streamlit
* joblib

---

## Project Structure

```
Heart-Disease-Prediction/
├── heart_disease_prediction.ipynb    # Jupyter notebook (ML workflow)
├── rf_model.pkl                      # Trained Random Forest model
├── scaler.pkl                        # Fitted StandardScaler object
├── app.py                            # Streamlit web application
├── report.pdf                        # Project report (optional)
├── requirements.txt                  # Required Python libraries
└── README.md                         # This file
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/hassaan-munir/Heart-Disease-Prediction.git
cd "Heart-Disease-Prediction"
```

### 2. Install the Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook heart_disease_Prediction.ipynb
```

### 4. Run the Web App (Optional)

```bash
streamlit run app.py
```

---

## Purpose

This project is built as part of my personal learning and portfolio to showcase my understanding of machine learning workflows, model deployment, and basic frontend integration using Streamlit.

---

## Connect with Me

**Muhammad Hassaan Munir** [LinkedIn Profile](https://www.linkedin.com/in/muhammad-hassaan-munir-79b5b2327/)

