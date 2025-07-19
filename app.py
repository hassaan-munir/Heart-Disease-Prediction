import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Custom CSS for white theme and clean UI
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #000000;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            padding: 0.5em 1.5em;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>💓 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px; margin-bottom: 30px;'>
Enter your health information below to check your risk level. This tool uses a trained machine learning model.
</div>
""", unsafe_allow_html=True)

# Input form
with st.form("heart_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 80, 30)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.slider("Max Heart Rate Achieved", 60, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3])  # 1 = fixed, 2 = normal, 3 = reversible

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Proper encoding of categorical features (same as training time)
        input_dict = {
            'age': age,
            'sex': 1 if sex == "Male" else 0,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'ca': ca,
            'cp_1': 1 if cp == 1 else 0,
            'cp_2': 1 if cp == 2 else 0,
            'cp_3': 1 if cp == 3 else 0,
            'restecg_1': 1 if restecg == 1 else 0,
            'restecg_2': 1 if restecg == 2 else 0,
            'slope_1': 1 if slope == 1 else 0,
            'slope_2': 1 if slope == 2 else 0,
            'thal_1': 1 if thal == 1 else 0,
            'thal_2': 1 if thal == 2 else 0,
            'thal_3': 1 if thal == 3 else 0
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        # Output result
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease Detected.\nProbability: {probability:.2f}%")
        else:
            st.success(f"✅ No Sign of Heart Disease.\nProbability: {probability:.2f}%")
