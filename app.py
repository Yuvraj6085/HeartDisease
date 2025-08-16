import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("❤️ Heart Disease Prediction App")
st.write("Predict the probability of **TenYearCHD** based on patient details.")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=45)
sex = st.selectbox("Sex", ["Female", "Male"])
cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0)
totChol = st.number_input("Total Cholesterol", min_value=100, max_value=700, value=200)
sysBP = st.number_input("Systolic Blood Pressure", min_value=90, max_value=250, value=120)
glucose = st.number_input("Glucose Level", min_value=40, max_value=300, value=80)

# Convert sex to numeric (as in dataset: male=1, female=0)
sex_val = 1 if sex == "Male" else 0

# Prepare data for model
features = np.array([[age, sex_val, cigsPerDay, totChol, sysBP, glucose]])
features_scaled = scaler.transform(features)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2f})")
