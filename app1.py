import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("Predict whether a patient has diabetes based on health metrics.")

# Input fields
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prepare input data
user_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Predict
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Patient **has diabetes** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Patient **does not have diabetes** (Probability: {probability:.2f})")
