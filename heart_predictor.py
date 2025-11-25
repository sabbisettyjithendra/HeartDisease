import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction App")

st.markdown("Enter your medical details:")

# Input fields
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert input
input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease.")
    else:
        st.success("✅ Low risk of heart disease.")
