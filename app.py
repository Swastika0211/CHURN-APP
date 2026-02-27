import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Churn Prediction", layout="centered")

@st.cache_resource
def load_model():
    with open("best_churn_pipeline.pkl", "rb") as f:
        obj = pickle.load(f)

    # 🔥 SAFETY CHECK
    if not hasattr(obj, "predict"):
        st.error("❌ Model file galat hai (numpy array mila, model nahi)")
        st.stop()

    return obj

model = load_model()

st.title("📊 Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, value=0)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

if st.button("Predict"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")
