import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Churn Prediction", layout="centered")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    model = joblib.load("best_churn_pipeline.pkl")
    return model

model = load_model()

st.title("📊 Customer Churn Prediction App")

st.write("Fill customer details below:")

# ----------- AUTO GET FEATURE NAMES ----------- #
try:
    feature_names = model.feature_names_in_
except:
    st.error("Model does not contain feature names. Retrain with sklearn >=1.0")
    st.stop()

input_data = {}

# ----------- CREATE INPUT FIELDS DYNAMICALLY ----------- #
for feature in feature_names:
    
    if "charge" in feature.lower() or "amount" in feature.lower():
        input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    elif "tenure" in feature.lower() or "age" in feature.lower():
        input_data[feature] = st.number_input(f"{feature}", value=0)
    
    else:
        input_data[feature] = st.text_input(f"{feature}")

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ----------- PREDICTION ----------- #
if st.button("Predict"):

    try:
        prediction = model.predict(input_df)
        proba = None
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error("⚠ Customer is likely to Churn")
        else:
            st.success("✅ Customer is Not likely to Churn")

        if proba is not None:
            st.info(f"Churn Probability: {proba:.2%}")

    except Exception as e:
        st.error("Error during prediction:")
        st.exception(e)
