import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Churn Prediction", layout="centered")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    try:
        with open("best_churn_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("Model load karne me error aa raha hai.")
        st.exception(e)
        st.stop()

model = load_model()

st.title("📊 Customer Churn Prediction")

st.write("Customer details enter karo:")

# -------- AUTO GET FEATURE NAMES -------- #
try:
    feature_names = model.feature_names_in_
except:
    st.error("Model me feature names nahi mile. Check training process.")
    st.stop()

input_data = {}

for feature in feature_names:

    if "charge" in feature.lower() or "amount" in feature.lower():
        input_data[feature] = st.number_input(feature, value=0.0)

    elif "tenure" in feature.lower() or "age" in feature.lower():
        input_data[feature] = st.number_input(feature, value=0)

    else:
        input_data[feature] = st.text_input(feature)

input_df = pd.DataFrame([input_data])

# -------- PREDICT -------- #
if st.button("Predict"):

    try:
        prediction = model.predict(input_df)

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            st.info(f"Churn Probability: {probability:.2%}")

        if prediction[0] == 1:
            st.error("⚠ Customer is likely to Churn")
        else:
            st.success("✅ Customer is Not likely to Churn")

    except Exception as e:
        st.error("Prediction ke waqt error aa gaya.")
        st.exception(e)
