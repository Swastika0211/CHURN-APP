
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline
try:
    pipeline = joblib.load('best_churn_pipeline.pkl')
except FileNotFoundError:
    st.error("Error: 'best_churn_pipeline.pkl' not found. Make sure the file is in the same directory as this script.")
    st.stop()

# Define the 75th percentile of balance from the training data (obtained from df.describe() output)
# This value is needed for the 'high_balance' feature engineering step
BALANCE_75TH_PERCENTILE = 127644.24 # Update if your training data's 75th percentile is different

# Feature engineering function (must match the one used during training)
def apply_feature_engineering(df_input):
    df_fe = df_input.copy()

    # Balance per product
    df_fe['balance_per_product'] = df_fe['balance'] / (df_fe['products_number'].replace(0, np.nan))
    df_fe['balance_per_product'].fillna(0, inplace=True)

    # Salary to balance ratio
    df_fe['salary_balance_ratio'] = df_fe['estimated_salary'] / (df_fe['balance'].replace(0, np.nan))
    df_fe['salary_balance_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # If balance is 0, salary_balance_ratio becomes inf, replacing with median from original training data
    # For a robust deployment, you'd typically precompute this median and store it.
    # For now, we'll use a placeholder or median from a representative dataset if 'balance' is 0.
    # A safer approach for deployment would be to save this median value along with the pipeline.
    # As an example, I'll use the median from the sample_df during notebook execution
    df_fe['salary_balance_ratio'].fillna(0.839258, inplace=True) # Using median from initial run in notebook for salary_balance_ratio

    # Age group
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=bins, labels=labels, right=True)

    # Tenure bucket
    df_fe['tenure_bucket'] = pd.cut(df_fe['tenure'], bins=[-1, 0, 2, 5, 10, 100], labels=['0', '1-2', '3-5', '6-10', '10+'], right=True)

    # Flag high balance - use the pre-defined 75th percentile from training data
    df_fe['high_balance'] = (df_fe['balance'] > BALANCE_75TH_PERCENTILE).astype(int)

    return df_fe


# Streamlit App Title
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn likelihood.')

# Input widgets
credit_score = st.slider('Credit Score', 350, 850, 650)
country = st.selectbox('Country', ['France', 'Germany', 'Spain'])
gender = st.radio('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92, 40)
tenure = st.slider('Tenure (years)', 0, 10, 3)
balance = st.number_input('Balance', 0.0, 250000.0, 50000.0, step=1000.0)
products_number = st.slider('Number of Products', 1, 4, 2)
credit_card = st.radio('Has Credit Card?', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')
active_member = st.radio('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x==1 else 'No')
estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 60000.0, step=1000.0)

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'credit_score': credit_score,
    'country': country,
    'gender': gender,
    'age': age,
    'tenure': tenure,
    'balance': balance,
    'products_number': products_number,
    'credit_card': credit_card,
    'active_member': active_member,
    'estimated_salary': estimated_salary
}])

# Drop 'customer_id' if present, as it's not used in prediction
# For this app, customer_id is not collected, so this line is technically not needed but good for robustness
# if 'customer_id' in input_data.columns:
#     input_data = input_data.drop(columns=['customer_id'])

# Button to predict
if st.button('Predict Churn'):
    # Apply feature engineering
    processed_input = apply_feature_engineering(input_data)

    # Make prediction
    prediction = pipeline.predict(processed_input)[0]
    probability = pipeline.predict_proba(processed_input)[0, 1]

    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error(f"The customer is predicted to CHURN with a probability of {probability:.2f}.")
    else:
        st.success(f"The customer is predicted NOT to churn with a probability of {probability:.2f}.")
