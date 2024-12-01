import streamlit as st
import pandas as pd
import pickle

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

feature_names = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]


def preprocess_input(input_data):
    # Apply encoding to categorical features
    for col, encoder in encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]
    return pd.DataFrame([input_data], columns=feature_names)


st.title("Customer Churn Prediction")

# Form for user inputs
with st.form("user_input_form"):
    st.write("### Enter Customer Details")
    user_input = {}

    for feature in feature_names:
        if feature in encoders:  # Categorical feature
            options = encoders[feature].classes_.tolist()
            user_input[feature] = st.selectbox(f"{feature}:", options)
        elif feature == "SeniorCitizen":  # Binary categorical feature (0 or 1)
            user_input[feature] = st.radio(f"{feature}:", [0, 1])
        else:  # Numerical feature
            user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = preprocess_input(user_input)
    prediction = model.predict(input_df)[0]
    result = "Churn" if prediction == 1 else "No Churn"
    st.write(f"### Prediction: {result}")
