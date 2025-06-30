import streamlit as st
import joblib
import numpy as np
import pandas as pd


models = {
    "K-Nearest Neighbors": joblib.load("kn_model.pkl"),
    "Logistic Regression": joblib.load("logreg_model.pkl"),
    "Random Forest": joblib.load("rf_model.pkl")
}


label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
accuracies = joblib.load("model_accuracies.pkl")


st.title(" Loan Approval Prediction App")
st.write("Fill in the details below to predict loan approval:")


married = st.selectbox("Married", label_encoders['Married'].classes_)
dependents = st.selectbox("Dependents", label_encoders['Dependents'].classes_)
education = st.selectbox("Education", label_encoders['Education'].classes_)
self_employed = st.selectbox("Self Employed", label_encoders['Self_Employed'].classes_)
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History", [0.0, 1.0])
property_area = st.selectbox("Property Area", label_encoders['Property_Area'].classes_)

model_choice = st.selectbox("Select Model", list(models.keys()))
st.write(f" Model Accuracy: **{accuracies[model_choice] * 100:.2f}%**")

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Married": label_encoders['Married'].transform([married])[0],
        "Dependents": label_encoders['Dependents'].transform([dependents])[0],
        "Education": label_encoders['Education'].transform([education])[0],
        "Self_Employed": label_encoders['Self_Employed'].transform([self_employed])[0],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": label_encoders['Property_Area'].transform([property_area])[0],
    }])

    input_scaled = scaler.transform(input_data)
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]

    pred_text = " Approved" if prediction == 1 else " Rejected"
    st.success(f"Prediction: {pred_text}")
