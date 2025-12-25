import os
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Online Fraud Detection",
    page_icon="🚨",
    layout="centered"
)

# ----------------------------------
# LOAD MODEL (ABSOLUTE PATH FIX)
# ----------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "fraud_pipeline.joblib")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found!")
        st.code(model_path)
        st.stop()

    return joblib.load(model_path)

model = load_model()

# ----------------------------------
# APP TITLE
# ----------------------------------
st.title("🚨 Online Fraud Detection System")
st.markdown("### Machine Learning–Based Transaction Risk Analysis")

st.divider()

# ----------------------------------
# INPUT FORM
# ----------------------------------
with st.form("fraud_form"):
    amount = st.number_input("Transaction Amount", min_value=1, step=100)
    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)
    location_risk_score = st.slider(
        "Location Risk Score", 0.0, 1.0, 0.5
    )

    device_type = st.selectbox(
        "Device Type",
        ["Mobile", "Desktop", "Tablet"]
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "Electronics",
            "Clothing",
            "Groceries",
            "Travel",
            "Entertainment"
        ]
    )

    failed_login_attempts = st.number_input(
        "Failed Login Attempts", min_value=0, max_value=10
    )

    velocity_transactions = st.number_input(
        "Velocity Transactions", min_value=1, max_value=50
    )

    submitted = st.form_submit_button("🔍 Check Fraud")

# ----------------------------------
# PREDICTION
# ----------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "amount": amount,
        "transaction_hour": transaction_hour,
        "location_risk_score": location_risk_score,
        "device_type": device_type,
        "merchant_category": merchant_category,
        "failed_login_attempts": failed_login_attempts,
        "velocity_transactions": velocity_transactions
    }])

    fraud_probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Fraud Probability",
        value=f"{fraud_probability:.2f}"
    )

    if fraud_probability > 0.7:
        st.error("🚨 FRAUD DETECTED — TRANSACTION BLOCKED")
    else:
        st.success("✅ TRANSACTION SAFE")

# ----------------------------------
# OPTIONAL: VIEW DATASET
# ----------------------------------
with st.expander("📂 View Sample Dataset"):
    try:
        df = pd.read_csv("online_fraud_dataset_10000.csv")
        st.dataframe(df.head(50))
    except:
        st.info("Dataset file not found.")
