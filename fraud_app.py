import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, time
from sklearn.preprocessing import LabelEncoder
import os


# Page config
st.set_page_config(page_title="AI Credit Card Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection System")

# Required files check
possible_model_files = ["fraud_rf_model_balanced.pkl", "fraud_rf_model.pkl", "fraud_rf_model_tuned.pkl"]
required_files = ["fraud_scaler.pkl", "fraud_categorical_cols.pkl", "fraud_feature_columns.pkl"]

# find available model file
model_file = next((f for f in possible_model_files if os.path.exists(f)), None)
missing = [f for f in required_files if not os.path.exists(f)]
if model_file is None:
    st.error(f"❌ Missing model file. Please add one of: {', '.join(possible_model_files)} to the app folder.")
    st.stop()
if missing:
    st.error(f"❌ Missing supporting files: {', '.join(missing)}. Please place them in this folder.")
    st.stop()

# Load artifacts (cached)
@st.cache_resource
def load_artifacts(mfile):
    model = joblib.load(mfile)
    scaler = joblib.load("fraud_scaler.pkl")
    cat_cols = joblib.load("fraud_categorical_cols.pkl")
    feature_cols = joblib.load("fraud_feature_columns.pkl")
    return model, scaler, cat_cols, feature_cols

rf_model, scaler, categorical_cols, feature_columns = load_artifacts(model_file)

# INPUT FORM
st.header("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    trans_date = st.date_input("Transaction Date", date(2025, 11, 6))

    # stable time input using session_state (prevents resetting on reruns)
    trans_time = st.time_input("Transaction Time", value=time(12, 0), key="trans_time")


    dob = st.date_input("Customer Date of Birth", date(1995, 1, 1))
    gender = st.selectbox("Gender", ["M", "F"])
    category = st.selectbox("Transaction Category", [
        "misc_net", "shopping_net", "shopping_pos", "gas_transport",
        "food_dining", "entertainment", "grocery_pos", "health_fitness"
    ])

with col2:
    try:
        amt = float(st.text_input("Transaction Amount", value="1200"))
        city_pop = float(st.text_input("City Population", value="50000"))
    except ValueError:
        st.error("Please enter numeric values for Transaction Amount and City Population.")
        st.stop()

    state = st.text_input("State (optional)", "MH")
    city = st.text_input("City (optional)", "Mumbai")

# geolocation (optional)
with st.expander("📍 Enter Geolocation Data (Customer & Merchant)"):
    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        lat = st.number_input("Customer Latitude", value=28.61, format="%.6f")
        long = st.number_input("Customer Longitude", value=77.23, format="%.6f")
    with col_geo2:
        merch_lat = st.number_input("Merchant Latitude", value=28.60, format="%.6f")
        merch_long = st.number_input("Merchant Longitude", value=77.21, format="%.6f")

# -------------------------
# Prediction
# -------------------------
if st.button("🔍 Predict Fraud"):
    try:
        # combine date+time
        trans_datetime = datetime.combine(trans_date, trans_time)

        # derived features
        weekday = trans_datetime.weekday()
        is_weekend = int(weekday in [5, 6])
        dob_ts = pd.to_datetime(dob)
        age = (pd.Timestamp(trans_datetime) - dob_ts).days // 365
        amt_per_pop = amt / (city_pop + 1)

        # build input dataframe
        data = {
            'Unnamed: 0': [0],
            'category': [category],
            'amt': [amt],
            'gender': [1 if gender == "M" else 0],
            'lat': [lat],
            'long': [long],
            'city_pop': [city_pop],
            'merch_lat': [merch_lat],
            'merch_long': [merch_long],
            'is_weekend': [is_weekend],
            'age': [age],
            'amt_per_pop': [amt_per_pop]
        }
        input_df = pd.DataFrame(data)

        # encode category — note: training encoder should ideally be saved; this uses local encoder for input mapping
        le = LabelEncoder()
        input_df['category'] = le.fit_transform(input_df['category'])

        # scale numeric cols (must match scaler used in training)
        num_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # ensure all model features exist (fill missing with 0)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_columns]  # reorder to training order

        # raw model probability
        pred_prob = rf_model.predict_proba(input_df)[0, 1]

        # Category-aware scaling (Option C)
        if category == "misc_net":
            scale_factor = 1.4
        elif category in ["shopping_net", "shopping_pos"]:
            scale_factor = 1.2
        elif category in ["grocery_pos", "gas_transport", "food_dining"]:
            scale_factor = 0.8
        else:
            scale_factor = 1.0

        adj_prob = min(pred_prob * scale_factor, 1.0)

        safe_cats = ["grocery_pos", "gas_transport", "food_dining"]
        moderate_cats = ["shopping_pos", "shopping_net", "entertainment"]
        risky_cats = ["misc_net"]

        if category in safe_cats:
            fraud_thresh = 0.7
            susp_thresh = 0.4
        elif category in moderate_cats:
            fraud_thresh = 0.55
            susp_thresh = 0.3
        else: 
            fraud_thresh = 0.35
            susp_thresh = 0.25

        reasons = []

        if amt >= 25000:
            reasons.append("High transaction amount")
        if city_pop <= 5000:
            reasons.append("Small city population")
        if amt_per_pop > 1.0:
            reasons.append("High amount-per-population ratio")
        if is_weekend:
            reasons.append("Weekend transaction")
        if age < 22 or age > 70:
            reasons.append("Unusual age for high-value transaction")

        if adj_prob >= fraud_thresh:
            st.error(f"🚨 Fraudulent Transaction Detected")
            reason_text = "High-risk pattern detected — immediate action recommended."
        elif adj_prob >= susp_thresh:
            st.warning(f"⚠️ Suspicious Transaction")
            reason_text = "Moderate risk — recommend manual review."
        else:
            st.success(f"✅ Legitimate Transaction")
            reason_text = "Low risk — transaction appears normal."

    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
