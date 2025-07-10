# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crypto_liquidity_model.pkl")

# Streamlit App
st.set_page_config(page_title="Crypto Liquidity Predictor", layout="centered")
st.title(" Cryptocurrency Liquidity Prediction App")

# User Inputs
volume = st.number_input("Trading Volume", min_value=0.0, step=0.1)
price = st.number_input(" Current Price", min_value=0.0, step=0.1)
volatility = st.number_input(" Price Volatility", min_value=0.0, step=0.01)

# Predict Button
if st.button("Predict Liquidity"):
    features = np.array([[volume, price, volatility]])
    prediction = model.predict(features)
    st.success(f" Predicted Liquidity: {prediction[0]:.4f}")
