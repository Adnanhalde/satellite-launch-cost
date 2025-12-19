import os
import streamlit as st
import numpy as np
import joblib

st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

model = joblib.load("satellite_launch_cost_model.pkl")
encoders = joblib.load("encoders.pkl")


st.title("🚀 Satellite Launch Cost Prediction")
st.write("Estimate the cost of launching a satellite (in Million USD)")

payload_weight = st.number_input("Payload Weight (kg)", 100, 10000, 1000)
mission_years = st.number_input("Mission Duration (Years)", 1, 30, 10)

orbit = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
rocket = st.selectbox("Rocket Class", ["Light", "Medium", "Heavy"])
fuel = st.selectbox("Fuel Type", ["Solid", "Liquid", "Cryogenic"])

# Encode inputs
orbit_enc = encoders["orbit_type"].transform([orbit])[0]
rocket_enc = encoders["rocket_class"].transform([rocket])[0]
fuel_enc = encoders["fuel_type"].transform([fuel])[0]

if st.button("Predict Launch Cost"):
    X = np.array([[payload_weight, orbit_enc, rocket_enc, fuel_enc, mission_years]])
    prediction = model.predict(X)[0]
    st.success(f"💰 Estimated Launch Cost: ${prediction:.2f} Million USD")
