import streamlit as st
import numpy as np
import joblib

model = joblib.load("satellite_launch_cost_model.pkl")

st.title("🚀 Satellite Launch Cost Prediction")
st.write("Estimate the cost of launching a satellite (in Million USD)")

payload_weight = st.number_input("Payload Weight (kg)", 100, 10000, 1000)
mission_years = st.number_input("Mission Duration (Years)", 1, 30, 10)

orbit = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
rocket = st.selectbox("Rocket Class", ["Light", "Medium", "Heavy"])
fuel = st.selectbox("Fuel Type", ["Solid", "Liquid", "Cryogenic"])

# Manual encoding
orbit_map = {"LEO": 0, "MEO": 1, "GEO": 2}
rocket_map = {"Light": 0, "Medium": 1, "Heavy": 2}
fuel_map = {"Solid": 0, "Liquid": 1, "Cryogenic": 2}

orbit_enc = orbit_map[orbit]
rocket_enc = rocket_map[rocket]
fuel_enc = fuel_map[fuel]

if st.button("Predict Launch Cost"):
    X = np.array([[payload_weight, orbit_enc, rocket_enc, fuel_enc, mission_years]])
    prediction = model.predict(X)[0]
    st.success(f"💰 Estimated Launch Cost: ${prediction:.2f} Million USD")
