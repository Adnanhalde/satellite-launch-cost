import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ---- Load trained model ----
model = joblib.load("satellite_launch_cost_model.pkl")

# ---- Page configuration ----
st.set_page_config(
    page_title="Satellite Launch Cost Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Header Banner ----
image = Image.open("/mnt/data/22fe56df-6027-4e5b-aad7-0af1a340960f.png")  # Replace with your uploaded banner
st.image(image, use_column_width=True)

st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>🚀 Satellite Launch Cost Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Estimate the cost of launching a satellite (in Million USD)</p>", unsafe_allow_html=True)
st.write("---")

# ---- Sidebar Inputs ----
st.sidebar.header("Satellite Parameters")

payload_weight = st.sidebar.number_input("Payload Weight (kg)", 100, 10000, 1000)
mission_years = st.sidebar.number_input("Mission Duration (Years)", 1, 30, 10)

orbit = st.sidebar.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
rocket = st.sidebar.selectbox("Rocket Class", ["Light", "Medium", "Heavy"])
fuel = st.sidebar.selectbox("Fuel Type", ["Solid", "Liquid", "Cryogenic"])

# ---- Manual Encoding ----
orbit_map = {"LEO": 0, "MEO": 1, "GEO": 2}
rocket_map = {"Light": 0, "Medium": 1, "Heavy": 2}
fuel_map = {"Solid": 0, "Liquid": 1, "Cryogenic": 2}

orbit_enc = orbit_map[orbit]
rocket_enc = rocket_map[rocket]
fuel_enc = fuel_map[fuel]

# ---- Prediction ----
if st.sidebar.button("Predict Launch Cost"):
    X = np.array([[payload_weight, orbit_enc, rocket_enc, fuel_enc, mission_years]])
    prediction = model.predict(X)[0]
    
    # Display prediction in a card-like format
    st.markdown(f"""
    <div style='background-color:#1F1F1F; padding:20px; border-radius:10px; text-align:center;'>
        <h2 style='color:#00FF99;'>💰 Estimated Launch Cost</h2>
        <h1 style='color:#FF4B4B;'>${prediction:.2f} Million USD</h1>
    </div>
    """, unsafe_allow_html=True)

# ---- About / Info ----
with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts the estimated cost of launching a satellite based on:
    - Payload weight (kg)
    - Mission duration (years)
    - Orbit type (LEO, MEO, GEO)
    - Rocket class (Light, Medium, Heavy)
    - Fuel type (Solid, Liquid, Cryogenic)

    The prediction is powered by a trained Linear Regression model.
    """)
