import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Satellite Launch Cost Predictor",
    page_icon="🚀",
    layout="centered"
)

# -------------------------------
# Custom CSS (Cards + Animations)
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    animation: fadeIn 1s ease-in-out;
}
.metric {
    font-size: 32px;
    font-weight: bold;
    color: #00ffcc;
}
.sub {
    color: #dddddd;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(15px);}
    to {opacity: 1; transform: translateY(0);}
}
button[kind="primary"] {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("satellite_launch_cost_model.pkl")

model = load_model()

# -------------------------------
# Header Card
# -------------------------------
st.markdown("""
<div class="card">
    <h1>🚀 Satellite Launch Cost Prediction</h1>
    <p class="sub">Predict satellite launch cost using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🛰 Mission Inputs")

payload_weight = st.sidebar.slider("Payload Weight (kg)", 100, 10000, 1200)
mission_years = st.sidebar.slider("Mission Duration (Years)", 1, 30, 8)

orbit = st.sidebar.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
rocket = st.sidebar.selectbox("Rocket Class", ["Light", "Medium", "Heavy"])
fuel = st.sidebar.selectbox("Fuel Type", ["Solid", "Liquid", "Cryogenic"])

# -------------------------------
# Encoding
# -------------------------------
orbit_map = {"LEO": 0, "MEO": 1, "GEO": 2}
rocket_map = {"Light": 0, "Medium": 1, "Heavy": 2}
fuel_map = {"Solid": 0, "Liquid": 1, "Cryogenic": 2}

# -------------------------------
# Prediction Card
# -------------------------------
if st.button("🚀 Predict Launch Cost", type="primary"):
    X = np.array([[
        payload_weight,
        orbit_map[orbit],
        rocket_map[rocket],
        fuel_map[fuel],
        mission_years
    ]])

    prediction_usd = model.predict(X)[0]
    inr_crore = (prediction_usd * 83) / 10

    st.markdown(f"""
    <div class="card">
        <h3>💰 Estimated Launch Cost</h3>
        <div class="metric">₹ {inr_crore:.2f} Crores</div>
        <p class="sub">(${prediction_usd:.2f} Million USD)</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Footer Card
# -------------------------------
st.markdown("""
<div class="card">
    <p class="sub">📊 Machine Learning Project | Streamlit UI | Final Year</p>
</div>
""", unsafe_allow_html=True)
