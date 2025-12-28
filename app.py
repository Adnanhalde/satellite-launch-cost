import streamlit as st
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="Satellite Launch Cost Predictor",
    page_icon="🚀",
    layout="centered"
)

st.markdown("""
<style>

/* App background */
.stApp {
    background: radial-gradient(circle at top, #0f2027, #000000);
    color: white;
}

/* Title animation */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    animation: fadeDown 1s ease;
}

/* Subtitle */
.subtitle {
    text-align: center;
    opacity: 0.8;
    margin-bottom: 25px;
}

/* Main card */
.main-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    border-radius: 24px;
    padding: 32px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.45);
    animation: slideUp 0.9s ease;
}

/* Input hover */
.stSlider > div:hover,
.stSelectbox > div:hover {
    transform: scale(1.01);
    transition: 0.2s ease-in-out;
}

/* Button */
button[kind="primary"] {
    width: 100%;
    border-radius: 16px;
    font-size: 20px;
    padding: 0.75em;
    margin-top: 15px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    transition: all 0.3s ease;
}

button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0,198,255,0.6);
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: #002b1b;
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    margin-top: 30px;
    animation: popIn 0.7s ease;
}

/* Glowing amount */
.result-amount {
    font-size: 44px;
    font-weight: 900;
    animation: glow 1.5s infinite alternate;
}

.result-sub {
    font-size: 16px;
    opacity: 0.9;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 30px;
    opacity: 0.6;
    font-size: 14px;
}

/* Animations */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-25px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes glow {
    from { text-shadow: 0 0 10px rgba(255,255,255,0.4); }
    to { text-shadow: 0 0 25px rgba(255,255,255,0.9); }
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("satellite_launch_cost_model.pkl")

model = load_model()

st.markdown('<div class="title">🚀 Satellite Launch Cost Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered estimation of satellite launch cost</div>', unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)

payload_weight = st.slider("Payload Weight (kg)", 100, 10000, 1000)
mission_years = st.slider("Mission Duration (Years)", 1, 30, 10)

orbit = st.selectbox("Orbit Type", ["LEO", "MEO", "GEO"])
rocket = st.selectbox("Rocket Class", ["Light", "Medium", "Heavy"])
fuel = st.selectbox("Fuel Type", ["Solid", "Liquid", "Cryogenic"])

orbit_map = {"LEO": 0, "MEO": 1, "GEO": 2}
rocket_map = {"Light": 0, "Medium": 1, "Heavy": 2}
fuel_map = {"Solid": 0, "Liquid": 1, "Cryogenic": 2}

# Predict Button
predict = st.button("🚀 Predict Launch Cost", type="primary")

st.markdown('</div>', unsafe_allow_html=True)

if predict:
    with st.spinner("Calculating launch cost..."):
        time.sleep(1)

    X = np.array([[
        payload_weight,
        orbit_map[orbit],
        rocket_map[rocket],
        fuel_map[fuel],
        mission_years
    ]])

    prediction_usd = model.predict(X)[0]
    prediction_inr = (prediction_usd * 83) / 10

    st.markdown(f"""
    <div class="result-card">
        <h2>💰 Estimated Launch Cost</h2>
        <div class="result-amount">₹ {prediction_inr:.2f} Crores</div>
        <div class="result-sub">(${prediction_usd:.2f} Million USD)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">📊 Machine Learning | Streamlit | Final Year Project</div>', unsafe_allow_html=True)
