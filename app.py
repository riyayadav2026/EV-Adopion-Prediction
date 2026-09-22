from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Voltwise | EV readiness",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


MODEL_PATH = Path(__file__).with_name("model.pkl")
FEATURES = [
    "Age",
    "Gender",
    "Annual_Income_USD",
    "City_Type",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Current_Car_Type",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Home_Charging_Possible",
    "Environmental_Concern_Level",
    "Subsidy_Available",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_input_data(values):
    return pd.DataFrame([values], columns=FEATURES)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
    :root { --ink: #13211f; --muted: #63716e; --mint: #a9f3c7; --orange: #ff7a45; --line: #d9e4df; }
    .stApp { background: #f4f7f3; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(244,247,243,.88); }
    [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 84% 7%, #d8f6df 0, transparent 24rem), #f4f7f3; }
    h1, h2, h3, p, label, button { font-family: 'Manrope', sans-serif !important; }
    h1 { letter-spacing: -1.6px; font-weight: 800 !important; }
    .eyebrow { color: #27805a; font: 500 12px 'DM Mono', monospace; letter-spacing: 1.7px; text-transform: uppercase; }
    .hero-copy { max-width: 740px; padding: 2.4rem 0 1.2rem; }
    .hero-copy p { color: var(--muted); font-size: 1.05rem; line-height: 1.65; max-width: 620px; }
    .panel-title { font-size: 1.2rem; font-weight: 800; margin-bottom: .2rem; }
    .panel-note { color: var(--muted); font-size: .88rem; margin-bottom: 1.2rem; }
    .result-label { color: var(--muted); font: 500 12px 'DM Mono', monospace; letter-spacing: 1.5px; text-transform: uppercase; }
    .result-value { font: 800 clamp(2.5rem, 6vw, 5rem) 'Manrope', sans-serif; letter-spacing: -3px; line-height: 1; margin: .45rem 0 .75rem; }
    .result-card { background: var(--ink); border-radius: 18px; color: white; padding: 1.8rem; min-height: 240px; }
    .result-card p { color: #b9cac4; }
    .footnote { color: var(--muted); font: 400 12px 'DM Mono', monospace; margin-top: 1.2rem; }
    div[data-testid="stForm"] { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 1.3rem 1.5rem 1.5rem; }
    div[data-testid="stMetric"] { background: #e6f6e9; border: 1px solid #cce9d3; border-radius: 12px; padding: .8rem 1rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button { background: var(--orange); color: white; border: 0; border-radius: 10px; font-weight: 800; min-height: 2.8rem; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { background: #e96231; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


st.markdown(
    """
    <div class="hero-copy">
      <div class="eyebrow">Voltwise / readiness engine</div>
      <h1>Is your next commute ready for electric?</h1>
      <p>Use the trained mobility model to estimate EV readiness from everyday driving habits, charging access, and household context.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

form_col, result_col = st.columns([1.35, 0.85], gap="large")

with form_col:
    with st.form("readiness_form"):
        st.markdown('<div class="panel-title">Tell us about the driver</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-note">All 12 fields match the model training schema.</div>', unsafe_allow_html=True)

        personal_left, personal_right = st.columns(2)
        with personal_left:
            age = st.number_input("Age", min_value=18, max_value=100, value=56, step=1)
            gender = st.selectbox("Gender code", [0, 1], index=1, help="Use the same encoded values used during training.")
            annual_income = st.number_input("Annual income (USD)", min_value=0.0, value=44906.0, step=1000.0)
        with personal_right:
            city_type = st.selectbox("City type code", [0, 1, 2], index=0)
            cars_owned = st.number_input("Cars owned", min_value=0, max_value=10, value=1, step=1)
            car_type = st.selectbox("Current car type code", [0, 1, 2, 3, 4], index=3)

        st.markdown("##### Commute and charging access")
        commute_left, commute_right = st.columns(2)
        with commute_left:
            commute = st.number_input("Daily commute (km)", min_value=0.0, value=18.1, step=0.5)
            home_stations = st.number_input("Stations near home", min_value=0, max_value=20, value=1, step=1)
            work_stations = st.number_input("Stations near work", min_value=0, max_value=20, value=0, step=1)
        with commute_right:
            home_charging = st.selectbox("Home charging possible", [0, 1], index=1)
            environmental = st.slider("Environmental concern level", 0.0, 5.0, 4.0, 0.5)
            subsidy = st.selectbox("Subsidy available", [0, 1], index=1)

        submitted = st.form_submit_button("Run readiness check", icon=":material/bolt:", width="stretch")

if submitted:
    values = [
        age,
        gender,
        annual_income,
        city_type,
        commute,
        cars_owned,
        car_type,
        home_stations,
        work_stations,
        home_charging,
        environmental,
        subsidy,
    ]
    input_data = build_input_data(values)
    prediction = int(model.predict(input_data)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_data)[0][1])
    st.session_state["prediction"] = prediction
    st.session_state["probability"] = probability
    st.session_state["input_data"] = input_data

with result_col:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Model output</div>', unsafe_allow_html=True)
    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]
        result_title = "EV-ready profile" if prediction == 1 else "More planning needed"
        result_text = (
            "The current profile aligns with the model's positive readiness class."
            if prediction == 1
            else "The current profile may benefit from stronger charging access or lower daily friction."
        )
        st.markdown(f'<div class="result-value">{result_title}</div>', unsafe_allow_html=True)
        st.markdown(f"<p>{result_text}</p>", unsafe_allow_html=True)
        if probability is not None:
            st.metric("Readiness confidence", f"{probability:.0%}")
    else:
        st.markdown('<div class="result-value">Awaiting input</div>', unsafe_allow_html=True)
        st.markdown("<p>Complete the profile and run a check to see the model's result here.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<div class='footnote'>Random forest classifier · 12 trained features · results are model estimates, not financial advice</div>",
    unsafe_allow_html=True,
)