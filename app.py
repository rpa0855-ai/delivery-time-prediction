import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(page_title="Delivery Time Predictor", page_icon="🛵")

st.title("🛵 Delivery Time Predictor")
st.write("Predicts delivery time (minutes) and explains why, using the model trained in the project notebook.")

# ---- load model + expected feature columns ----
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, explainer

try:
    model, feature_columns, explainer = load_model()
except FileNotFoundError:
    st.error("Model files not found. Put best_model.pkl and feature_columns.pkl "
             "in the same folder as this app (see README for how to generate them).")
    st.stop()

# ---- input form ----
st.header("Order details")

col1, col2 = st.columns(2)
with col1:
    distance_km = st.slider("Distance (km)", 1.0, 25.0, 8.0, 0.5)
    age = st.slider("Delivery person age", 18, 50, 30)
    ratings = st.slider("Delivery person rating", 1.0, 5.0, 4.5, 0.1)
    vehicle_condition = st.selectbox("Vehicle condition (0=worst, 2=best)", [0, 1, 2], index=1)
    multiple_deliveries = st.selectbox("Multiple deliveries on this trip", [0, 1, 2, 3], index=0)

with col2:
    traffic = st.selectbox("Traffic density", ["Low", "Medium", "High", "Jam"])
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"])
    festival = st.selectbox("Festival day?", ["No", "Yes"])
    city = st.selectbox("City type", ["Metropolitian", "Urban", "Semi-Urban"])

# ---- build the one-hot feature row exactly matching training columns ----
def build_feature_row():
    row = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=float)

    row["Distance_km"] = distance_km
    row["Delivery_person_Age"] = age
    row["Delivery_person_Ratings"] = ratings
    row["Vehicle_condition"] = vehicle_condition
    row["multiple_deliveries"] = multiple_deliveries

    # one-hot columns were built with drop_first=True in training, so the
    # first alphabetical category of each has no column -- leaving all zeros
    # for that category is correct, not a bug
    traffic_col = f"Traffic_{traffic}"
    if traffic_col in row.columns:
        row[traffic_col] = 1

    weather_col = f"Weather_{weather}"
    if weather_col in row.columns:
        row[weather_col] = 1

    if festival == "Yes" and "Festival_Yes" in row.columns:
        row["Festival_Yes"] = 1

    city_col = f"City_{city}"
    if city_col in row.columns:
        row[city_col] = 1

    return row

# ---- predict + explain ----
if st.button("Predict delivery time", type="primary"):
    row = build_feature_row()
    pred = model.predict(row)[0]

    st.header(f"⏱️ Predicted delivery time: {pred:.0f} minutes")

    row_shap = explainer.shap_values(row)[0]
    contributions = pd.Series(row_shap, index=row.columns).sort_values(key=abs, ascending=False)
    top_features = contributions.head(4)

    st.subheader("Why this estimate?")
    for feat, val in top_features.items():
        direction = "increased" if val > 0 else "decreased"
        arrow = "🔺" if val > 0 else "🔻"
        st.write(f"{arrow} **{feat}** {direction} the estimate by **{abs(val):.1f} min**")

st.caption("Model and explanation logic come directly from the project notebook "
           "(Part 2: ML/DL model comparison + SHAP-based Delay Explanation Agent).")
