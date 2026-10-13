"""
Florida House Price Prediction – Streamlit frontend

"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = "artifacts"
RIDGE_PATH = "hybrid_ridge_model.joblib" if os.path.exists("hybrid_ridge_model.joblib") else f"{ART}/stacking.joblib"

st.set_page_config(page_title="Florida House Price Predictor", page_icon="🏠", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    from tensorflow.keras.models import load_model

    return {
        "poly": joblib.load(f"{ART}/poly.joblib"),
        "scaler": joblib.load(f"{ART}/scaler.joblib"),
        "lr": joblib.load(f"{ART}/lr.joblib"),
        "dt": joblib.load(f"{ART}/dt.joblib"),
        "stacking": joblib.load(RIDGE_PATH),
        "ann": load_model(f"{ART}/ann.keras"),
    }


@st.cache_data
def load_meta():
    with open(f"{ART}/meta.json") as f:
        return json.load(f)


@st.cache_data
def load_city_agg():
    return pd.read_csv(f"{ART}/city_aggregation.csv")


@st.cache_data
def load_test_preds():
    return pd.read_csv(f"{ART}/test_predictions.csv")


if not os.path.exists(f"{ART}/meta.json"):
    st.error("Model artifacts not found. Run `python train_models.py` first, then restart the app.")
    st.stop()

models = load_models()
meta = load_meta()
city_agg = load_city_agg()
FEATURES = meta["feature_columns"]
CITIES = sorted(city_agg["City"].tolist())
DEFAULTS = meta["defaults"]


# ---------------------------------------------------------------- prediction
def predict(inputs: dict, city: str) -> float:
    row = {c: 0.0 for c in FEATURES}
    row.update(DEFAULTS)          # fill anything the UI doesn't ask for
    row.update(inputs)
    city_col = f"City_{city}"
    if city_col in row:           # the dropped (first) city is the all-zero baseline
        row[city_col] = 1.0

    x = pd.DataFrame([row])[FEATURES]
    x_poly = models["poly"].transform(x)

    lr_p = float(models["lr"].predict(x_poly)[0])
    dt_p = float(models["dt"].predict(x_poly)[0])
    ann_p = float(models["ann"].predict(models["scaler"].transform(x_poly), verbose=0).flatten()[0])
    # Base-model outputs are only the input features of the Ridge model
    return float(models["stacking"].predict(np.array([[lr_p, dt_p, ann_p]]))[0])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🏠 Florida Housing")
page = st.sidebar.radio("Navigate", ["Price Predictor", "City Insights", "About"])


# ================================================================ PAGE 1
if page == "Price Predictor":
    st.title("City Wise House Price Prediction")
    st.caption("This AI based WebApp provides an Aggrigated Price for the type of House you are looking for in any city of the Florida State.")
    st.caption("Enter the property details and get an estimated price from the proposed hybrid LR-DT-ANN model.")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", CITIES, index=0)
        sqft = st.number_input("Living area (Sq Ft)", 300, 20000, 1800, step=50)
        lot = st.number_input("Lot size (Sq Ft)", 0, 500000, 8000, step=250)
        built = st.number_input("Year built", 1900, 2026, 2005, step=1)
    with col2:
        beds = st.number_input("Bedrooms", 0, 12, 3)
        baths = st.number_input("Bathrooms", 0.0, 12.0, 2.0, step=0.5)
        garages = st.number_input("Garages", 0, 6, 2)

        city_ppsf = city_agg.loc[city_agg["City"] == city, "Price per Sq Ft"]
        city_ppsf = float(city_ppsf.iloc[0]) if len(city_ppsf) else float(DEFAULTS.get("Price per Sq Ft", 200))
        use_city_avg = st.checkbox(f"Use {city} average price per Sq Ft (${city_ppsf:,.0f})", value=True)
        ppsf = city_ppsf if use_city_avg else st.number_input(
            "Price per Sq Ft ($)", 10.0, 5000.0, city_ppsf, step=10.0
        )

    if st.button("Predict price", type="primary", use_container_width=True):
        inputs = {
            "Sq Ft": sqft, "Price per Sq Ft": ppsf, "Lot Sq Ft": lot,
            "Beds": beds, "Baths": baths, "Garages": garages, "Built": built,
        }
        inputs = {k: v for k, v in inputs.items() if k in FEATURES}
        price = predict(inputs, city)

        st.divider()
        st.metric("Estimated price (Hybrid Ridge model)", f"${price:,.0f}")

        avg_city_price = city_agg.loc[city_agg["City"] == city, "Price"]
        if len(avg_city_price):
            st.info(f"Average listed price in {city}: **${float(avg_city_price.iloc[0]):,.0f}**")


# ================================================================ PAGE 2
elif page == "City Insights":
    st.title("City-wise Insights")

    metric_labels = {
        "Price": "Average Price",
        "Price per Sq Ft": "Average Price per Sq Ft",
        "Sq Ft": "Average Square Footage",
        "Lot Sq Ft": "Average Lot Square Footage",
        "Beds": "Average Beds",
        "Baths": "Average Baths",
        "Garages": "Average Garages",
    }
    c1, c2, c3 = st.columns([2, 1, 1])
    metric = c1.selectbox("Feature", list(metric_labels), format_func=metric_labels.get)
    order = c2.radio("Show", ["Highest", "Lowest"], horizontal=True)
    n = c3.slider("Number of cities", 5, 25, 10)

    top = city_agg.sort_values(metric, ascending=(order == "Lowest")).head(n)
    st.subheader(f"{n} cities with the {order.lower()} {metric_labels[metric].lower()}")
    st.bar_chart(top.set_index("City")[metric])

    with st.expander("Full city-wise aggregation table"):
        st.dataframe(city_agg, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", city_agg.to_csv(index=False),
                           "city_wise_aggregation.csv", "text/csv")

# ================================================================ PAGE 4
else:
    st.title("About This Project")
    st.markdown(
        """
**Note**

1. The dataset used for this project was compiled in 2024.
2. Missing values are imputed.
3. `City` is one-hot encoded.
4. Three base models are trained: **Linear Regression**, **Decision Tree**, **ANN**.
5. Their predictions are combined by a **Ridge** meta-model to give the hybrid prediction.

**Dataset:** Florida housing listings (city, area, lot size, beds, baths, garages, year built, price).
        """
    )
