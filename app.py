# ==========================================================
# CAR PRICE PREDICTION WEB APP
# ==========================================================

import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🚗 Car Price Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚗 Predict Price",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Performance")

st.sidebar.metric("R² Score", "0.8359")
st.sidebar.metric("MAE", "1.3026")
st.sidebar.metric("RMSE", "1.9646")

st.sidebar.markdown("---")

st.sidebar.subheader("📁 Dataset")

st.sidebar.write("Rows : **301**")
st.sidebar.write("Features : **8**")

st.sidebar.markdown("---")

st.sidebar.success("Developed by")
st.sidebar.write("**Ashi Saini**")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🚗 Car Price Prediction using Machine Learning")

    st.markdown("---")

    st.write("""
Welcome to the **Car Price Prediction Web Application**.

This application predicts the **selling price of a used car** using a Machine Learning model trained on historical car sales data.

The model considers several important factors before estimating the selling price.
""")

    st.subheader("📌 Features Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
- 🚗 Present Price
- 📅 Car Age
- 🛣️ Driven Kilometers
- ⛽ Fuel Type
""")

    with col2:
        st.markdown("""
- 🔄 Transmission
- 🏪 Seller Type
- 👤 Number of Owners
- 🤖 Linear Regression Model
""")

    st.markdown("---")

    st.info("👈 Use the navigation menu on the left to predict the selling price of a car.")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Dataset Size", "301")

    with c2:
        st.metric("Features", "8")

    with c3:
        st.metric("Model", "Linear Regression")

# ==========================================================
# PREDICT PRICE PAGE
# ==========================================================

elif page == "🚗 Predict Price":

    st.title("🚗 Predict Car Selling Price")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ==========================
    # LEFT COLUMN
    # ==========================

    with col1:

        present_price = st.number_input(
            "Present Price (Lakhs)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

        driven_kms = st.number_input(
            "Driven Kilometers",
            min_value=0,
            max_value=500000,
            value=20000,
            step=1000
        )

        owner = st.selectbox(
            "Number of Previous Owners",
            [0, 1, 2, 3]
        )

        car_age = st.slider(
            "Car Age (Years)",
            min_value=0,
            max_value=25,
            value=5
        )

    # ==========================
    # RIGHT COLUMN
    # ==========================

    with col2:

        fuel = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG"]
        )

        selling_type = st.selectbox(
            "Seller Type",
            ["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    # ==========================
    # ENCODING
    # ==========================

    fuel_diesel = 0
    fuel_petrol = 0

    if fuel == "Diesel":
        fuel_diesel = 1

    elif fuel == "Petrol":
        fuel_petrol = 1

    seller_individual = 1 if selling_type == "Individual" else 0

    transmission_manual = 1 if transmission == "Manual" else 0

    # ==========================
    # CREATE INPUT DATAFRAME
    # ==========================

    input_df = pd.DataFrame({

        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Owner": [owner],
        "Car_Age": [car_age],
        "Fuel_Type_Diesel": [fuel_diesel],
        "Fuel_Type_Petrol": [fuel_petrol],
        "Selling_type_Individual": [seller_individual],
        "Transmission_Manual": [transmission_manual]

    })

    st.markdown("---")

    st.subheader("📋 Input Summary")

    st.dataframe(input_df, use_container_width=True)

    # ==========================
    # PREDICT BUTTON
    # ==========================

    if st.button("🚀 Predict Selling Price", use_container_width=True):

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)

        st.balloons()

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        st.metric(
            "Estimated Selling Price",
            f"₹ {prediction[0]:.2f} Lakhs"
        )

        if prediction[0] >= 10:
            st.success("✅ This car has a high resale value.")

        elif prediction[0] >= 5:
            st.info("ℹ️ This car has a moderate resale value.")

        else:
            st.warning("⚠️ This car has a relatively lower resale value.")

# ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Actual vs Predicted")

        st.image(
            "images/prediction.png",
            use_container_width=True
        )

    with col2:

        st.subheader("📈 Feature Importance")

        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📊 Model Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("R² Score", "0.8359")

    with c2:
        st.metric("MAE", "1.3026")

    with c3:
        st.metric("RMSE", "1.9646")

    st.success("The Linear Regression model explains approximately 83.6% of the variation in car selling prices.")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.write("""
## 🚗 Car Price Prediction using Machine Learning

This project predicts the selling price of used cars using a Machine Learning model trained on historical car data.

The application allows users to enter various car details and instantly receive an estimated selling price.

### Machine Learning Workflow

✔ Data Collection

✔ Data Cleaning

✔ Feature Engineering

✔ One-Hot Encoding

✔ Feature Scaling

✔ Model Training

✔ Model Evaluation

✔ Streamlit Deployment

---

### Algorithm Used

• Linear Regression

---

### Python Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- Streamlit

---

### Dataset Information

- Total Records : 301
- Total Features : 8

---

### Developer

**Ashi Saini**

B.Tech CSE Student

Machine Learning Enthusiast 🚀
""")