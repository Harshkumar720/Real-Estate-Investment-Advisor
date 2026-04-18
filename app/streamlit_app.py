import streamlit as st
import pandas as pd
import pickle
import json
import os


# PATH SETUP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")


# LOAD DATA

data = pd.read_csv(DATA_PATH)


# LOAD MODELS

with open(os.path.join(MODEL_DIR, "classification_model.pkl"), "rb") as f:
    clf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "regression_model.pkl"), "rb") as f:
    reg_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "clf_features.json"), "r") as f:
    CLF_COLUMNS = json.load(f)

with open(os.path.join(MODEL_DIR, "reg_features.json"), "r") as f:
    REG_COLUMNS = json.load(f)


# LOAD ENCODERS

city_encoder = pickle.load(open(os.path.join(MODEL_DIR, "city_encoder.pkl"), "rb"))
type_encoder = pickle.load(open(os.path.join(MODEL_DIR, "type_encoder.pkl"), "rb"))
furnish_encoder = pickle.load(open(os.path.join(MODEL_DIR, "furnish_encoder.pkl"), "rb"))


# UI

st.set_page_config(page_title="Real Estate Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")
st.markdown("### 👨‍💻 Developed by **Harsh Kumar**")

st.subheader("📋 Enter Property Details")


# STATE & CITY

states = sorted(data["State"].dropna().unique())
state = st.selectbox("State", states)

cities = sorted(data[data["State"] == state]["City"].dropna().unique())
city = st.selectbox("City", cities)


# PROPERTY DETAILS

property_type = st.selectbox(
    "Property Type",
    sorted(data["Property_Type"].dropna().unique())
)

furnishing = st.selectbox(
    "Furnishing Status",
    sorted(data["Furnished_Status"].dropna().unique())
)


# NUMERIC INPUTS

col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=3, step=1)
    size = st.number_input("Size (SqFt)", min_value=500, max_value=10000, value=1600, step=50)
    price = st.number_input("Current Price (Lakhs)", min_value=10.0, max_value=1000.0, value=50.0, step=1.0)
    year = st.number_input("Year Built", min_value=1990, max_value=2025, value=2015, step=1)

with col2:
    floor = st.number_input("Floor No", min_value=0, max_value=50, value=2, step=1)
    total_floors = st.number_input("Total Floors", min_value=1, max_value=100, value=6, step=1)
    schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=2, step=1)
    hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=1, step=1)

budget = st.number_input("Your Budget (Lakhs)", min_value=10, max_value=500, value=72, step=1)


# BUTTON

if st.button("🔍 Analyze Property"):

    input_data = {
        "City": city_encoder.transform([city])[0],
        "Property_Type": type_encoder.transform([property_type])[0],
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Year_Built": year,
        "Furnished_Status": furnish_encoder.transform([furnishing])[0],
        "Floor_No": floor,
        "Total_Floors": total_floors,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals
    }

    df = pd.DataFrame([input_data])

    df["Property_Age"] = 2025 - df["Year_Built"]
    df["Age_of_Property"] = df["Property_Age"]

    price_per_sqft = price / size


    # DISPLAY INPUT

    st.json({
        "State": state,
        "City": city,
        "Property_Type": property_type,
        "Furnishing": furnishing,
        "BHK": bhk,
        "Size": size,
        "Price": price,
        "Year Built": year
    })

    st.write(f"🏠 Property Age: {int(df['Property_Age'][0])} years")
    st.write(f"📐 Price per SqFt: {round(price_per_sqft, 2)}")


    # MODEL INPUT

    df_clf = df.copy()
    for col in CLF_COLUMNS:
        if col not in df_clf.columns:
            df_clf[col] = 0
    df_clf = df_clf[CLF_COLUMNS]

    df_reg = df.copy()
    for col in REG_COLUMNS:
        if col not in df_reg.columns:
            df_reg[col] = 0
    df_reg = df_reg[REG_COLUMNS]


    # PREDICTION

    pred = clf_model.predict(df_clf)[0]
    prob = clf_model.predict_proba(df_clf)[0][1]
    future_price_5yr = reg_model.predict(df_reg)[0]


    # RESULTS

    st.subheader("📊 Results")

    if pred == 1:
        st.success("✅ Good Investment")
    else:
        st.error("❌ Not a Good Investment")

    st.progress(float(prob))
    st.write(f"📊 Confidence: {round(prob * 100, 2)}%")

    if prob > 0.7:
        st.success("🟢 Low Risk")
    elif prob > 0.4:
        st.warning("🟡 Moderate Risk")
    else:
        st.error("🔴 High Risk")


    # 🔥 YEAR-WISE PRICE PREDICTION

    st.subheader("📈 Price Growth Over 5 Years")

    growth_rate = 1.08
    yearly_prices = []
    current_price = price

    for i in range(1, 6):
        current_price = current_price * growth_rate
        yearly_prices.append(round(current_price, 2))

    # Show table
    price_df = pd.DataFrame({
        "Year": [1, 2, 3, 4, 5],
        "Predicted Price (Lakhs)": yearly_prices
    })

    st.table(price_df)

    # Show graph
    st.line_chart(price_df.set_index("Year"))


    # 🔥 DYNAMIC REASONING

    st.subheader("🧠 Why this prediction?")

    reasons = []

    if price_per_sqft > 0.1:
        reasons.append("Property is relatively expensive per SqFt")
    else:
        reasons.append("Property is reasonably priced per SqFt")

    if size > 2000:
        reasons.append("Large property size increases investment potential")
    elif size < 800:
        reasons.append("Smaller size may reduce investment value")

    if df["Property_Age"][0] < 10:
        reasons.append("Newer property is more attractive for buyers")
    else:
        reasons.append("Older property may affect future appreciation")

    if schools >= 3:
        reasons.append("Good nearby schools improve desirability")

    if hospitals >= 2:
        reasons.append("Access to hospitals adds convenience")

    if floor > total_floors / 2:
        reasons.append("Higher floor can increase demand")

    if not reasons:
        reasons.append("Balanced factors influenced the prediction")

    for r in reasons:
        st.write(f"• {r}")


    # INSIGHTS

    st.subheader("📊 Insights")

    st.write("• Larger size increases investment potential")
    st.write("• Location plays a major role in pricing")
    st.write("• Newer properties are generally better investments")

    st.markdown("---")
    st.markdown("© 2026 Harsh Kumar")