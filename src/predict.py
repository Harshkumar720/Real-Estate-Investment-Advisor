import pandas as pd
import pickle
import json


# 🔹 Load models
with open("models/classification_model.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("models/regression_model.pkl", "rb") as f:
    reg_model = pickle.load(f)


# 🔹 Load feature columns separately
with open("models/clf_features.json", "r") as f:
    CLF_COLUMNS = json.load(f)

with open("models/reg_features.json", "r") as f:
    REG_COLUMNS = json.load(f)


# 🔹 Preprocess input
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Feature engineering
    df["Property_Age"] = 2025 - df["Year_Built"]

    if "Age_of_Property" not in df.columns:
        df["Age_of_Property"] = df["Property_Age"]

    return df


# 🔹 Prediction function
def predict(input_dict):
    df = preprocess_input(input_dict)

    # -------------------------
    # Classification input
    # -------------------------
    df_clf = df.copy()
    for col in CLF_COLUMNS:
        if col not in df_clf.columns:
            df_clf[col] = 0
    df_clf = df_clf[CLF_COLUMNS]

    # -------------------------
    # Regression input
    # -------------------------
    df_reg = df.copy()
    for col in REG_COLUMNS:
        if col not in df_reg.columns:
            df_reg[col] = 0
    df_reg = df_reg[REG_COLUMNS]

    # Predictions
    investment_pred = clf_model.predict(df_clf)[0]
    future_price = reg_model.predict(df_reg)[0]

    return {
        "Good_Investment": int(investment_pred),
        "Future_Price": round(float(future_price), 2)
    }


# 🔹 Test
if __name__ == "__main__":
    sample_input = {
        "City": 5,
        "Property_Type": 1,
        "BHK": 3,
        "Size_in_SqFt": 2000,
        "Year_Built": 2010,
        "Furnished_Status": 1,
        "Floor_No": 2,
        "Total_Floors": 5,
        "Nearby_Schools": 3,
        "Nearby_Hospitals": 2
    }

    print(predict(sample_input))