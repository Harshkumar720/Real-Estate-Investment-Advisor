import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


# 🔹 Load data

df = pd.read_csv("data/processed/cleaned_data.csv")


# 🔹 Clean data (same as training)

df = df.drop(columns=[
    "ID",
    "State",
    "Locality",
    "Owner_Type",
    "Availability_Status",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing"
], errors='ignore')


# 🔹 Load models

clf_model = pickle.load(open("models/classification_model.pkl", "rb"))
reg_model = pickle.load(open("models/regression_model.pkl", "rb"))


# 🔹 Classification Evaluation

median = df["Price_per_SqFt"].median()
df["Good_Investment"] = (df["Price_per_SqFt"] < median).astype(int)

X_clf = df.drop([
    "Good_Investment",
    "Price_per_SqFt",
    "Price_in_Lakhs"
], axis=1)

y_clf = df["Good_Investment"]

y_pred_clf = clf_model.predict(X_clf)

print("\n🔹 Classification Evaluation")
print("Accuracy:", accuracy_score(y_clf, y_pred_clf))
print(classification_report(y_clf, y_pred_clf))


# 🔹 Regression Evaluation

df["Future_Price"] = df["Price_in_Lakhs"] * (1.08 ** 5)

X_reg = df.drop([
    "Future_Price",
    "Good_Investment"   # 🔥 remove this
], axis=1)
y_reg = df["Future_Price"]

y_pred_reg = reg_model.predict(X_reg)

rmse = np.sqrt(mean_squared_error(y_reg, y_pred_reg))
r2 = r2_score(y_reg, y_pred_reg)

print("\n🔹 Regression Evaluation")
print("RMSE:", rmse)
print("R2 Score:", r2)