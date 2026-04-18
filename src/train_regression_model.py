import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np
import json


# 🔹 STEP 1: Load data

df = pd.read_csv("data/processed/cleaned_data.csv")

print("Initial Shape:", df.shape)


# 🔹 STEP 2: Drop unnecessary columns

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

print("After Cleaning Shape:", df.shape)


# 🔹 STEP 3: Load SAME encoders (IMPORTANT)

city_encoder = pickle.load(open("models/city_encoder.pkl", "rb"))
type_encoder = pickle.load(open("models/type_encoder.pkl", "rb"))
furnish_encoder = pickle.load(open("models/furnish_encoder.pkl", "rb"))


# 🔹 STEP 4: Apply encoding

df["City"] = city_encoder.transform(df["City"])
df["Property_Type"] = type_encoder.transform(df["Property_Type"])
df["Furnished_Status"] = furnish_encoder.transform(df["Furnished_Status"])


# 🔹 STEP 5: Create target (Future Price)

df["Future_Price"] = df["Price_in_Lakhs"] * (1.08 ** 5)


# 🔹 STEP 6: Define features & target

X = df.drop("Future_Price", axis=1)
y = df["Future_Price"]

print("Feature Shape:", X.shape)


# 🔹 STEP 7: Train-test split (80-20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 🔹 STEP 8: Train model

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=50,
    n_jobs=-1
)

model.fit(X_train, y_train)


# 🔹 STEP 9: Predictions

y_pred = model.predict(X_test)


# 🔹 STEP 10: Evaluation

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRMSE:", rmse)
print("R2 Score:", r2)


# 🔹 STEP 11: Save model

with open("models/regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Regression model saved successfully")


# 🔹 STEP 12: Save feature columns

with open("models/reg_features.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Regression feature columns saved")