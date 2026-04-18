import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
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


# 🔹 STEP 3: Encode categorical columns (IMPORTANT FIX)

city_encoder = LabelEncoder()
type_encoder = LabelEncoder()
furnish_encoder = LabelEncoder()

df["City"] = city_encoder.fit_transform(df["City"])
df["Property_Type"] = type_encoder.fit_transform(df["Property_Type"])
df["Furnished_Status"] = furnish_encoder.fit_transform(df["Furnished_Status"])


# 🔹 Save encoders (VERY IMPORTANT for Streamlit)

pickle.dump(city_encoder, open("models/city_encoder.pkl", "wb"))
pickle.dump(type_encoder, open("models/type_encoder.pkl", "wb"))
pickle.dump(furnish_encoder, open("models/furnish_encoder.pkl", "wb"))

print("✅ Encoders saved")


# 🔹 STEP 4: Create target

median = df["Price_per_SqFt"].median()

df["Good_Investment"] = (df["Price_per_SqFt"] < median).astype(int)


# 🔹 STEP 5: Define features & target

X = df.drop([
    "Good_Investment",
    "Price_per_SqFt",   # ❌ remove leakage
    "Price_in_Lakhs"    # optional but improves realism
], axis=1)

y = df["Good_Investment"]

print("Feature Shape:", X.shape)


# 🔹 STEP 6: Train-test split (80-20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 🔹 STEP 7: Train model

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=50,
    n_jobs=-1
)

model.fit(X_train, y_train)


# 🔹 STEP 8: Predictions

y_pred = model.predict(X_test)


# 🔹 STEP 9: Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 🔹 STEP 10: Save model

with open("models/classification_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Classification model saved successfully")


# 🔹 STEP 11: Save feature column order

with open("models/clf_features.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Classification feature columns saved")