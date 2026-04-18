import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 🔹 STEP 1: Load cleaned data

df = pd.read_csv("data/processed/cleaned_data.csv")

print("Initial Shape:", df.shape)


# 🔹 STEP 2: Create new features

# Price per square foot
df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

# Property age
df["Property_Age"] = 2025 - df["Year_Built"]


# 🔹 STEP 3: Save updated data

df.to_csv("data/processed/cleaned_data.csv", index=False)

print("Final Shape:", df.shape)
print("✅ Feature engineering completed successfully")