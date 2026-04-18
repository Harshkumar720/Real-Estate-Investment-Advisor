import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/india_housing_prices.csv")

print("Original Shape:", df.shape)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Clean column names
df.columns = df.columns.str.strip()

print("After Cleaning:", df.shape)

# Save cleaned data
df.to_csv("data/processed/cleaned_data.csv", index=False)

print("✅ Cleaned data saved to data/processed/")