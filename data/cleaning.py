import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === Step 1: Load Raw Data ===
# (Example: Replace file paths with your dataset files)
genotype_df = pd.read_csv("data/genotype.csv")  # e.g., SNPs, QTL markers
phenotype_df = pd.read_csv("data/phenotype.csv")  # e.g., yield, height, tolerance
environment_df = pd.read_csv("data/environment.csv")  # e.g., rainfall, temperature

# === Step 2: Merge datasets on common key (e.g., plant_id) ===
merged_df = pd.merge(genotype_df, phenotype_df, on="plant_id", how="inner")
merged_df = pd.merge(merged_df, environment_df, on="plant_id", how="inner")

# === Step 3: Data Cleaning ===
# Handle missing values (replace with mean for numeric, mode for categorical)
for col in merged_df.columns:
    if merged_df[col].dtype in ["int64", "float64"]:
        merged_df[col].fillna(merged_df[col].mean(), inplace=True)
    else:
        merged_df[col].fillna(merged_df[col].mode()[0], inplace=True)

# Remove duplicate rows
merged_df.drop_duplicates(inplace=True)

# Standardize column names (lowercase, underscores)
merged_df.columns = [col.strip().lower().replace(" ", "_") for col in merged_df.columns]

# === Step 4: Normalize numeric values ===
scaler = MinMaxScaler()
numeric_cols = merged_df.select_dtypes(include=["int64", "float64"]).columns
merged_df[numeric_cols] = scaler.fit_transform(merged_df[numeric_cols])

# === Step 5: Save cleaned dataset ===
merged_df.to_csv("data/dataset.csv", index=False)

print("✅ Data preprocessing complete. Clean dataset saved as 'data/dataset.csv'")
