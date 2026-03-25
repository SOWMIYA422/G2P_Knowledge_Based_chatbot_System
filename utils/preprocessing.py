import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def load_and_clean(path="data/dataset.csv"):
    df = pd.read_csv(path)

    # Handle SNPs (assume numeric 0/1/2 encoding, missing as NaN)
    snp_cols = [c for c in df.columns if c.startswith("SNP")]
    snp_imputer = SimpleImputer(strategy="most_frequent")
    df[snp_cols] = snp_imputer.fit_transform(df[snp_cols])

    # Handle categorical environment info (soil, country etc.)
    for col in ["SoilType", "Country", "Environment"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Handle numeric traits (yield, rainfall, temp)
    for col in ["Yield", "Rainfall", "Temperature"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    return df
