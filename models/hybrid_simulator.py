import numpy as np
import pandas as pd
import random


def cross_genotypes(parent1, parent2):
    """
    Simulate hybrid by averaging numeric features and randomly inheriting categorical features.
    parent1, parent2: pandas Series
    """
    hybrid = {}
    for col in parent1.index:
        val1, val2 = parent1[col], parent2[col]
        try:
            # If both are numbers, average them
            if pd.api.types.is_numeric_dtype(
                type(val1)
            ) and pd.api.types.is_numeric_dtype(type(val2)):
                hybrid[col] = (float(val1) + float(val2)) / 2
            else:
                # For non-numeric values (like Soil_Type, Country, SNP strings), randomly pick one
                hybrid[col] = random.choice([val1, val2])
        except Exception:
            # fallback in case of mixed/NaN
            hybrid[col] = val1 if pd.notna(val1) else val2

    return pd.Series(hybrid)
