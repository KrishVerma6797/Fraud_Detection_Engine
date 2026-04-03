import pandas as pd
import numpy as np

np.random.seed(42)
n = 200000

data = pd.DataFrame({
    "transaction_id": range(1, n + 1),
    "amount": np.random.exponential(scale=3000, size=n),
    "hour": np.random.randint(0, 24, n),
    "is_international": np.random.binomial(1, 0.15, n),
    "transaction_gap_minutes": np.random.exponential(scale=60, size=n),
    "location_risk_score": np.random.uniform(0, 1, n),
    "device_risk_score": np.random.uniform(0, 1, n),
    "merchant_risk_score": np.random.uniform(0, 1, n),
})

# -----------------------------
# SOFT FRAUD PROBABILITY (IMPORTANT)
# -----------------------------

fraud = (
    (data["amount"] > 9000) & (data["is_international"] == 1)
) | (
    (data["transaction_gap_minutes"] < 2) & (data["device_risk_score"] > 0.7)
) | (
    (data["location_risk_score"] > 0.8) & (data["merchant_risk_score"] > 0.8)
)

# Convert to int
data["fraud"] = fraud.astype(int)

# Add SMALL noise (very important)
noise_idx = data.sample(frac=0.03).index
data.loc[noise_idx, "fraud"] = 1 - data.loc[noise_idx, "fraud"]

# Save
data.to_csv("data/realistic_fraud_dataset_200k.csv", index=False)

print("Dataset regenerated with realistic noise.")
