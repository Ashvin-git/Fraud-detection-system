import pandas as pd
import numpy as np

np.random.seed(42)
rows = 10000

data = {
    "transaction_id": range(1, rows + 1),
    "amount": np.random.randint(100, 100000, rows),
    "transaction_hour": np.random.randint(0, 24, rows),
    "location_risk_score": np.round(np.random.uniform(0, 1, rows), 2),
    "device_type": np.random.choice(
        ["Mobile", "Desktop", "Tablet"], rows, p=[0.6, 0.3, 0.1]
    ),
    "merchant_category": np.random.choice(
        ["Electronics", "Clothing", "Groceries", "Travel", "Entertainment"],
        rows
    ),
    "failed_login_attempts": np.random.randint(0, 6, rows),
    "velocity_transactions": np.random.randint(1, 20, rows),
}

df = pd.DataFrame(data)

# FRAUD LOGIC (IMPORTANT)
df["is_fraud"] = (
    (df["amount"] > 50000) |
    (df["location_risk_score"] > 0.8) |
    (df["failed_login_attempts"] > 3) |
    ((df["velocity_transactions"] > 12) &
     (df["transaction_hour"].isin([0, 1, 2, 3, 4])))
).astype(int)

df.to_csv("online_fraud_dataset_10000.csv", index=False)

print("✅ Dataset created: online_fraud_dataset_10000.csv")
