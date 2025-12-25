import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# LOAD DATASET
df = pd.read_csv("online_fraud_dataset_10000.csv")

X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]

numeric_features = [
    "amount",
    "transaction_hour",
    "location_risk_score",
    "failed_login_attempts",
    "velocity_transactions"
]

categorical_features = [
    "device_type",
    "merchant_category"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ), categorical_features)
    ]
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

# SAVE MODEL (THIS CREATES THE FILE)
joblib.dump(pipeline, "fraud_pipeline.joblib")

print("✅ Model created: fraud_pipeline.joblib")
