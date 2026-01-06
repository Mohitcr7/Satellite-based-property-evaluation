import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgboost_final_model.pkl")

# Load selected features (from Sentinel-optimized run)
selected_features = pd.read_csv("selected_features.csv")["feature"].tolist()

# ---------------- LOAD TEST DATA ----------------
# IMPORTANT: this must already include Sentinel features
test_df = pd.read_csv("test_with_sentinel.csv")

# Safety check
missing = set(selected_features) - set(test_df.columns)
if missing:
    raise ValueError(f"Missing features in test data: {missing}")

# ---------------- PREPARE INPUT ----------------
X_test = test_df[selected_features]

# ---------------- PREDICT ----------------
log_preds = model.predict(X_test)
price_preds = np.expm1(log_preds)  # back to original price scale

# ---------------- SAVE SUBMISSION ----------------
submission = pd.DataFrame({
    "id": test_df["id"],
    "predicted_price": price_preds
})

submission.to_csv("submission_tabular_sentinel.csv", index=False)

print("✓ Test predictions saved: submission_tabular_sentinel.csv")