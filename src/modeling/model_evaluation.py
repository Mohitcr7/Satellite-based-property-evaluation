"""
ML Model Evaluation Pipeline
Compares multiple regression algorithms using cross-validation
Now includes XGBoost!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv("train_cleaned.csv")

# --- Target / leakage-safe features ---
y = df["price_log"]  # log-transformed target
X = df.drop(columns=["id", "price", "price_log"])

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: price_log (log-transformed prices)\n")

# --- CV setup ---
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# --- Models dictionary ---
models = {
    "Ridge": Ridge(
        alpha=10.0, 
        random_state=42
    ),
    
    "ElasticNet": ElasticNet(
        alpha=0.01, 
        l1_ratio=0.2, 
        random_state=42, 
        max_iter=20000
    ),
    
    "RandomForest": RandomForestRegressor(
        n_estimators=600, 
        random_state=42, 
        n_jobs=-1,
        max_depth=None, 
        min_samples_leaf=2
    ),
    
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=800, 
        random_state=42, 
        n_jobs=-1,
        max_depth=None, 
        min_samples_leaf=2
    ),
    
    "HistGB": HistGradientBoostingRegressor(
        learning_rate=0.05, 
        max_depth=None, 
        max_iter=600,
        min_samples_leaf=20, 
        random_state=42
    ),
    
    # NEW: XGBoost
    "XGBoost": XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Fast histogram-based algorithm
    )
}

# --- Scoring metrics ---
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2"
}

# --- Cross-validation evaluation ---
print("="*70)
print("EVALUATING MODELS (5-Fold Cross-Validation)")
print("="*70)

rows = []
for name, model in models.items():
    print(f"\nEvaluating {name}...", end=" ")
    
    res = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1, 
        return_train_score=False
    )
    
    rows.append({
        "Model": name,
        "RMSE_log_mean": -res["test_rmse"].mean(),
        "RMSE_log_std": res["test_rmse"].std(),
        "MAE_log_mean": -res["test_mae"].mean(),
        "MAE_log_std": res["test_mae"].std(),
        "R2_mean": res["test_r2"].mean(),
        "R2_std": res["test_r2"].std()
    })
    
    print("âœ“")

# --- Results table ---
report = pd.DataFrame(rows).sort_values("RMSE_log_mean")

print("\n" + "="*70)
print("RESULTS (sorted by RMSE - lower is better)")
print("="*70)
print(report.to_string(index=False))

# --- Winner announcement ---
best_model = report.iloc[0]["Model"]
best_rmse = report.iloc[0]["RMSE_log_mean"]
best_r2 = report.iloc[0]["R2_mean"]

print("\n" + "="*70)
print(f"ðŸ† BEST MODEL: {best_model}")
print(f"   RMSE (log): {best_rmse:.6f}")
print(f"   RÂ² Score: {best_r2:.6f}")
print("="*70)

# --- Convert log metrics to actual price scale ---
print("\nðŸ“Š INTERPRETATION (approximate actual price errors):")
print("   If RMSE_log â‰ˆ 0.15, then typical error â‰ˆ e^0.15 - 1 â‰ˆ 16% of price")
print("   If RMSE_log â‰ˆ 0.10, then typical error â‰ˆ e^0.10 - 1 â‰ˆ 10% of price")

# --- Save results ---
report.to_csv("model_comparison_results.csv", index=False)
print(f"\nðŸ’¾ Results saved to: model_comparison_results.csv")