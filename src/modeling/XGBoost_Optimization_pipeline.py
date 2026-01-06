"""
XGBoost Optimization Pipeline with Feature Selection
Step 1: Train baseline model and identify low-importance features
Step 2: Drop bottom 5-10 features
Step 3: Run Optuna optimization on cleaned feature set

This approach reduces training time and often improves performance!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from xgboost import XGBRegressor
import xgboost as xgb
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
N_FEATURES_TO_DROP = 15  # Drop bottom 15 features (adjust as needed)
N_TRIALS = 100          # Optuna trials
N_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = -1

print("="*70)
print("XGBOOST FEATURE SELECTION + OPTUNA OPTIMIZATION PIPELINE")
print("="*70)

# ==================== LOAD DATA ====================
print("\n[STEP 0] Loading data...")
df = pd.read_csv("satellite_based_property_evaluation/data/processed/train_with_sentinel.csv")

# Check for null values and drop rows with any nulls
null_count = df.isnull().sum().sum()
if null_count > 0:
    print(f"  Found {null_count} null values in dataset")
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"  Dropped {rows_before - rows_after} rows with null values")
    print(f"  Remaining: {rows_after} rows")

y = df["price_log"]
X = df.drop(columns=["id", "price", "price_log"])

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Baseline RMSE: 0.305077")

# ==================== STEP 1: FEATURE IMPORTANCE ====================
print("\n" + "="*70)
print("[STEP 1] ANALYZING FEATURE IMPORTANCE")
print("="*70)

# Train baseline XGBoost model
print("\nTraining baseline model to identify important features...")
baseline_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    tree_method='hist'
)

baseline_model.fit(X, y)

# Get feature importances
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': baseline_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance Summary:")
print(f"  Total features: {len(importance_df)}")
print(f"  Features with zero importance: {(importance_df['importance'] == 0).sum()}")
print(f"  Features with importance < 0.001: {(importance_df['importance'] < 0.001).sum()}")

# Show top 10 and bottom 10 features
print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
print(importance_df.head(10).to_string(index=False))

print(f"\n⬇️ BOTTOM {N_FEATURES_TO_DROP} LEAST IMPORTANT FEATURES (TO BE DROPPED):")
bottom_features = importance_df.tail(N_FEATURES_TO_DROP)
print(bottom_features.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_20 = importance_df.head(20)
plt.barh(range(len(top_20)), top_20['importance'].values)
plt.yticks(range(len(top_20)), top_20['feature'].values)
plt.xlabel('Importance Score')
plt.title('Top 20 Most Important Features (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=150, bbox_inches='tight')
print("\n✓ Feature importance plot saved: feature_importance_plot.png")

# Save full importance ranking
importance_df.to_csv('feature_importance_ranking.csv', index=False)
print("✓ Feature importance ranking saved: feature_importance_ranking.csv")

# ==================== STEP 2: DROP LOW-IMPORTANCE FEATURES ====================
print("\n" + "="*70)
print("[STEP 2] DROPPING LOW-IMPORTANCE FEATURES")
print("="*70)

features_to_drop = bottom_features['feature'].tolist()
X_reduced = X.drop(columns=features_to_drop)

print(f"\n✂️ Dropped {len(features_to_drop)} features:")
for feat in features_to_drop:
    print(f"  - {feat}")

print(f"\n📉 Feature count: {X.shape[1]} → {X_reduced.shape[1]}")
print(f"   Reduction: {(1 - X_reduced.shape[1]/X.shape[1]) * 100:.1f}%")

# Test performance with reduced features
print("\n🧪 Testing performance with reduced feature set...")
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
scores = cross_val_score(
    baseline_model, X_reduced, y,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=1
)
reduced_rmse = -scores.mean()

print(f"\n📊 COMPARISON:")
print(f"  Full features ({X.shape[1]}):     RMSE = 0.305077 (baseline)")
print(f"  Reduced features ({X_reduced.shape[1]}): RMSE = {reduced_rmse:.6f}")
print(f"  Difference: {(reduced_rmse - 0.305077):.6f}")

if reduced_rmse < 0.305077:
    print("  ✅ Feature reduction IMPROVED performance!")
else:
    print("  ⚠️ Slight performance trade-off, but will optimize in next step")

# ==================== STEP 3: OPTUNA OPTIMIZATION ====================
print("\n" + "="*70)
print("[STEP 3] RUNNING OPTUNA OPTIMIZATION ON REDUCED FEATURES")
print("="*70)
print(f"\nThis should be ~{(1 - X_reduced.shape[1]/X.shape[1]) * 100:.0f}% faster than before!")
print(f"Running {N_TRIALS} trials...\n")

def objective(trial):
    """Optuna objective for reduced feature set"""
    params = {
        # Key parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        
        # Additional parameters
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        
        # Fixed parameters
        'n_estimators': 800,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0
    }
    
    model = XGBRegressor(**params)
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, X_reduced, y,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=1
    )
    
    return -scores.mean()

# Run optimization
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    n_jobs=1
)

# ==================== RESULTS ====================
print("\n" + "="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)

best_params = study.best_params
best_rmse = study.best_value

print(f"\n🎯 BEST RMSE (log): {best_rmse:.6f}")
print(f"📉 Improvement over baseline: {(0.305077 - best_rmse) / 0.305077 * 100:.2f}%")

print(f"\n🏆 BEST HYPERPARAMETERS:")
print("-" * 70)
for param, value in best_params.items():
    print(f"  {param:20s}: {value}")

# ==================== TRAIN FINAL MODEL ====================
print("\n" + "="*70)
print("[FINAL] TRAINING OPTIMIZED MODEL")
print("="*70)

final_model = XGBRegressor(
    n_estimators=800,
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    tree_method='hist',
    **best_params
)

final_model.fit(X_reduced, y)

# Final cross-validation
scoring = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2'
}

cv_results = cross_validate(
    final_model, X_reduced, y,
    cv=cv,
    scoring=scoring,
    n_jobs=1
)

final_rmse = -cv_results['test_rmse'].mean()
final_rmse_std = cv_results['test_rmse'].std()
final_mae = -cv_results['test_mae'].mean()
final_r2 = cv_results['test_r2'].mean()

print(f"\n📊 FINAL MODEL PERFORMANCE:")
print("-" * 70)
print(f"  Features used:  {X_reduced.shape[1]} (dropped {len(features_to_drop)})")
print(f"  RMSE (log):     {final_rmse:.6f} ± {final_rmse_std:.6f}")
print(f"  MAE (log):      {final_mae:.6f}")
print(f"  R² Score:       {final_r2:.6f}")

# ==================== SAVE EVERYTHING ====================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save reduced feature list
pd.DataFrame({'feature': X_reduced.columns}).to_csv('selected_features.csv', index=False)
print("✓ Selected features saved: selected_features.csv")

# Save dropped features list
pd.DataFrame({'dropped_feature': features_to_drop}).to_csv('dropped_features.csv', index=False)
print("✓ Dropped features saved: dropped_features.csv")

# Save best parameters
params_df = pd.DataFrame([best_params])
params_df['best_rmse'] = best_rmse
params_df['final_rmse'] = final_rmse
params_df['final_r2'] = final_r2
params_df['features_used'] = X_reduced.shape[1]
params_df['features_dropped'] = len(features_to_drop)
params_df.to_csv('xgboost_optimized_params.csv', index=False)
print("✓ Best parameters saved: xgboost_optimized_params.csv")

# Save optimization history
study.trials_dataframe().to_csv('optimization_history.csv', index=False)
print("✓ Optimization history saved: optimization_history.csv")

# Save final model
joblib.dump(final_model, 'xgboost_final_model.pkl')
print("✓ Final model saved: xgboost_final_model.pkl")

# Save feature importance of final model
final_importance = pd.DataFrame({
    'feature': X_reduced.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)
final_importance.to_csv('final_feature_importance.csv', index=False)
print("✓ Final feature importance saved: final_feature_importance.csv")

# ==================== VISUALIZATIONS ====================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    
    fig1 = plot_optimization_history(study)
    fig1.write_html('optimization_history.html')
    print("✓ Optimization history plot: optimization_history.html")
    
    fig2 = plot_param_importances(study)
    fig2.write_html('param_importances.html')
    print("✓ Parameter importance plot: param_importances.html")
except:
    print("⚠ Install plotly for interactive plots: pip install plotly")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print("🎉 PIPELINE COMPLETE - FINAL SUMMARY")
print("="*70)
print(f"\n📈 PERFORMANCE JOURNEY:")
print(f"  Original baseline:    RMSE = 0.305077  ({X.shape[1]} features)")
print(f"  After dropping {N_FEATURES_TO_DROP}:     RMSE = {reduced_rmse:.6f}  ({X_reduced.shape[1]} features)")
print(f"  After optimization:   RMSE = {final_rmse:.6f}  ({X_reduced.shape[1]} features)")
print(f"\n🏆 TOTAL IMPROVEMENT:   {(0.305077 - final_rmse) / 0.305077 * 100:.2f}%")
print(f"💡 R² Score:            {final_r2:.6f}")
print(f"⚡ Speed improvement:   ~{(1 - X_reduced.shape[1]/X.shape[1]) * 100:.0f}% faster training")
print("="*70)

print("\n📝 TO USE THE FINAL MODEL ON TEST DATA:")
print("="*70)
print("""
import pandas as pd
import joblib

# Load model and selected features
model = joblib.load('xgboost_final_model.pkl')
selected_features = pd.read_csv('selected_features.csv')['feature'].tolist()

# Load and prepare test data
test_df = pd.read_csv('test_cleaned.csv')
X_test = test_df[selected_features]  # Use only selected features

# Predict
predictions = model.predict(X_test)
actual_prices = np.expm1(predictions)  # Convert from log scale
""")