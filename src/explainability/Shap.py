import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs("shap_outputs", exist_ok=True)

# Load model
model = joblib.load("xgboost_final_model.pkl")

# Load selected features
selected_features = pd.read_csv("selected_features.csv")["feature"].tolist()

# Load training data (with Sentinel features)
df = pd.read_csv("train_with_sentinel.csv")

X = df[selected_features]
y = df["price_log"]

# Use TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# GLOBAL SHAP SUMMARY PLOT
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X,
    show=False
)
plt.tight_layout()
plt.savefig("shap_outputs/01_shap_summary.png", dpi=200)
plt.close()

# GLOBAL FEATURE IMPORTANCE (BAR CHART)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("shap_outputs/02_shap_feature_importance.png", dpi=200)
plt.close()

# SENTINEL FEATURE DEPENDENCE PLOTS

# NDVI – Neighborhood Greenness
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "ndvi_mean_500m",
    shap_values,
    X,
    show=False
)
plt.tight_layout()
plt.savefig("shap_outputs/03_shap_ndvi_dependence.png", dpi=200)
plt.close()

# NDBI – Built-up Density
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    "ndbi_mean_500m",
    shap_values,
    X,
    show=False
)
plt.tight_layout()
plt.savefig("shap_outputs/04_shap_ndbi_dependence.png", dpi=200)
plt.close()

# NDWI – Water Proximity (if present)
if "ndwi_mean_500m" in X.columns:
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        "ndwi_mean_500m",
        shap_values,
        X,
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_outputs/05_shap_ndwi_dependence.png", dpi=200)
    plt.close()

#  SINGLE-PROPERTY EXPLANATION (FORCE PLOT)
shap.initjs()

idx = 0  # any property index

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[idx],
    X.iloc[idx],
    matplotlib=False
)

shap.save_html(
    "shap_outputs/06_shap_force_plot_single_property.html",
    force_plot
)

# EXPORT SHAP IMPORTANCE TABLE (REPORT-READY)
shap_table = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_table.to_csv(
    "shap_outputs/07_shap_importance_table.csv",
    index=False
)