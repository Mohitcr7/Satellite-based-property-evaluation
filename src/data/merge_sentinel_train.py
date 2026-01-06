import pandas as pd
import numpy as np

print("="*70)
print("MERGE: train_cleaned + sentinel_features_train")
print("="*70)

# ---------------- LOAD DATA ----------------
base = pd.read_csv("satellite_based_property_evaluation/data/processed/train_cleaned.csv")
sentinel = pd.read_csv("satellite_based_property_evaluation/data/external/sentinel_features_train.csv")

print(f"\n[STEP 1] Loading data...")
print(f"  train_cleaned: {base.shape[0]} rows, {base.shape[1]} columns")
print(f"  sentinel_features_train: {sentinel.shape[0]} rows, {sentinel.shape[1]} columns")

# ---------------- IDENTIFY DUPLICATE COLUMNS ----------------
cols_base = set(base.columns)
cols_sentinel = set(sentinel.columns)
common_cols = cols_base & cols_sentinel

print(f"\n[STEP 2] Identifying duplicate columns...")
print(f"  Common columns (will be dropped from sentinel_features_train): {len(common_cols)}")
if common_cols:
    print(f"  Common columns: {sorted(list(common_cols))}")

# ---------------- DROP DUPLICATES (KEEP id) ----------------
cols_to_drop = common_cols - {"id"}  # keep id for merging
sentinel_unique = sentinel.drop(columns=cols_to_drop)

print(f"\n[STEP 3] Dropping duplicate columns from sentinel_features_train...")
print(f"  Dropped columns: {sorted(list(cols_to_drop))}")
print(f"  Remaining columns: {sentinel_unique.shape[1]}")
print(f"  Sentinel columns added: {list(sentinel_unique.columns)}")

# ---------------- DUPLICATE ID CHECK ----------------
base_dup = base.duplicated(subset=["id"]).sum()
sentinel_dup = sentinel_unique.duplicated(subset=["id"]).sum()

print(f"\n[STEP 4] Checking for duplicate IDs...")
print(f"  Duplicate IDs in train_cleaned: {base_dup}")
print(f"  Duplicate IDs in sentinel_features_train: {sentinel_dup}")

if base_dup > 0:
    base = base.drop_duplicates(subset=["id"], keep="first")
    print(f"  ✓ Removed {base_dup} duplicate IDs from train_cleaned")

if sentinel_dup > 0:
    sentinel_unique = sentinel_unique.drop_duplicates(subset=["id"], keep="first")
    print(f"  ✓ Removed {sentinel_dup} duplicate IDs from sentinel_features_train")

# ---------------- ID OVERLAP ANALYSIS ----------------
ids_base = set(base["id"])
ids_sentinel = set(sentinel_unique["id"])

print(f"\n[STEP 5] ID overlap analysis...")
print(f"  IDs in train_cleaned: {len(ids_base)}")
print(f"  IDs in sentinel_features_train: {len(ids_sentinel)}")
print(f"  IDs in both: {len(ids_base & ids_sentinel)}")

if len(ids_base - ids_sentinel) > 0:
    print(f"  ⚠ IDs only in train_cleaned: {len(ids_base - ids_sentinel)}")

if len(ids_sentinel - ids_base) > 0:
    print(f"  ⚠ IDs only in sentinel_features_train: {len(ids_sentinel - ids_base)}")

# ---------------- MERGE ----------------
print(f"\n[STEP 6] Performing merge on 'id'...")
df = base.merge(
    sentinel_unique,
    on="id",
    how="left"
)

print(f"  ✓ Merged result: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------- NULL CHECK ----------------
print(f"\n[STEP 7] Checking for null values...")
null_count = df.isnull().sum().sum()

if null_count > 0:
    print(f"  ⚠ Found {null_count} null values")
    null_cols = df.isnull().sum()[df.isnull().sum() > 0]
    print(f"  Columns with nulls: {list(null_cols.index)}")

    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)

    print(f"  ✓ Dropped {rows_before - rows_after} rows with null values")
    print(f"    Remaining rows: {rows_after}")
else:
    print(f"  ✓ No null values found")

# ---------------- FINAL VALIDATION ----------------
print(f"\n[STEP 8] Final validation...")
final_dup = df.duplicated(subset=["id"]).sum()
final_null = df.isnull().sum().sum()

if final_dup == 0:
    print(f"  ✓ No duplicate IDs")
else:
    print(f"  ⚠ WARNING: Found {final_dup} duplicate IDs")

print(f"\n[FINAL] Dataset summary:")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")
print(f"    - Base features: {base.shape[1]}")
print(f"    - Sentinel features added: {df.shape[1] - base.shape[1]}")

#-----------Feature Engineering------------
# Create derived features
df_engineered = df.copy()

# Vegetation intensity
df_engineered['vegetation_score'] = (df_engineered['ndvi_mean_500m'] + df_engineered['ndvi_max_500m']) / 2

# Urban intensity
df_engineered['urban_score'] = (df_engineered['ndbi_mean_500m'] + df_engineered['ndbi_max_500m']) / 2

# Water risk (variability)
df_engineered['water_variability'] = df_engineered['ndwi_max_500m'] - df_engineered['ndwi_mean_500m']

# Environment balance
df_engineered['green_vs_urban'] = df_engineered['ndvi_mean_500m'] - df_engineered['ndbi_mean_500m']

# Spatial heterogeneity (mean-max diff indicates diversity)
df_engineered['ndvi_heterogeneity'] = df_engineered['ndvi_max_500m'] - df_engineered['ndvi_mean_500m']
df_engineered['ndbi_heterogeneity'] = df_engineered['ndbi_max_500m'] - df_engineered['ndbi_mean_500m']
df_engineered['ndwi_heterogeneity'] = df_engineered['ndwi_max_500m'] - df_engineered['ndwi_mean_500m']

# Flood vulnerability score (low vegetation + high water + high variability)
df_engineered['flood_vulnerability'] = (
    (1 - (df_engineered['ndvi_mean_500m'] + 1) / 2) * 0.4 +  # Low vegetation is risky
    ((df_engineered['ndwi_mean_500m'] + 1) / 2) * 0.4 +  # High water presence is risky
    np.abs(df_engineered['water_variability']) * 0.2  # High variability adds uncertainty
)

print("="*70)
print("ENGINEERED SENTINEL FEATURES")
print("="*70)

# ---------------- SAVE ----------------
output_file = "train_with_sentinel.csv"
# df.to_csv(output_file, index=False)
df_engineered.to_csv('train_with_sentinel_engineered.csv',index=False)

print(f"\n✓ Saved merged data to {output_file}")
print("="*70)