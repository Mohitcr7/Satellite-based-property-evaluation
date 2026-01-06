import pandas as pd

print("="*70)
print("MERGE: test_cleaned + sentinel_features_test")
print("="*70)

# Load the datasets
base = pd.read_csv('test_cleaned.csv')
sentinel = pd.read_csv('sentinel_features_test.csv')

print(f"\n[STEP 1] Loading data...")
print(f"  test_cleaned: {base.shape[0]} rows, {base.shape[1]} columns")
print(f"  sentinel_features_test: {sentinel.shape[0]} rows, {sentinel.shape[1]} columns")

# Identify common columns (duplicates to drop from sentinel_features_test)
cols_base = set(base.columns)
cols_sentinel = set(sentinel.columns)
common_cols = cols_base & cols_sentinel
print(f"\n[STEP 2] Identifying duplicate columns...")
print(f"  Common columns (will be dropped from sentinel_features_test): {len(common_cols)}")
if len(common_cols) > 0:
    print(f"  Common columns: {sorted(list(common_cols))}")

# Keep only unique columns from sentinel_features_test (plus 'id' for merging)
# Drop: all common columns except 'id' (keep 'id' for merging)
cols_to_drop = common_cols - {'id'}  # Keep 'id' for merging
sentinel_unique = sentinel.drop(columns=cols_to_drop)

print(f"\n[STEP 3] Dropping duplicate columns from sentinel_features_test...")
print(f"  Dropped columns: {sorted(list(cols_to_drop))}")
print(f"  Remaining columns in sentinel_features_test: {sentinel_unique.shape[1]}")
print(f"  Unique columns: {list(sentinel_unique.columns)}")

# Check for duplicates
base_dup = base.duplicated(subset=['id']).sum()
sentinel_dup = sentinel_unique.duplicated(subset=['id']).sum()
print(f"\n[STEP 4] Checking for duplicate IDs...")
print(f"  Duplicate IDs in test_cleaned: {base_dup}")
print(f"  Duplicate IDs in sentinel_features_test: {sentinel_dup}")

# Remove duplicates if any (keep first occurrence)
if base_dup > 0:
    base = base.drop_duplicates(subset=['id'], keep='first')
    print(f"  ✓ Removed {base_dup} duplicate IDs from test_cleaned")

if sentinel_dup > 0:
    sentinel_unique = sentinel_unique.drop_duplicates(subset=['id'], keep='first')
    print(f"  ✓ Removed {sentinel_dup} duplicate IDs from sentinel_features_test")

# Check ID overlap
ids_base = set(base['id'])
ids_sentinel = set(sentinel_unique['id'])
print(f"\n[STEP 5] ID overlap analysis...")
print(f"  IDs in test_cleaned: {len(ids_base)}")
print(f"  IDs in sentinel_features_test: {len(ids_sentinel)}")
print(f"  IDs in both: {len(ids_base & ids_sentinel)}")
if len(ids_base - ids_sentinel) > 0:
    print(f"  ⚠ IDs only in test_cleaned: {len(ids_base - ids_sentinel)}")
if len(ids_sentinel - ids_base) > 0:
    print(f"  ⚠ IDs only in sentinel_features_test: {len(ids_sentinel - ids_base)}")

# Perform merge on 'id'
print(f"\n[STEP 6] Performing merge on 'id'...")
df = base.merge(
    sentinel_unique,
    on='id',
    how='left'  # Keep all rows from test_cleaned
)

print(f"  ✓ Merged result: {df.shape[0]} rows, {df.shape[1]} columns")

# Check for null values and drop rows with any nulls
print(f"\n[STEP 7] Checking for null values...")
null_count = df.isnull().sum().sum()
if null_count > 0:
    print(f"  ⚠ Found {null_count} null values")
    null_cols = df.isnull().sum()[df.isnull().sum() > 0]
    print(f"  Columns with nulls: {list(null_cols.index)}")
    
    # Drop all rows with any null values
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"  ✓ Dropped {rows_before - rows_after} rows with null values")
    print(f"    Remaining: {rows_after} rows")
else:
    print(f"  ✓ No null values found")

# Final validation
print(f"\n[STEP 8] Final validation...")
final_dup = df.duplicated(subset=['id']).sum()
final_null = df.isnull().sum().sum()

if final_dup == 0:
    print(f"  ✓ No duplicate IDs: {final_dup}")
else:
    print(f"  ⚠ WARNING: Found {final_dup} duplicate IDs")

print(f"\n[FINAL] Dataset summary:")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")
print(f"  Column breakdown:")
print(f"    - Base features from test_cleaned: {base.shape[1]}")
print(f"    - Sentinel features added: {df.shape[1] - base.shape[1]}")

# Save the merged dataset
output_file = "test_final_merged.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Saved merged data to {output_file}")
print("="*70)