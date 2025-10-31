"""
Script to convert Diginetica dataset to RecBole .inter format
- Creates raw .inter file with all data
- Creates processed .inter file with duplicate removal and filtering
"""

import pandas as pd
import os

# Paths
DATA_DIR = "/home/pierre/workspace/RecBole/dataset/digi"
INPUT_FILE = os.path.join(DATA_DIR, "train-item-views.csv")
OUTPUT_RAW = os.path.join(DATA_DIR, "diginetica_raw.inter")
OUTPUT_PROCESSED = os.path.join(DATA_DIR, "diginetica.inter")

print("Diginetica Dataset Conversion")
print()

# Step 1: Load the data
print("[1/5] Loading train-item-views.csv...")
df = pd.read_csv(INPUT_FILE, sep=';')
print(f"   Loaded {len(df):,} interactions")
print(f"   Columns: {list(df.columns)}")
print()
print("   Sample data:")
print(df.head())

# Step 2: Initial statistics
print()
print("[2/5] Initial statistics:")
print(f"   Total interactions: {len(df):,}")
print(f"   Unique sessions: {df['sessionId'].nunique():,}")
print(f"   Unique users: {df['userId'].nunique():,}")
print(f"   Missing user IDs (NA): {(df['userId'] == 'NA').sum():,}")
print(f"   Unique items: {df['itemId'].nunique():,}")

# Step 3: Create raw .inter file
print()
print("[3/5] Creating raw .inter file...")
df_raw = df.copy()
df_raw['user_id:token'] = df_raw['userId'].astype(str)
df_raw['session_id:token'] = df_raw['sessionId'].astype(str)
df_raw['item_id:token'] = df_raw['itemId'].astype(str)
df_raw['timestamp:float'] = df_raw['timeframe'].astype(float)

# Select only the RecBole columns
df_raw_output = df_raw[['user_id:token', 'session_id:token', 'item_id:token', 'timestamp:float']]

# Save raw file
df_raw_output.to_csv(OUTPUT_RAW, sep='\t', index=False)
print(f"   Saved raw file: {OUTPUT_RAW}")
print(f"   Total interactions: {len(df_raw_output):,}")

# Step 4: Create processed .inter file with filtering
print()
print("[4/5] Creating processed .inter file...")
df_processed = df.copy()

# Filter out interactions with no user ID (where userId is 'NA' or NaN)
initial_count = len(df_processed)
df_processed = df_processed[(df_processed['userId'] != 'NA') & (df_processed['userId'].notna())]
filtered_na = initial_count - len(df_processed)
print(f"   Filtered {filtered_na:,} interactions with missing user IDs (NA or NaN)")

# Filter out missing session IDs
df_processed = df_processed[df_processed['sessionId'].notna()]
filtered_session = initial_count - filtered_na - len(df_processed)
print(f"   Filtered {filtered_session:,} interactions with missing session IDs")

# Remove duplicates (same user, session, item, and timeframe)
before_dedup = len(df_processed)
df_processed = df_processed.drop_duplicates(subset=['userId', 'sessionId', 'itemId', 'timeframe'])
duplicates_removed = before_dedup - len(df_processed)
print(f"   Removed {duplicates_removed:,} duplicate interactions")

# Convert to RecBole format
df_processed['user_id:token'] = df_processed['userId'].astype(str)
df_processed['session_id:token'] = df_processed['sessionId'].astype(str)
df_processed['item_id:token'] = df_processed['itemId'].astype(str)
df_processed['timestamp:float'] = df_processed['timeframe'].astype(float)

# Select only the RecBole columns
df_processed_output = df_processed[['user_id:token', 'session_id:token', 'item_id:token', 'timestamp:float']]

# Save processed file
df_processed_output.to_csv(OUTPUT_PROCESSED, sep='\t', index=False)
print(f"   Saved processed file: {OUTPUT_PROCESSED}")
print(f"   Total interactions: {len(df_processed_output):,}")

# Step 5: Final statistics
print()
print("[5/5] Final statistics:")
print()
print(f"   RAW FILE ({OUTPUT_RAW}):")
print(f"   - Interactions: {len(df_raw_output):,}")
print(f"   - Unique users: {df_raw_output['user_id:token'].nunique():,}")
print(f"   - Unique sessions: {df_raw_output['session_id:token'].nunique():,}")
print(f"   - Unique items: {df_raw_output['item_id:token'].nunique():,}")

print()
print(f"   PROCESSED FILE ({OUTPUT_PROCESSED}):")
print(f"   - Interactions: {len(df_processed_output):,}")
print(f"   - Unique users: {df_processed_output['user_id:token'].nunique():,}")
print(f"   - Unique sessions: {df_processed_output['session_id:token'].nunique():,}")
print(f"   - Unique items: {df_processed_output['item_id:token'].nunique():,}")
print(f"   - Reduction: {(1 - len(df_processed_output)/len(df_raw_output))*100:.2f}%")

# Additional statistics
print()
print("   Data density:")
n_users = df_processed_output['user_id:token'].nunique()
n_items = df_processed_output['item_id:token'].nunique()
n_interactions = len(df_processed_output)
sparsity = 1 - (n_interactions / (n_users * n_items))
print(f"   - Sparsity: {sparsity*100:.4f}%")
print(f"   - Avg interactions per user: {n_interactions/n_users:.2f}")
print(f"   - Avg interactions per item: {n_interactions/n_items:.2f}")

# Show interaction distribution
interactions_per_user = df_processed_output.groupby('user_id:token').size()
print()
print("   User interaction distribution:")
print(f"   - Min: {interactions_per_user.min()}")
print(f"   - Max: {interactions_per_user.max()}")
print(f"   - Mean: {interactions_per_user.mean():.2f}")
print(f"   - Median: {interactions_per_user.median():.2f}")

print()
print("Conversion completed successfully!")
