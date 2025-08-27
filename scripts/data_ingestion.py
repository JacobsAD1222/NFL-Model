# --- Step 1: Imports ---
import os
import pandas as pd
import time
import nfl_data_py as nfl
from google.cloud import storage
import urllib.error

# --- Step 2: Firebase Storage Setup ---
# The service account key JSON will be supplied as a GitHub secret and written to runtime
FIREBASE_KEY_PATH = "firebase_key.json"
BUCKET_NAME = "sports-betting-model-fea2f.firebasestorage.app"
SPORT_FOLDER = "nfl/raw_data"

# Initialize Firebase client
client = storage.Client.from_service_account_json(FIREBASE_KEY_PATH)
bucket = client.bucket(BUCKET_NAME)

# --- Step 3: Define function to fetch & update data ---
def fetch_and_update_nfl_data(data_type, year):
    """Fetch NFL data for a given type and year, merge with existing Parquet, and update Firebase."""
    local_filename = f"nfl_{data_type}_{year}.parquet"
    firebase_path = f"{SPORT_FOLDER}/{local_filename}"
    
    # --- Download existing file from Firebase if it exists ---
    blob = bucket.blob(firebase_path)
    if blob.exists():
        print(f"Existing data found in Firebase for {data_type} {year}. Downloading...")
        blob.download_to_filename(local_filename)
        df_existing = pd.read_parquet(local_filename)
        print(f"Loaded existing {data_type} data: {df_existing.shape[0]} rows")
    else:
        df_existing = pd.DataFrame()
        print(f"No existing data found for {data_type} {year}. Will create new file.")
    
    # --- Fetch latest data from nfl_data_py with error handling ---
    print(f"Fetching {data_type} for {year} from nfl_data_py...")
    df_new = pd.DataFrame() # Initialize df_new to an empty DataFrame
    try:
        if data_type == 'schedules':
            df_new = nfl.import_schedules([year])
        elif data_type == 'weekly_stats':
            df_new = nfl.import_weekly_data([year])
        elif data_type == 'pbp':
            # Specific handling for the potentially flaky pbp data source
            nfl.cache_pbp([year])
            df_new = nfl.import_pbp_data([year])
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")
        print(f"Fetched {data_type} {year}: {df_new.shape[0]} rows")
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} occurred while fetching {data_type} data for {year}. The file may be temporarily unavailable. Skipping this step.")
        # We can continue with the loop, as df_new is still an empty DataFrame
    except Exception as e:
        # Catch any other unexpected errors, including the NameError from the library
        print(f"An unexpected error occurred while fetching {data_type} data for {year}: {e}. Skipping this step.")

    # --- Combine with existing data if applicable ---
    if not df_new.empty:
        if not df_existing.empty:
            if data_type == 'weekly_stats':
                key_cols = ['player_id', 'season', 'week']
            elif data_type == 'pbp':
                key_cols = ['game_id', 'play_id']
            else:
                key_cols = ['game_id'] if 'game_id' in df_new.columns else df_new.columns.tolist()
            
            # Remove duplicate rows in new data (already in existing)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=key_cols)
            print(f"Combined dataset: {df_combined.shape[0]} rows")
        else:
            df_combined = df_new
            print(f"Combined dataset: {df_combined.shape[0]} rows")

        # --- Save locally and upload to Firebase ---
        df_combined.to_parquet(local_filename, index=False)
        blob.upload_from_filename(local_filename)
        print(f"Updated {data_type} {year} uploaded to Firebase at {firebase_path}\n")
    else:
        print(f"No new data to upload for {data_type} {year}. The script will continue.")
    
    # Polite delay
    time.sleep(3 if data_type != 'pbp' else 10)

# --- Step 4: Main execution ---
if __name__ == "__main__":
    START_YEAR = 2024
    END_YEAR = 2024  # Update as the season progresses
    data_types_to_fetch = ['schedules', 'weekly_stats', 'pbp']
    
    for year in range(START_YEAR, END_YEAR + 1):
        for data_type in data_types_to_fetch:
            fetch_and_update_nfl_data(data_type, year)
