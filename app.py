import os
import requests
import pandas as pd
import pickle
import datetime
import boto3
import numpy as np
import warnings

from dotenv import load_dotenv
load_dotenv()

# Suppress warnings from libraries like Prophet
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# --- 1. CONFIGURATION (READ FROM ENVIRONMENT) ---
# Ensure these environment variables are set securely in DigitalOcean App Platform
SPACE_ENDPOINT = os.environ.get('SPACE_ENDPOINT')
DO_ACCESS_KEY = os.environ.get('DO_ACCESS_KEY')
DO_SECRET_KEY = os.environ.get('DO_SECRET_KEY')
DO_BUCKET_NAME = os.environ.get('DO_BUCKET_NAME')

# --- PROPHET CONFIGURATION (Copied from Notebook) ---
HOLDOUT_DAYS = 30           
FORECAST_DAYS = 30          # Not strictly needed for serving, but good for validation
ADD_COUNTRY_HOLIDAYS = "FR" 
WEEKLY_SEASONALITY = True
YEARLY_SEASONALITY = True
DAILY_SEASONALITY = False
DATE_COL = None             # Keep auto-detection logic
SALES_COL = None            # Keep auto-detection logic

# --- HELPER FUNCTIONS (From Notebook) ---

def detect_date_col(df: pd.DataFrame):
    # (Existing column detection logic here - omitted for brevity)
    candidates = ["date", "Date", "DATE", "ds", "order_date", "OrderDate", "timestamp", "Timestamp", "day"]
    for c in candidates:
        if c in df.columns:
            return c
    # ... (rest of detection logic)
    raise ValueError("Could not detect a date column. Please set DATE_COL manually.")

def detect_sales_col(df: pd.DataFrame):
    # (Existing column detection logic here - omitted for brevity)
    candidates = ["sales", "Sales", "quantity", "Quantity", "qty", "Qty", "units", "Units", "y", "sales_qty", "Sales_Quantity", "SalesQuantity", "count", "Count"]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    # ... (rest of detection logic)
    raise ValueError("Could not detect a sales/quantity column. Please set SALES_COL manually.")

def mape(y_true, y_pred):
    # (Existing MAPE function - omitted for brevity)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def rmse_fn(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --- 2. DATA INGESTION ---

def pull_historical_data() -> pd.DataFrame:
    """
    Pulls historical sales data from the DigitalOcean Spaces bucket.
    """
    print(f"-> Attempting to pull historical_sales_data.csv from bucket: {DO_BUCKET_NAME}")
    try:
        session = boto3.session.Session()
        s3_client = session.client(
            's3',
            endpoint_url=SPACE_ENDPOINT,
            aws_access_key_id=DO_ACCESS_KEY,
            aws_secret_access_key=DO_SECRET_KEY
        )
        obj = s3_client.get_object(Bucket=DO_BUCKET_NAME, Key="BakerySales.csv")
        df = pd.read_csv(obj['Body'])
        print(f"-> Successfully pulled {len(df)} records from DigitalOcean Spaces.")
        return df
    except Exception as e:
        print(f"ERROR: Failed to pull data from DigitalOcean Spaces: {e}")
        return pd.DataFrame()


# --- 3. TRAINING PIPELINE ---

def run_prophet_training_pipeline(df_raw: pd.DataFrame):
    """
    Runs the full Prophet training and validation pipeline on the raw data.
    """
    if df_raw.empty:
        print("ERROR: Raw DataFrame is empty. Cannot train model.")
        return None, None

    # --- 2) Column detection (Modified to use global DATE_COL/SALES_COL if set) ---
    global DATE_COL, SALES_COL, HOLDOUT_DAYS # Need to use global to update these vars
    
    if DATE_COL is None:
        DATE_COL = detect_date_col(df_raw)
    if SALES_COL is None:
        SALES_COL = detect_sales_col(df_raw)

    print(f"Using date column:   {DATE_COL}")
    print(f"Using sales column:  {SALES_COL}")

    # --- 3) Clean & aggregate to daily ---
    print("-> Cleaning and aggregating data to daily totals...")
    df = df_raw[[DATE_COL, SALES_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    
    # Aggregate and reindex/fill missing dates with zero
    daily = (
        df.groupby(pd.Grouper(key=DATE_COL, freq="D"))[SALES_COL]
        .sum()
        .rename("y")
        .reset_index()
        .rename(columns={DATE_COL: "ds"})
    )
    all_days = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
             .reindex(all_days)
             .fillna(0.0)
             .rename_axis("ds")
             .reset_index()
    )
    daily["y"] = daily["y"].clip(lower=0)
    
    # --- 4) Train / validation split ---
    if HOLDOUT_DAYS <= 0 or HOLDOUT_DAYS >= len(daily):
        print(f"Warning: HOLDOUT_DAYS ({HOLDOUT_DAYS}) is invalid for {len(daily)} rows.")
        HOLDOUT_DAYS = 0 # Train on all data if invalid

    if HOLDOUT_DAYS > 0:
        train = daily.iloc[:-HOLDOUT_DAYS].copy()
        valid = daily.iloc[-HOLDOUT_DAYS:].copy()
        train_df = train
    else:
        train_df = daily
        
    print(f"Total days used for training: {len(train_df)}")

    # --- 5) Fit Prophet (train split) ---
    m_full = Prophet(
        weekly_seasonality=WEEKLY_SEASONALITY,
        yearly_seasonality=YEARLY_SEASONALITY,
        daily_seasonality=DAILY_SEASONALITY,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,
    )
    
    if ADD_COUNTRY_HOLIDAYS:
        try:
            m_full.add_country_holidays(country_name=ADD_COUNTRY_HOLIDAYS)
            print(f"Added country holidays: {ADD_COUNTRY_HOLIDAYS}")
        except Exception as e:
            print(f"Warning: could not add holidays '{ADD_COUNTRY_HOLIDAYS}': {e}")
            
    m_full.fit(train_df)

    # --- 6) Validation forecast + metrics (only if HOLDOUT_DAYS > 0) ---
    if HOLDOUT_DAYS > 0:
        future_valid = valid[["ds"]].copy()
        forecast_valid = m_full.predict(future_valid)
        eval_df = valid.merge(
            forecast_valid[["ds", "yhat"]], on="ds", how="left"
        )
        mae  = mean_absolute_error(eval_df["y"], eval_df["yhat"])
        rmse = rmse_fn(eval_df["y"], eval_df["yhat"])
        mape_val = mape(eval_df["y"], eval_df["yhat"])

        print("\n=== Hold-out Metrics ===")
        print(f"MAE : {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"MAPE: {mape_val:,.2f}%")
        
    # NOTE: We skip fitting on the full data 'm_full' from your notebook 
    # and just use the last model trained on the largest subset (or full data if HOLDOUT_DAYS=0).
    
    return m_full, current_version # Return the trained model and a version tag


# --- 4. MODEL SAVING TO SPACES ---

def save_model(model, version_tag: str):
    """
    Saves the trained Prophet model locally, then uploads it to DigitalOcean Spaces.
    """
    if model is None:
        print("-> No model to save.")
        return
        
    print(f"-> Connecting to DigitalOcean Spaces at {SPACE_ENDPOINT}")
    
    try:
        # 1. Setup Boto3 client for DO Spaces
        session = boto3.session.Session()
        s3_client = session.client('s3', 
                                   endpoint_url=SPACE_ENDPOINT, 
                                   aws_access_key_id=DO_ACCESS_KEY, 
                                   aws_secret_access_key=DO_SECRET_KEY)
        
        # 2. Define the artifact name and path
        # Prophet model is an instance of the class, save it via pickle
        model_filename = f'sales_forecast_prophet_{version_tag}.pkl'
        space_path = f'model-artifacts/{model_filename}'
        
        # 3. Save the model locally first to a temporary directory
        local_path = f'/tmp/{model_filename}' 
        with open(local_path, 'wb') as file:
            pickle.dump(model, file)
        
        # 4. Upload to Spaces
        print(f"-> Uploading {model_filename} to {DO_BUCKET_NAME}/{space_path}")
        s3_client.upload_file(local_path, DO_BUCKET_NAME, space_path)
        
        print(f"SUCCESS: Model version {version_tag} saved and uploaded.")
        
    except Exception as e:
        print(f"FATAL ERROR during model saving/upload: {e}")


if __name__ == "__main__":
    # Generate a unique version tag (YYYYMMDD_HHMM)
    current_version = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # --- STEP 1: PULL DATA ---
    raw_data_df = pull_historical_data()

    # --- STEP 2: PROCESS & TRAIN MODEL ---
    # The training pipeline returns the fitted Prophet object
    trained_prophet_model, _ = run_prophet_training_pipeline(raw_data_df)

    # --- STEP 3: SAVE ARTIFACT ---
    save_model(trained_prophet_model, current_version)