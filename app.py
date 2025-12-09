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
ITEM_COL = None             # New item column for bakery items

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

def detect_item_col(df: pd.DataFrame):
    candidates = ["item", "Item", "product", "Product", "sku", "SKU", "product_name", "ProductName", "article", "Article"]
    for c in candidates:
        if c in df.columns:
            return c
    return None  # Item column is optional

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

def get_season(date):
    """
    Map a date to a season string: Winter, Spring, Summer, Fall
    """
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

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

# --- 3. DATA PREPROCESSING ---

def preprocess_bakery_data(df_raw: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    """
    Preprocess bakery data DataFrame:
    - Detect columns (date, sales, item)
    - Convert date column to datetime
    - Aggregate daily sales (optionally by item)
    - Add season column
    - Split into train / validation sets based on holdout_days
    Returns:
        train_df, valid_df (both DataFrames with columns: ds, y, optionally item and season)
    """
    if df_raw.empty:
        print("ERROR: Raw DataFrame is empty. Cannot preprocess data.")
        return None, None

    global DATE_COL, SALES_COL, ITEM_COL

    if DATE_COL is None:
        DATE_COL = detect_date_col(df_raw)
    if SALES_COL is None:
        SALES_COL = detect_sales_col(df_raw)
    if ITEM_COL is None:
        ITEM_COL = detect_item_col(df_raw)

    print(f"Using date column:   {DATE_COL}")
    print(f"Using sales column:  {SALES_COL}")
    print(f"Using item column:   {ITEM_COL if ITEM_COL else 'None'}")

    df = df_raw[[DATE_COL, SALES_COL] + ([ITEM_COL] if ITEM_COL else [])].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    # Aggregate daily sales
    if ITEM_COL:
        daily = (
            df.groupby([pd.Grouper(key=DATE_COL, freq="D"), ITEM_COL])[SALES_COL]
            .sum()
            .rename("y")
            .reset_index()
            .rename(columns={DATE_COL: "ds"})
        )
    else:
        daily = (
            df.groupby(pd.Grouper(key=DATE_COL, freq="D"))[SALES_COL]
            .sum()
            .rename("y")
            .reset_index()
            .rename(columns={DATE_COL: "ds"})
        )

    # Fill missing dates with zero sales
    if ITEM_COL:
        # For each item, reindex dates
        all_dates = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        items = daily[ITEM_COL].unique()
        daily_full = []
        for item in items:
            item_df = daily[daily[ITEM_COL] == item].set_index("ds").reindex(all_dates).fillna(0.0)
            item_df[ITEM_COL] = item
            item_df = item_df.rename_axis("ds").reset_index()
            daily_full.append(item_df)
        daily = pd.concat(daily_full, ignore_index=True)
    else:
        all_days = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
        daily = (
            daily.set_index("ds")
                 .reindex(all_days)
                 .fillna(0.0)
                 .rename_axis("ds")
                 .reset_index()
        )
    daily["y"] = daily["y"].clip(lower=0)

    # Add season column
    daily["season"] = daily["ds"].apply(get_season)

    # Train/validation split
    if holdout_days <= 0 or holdout_days >= len(daily):
        print(f"Warning: holdout_days ({holdout_days}) is invalid for {len(daily)} rows.")
        holdout_days = 0  # Train on all data if invalid

    if holdout_days > 0:
        train_df = daily.iloc[:-holdout_days].copy()
        valid_df = daily.iloc[-holdout_days:].copy()
    else:
        train_df = daily
        valid_df = pd.DataFrame()

    print(f"Total days used for training: {len(train_df)}")
    if holdout_days > 0:
        print(f"Total days used for validation: {len(valid_df)}")
    else:
        print("No validation split due to holdout_days=0")

    return train_df, valid_df

# --- 4. TRAINING PIPELINE ---

def run_prophet_training_pipeline(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """
    Runs the full Prophet training and validation pipeline on the preprocessed data.
    """
    if train_df.empty:
        print("ERROR: Training DataFrame is empty. Cannot train model.")
        return None, None

    # Fit Prophet model
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

    # Validation forecast + metrics (only if validation set exists)
    if not valid_df.empty:
        future_valid = valid_df[["ds"]].copy()
        forecast_valid = m_full.predict(future_valid)
        eval_df = valid_df.merge(
            forecast_valid[["ds", "yhat"]], on="ds", how="left"
        )
        mae  = mean_absolute_error(eval_df["y"], eval_df["yhat"])
        rmse = rmse_fn(eval_df["y"], eval_df["yhat"])
        mape_val = mape(eval_df["y"], eval_df["yhat"])

        print("\n=== Hold-out Metrics ===")
        print(f"MAE : {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"MAPE: {mape_val:,.2f}%")

    return m_full, current_version # Return the trained model and a version tag

# --- 5. MODEL SAVING TO SPACES ---

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

    # --- STEP 2: PREPROCESS DATA ---
    train_df, valid_df = preprocess_bakery_data(raw_data_df, holdout_days=HOLDOUT_DAYS)

    # --- STEP 3: PROCESS & TRAIN MODEL ---
    if train_df is None or train_df.empty:
        print("No training data available. Exiting.")
        trained_models = None
    else:
        if ITEM_COL is None:
            print("No item column detected; training a single Prophet model for all data.")
            trained_model, _ = run_prophet_training_pipeline(train_df, valid_df)
            trained_models = {"all_items": trained_model}
            df_items = train_df  # Save entire training data
        else:
            trained_models = {}
            # Extract unique items
            unique_items = train_df[ITEM_COL].unique()
            print(f"Training Prophet models for {len(unique_items)} items.")
            for item in unique_items:
                print(f"Training model for item: {item}")
                item_train_df = train_df[train_df[ITEM_COL] == item][["ds", "y"]].copy()
                item_valid_df = pd.DataFrame()
                if not valid_df.empty:
                    item_valid_df = valid_df[valid_df[ITEM_COL] == item][["ds", "y"]].copy()
                m = Prophet(
                    weekly_seasonality=WEEKLY_SEASONALITY,
                    yearly_seasonality=YEARLY_SEASONALITY,
                    daily_seasonality=DAILY_SEASONALITY,
                    seasonality_mode="additive",
                    changepoint_prior_scale=0.5,
                )
                if ADD_COUNTRY_HOLIDAYS:
                    try:
                        m.add_country_holidays(country_name=ADD_COUNTRY_HOLIDAYS)
                    except Exception as e:
                        print(f"Warning: could not add holidays '{ADD_COUNTRY_HOLIDAYS}': {e}")
                m.fit(item_train_df)

                # Validation metrics per item
                if not item_valid_df.empty:
                    future_valid = item_valid_df[["ds"]].copy()
                    forecast_valid = m.predict(future_valid)
                    eval_df = item_valid_df.merge(
                        forecast_valid[["ds", "yhat"]], on="ds", how="left"
                    )
                    mae  = mean_absolute_error(eval_df["y"], eval_df["yhat"])
                    rmse = rmse_fn(eval_df["y"], eval_df["yhat"])
                    mape_val = mape(eval_df["y"], eval_df["yhat"])
                    print(f"Item: {item} - MAE: {mae:,.2f}, RMSE: {rmse:,.2f}, MAPE: {mape_val:,.2f}%")

                trained_models[item] = m

            # Save the full item-level training data for use in predictions
            df_items = train_df.copy()

    # --- STEP 4: SAVE ARTIFACT ---
    # Save a dictionary containing all trained models and the historical item-level data
    model_artifact = {
        "models": trained_models,
        "historical_data": df_items if 'df_items' in locals() else None,
        "item_col": ITEM_COL,
        "date_col": DATE_COL,
        "sales_col": SALES_COL,
    }
    save_model(model_artifact, current_version)