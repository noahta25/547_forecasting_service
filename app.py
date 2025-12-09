import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from prophet import Prophet
import numpy as np

app = Flask(__name__)

# Load model artifact on startup
MODEL_PATH = os.environ.get("MODEL_PATH", "model-artifacts/sales_forecast_prophet_latest.pkl")

print(f"Loading model artifact from {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model_artifact = pickle.load(f)

trained_models = model_artifact.get("models", {})
df_items = model_artifact.get("historical_data", pd.DataFrame())
ITEM_COL = model_artifact.get("item_col", None)
DATE_COL = model_artifact.get("date_col", None)
SALES_COL = model_artifact.get("sales_col", None)

@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.get_json()
    days = data.get("days", 30)
    item = data.get("item", None)

    if ITEM_COL and item:
        if item not in trained_models:
            return jsonify({"error": f"No model found for item '{item}'"}), 400
        model = trained_models[item]
    else:
        # Use aggregate model if no item specified or no item column
        model = trained_models.get("all_items", None)
        if model is None:
            return jsonify({"error": "No aggregate model available"}), 400

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # Filter forecast to only future dates
    forecast = forecast[forecast['ds'] > forecast['ds'].max() - pd.Timedelta(days=days)]

    response = {
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
    }

    # Include historical sales if available
    if not df_items.empty:
        if ITEM_COL and item:
            hist = df_items[df_items[ITEM_COL] == item][["ds", "y"]]
        else:
            hist = df_items[["ds", "y"]]
        response["historical"] = hist.to_dict(orient="records")

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)