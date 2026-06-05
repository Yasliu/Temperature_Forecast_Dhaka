# Dhaka Weather AI Engine

A production-ready machine learning pipeline that predicts 24-hour peak daytime temperatures. 

Instead of relying on standard meteorological models, this engine engineers physical atmospheric approximations (diurnal ranges, moisture flux, volatility trends) and feeds them into a tuned XGBoost architecture.

## Architecture
* **Data Engineering:** Automated cyclical time encoding and rolling 168-hour window feature generation via Open-Meteo API.
* **Model:** XGBoost Regressor.
* **MLOps:** SQLite3 experiment tracking for R&D versioning.
* **Serving:** FastAPI backend implementing strict schema validation and realtime data ingestion.

## Running Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the inference server
python main.py

# 3. Test the endpoint
curl -X GET "http://localhost:8000/predict_temp"