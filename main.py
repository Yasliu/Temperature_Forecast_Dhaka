from fastapi import FastAPI, HTTPException
import joblib
import json
from datetime import datetime, timedelta, UTC
from zoneinfo import ZoneInfo

from data_pipeline import fetch_weather_data, build_weather_features, encode_cyclical_features

app = FastAPI(title='Dhaka Weather AI Engine')

# loading the model.

print("Loading model and its features...")
try:
    model = joblib.load('xgboost_dhaka_weather_v1.pkl')
    with open('model_features.json', 'r') as f:
        expected_columns = json.load(f)
except FileNotFoundError:
    raise RuntimeError("ERROR. Model artifacts not found. Run training script first.")

@app.get("/predict_temp")
def predict_temp():
    try:
        # Calculating the 7 day time window
        dhaka_tz = ZoneInfo("Asia/Dhaka")
        today = datetime.now(dhaka_tz)
        start_date = today - timedelta(days=7) 
        
        raw_df = fetch_weather_data(
            lat=23.8103, 
            lon=90.4125
        )
        
        df_encode = encode_cyclical_features(raw_df)
        processed_df = build_weather_features(df_encode)
        
        daytime_df = processed_df[processed_df['time'].dt.hour == 14]
        current_state = daytime_df.tail(1).copy()
        
        X_live = current_state[expected_columns].ffill()
        preds = model.predict(X_live)
        
        return {
            'status': 'success',
            'forecast_date': (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            'predicted_temperature_celcius': round(float(preds[0]),2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)