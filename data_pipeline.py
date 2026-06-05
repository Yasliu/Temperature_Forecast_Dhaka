import pandas as pd
import numpy as np
import requests

def fetch_weather_data(lat: float, lon: float) -> pd.DataFrame: 
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=14&forecast_days=0&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation&timezone=auto"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed API Call: {response.status_code}")
    
    raw_data = response.json()
    
    if 'hourly' not in raw_data:
        raise Exception(f"API Error: 'hourly' data missing from response. keys received {list(raw_data.keys())}")
    
    df = pd.DataFrame(raw_data['hourly'])
    return df

def encode_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df['time'] = pd.to_datetime(df['time'])
    
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    
    df['month_sin'] = np.sin( 2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos( 2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin( 2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos( 2 * np.pi * df['hour'] / 24)
    
    return df

def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    target_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']

    for col in target_columns:
        df[f"{col}_lag_24h"] = df[f'{col}'].shift(24)
        df[f"{col}_rolling_avg"] = df[f"{col}"].rolling(window=168, closed="left").mean()
        
    df['temp_delta_1h'] = df['temperature_2m'].diff(periods=1)
    df['temp_volatility_24h'] = df['temperature_2m'].rolling(window=24, closed='left').std()
    
    rolling_max = df['temperature_2m'].rolling(window=24, closed='left').max()
    rolling_min = df['temperature_2m'].rolling(window=24, closed='left').min()
    df['temp_diurnal_range'] = rolling_max - rolling_min
    
    df['Atm_moisture_flux'] = df['wind_speed_10m'] * df['relative_humidity_2m']
    df['Evaporative_potential_index'] = df['wind_speed_10m'] / df['relative_humidity_2m']
    df['dew_point_approx'] = df['temperature_2m'] - ((100 - df['relative_humidity_2m']) / 5)
    df['pressure_delta_3h'] = df['surface_pressure'].diff(periods=3).fillna(0)
    df['humidity_delta_1h'] = df['relative_humidity_2m'].diff(periods=1).fillna(0)
    df['diurnal_position_score'] = df['hour_cos'] * df['temp_diurnal_range']
    
    return df

   