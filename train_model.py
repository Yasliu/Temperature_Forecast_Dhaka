import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import optuna
from sklearn.model_selection import TimeSeriesSplit
import json
import joblib


# =================================================
# DATA PRE PROCESSING
# =================================================

# Fetching the data from API (open-meteo)
def fetch_weather_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame: 
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation&timezone=auto"
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Connection successful")
    else:
        print(f"Failed with status code: {response.status_code}")
    
    
    raw_data = response.json()
    
    hourly_dictionary = raw_data['hourly']
    df = pd.DataFrame(hourly_dictionary)

    return df

df = fetch_weather_data(23.8667, 90.46, "2018-01-01", "2023-12-31")

# Cyclical Encoding
def encode_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Converting time to date_time format
    df['time'] = pd.to_datetime(df['time'])
    
    df['month'] = df['time'].dt.month
    df['hour'] = df['time'].dt.hour
    
    # Math for month
    df['month_sin'] = np.sin( 2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos( 2 * np.pi * df['month'] / 12)
    
    # Math for Hour
    df['hour_sin'] = np.sin( 2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos( 2 * np.pi * df['hour'] / 24)
    
    return df
df = encode_cyclical_features(df)

# Sorting the data based on time

def sort_time_series_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['time'])
    
    # sorting based on time
    df = df.sort_values(by='time', ascending=True)
    
    # Resetting index
    df = df.reset_index(drop=True)
    
    return df
df = sort_time_series_data(df)

# ==================================================
# FEATURE BUILDING
# ==================================================

def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    target_columns = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']

    for col in target_columns:
        df[f"{col}_lag_24h"] = df[f'{col}'].shift(24)
        
        df[f"{col}_rolling_avg"] = (
            df[f"{col}"].rolling(window=168, closed="left").mean()
        )
        
    df['temp_delta_1h'] = df['temperature_2m'].diff(periods=1)
    
    # Calculation of Temperature volatility over last 24H
    df['temp_volatility_24h'] = df['temperature_2m'].rolling(window=24, closed='left').std()
    
    rolling_max = df['temperature_2m'].rolling(window=24, closed='left').max()
    rolling_min = df['temperature_2m'].rolling(window=24, closed='left').min()
    df['temp_diurnal_range'] = rolling_max - rolling_min
    
    df['Atm_moisture_flux'] = df['wind_speed_10m'] * df['relative_humidity_2m']
    df['Evaporative_potential_index'] = df['wind_speed_10m'] / df['relative_humidity_2m']

    # 1. Physical Approximations
    df['dew_point_approx'] = df['temperature_2m'] - ((100 - df['relative_humidity_2m']) / 5)

    # 2. Advanced Multi-Column Interactions
    # df['moisture_flux_rolling_avg_24h'] = df['Atm_moisture_flux'].rolling(window=24, min_periods=1, closed='left').mean()
    # df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']
    # df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']

    # 3. Microclimate & Volatility Trends
    df['pressure_delta_3h'] = df['surface_pressure'].diff(periods=3).fillna(0)
    df['humidity_delta_1h'] = df['relative_humidity_2m'].diff(periods=1).fillna(0)
    # df['temp_departure_from_rolling'] = df['temperature_2m'] - df['temperature_2m_rolling_avg']

    # 4. Cyclical Diurnal Anchor
    df['diurnal_position_score'] = df['hour_cos'] * df['temp_diurnal_range']
   
    return df

df = build_weather_features(df)

def clean_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """We will be surgically removing NaNs from specific columns"""
    longest_window_col = "temperature_2m_rolling_avg"
    
    df_clean = df.dropna(subset=[longest_window_col]).copy()
    
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean
df_clean = clean_data_pipeline(df)

def target_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """ Here we will create a 24 hour window for the future prediction by shift(-24) """
    df['target_temperature_T24'] = df['temperature_2m'].shift(-24)
    
    # This creates NaNs at the bottom of the dataset - we remove those NaNs.
    df_clean = df.dropna(subset=['target_temperature_T24']).copy()
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean
    
df_clean = target_temperature(df_clean)

# Data Splitting
def data_splitting(df):
    train_df = df_clean[df_clean['time'] < '2023-01-01 00:00:00']
    test_df = df_clean[df_clean['time'] >= '2022-01-01 00:00:00']
    
    X_train = train_df.drop(columns=['time', 'target_temperature_T24'])
    y_train = train_df['target_temperature_T24']
    
    X_test = test_df.drop(columns=['time', 'target_temperature_T24'])
    y_test = test_df['target_temperature_T24']

    return X_train, y_train, X_test, y_test    

X_train, y_train, X_test, y_test = data_splitting(df_clean)

# print(df_clean.head(10))    
# print(df_clean.info())

# =================================================
# MAKING THE MODEL
# =================================================

def run_baseline_model(X_train, y_train, X_test, y_test) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"Base line model MAE score: {mae}")
    
    return model, mae

# ================================================
# OPTUNA - HYPERPARAMETER FINDING
# ================================================

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.75, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 1.0),
        
        'random_state': 42,
        'n_jobs': -1
    }
    
    tcsv = TimeSeriesSplit(n_splits=3)
    score = []
    for train_idx, val_idx in tcsv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        
        y_tr = y_train.iloc[train_idx]
        y_va = y_train.iloc[val_idx]
        
        model = XGBRegressor(**params)        
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_va)
        score_mae = mean_absolute_error(y_va, preds)
        
        score.append(score_mae)
        
    return sum(score) / len(score) 

# ================================================
# LET IT RIP
# ================================================

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_splitting(df_clean)
    
    # Now we run the baseline model (for comparison)
    print("Evaluating the baseline...")
    baseline_model, baseline_mae = run_baseline_model(X_train, y_train, X_test, y_test)
    
    RUN_OPTUNA = False
    
    if RUN_OPTUNA:
        print("Initializing Optuna Hyperparameter Search...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        
        print(f"Best Score: {study.best_value}")
        print(f"Best params: {best_params}")
        
        with open('best_xgb_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # with open('best_xgb_params.json', 'r') as f:
        #     json.load(f)
        
        print("Evaluating final tuned model")
        tuned_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        tuned_model.fit(X_train, y_train)
    
        tuned_preds = tuned_model.predict(X_test)
        tuned_mae = mean_absolute_error(y_test, tuned_preds)
    
        print(f"Final Tuned MAE:{tuned_mae:.5f}")
        print(f"Total imporvement: {baseline_mae - tuned_mae:.5f}")
        
    joblib.dump(baseline_model, 'xgboost_dhaka_weather_v1.pkl')
    
    #Saving the original columns
    expected_columns = X_train.columns.tolist()
    with open('model_features.json', 'w') as f:
        json.dump(expected_columns, f, indent=4)
        
    print("Production baseline and artifacts sucessfully written to disk.")
        