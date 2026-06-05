import sqlite3

with sqlite3.connect('version_log.db') as conn:
    cursor = conn.cursor()
    
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS model_logs (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       model_name TEXT,
                       feature_set TEXT UNIQUE,
                       hyperparameters TEXT,
                       mae_score REAL
                   )
                   """)
    
    experiment_history = [
        ("XGBRegressor", "Raw Baseline (5 basic features)", "{}", 0.9359),
        ("XGBRegressor", "Optuna Attempt 1 (Underfit)", '{"n_estimators": 185, "max_depth": 3, "learning_rate": 0.04}', 1.0335),
        ("XGBRegressor", "Physics Baseline v1 (Volatility, Diurnal Range, Flux)", "{}", 0.9019),
        ("XGBRegressor", "Oversaturated Physics (10+ external features)", "{}", 0.9121),
        ("XGBRegressor", "Dynamic Feature Selection (0.005 threshold)", "{}", 0.9977),
        ("XGBRegressor", "Purged Physics (Removed multicollinearity)", "{}", 0.9045)
    ]
    
    cursor.executemany("""
                       INSERT OR IGNORE INTO model_logs (model_name, feature_set, hyperparameters, mae_score)
                       VALUES (?, ?, ?, ?)
                       """, experiment_history)

    conn.commit()
    