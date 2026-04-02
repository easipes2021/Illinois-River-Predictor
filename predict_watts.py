import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib # You may need to run 'pip install joblib'

def train_refining_model():
    # 1. Load data
    df = pd.read_csv('master_training_data.csv', index_col=0, parse_dates=True).dropna(subset=['watts_ok_height'])
    
    # 2. Target: Predict Watts height 6 hours into the future
    df['target_watts_6h'] = df['watts_ok_height'].shift(-6)
    
    # 3. Features: The "Why" behind the "What"
    features = [
        'savoy_height', 'osage_creek_flow', 'hwy_59_height', 
        'precip_expected_mm', 'soil_saturation_index', 
        'savoy_trend', 'seasonal_cycle', 'lake_headroom'
    ]
    
    df_clean = df.dropna(subset=['target_watts_6h'] + features)
    X = df_clean[features]
    y = df_clean['target_watts_6h']

    # 4. Train the Model
    # We use more estimators (trees) to capture the "flashiness"
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # 5. Save for "Continual Refining"
    joblib.dump(model, 'illinois_river_model.pkl')
    
    # 6. Self-Correction Logic
    predictions = model.predict(X)
    error = mean_absolute_error(y, predictions)
    
    print(f"Model trained! Average error: {error:.2f} feet")
    print("Top factors influencing the Illinois River at Watts:")
    for feat, importance in zip(features, model.feature_importances_):
        print(f"- {feat}: {importance:.2%}")

if __name__ == "__main__":
    train_refining_model()