import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

def retrain_models():
    # Load your permanent history
    df = pd.read_csv('master_training_data.csv', index_col=0)
    
    # 1. Define the full list of what you want to predict
    locations = {
        'hwy_16_flow': 'Hwy 16',
        'hwy_59_flow_est': 'Hwy 59',
        'lake_francis_height': 'Lake Francis',
        'watts_ok_flow': 'Watts Bridge'
    }

    # Define features
    features = [
        'savoy_height', 'osage_creek_flow', 'hwy_59_height',
        'savoy_height_3h_ago', 'savoy_height_6h_ago', 
        'osage_creek_flow_3h_ago', 'osage_creek_flow_6h_ago',
        'precip_fayetteville', 'precip_springdale', 
        'precip_bentonville', 'precip_siloam',
        'precip_fayetteville_saturation', 'seasonal_cycle', 'lake_headroom'
    ]
    
    # Loop through ALL targets defined in the locations dictionary
    for target in locations.keys():
        print(f"--- Training model for {target} ---")
        
        # Check if the column exists in the data
        if target not in df.columns:
            print(f"⚠️ Warning: Column '{target}' not found in CSV. Skipping.")
            continue
            
        # Create a clean subset for this specific model
        model_df = df.dropna(subset=features + [target])
        
        model = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_depth=10)
        model.fit(model_df[features], model_df[target])
        
        # Save the model
        joblib.dump(model, f'model_{target}.pkl')
        print(f"✅ Saved model_{target}.pkl")

        # DEBUG: Show importance and sensitivity for each model
        print(f"DEBUG: Feature importance for {target}: {dict(zip(features, model.feature_importances_))}")
        
        test_low = pd.DataFrame([[1.0] * len(features)], columns=features)
        test_high = pd.DataFrame([[5.0] * len(features)], columns=features)
        print(f"DEBUG: Prediction at 1ft/val: {model.predict(test_low)[0]:.2f}")
        print(f"DEBUG: Prediction at 5ft/val: {model.predict(test_high)[0]:.2f}")

if __name__ == "__main__":
    retrain_models()