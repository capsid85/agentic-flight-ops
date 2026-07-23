import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'Sprint2/models/delay_model.json')
encoders_path = os.path.join(base_dir, 'Sprint2/models/encoders.pkl')

encoders = joblib.load(encoders_path)
reg = xgb.XGBRegressor()
reg.load_model(model_path)

# Ensure all 23 features used in training are present in the exact order
features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 'MINUTE', 'TIME_OF_DAY', 'SIN_TIME', 'COS_TIME', 
            'IS_RUSH_HOUR', 'CONCURRENT_DELAYS_AT_ORIGIN', 'IS_HOLIDAY', 'IS_HOLIDAY_WEEK', 
            'ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND', 'DISTANCE', 'CRS_ELAPSED_TIME', 
            'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']

flight_data = {
    'MONTH': 7, 
    'DAY': 23, 
    'DAY_OF_WEEK': 4, 
    'HOUR': 7, 
    'MINUTE': 10, 
    'TIME_OF_DAY': 7 * 60 + 10, 
    'SIN_TIME': np.sin(2*np.pi*(7 * 60 + 10)/1440), 
    'COS_TIME': np.cos(2*np.pi*(7 * 60 + 10)/1440), 
    'IS_RUSH_HOUR': 1, 
    'CONCURRENT_DELAYS_AT_ORIGIN': 0, # Early morning, usually low congestion
    'IS_HOLIDAY': 0, 
    'IS_HOLIDAY_WEEK': 0, 
    'ORIGIN_PRECIP': 0.0, 
    'ORIGIN_WIND': 5.0, 
    'DEST_PRECIP': 0.0, 
    'DEST_WIND': 10.0, 
    'DISTANCE': 2704, 
    'CRS_ELAPSED_TIME': 335, 
    'AIRLINE': 'DL', 
    'ORIGIN_AIRPORT': 'SFO', 
    'DESTINATION_AIRPORT': 'BOS', 
    'FLIGHT_NUMBER': '400', 
    'ROUTE': 'SFO_BOS'
}

df = pd.DataFrame([flight_data])[features]

# Apply target encoding
for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']:
    mapping = encoders.get(col, {})
    gm = mapping.get('_GLOBAL_MEAN_', 10.0)
    df[col] = df[col].map(mapping).fillna(gm)

preds = reg.predict(df)
print(f"Live Flight Prediction:\nRoute: {flight_data['ORIGIN_AIRPORT']} -> {flight_data['DESTINATION_AIRPORT']}\nAirline: {flight_data['AIRLINE']} {flight_data['FLIGHT_NUMBER']}\nScheduled Departure: 07:10 AM\nWeather: Clear at Origin (SFO)\nPredicted Departure Delay: {preds[0]:.1f} minutes")
