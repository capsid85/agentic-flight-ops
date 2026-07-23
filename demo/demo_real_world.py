import requests
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import json
from datetime import datetime

print("=======================================================")
print("      LIVE PRODUCTION SYSTEM DEMONSTRATION      ")
print("=======================================================\n")

# 1. LIVE WEATHER INGESTION (NOAA METAR API)
print("[STEP 1] Fetching Live FAA Data for JFK...")
url = "https://aviationweather.gov/api/data/metar?ids=KJFK&format=json"
try:
    res = requests.get(url, timeout=3)
    raw_metar = res.json()[0]['rawOb']
    print(f"--> LIVE METAR RECEIVED: {raw_metar}")
except Exception as e:
    print(f"--> Failed to fetch METAR: {e}")
    raw_metar = "KJFK AUTO 10SM CLR"

# 2. XGBOOST PREDICTION MODEL
print("\n[STEP 2] Running XGBoost Inference...")
features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 'MINUTE', 'TIME_OF_DAY', 'SIN_TIME', 'COS_TIME', 
            'IS_RUSH_HOUR', 'CONCURRENT_DELAYS_AT_ORIGIN', 'IS_HOLIDAY', 'IS_HOLIDAY_WEEK', 
            'ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND', 'DISTANCE', 'CRS_ELAPSED_TIME', 
            'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']

import os

encoders = joblib.load(os.path.join(os.path.dirname(__file__), '../backend/models/encoders.pkl'))
reg = xgb.XGBRegressor()
reg.load_model(os.path.join(os.path.dirname(__file__), '../backend/models/delay_model.json'))

# Simulate a flight right now
now = datetime.now()
flight_data = {
    'MONTH': now.month, 'DAY': now.day, 'DAY_OF_WEEK': now.weekday() + 1,
    'HOUR': now.hour, 'MINUTE': now.minute, 'TIME_OF_DAY': now.hour * 60 + now.minute,
    'SIN_TIME': np.sin(2*np.pi*(now.hour * 60 + now.minute)/1440), 
    'COS_TIME': np.cos(2*np.pi*(now.hour * 60 + now.minute)/1440),
    'IS_RUSH_HOUR': 1 if (7 <= now.hour <= 9 or 16 <= now.hour <= 19) else 0,
    'CONCURRENT_DELAYS_AT_ORIGIN': 15.0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 0,
    'ORIGIN_PRECIP': 0.0, 'ORIGIN_WIND': 10.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 5.0,
    'DISTANCE': 3451, 'CRS_ELAPSED_TIME': 420, 'AIRLINE': 'DL', 'ORIGIN_AIRPORT': 'JFK',
    'DESTINATION_AIRPORT': 'LHR', 'FLIGHT_NUMBER': '1', 'ROUTE': 'JFK_LHR'
}

df = pd.DataFrame([flight_data])[features]
for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']:
    mapping = encoders.get(col, {})
    df[col] = df[col].map(mapping).fillna(mapping.get('_GLOBAL_MEAN_', 10.0))
    
prediction = reg.predict(df)[0]
print(f"--> PREDICTED DELAY: {prediction:.1f} minutes")

# 3. WEBSOCKET STREAM PAYLOAD (KAFKA SIMULATION)
print("\n[STEP 3] Pushing to WebSocket Stream (Frontend Broadcast)...")
payload = {
    "type": "STREAM_UPDATE",
    "timestamp": now.isoformat(),
    "updates": [
        {
            "flight_id": "DL1",
            "route": "JFK -> LHR",
            "predicted_delay": round(float(prediction), 1),
            "status": "DELAYED" if prediction > 15 else "ON TIME",
            "metar": raw_metar
        }
    ]
}
print(json.dumps(payload, indent=2))
print("\n=======================================================")
