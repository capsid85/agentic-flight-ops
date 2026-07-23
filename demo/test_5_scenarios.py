import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, '../backend/models/delay_model.json')
encoders_path = os.path.join(base_dir, '../backend/models/encoders.pkl')

encoders = joblib.load(encoders_path)
reg = xgb.XGBRegressor()
reg.load_model(model_path)

features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 'MINUTE', 'TIME_OF_DAY', 'SIN_TIME', 'COS_TIME', 
            'IS_RUSH_HOUR', 'CONCURRENT_DELAYS_AT_ORIGIN', 'IS_HOLIDAY', 'IS_HOLIDAY_WEEK', 
            'ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND', 'DISTANCE', 'CRS_ELAPSED_TIME', 
            'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']

scenarios = [
    {
        "name": "1. The Early Bird (Clear Weather, No Congestion)",
        "desc": "An early morning flight from SFO to BOS with perfect weather.",
        "data": {'MONTH': 7, 'DAY': 23, 'DAY_OF_WEEK': 4, 'HOUR': 7, 'MINUTE': 10, 'TIME_OF_DAY': 430,
                 'SIN_TIME': np.sin(2*np.pi*430/1440), 'COS_TIME': np.cos(2*np.pi*430/1440), 
                 'IS_RUSH_HOUR': 1, 'CONCURRENT_DELAYS_AT_ORIGIN': 0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 0, 
                 'ORIGIN_PRECIP': 0.0, 'ORIGIN_WIND': 5.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 5.0, 
                 'DISTANCE': 2704, 'CRS_ELAPSED_TIME': 335, 'AIRLINE': 'DL', 'ORIGIN_AIRPORT': 'SFO', 
                 'DESTINATION_AIRPORT': 'BOS', 'FLIGHT_NUMBER': '400', 'ROUTE': 'SFO_BOS'}
    },
    {
        "name": "2. The Rush Hour Thunderstorm",
        "desc": "A late afternoon flight from JFK to LAX during a severe rainstorm and high airport congestion.",
        "data": {'MONTH': 7, 'DAY': 23, 'DAY_OF_WEEK': 4, 'HOUR': 17, 'MINUTE': 30, 'TIME_OF_DAY': 1050,
                 'SIN_TIME': np.sin(2*np.pi*1050/1440), 'COS_TIME': np.cos(2*np.pi*1050/1440), 
                 'IS_RUSH_HOUR': 1, 'CONCURRENT_DELAYS_AT_ORIGIN': 45.0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 0, 
                 'ORIGIN_PRECIP': 1.5, 'ORIGIN_WIND': 30.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 5.0, 
                 'DISTANCE': 2475, 'CRS_ELAPSED_TIME': 390, 'AIRLINE': 'AA', 'ORIGIN_AIRPORT': 'JFK', 
                 'DESTINATION_AIRPORT': 'LAX', 'FLIGHT_NUMBER': '100', 'ROUTE': 'JFK_LAX'}
    },
    {
        "name": "3. The Perfect Mid-Day Cruise",
        "desc": "A 2:00 PM flight out of Denver, well outside rush hour, with zero weather issues.",
        "data": {'MONTH': 7, 'DAY': 23, 'DAY_OF_WEEK': 4, 'HOUR': 14, 'MINUTE': 0, 'TIME_OF_DAY': 840,
                 'SIN_TIME': np.sin(2*np.pi*840/1440), 'COS_TIME': np.cos(2*np.pi*840/1440), 
                 'IS_RUSH_HOUR': 0, 'CONCURRENT_DELAYS_AT_ORIGIN': 5.0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 0, 
                 'ORIGIN_PRECIP': 0.0, 'ORIGIN_WIND': 5.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 5.0, 
                 'DISTANCE': 888, 'CRS_ELAPSED_TIME': 150, 'AIRLINE': 'UA', 'ORIGIN_AIRPORT': 'DEN', 
                 'DESTINATION_AIRPORT': 'ORD', 'FLIGHT_NUMBER': '200', 'ROUTE': 'DEN_ORD'}
    },
    {
        "name": "4. The Holiday Cascade",
        "desc": "An evening flight out of Chicago during a holiday week. Air traffic control is backed up.",
        "data": {'MONTH': 12, 'DAY': 23, 'DAY_OF_WEEK': 3, 'HOUR': 19, 'MINUTE': 15, 'TIME_OF_DAY': 1155,
                 'SIN_TIME': np.sin(2*np.pi*1155/1440), 'COS_TIME': np.cos(2*np.pi*1155/1440), 
                 'IS_RUSH_HOUR': 1, 'CONCURRENT_DELAYS_AT_ORIGIN': 60.0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 1, 
                 'ORIGIN_PRECIP': 0.0, 'ORIGIN_WIND': 10.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 5.0, 
                 'DISTANCE': 990, 'CRS_ELAPSED_TIME': 160, 'AIRLINE': 'WN', 'ORIGIN_AIRPORT': 'MDW', 
                 'DESTINATION_AIRPORT': 'MCO', 'FLIGHT_NUMBER': '500', 'ROUTE': 'MDW_MCO'}
    },
    {
        "name": "5. The Windy Destination",
        "desc": "Morning flight from Boston to San Francisco. Clear in Boston, but severe wind shear at SFO.",
        "data": {'MONTH': 7, 'DAY': 23, 'DAY_OF_WEEK': 4, 'HOUR': 9, 'MINUTE': 0, 'TIME_OF_DAY': 540,
                 'SIN_TIME': np.sin(2*np.pi*540/1440), 'COS_TIME': np.cos(2*np.pi*540/1440), 
                 'IS_RUSH_HOUR': 1, 'CONCURRENT_DELAYS_AT_ORIGIN': 10.0, 'IS_HOLIDAY': 0, 'IS_HOLIDAY_WEEK': 0, 
                 'ORIGIN_PRECIP': 0.0, 'ORIGIN_WIND': 5.0, 'DEST_PRECIP': 0.0, 'DEST_WIND': 35.0, 
                 'DISTANCE': 2704, 'CRS_ELAPSED_TIME': 390, 'AIRLINE': 'B6', 'ORIGIN_AIRPORT': 'BOS', 
                 'DESTINATION_AIRPORT': 'SFO', 'FLIGHT_NUMBER': '300', 'ROUTE': 'BOS_SFO'}
    }
]

print("--- 5 REAL-WORLD FLIGHT PREDICTION SCENARIOS ---\n")

for s in scenarios:
    df = pd.DataFrame([s['data']])[features]
    
    # Target encoding
    for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']:
        mapping = encoders.get(col, {})
        gm = mapping.get('_GLOBAL_MEAN_', 10.0)
        df[col] = df[col].map(mapping).fillna(gm)
        
    pred = reg.predict(df)[0]
    
    print(f"{s['name']}")
    print(f"Description: {s['desc']}")
    print(f"Details: {s['data']['AIRLINE']} {s['data']['FLIGHT_NUMBER']} | {s['data']['ORIGIN_AIRPORT']} -> {s['data']['DESTINATION_AIRPORT']} | {s['data']['HOUR']:02d}:{s['data']['MINUTE']:02d}")
    print(f"--> PREDICTED DELAY: {pred:.1f} minutes\n")
