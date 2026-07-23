import os
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import holidays
import requests
import airportsdata

from state import UnifiedEventState, RiskState

CLASSIFIER = None
MODEL = None
ENCODERS = None

def load_ml_assets():
    global CLASSIFIER, MODEL, ENCODERS
    if MODEL is None or ENCODERS is None:
        base_dir = os.path.dirname(__file__)
        clf_path = os.path.abspath(os.path.join(base_dir, 'models/classifier_model.json'))
        model_path = os.path.abspath(os.path.join(base_dir, 'models/delay_model.json'))
        encoders_path = os.path.abspath(os.path.join(base_dir, 'models/encoders.pkl'))
        
        if os.path.exists(model_path) and os.path.exists(encoders_path):
            MODEL = xgb.XGBRegressor()
            MODEL.load_model(model_path)
            ENCODERS = joblib.load(encoders_path)
            if os.path.exists(clf_path):
                CLASSIFIER = xgb.XGBClassifier()
                CLASSIFIER.load_model(clf_path)
            print("Risk Agent: ML Models Loaded Successfully")
        else:
            print("Risk Agent: ML Model files not found. Using fallback heuristics.")

def get_live_weather(airport_code):
    try:
        airports = airportsdata.load('IATA')
        if airport_code not in airports:
            return 0.0, 0.0
        
        lat = airports[airport_code]['lat']
        lon = airports[airport_code]['lon']
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=precipitation,wind_speed_10m"
        response = requests.get(url, timeout=2).json()
        precip = response['current'].get('precipitation', 0.0)
        wind = response['current'].get('wind_speed_10m', 0.0)
        return float(precip), float(wind)
    except Exception:
        return 0.0, 0.0

def risk_assessment_node(state: UnifiedEventState) -> UnifiedEventState:
    """
    Risk Assessment Agent: Reads flight and weather state, predicts delay using XGBoost.
    """
    global MODEL, ENCODERS
    
    flight = state.get("flight", {})
    weather = state.get("weather", {})
    timestamp = state.get("timestamp", "")
    
    # Try loading ML model
    load_ml_assets()
    
    delay = 0
    confidence = 0.90
    
    if MODEL and ENCODERS:
        try:
            # Parse Date
            if timestamp:
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d")
                month = dt.month
                day = dt.day
                day_of_week = dt.weekday() + 1 # Monday is 1 in pandas, weekday() is 0-6
            else:
                dt = datetime.datetime(2023, 1, 1)
                month = 1
                day = 1
                day_of_week = 1
                
            # Parse Hour and Minute
            sched = str(flight.get("scheduled_departure", "0"))
            try:
                # Clean up float string if necessary
                sched_val = int(float(sched))
                hour = sched_val // 100
                minute = sched_val % 100
            except ValueError:
                hour = 12
                minute = 0
                
            time_of_day = hour * 60 + minute
            sin_time = np.sin(2 * np.pi * time_of_day / 1440)
            cos_time = np.cos(2 * np.pi * time_of_day / 1440)
                
            airline_iata = str(flight.get("carrier", "UNK")).strip()
            # Map IATA to full name for the ML model
            airline_map = {
                "DL": "Delta Air Lines Inc.",
                "AA": "American Airlines Inc.",
                "UA": "United Air Lines Inc.",
                "WN": "Southwest Airlines Co.",
                "B6": "JetBlue Airways",
                "AS": "Alaska Airlines Inc.",
                "NK": "Spirit Air Lines",
                "F9": "Frontier Airlines Inc.",
                "HA": "Hawaiian Airlines Inc.",
                "MQ": "Envoy Air",
                "YX": "Republic Airline",
                "OO": "SkyWest Airlines Inc.",
                "OH": "PSA Airlines Inc.",
                "YV": "Mesa Airlines Inc.",
                "9E": "Endeavor Air Inc.",
                "QX": "Horizon Air",
                "G4": "Allegiant Air"
            }
            airline = airline_map.get(airline_iata, airline_iata)
            
            origin = str(flight.get("origin", "UNK")).strip()
            dest = str(flight.get("destination", "UNK")).strip()
            
            flight_id = str(flight.get("flight_id", "000")).strip()
            flight_number = flight_id.replace(airline, "").strip()
            def safe_encode(encoder_dict, val):
                return encoder_dict.get(val, encoder_dict.get('_GLOBAL_MEAN_', 0.0))
                
            route_str = f"{origin}_{dest}"
            is_rush = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
            
            us_holidays = holidays.US(years=[dt.year, dt.year - 1, dt.year + 1])
            is_holiday = 1 if dt.date() in us_holidays else 0
            holiday_dates = list(us_holidays.keys())
            if holiday_dates:
                days_to_nearest = min([abs((dt.date() - h).days) for h in holiday_dates])
            else:
                days_to_nearest = 999
            is_holiday_week = 1 if days_to_nearest <= 3 else 0

            
            encoded_airline = safe_encode(ENCODERS.get('AIRLINE', {}), airline)
            encoded_origin = safe_encode(ENCODERS.get('ORIGIN_AIRPORT', {}), origin)
            encoded_dest = safe_encode(ENCODERS.get('DESTINATION_AIRPORT', {}), dest)
            encoded_flight_num = safe_encode(ENCODERS.get('FLIGHT_NUMBER', {}), flight_number)
            encoded_route = safe_encode(ENCODERS.get('ROUTE', {}), route_str)
            
            # Live Weather Fetch
            origin_precip, origin_wind = get_live_weather(origin)
            dest_precip, dest_wind = get_live_weather(dest)
            
            # Create feature vector
            feature_dict = {
                'MONTH': month,
                'DAY': day,
                'DAY_OF_WEEK': day_of_week,
                'HOUR': hour,
                'MINUTE': minute,
                'TIME_OF_DAY': time_of_day,
                'SIN_TIME': sin_time,
                'COS_TIME': cos_time,
                'IS_RUSH_HOUR': is_rush,
                'IS_HOLIDAY': is_holiday,
                'IS_HOLIDAY_WEEK': is_holiday_week,
                'ORIGIN_PRECIP': origin_precip,
                'ORIGIN_WIND': origin_wind,
                'DEST_PRECIP': dest_precip,
                'DEST_WIND': dest_wind,
                'DISTANCE': 500.0,
                'CRS_ELAPSED_TIME': 120.0,
                'AIRLINE': encoded_airline,
                'ORIGIN_AIRPORT': encoded_origin,
                'DESTINATION_AIRPORT': encoded_dest,
                'FLIGHT_NUMBER': encoded_flight_num,
                'ROUTE': encoded_route
            }
            
            if hasattr(MODEL, 'feature_names_in_'):
                # Align exact feature column order expected by XGBoost
                cols = list(MODEL.feature_names_in_)
                for c in cols:
                    if c not in feature_dict:
                        feature_dict[c] = 0.0
                features = pd.DataFrame([feature_dict])[cols]
            else:
                features = pd.DataFrame([feature_dict])
            
            # Inference baseline (Two-Stage Prediction)
            if CLASSIFIER is not None:
                prob_delay = CLASSIFIER.predict_proba(features)[0, 1]
                if prob_delay >= 0.38:
                    raw_pred = MODEL.predict(features)[0]
                    predicted = np.expm1(raw_pred)
                else:
                    predicted = 0.0
            else:
                predicted = MODEL.predict(features)[0]
                
            baseline = max(0, int(predicted))
            
            # Pure Data-Driven Weather Fusion
            dest_risk = weather.get("dest_risk", "Low")
            origin_risk = weather.get("origin_risk", "Low")
            
            weather_delay = 0
            # Load weather profiles
            base_dir = os.path.dirname(__file__)
            profile_path = os.path.join(base_dir, 'models/weather_profiles.json')
            try:
                with open(profile_path, 'r') as f:
                    profiles = json.load(f)
            except Exception:
                profiles = {'_GLOBAL_MEAN_': 30.0}
                
            glob_delay = profiles.get('_GLOBAL_MEAN_', 45.0)
            if dest_risk == "High":
                weather_delay = max(weather_delay, profiles.get(dest) or glob_delay, 60.0)
            elif dest_risk == "Moderate":
                weather_delay = max(weather_delay, (profiles.get(dest) or glob_delay) * 0.5, 30.0)
                
            if origin_risk == "High":
                weather_delay = max(weather_delay, profiles.get(origin) or glob_delay, 60.0)
            elif origin_risk == "Moderate":
                weather_delay = max(weather_delay, (profiles.get(origin) or glob_delay) * 0.5, 30.0)
                
            delay = max(baseline, int(weather_delay))
            
            # Live Airborne Landing Telemetry Fusion
            status_text = str(flight.get("status", ""))
            if "About to Land" in status_text:
                # Airborne flight approaching touchdown
                confidence = 0.94 # High confidence due to real-time radar proximity
                if dest_risk == "High":
                    delay = max(delay, 25) # Expect holding pattern buffer
            else:
                # Dynamic Confidence Score Calculation
                confidence = 0.85
                if airline not in ENCODERS['AIRLINE']:
                    confidence -= 0.05
                if origin not in ENCODERS['ORIGIN_AIRPORT']:
                    confidence -= 0.05
                if dest not in ENCODERS['DESTINATION_AIRPORT']:
                    confidence -= 0.05
                    
                if origin_risk == "High" or dest_risk == "High":
                    confidence -= 0.15
                elif origin_risk == "Low" and dest_risk == "Low":
                    confidence += 0.05
                    
            confidence = min(0.95, max(0.50, confidence))
            print(f"Risk Agent: Predicted {delay} mins (Confidence: {confidence * 100:.1f}%)")
            
        except Exception as e:
            print(f"Risk Agent: ML Inference Failed: {e}. Falling back to heuristic.")
            MODEL = None # Trigger fallback
            
    if not MODEL or not ENCODERS:
        # Fallback Heuristic
        origin_risk = weather.get("origin_risk", "Low")
        dest_risk = weather.get("dest_risk", "Low")
        
        if origin_risk == "High" or dest_risk == "High":
            delay = 90
            confidence = 0.70  # Lower base confidence for fallback heuristic with severe weather
        elif origin_risk == "Moderate" or dest_risk == "Moderate":
            delay = 30
            confidence = 0.60  # Lower base confidence for moderate weather fallback
        else:
            delay = 0
            confidence = 0.80  # Default fallback for calm weather
            
    accumulated_delay = flight.get("accumulated_delay", 0)
    delay += accumulated_delay
            
    state["risk"] = RiskState(
        predicted_delay_minutes=delay,
        confidence_score=confidence
    )
    
    return state
