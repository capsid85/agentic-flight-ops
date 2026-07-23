import pandas as pd
import os
import time
import requests
import airportsdata

def get_real_weather_data():
    base_dir = os.path.dirname(__file__)
    flights_path = os.path.join(base_dir, 'flights_sample_3m.csv')
    out_path = os.path.join(base_dir, 'weather_2023.csv')
    progress_file = os.path.join(base_dir, 'weather_progress.csv')
    
    print("Loading unique airports from flights data...")
    df = pd.read_csv(flights_path, usecols=['ORIGIN', 'DEST'])
    unique_airports = list(set(df['ORIGIN'].unique()).union(set(df['DEST'].unique())))
    print(f"Found {len(unique_airports)} unique airports.")
    
    airports = airportsdata.load('IATA')
    valid_airports = sorted([iata for iata in unique_airports if iata in airports])
    print(f"Fetching real 2023 hourly weather for {len(valid_airports)} valid airports from Open-Meteo REST API...")
    
    # Checkpoint loading
    completed_airports = set()
    if os.path.exists(progress_file):
        progress_df = pd.read_csv(progress_file, usecols=['AIRPORT'])
        completed_airports = set(progress_df['AIRPORT'].unique())
        print(f"Resuming from checkpoint: {len(completed_airports)} airports already downloaded.")
    
    session = requests.Session()
    
    for i, iata in enumerate(valid_airports):
        if iata in completed_airports:
            continue
            
        print(f"Processing {i+1}/{len(valid_airports)}: {iata}...")
        lat = airports[iata]['lat']
        lon = airports[iata]['lon']
        
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date=2023-12-31&hourly=precipitation,wind_speed_10m&timezone=UTC"
        
        while True:
            try:
                res = session.get(url, timeout=15)
                if res.status_code == 429:
                    print("Hit Open-Meteo Rate Limit (429). Sleeping for 65 seconds...")
                    time.sleep(65)
                    continue
                
                if res.status_code == 200:
                    data = res.json()
                    hourly = data.get("hourly", {})
                    times = hourly.get("time", [])
                    precip = hourly.get("precipitation", [])
                    wind = hourly.get("wind_speed_10m", [])
                    
                    if not times:
                        print(f"  Warning: No hourly data for {iata}")
                        break
                        
                    weather_df = pd.DataFrame({
                        "time": pd.to_datetime(times)
                    })
                    weather_df['AIRPORT'] = iata
                    weather_df['YEAR'] = weather_df['time'].dt.year
                    weather_df['MONTH'] = weather_df['time'].dt.month
                    weather_df['DAY'] = weather_df['time'].dt.day
                    weather_df['HOUR'] = weather_df['time'].dt.hour
                    
                    # Fill missing with 0 for precipitation and average for wind
                    weather_df['precipitation'] = pd.Series(precip).fillna(0.0)
                    weather_df['wind_speed_10m'] = pd.Series(wind).fillna(10.0)
                    
                    # Append to checkpoint file dynamically
                    header = not os.path.exists(progress_file)
                    weather_df[['AIRPORT', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'precipitation', 'wind_speed_10m']].to_csv(progress_file, mode='a', index=False, header=header)
                    
                    # Safety sleep between successful requests
                    time.sleep(2)
                    break
                else:
                    print(f"  Error HTTP {res.status_code} for {iata}: {res.text}")
                    break
            except Exception as e:
                print(f"  Network error for {iata}: {e}. Retrying in 10s...")
                time.sleep(10)
    
    print("All airports processed! Verifying and finalizing dataset...")
    if os.path.exists(progress_file):
        final_df = pd.read_csv(progress_file)
        final_df.to_csv(out_path, index=False)
        print(f"Real weather dataset complete! Shape: {final_df.shape}")
        print(f"Saved to {out_path}")
    else:
        print("Failed to download any data.")

if __name__ == "__main__":
    get_real_weather_data()
