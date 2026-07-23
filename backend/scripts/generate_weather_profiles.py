import os
import pandas as pd
import json

def main():
    print("Loading BTS dataset for Weather Profile extraction...")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, '../../data/bts_sample.csv'))
    
    # Read the dataset
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter for flights that actually experienced a weather delay
    weather_delayed_flights = df[(df['WEATHER_DELAY'] > 0)]
    print(f"Found {len(weather_delayed_flights)} flights with recorded weather delays.")
    
    # Group by Origin Airport and calculate the mean weather delay
    profiles = weather_delayed_flights.groupby('ORIGIN_AIRPORT')['WEATHER_DELAY'].mean().to_dict()
    
    # Calculate a global fallback average for airports not in the list
    global_average = weather_delayed_flights['WEATHER_DELAY'].mean()
    if pd.isna(global_average):
        global_average = 45.0
        
    profiles['_GLOBAL_MEAN_'] = global_average
    
    # Save the profiles to models directory
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    out_path = os.path.join(base_dir, 'models', 'weather_profiles.json')
    
    with open(out_path, 'w') as f:
        json.dump(profiles, f, indent=4)
        
    print(f"Weather Profiles saved to {out_path}")
    print(f"Global Average Weather Delay: {global_average:.1f} mins")

if __name__ == "__main__":
    main()
