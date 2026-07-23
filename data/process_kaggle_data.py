import pandas as pd
import os

def process_kaggle_data():
    base_dir = os.path.dirname(__file__)
    kaggle_path = os.path.join(base_dir, 'flights_sample_3m.csv')
    out_path = os.path.join(base_dir, 'bts_sample.csv')
    
    print("Loading 3M flights dataset... This might take a moment.")
    df = pd.read_csv(kaggle_path, low_memory=False)
    
    # The dataset spans 2019-2023. Filter for 2023 only to get modern data.
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df[df['FL_DATE'].dt.year == 2023]
    
    print(f"Filtered to 2023 flights. Total records: {len(df)}")
    
    # Map columns to match the old bts_sample format
    df['YEAR'] = df['FL_DATE'].dt.year
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY'] = df['FL_DATE'].dt.day
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1
    
    df = df.rename(columns={
        'FL_NUMBER': 'FLIGHT_NUMBER',
        'ORIGIN': 'ORIGIN_AIRPORT',
        'DEST': 'DESTINATION_AIRPORT',
        'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
        'DEP_DELAY': 'DEPARTURE_DELAY',
        'DELAY_DUE_WEATHER': 'WEATHER_DELAY'
    })
    
    # Keep only the necessary columns to save space
    cols_to_keep = [
        'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'FLIGHT_NUMBER', 
        'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 
        'DEPARTURE_DELAY', 'CANCELLED', 'WEATHER_DELAY'
    ]
    
    df = df[cols_to_keep]
    
    # Fill NAs in delays
    df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].fillna(0)
    
    # Sample down to 100,000 flights to keep it lightweight for the demo and fast for XGBoost
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
        
    print("Saving normalized 2023 dataset to bts_sample.csv...")
    df.to_csv(out_path, index=False)
    print("Done! You can now delete flights_sample_3m.csv to save disk space.")

if __name__ == "__main__":
    process_kaggle_data()
