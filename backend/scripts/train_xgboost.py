import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import holidays

def main():
    print("Loading Full Massive BTS dataset (3 Million Rows)...")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(os.path.join(base_dir, '../../data/flights_sample_3m.csv'))
    
    # Read full 3M rows dataset
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Initial shape: {df.shape}")
    
    # Clean and prepare target variable
    df = df.dropna(subset=['DEP_DELAY'])
    df = df[(df['DEP_DELAY'] >= -30) & (df['DEP_DELAY'] <= 720)]
    
    df = df.rename(columns={
        'DEP_DELAY': 'DEPARTURE_DELAY',
        'AIRLINE': 'AIRLINE',
        'ORIGIN': 'ORIGIN_AIRPORT',
        'DEST': 'DESTINATION_AIRPORT',
        'FL_NUMBER': 'FLIGHT_NUMBER',
        'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE'
    })
    
    print(f"Shape after dropping NaNs and outliers: {df.shape}")
    
    print("Engineering Temporal & Congestion Features...")
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df = df.dropna(subset=['FL_DATE'])
    
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY'] = df['FL_DATE'].dt.day
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1
    
    df['HOUR'] = (df['SCHEDULED_DEPARTURE'] // 100).fillna(0).astype(int)
    df['MINUTE'] = (df['SCHEDULED_DEPARTURE'] % 100).fillna(0).astype(int)
    df['TIME_OF_DAY'] = df['HOUR'] * 60 + df['MINUTE']
    df['SIN_TIME'] = np.sin(2 * np.pi * df['TIME_OF_DAY'] / 1440)
    df['COS_TIME'] = np.cos(2 * np.pi * df['TIME_OF_DAY'] / 1440)
    df['IS_RUSH_HOUR'] = df['HOUR'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    
    print("Engineering Systemic Congestion Feature (Rolling Delay Cascade)...")
    # Build proper datetime for sorting
    df['CRS_DEP_TIME_STR'] = df['SCHEDULED_DEPARTURE'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4).replace('2400', '0000')
    
    # Handle bad times gracefully (e.g., 'nan', '0nan')
    valid_time_mask = df['CRS_DEP_TIME_STR'].str.match(r'^\d{4}$')
    df.loc[~valid_time_mask, 'CRS_DEP_TIME_STR'] = '0000'
    
    df['SCHEDULED_DATETIME'] = pd.to_datetime(
        df['FL_DATE'].dt.strftime('%Y-%m-%d') + ' ' + 
        df['CRS_DEP_TIME_STR'].str[:2] + ':' + df['CRS_DEP_TIME_STR'].str[2:], 
        errors='coerce'
    )
    
    df = df.sort_values(by='SCHEDULED_DATETIME')
    df = df.set_index('SCHEDULED_DATETIME')
    
    df['CONCURRENT_DELAYS_AT_ORIGIN'] = df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].transform(
        lambda x: x.rolling('2h', closed='left').mean()
    )
    df['CONCURRENT_DELAYS_AT_ORIGIN'] = df['CONCURRENT_DELAYS_AT_ORIGIN'].fillna(0)
    df = df.reset_index(drop=True)
    
    print("Engineering Holiday Features...")
    us_holidays = holidays.US(years=range(2019, 2025))
    df['IS_HOLIDAY'] = df['FL_DATE'].dt.date.isin(us_holidays).astype(int)
    
    holiday_dates = pd.to_datetime(list(us_holidays.keys())).sort_values()
    df_holidays = pd.DataFrame({'HOLIDAY_DATE': holiday_dates})
    
    df_sorted = df.sort_values('FL_DATE')
    df_sorted = pd.merge_asof(df_sorted, df_holidays, left_on='FL_DATE', right_on='HOLIDAY_DATE', direction='backward')
    df_sorted['DAYS_SINCE'] = (df_sorted['FL_DATE'] - df_sorted['HOLIDAY_DATE']).dt.days
    df_sorted = df_sorted.drop(columns=['HOLIDAY_DATE'])
    
    df_sorted = pd.merge_asof(df_sorted, df_holidays, left_on='FL_DATE', right_on='HOLIDAY_DATE', direction='forward')
    df_sorted['DAYS_TO'] = (df_sorted['HOLIDAY_DATE'] - df_sorted['FL_DATE']).dt.days
    df_sorted = df_sorted.drop(columns=['HOLIDAY_DATE'])
    
    df_sorted['DAYS_TO_NEAREST_HOLIDAY'] = df_sorted[['DAYS_SINCE', 'DAYS_TO']].min(axis=1)
    df_sorted['DAYS_TO_NEAREST_HOLIDAY'] = df_sorted['DAYS_TO_NEAREST_HOLIDAY'].fillna(999)
    df_sorted['IS_HOLIDAY_WEEK'] = (df_sorted['DAYS_TO_NEAREST_HOLIDAY'] <= 3).astype(int)
    
    df = df_sorted.drop(columns=['DAYS_SINCE', 'DAYS_TO', 'DAYS_TO_NEAREST_HOLIDAY'])

    print("Engineering Weather Features...")
    weather_path = os.path.abspath(os.path.join(base_dir, '../../data/weather_2023.csv'))
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path)
        
        df['YEAR'] = df['FL_DATE'].dt.year
        
        # Merge Origin Weather
        weather_df_origin = weather_df.rename(columns={
            'AIRPORT': 'ORIGIN_AIRPORT',
            'precipitation': 'ORIGIN_PRECIP',
            'wind_speed_10m': 'ORIGIN_WIND'
        })
        df = df.merge(weather_df_origin, on=['ORIGIN_AIRPORT', 'YEAR', 'MONTH', 'DAY', 'HOUR'], how='left')
        
        # Merge Dest Weather
        weather_df_dest = weather_df.rename(columns={
            'AIRPORT': 'DESTINATION_AIRPORT',
            'precipitation': 'DEST_PRECIP',
            'wind_speed_10m': 'DEST_WIND'
        })
        df = df.merge(weather_df_dest, on=['DESTINATION_AIRPORT', 'YEAR', 'MONTH', 'DAY', 'HOUR'], how='left')
        
        # Fill NAs
        for col in ['ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND']:
            df[col] = df[col].fillna(0)
    else:
        print("Warning: weather_2023.csv not found.")
        for col in ['ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND']:
            df[col] = 0.0

    df['AIRLINE'] = df['AIRLINE'].astype(str).str.strip()
    df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str).str.strip()
    df['DESTINATION_AIRPORT'] = df['DESTINATION_AIRPORT'].astype(str).str.strip()
    df['FLIGHT_NUMBER'] = df['FLIGHT_NUMBER'].astype(str).str.strip()
    df['ROUTE'] = df['ORIGIN_AIRPORT'] + "_" + df['DESTINATION_AIRPORT']
    
    if 'DISTANCE' in df.columns:
        df['DISTANCE'] = df['DISTANCE'].fillna(df['DISTANCE'].median())
    else:
        df['DISTANCE'] = 500
        
    if 'CRS_ELAPSED_TIME' in df.columns:
        df['CRS_ELAPSED_TIME'] = df['CRS_ELAPSED_TIME'].fillna(df['CRS_ELAPSED_TIME'].median())
    else:
        df['CRS_ELAPSED_TIME'] = 120

    features = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'HOUR', 'MINUTE', 'TIME_OF_DAY', 'SIN_TIME', 'COS_TIME', 
                'IS_RUSH_HOUR', 'CONCURRENT_DELAYS_AT_ORIGIN', 'IS_HOLIDAY', 'IS_HOLIDAY_WEEK', 
                'ORIGIN_PRECIP', 'ORIGIN_WIND', 'DEST_PRECIP', 'DEST_WIND', 'DISTANCE', 'CRS_ELAPSED_TIME', 
                'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']
    
    X = df[features]
    y = df['DEPARTURE_DELAY']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Target encoding
    encoders = {}
    cat_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'FLIGHT_NUMBER', 'ROUTE']
    global_mean = y_train.mean()
    m_smooth = 10.0
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    for col in cat_cols:
        stats = y_train.groupby(X_train[col]).agg(['count', 'mean'])
        smoothed = ((stats['count'] * stats['mean']) + (m_smooth * global_mean)) / (stats['count'] + m_smooth)
        mapping = smoothed.to_dict()
        mapping['_GLOBAL_MEAN_'] = global_mean
        
        X_train[col] = X_train[col].map(mapping).fillna(global_mean)
        X_test[col] = X_test[col].map(mapping).fillna(global_mean)
        encoders[col] = mapping

    print("\n--- Training Single XGBoost Regressor ---")
    reg = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='reg:pseudohubererror',
        random_state=42,
        n_jobs=-1
    )
    # Train on the full dataset
    reg.fit(X_train, y_train)
    print(f"Regressor Training Complete.")

    print("\n--- Evaluation ---")
    final_preds = reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, final_preds)
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    
    print(f"==========================================")
    print(f"OPTIMIZED MAE:  {mae:.2f} minutes (Target: < 12.0)")
    print(f"OPTIMIZED RMSE: {rmse:.2f} minutes (Target: < 20.0)")
    print(f"==========================================")
    
    # Save artifacts
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    clf_path = os.path.join(models_dir, 'classifier_model.json')
    reg_path = os.path.join(models_dir, 'delay_model.json')
    encoders_path = os.path.join(models_dir, 'encoders.pkl')
    
    if os.path.exists(clf_path):
        os.remove(clf_path)
        
    reg.save_model(reg_path)
    joblib.dump(encoders, encoders_path)
    
    print(f"\nModels successfully saved to {models_dir}")

if __name__ == "__main__":
    main()
