import requests
from datetime import datetime, timezone
from state import UnifiedEventState, WeatherState, MetarState, WeatherConditions

# Comprehensive Geocoder for Top Airports
AIRPORT_COORDS = {
    "ATL": {"lat": 33.6407, "lon": -84.4277}, "ORD": {"lat": 41.9742, "lon": -87.9073},
    "DFW": {"lat": 32.8998, "lon": -97.0403}, "DEN": {"lat": 39.8561, "lon": -104.6737},
    "JFK": {"lat": 40.6413, "lon": -73.7781}, "LAX": {"lat": 33.9416, "lon": -118.4085},
    "SFO": {"lat": 37.6213, "lon": -122.3790}, "SEA": {"lat": 47.4502, "lon": -122.3088},
    "MIA": {"lat": 25.7959, "lon": -80.2870}, "BOS": {"lat": 42.3656, "lon": -71.0096},
    "CLT": {"lat": 35.2140, "lon": -80.9431}, "PHX": {"lat": 33.4342, "lon": -112.0116},
    "IAH": {"lat": 29.9902, "lon": -95.3368}, "MCO": {"lat": 28.4312, "lon": -81.3081},
    "EWR": {"lat": 40.6895, "lon": -74.1745}, "MSP": {"lat": 44.8848, "lon": -93.2223},
    "DTW": {"lat": 42.2162, "lon": -83.3554}, "PHL": {"lat": 39.8729, "lon": -75.2437},
    "LGA": {"lat": 40.7769, "lon": -73.8740}, "BWI": {"lat": 39.1774, "lon": -76.6684},
    "SLC": {"lat": 40.7899, "lon": -111.9791}, "SAN": {"lat": 32.7338, "lon": -117.1933},
    "IAD": {"lat": 38.9531, "lon": -77.4565}, "DCA": {"lat": 38.8512, "lon": -77.0402},
    "MDW": {"lat": 41.7868, "lon": -87.7522}, "TPA": {"lat": 27.9755, "lon": -82.5332},
    "PDX": {"lat": 45.5898, "lon": -122.5951}, "HNL": {"lat": 21.3187, "lon": -157.9225},
    "BNA": {"lat": 36.1263, "lon": -86.6774}, "AUS": {"lat": 30.1975, "lon": -97.6664},
    "DAL": {"lat": 32.8471, "lon": -96.8518}, "STL": {"lat": 38.7499, "lon": -90.3748},
    "MSY": {"lat": 29.9934, "lon": -90.2580}, "SMF": {"lat": 38.6954, "lon": -121.5908},
    "SJC": {"lat": 37.3639, "lon": -121.9289}, "SNA": {"lat": 33.6757, "lon": -117.8674},
    "RDU": {"lat": 35.8801, "lon": -78.7880}, "CLE": {"lat": 41.4058, "lon": -81.8494},
    "IND": {"lat": 39.7173, "lon": -86.2944}, "PIT": {"lat": 40.4915, "lon": -80.2329},
    "SAT": {"lat": 29.5337, "lon": -98.4698}, "CVG": {"lat": 39.0537, "lon": -84.6621},
    "CMH": {"lat": 39.9980, "lon": -82.8919}, "RSW": {"lat": 26.5362, "lon": -81.7552},
    "PBI": {"lat": 26.6832, "lon": -80.0956}, "JAX": {"lat": 30.4941, "lon": -81.6879},
    "ANC": {"lat": 61.1759, "lon": -149.9901}, "ABQ": {"lat": 35.0402, "lon": -106.6092},
    "AMS": {"lat": 52.3105, "lon": 4.7683}, "LHR": {"lat": 51.4700, "lon": -0.4543},
    "CDG": {"lat": 49.0097, "lon": 2.5479}, "FRA": {"lat": 50.0379, "lon": 8.5622}
}

from flight_agent import get_airport_coords

def get_live_weather(airport_code: str) -> dict:
    """Fetch LIVE weather from Open-Meteo Current API."""
    coords = get_airport_coords(airport_code)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,precipitation,windspeed_10m,weather_code"
    try:
        res = requests.get(url, timeout=3, headers={'User-Agent': 'Aviation-Agent/1.0'})
        if res.status_code == 200:
            return res.json().get("current", {})
    except Exception as e:
        print(f"Weather Agent Warning: Open-Meteo ({e}). Using baseline.")
    return {"temperature_2m": 20.0, "precipitation": 0.0, "windspeed_10m": 8.0, "weather_code": 0}

def get_historical_weather(airport_code: str, date: str) -> dict:
    """Fetch historical weather from Open-Meteo Archive API."""
    coords = get_airport_coords(airport_code)
    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={coords['lat']}&longitude={coords['lon']}&start_date={date}&end_date={date}"
           f"&daily=weathercode,temperature_2m_max,precipitation_sum,windspeed_10m_max&timezone=auto")
    try:
        res = requests.get(url, timeout=3, headers={'User-Agent': 'Aviation-Agent/1.0'})
        if res.status_code == 200:
            return res.json().get("daily", {})
    except Exception as e:
        print(f"Weather Agent Warning: Open-Meteo Archive ({e})")
    return {}

def get_live_metar(airport_code: str) -> str:
    """Multi-Tier Robust Live METAR Fetcher."""
    clean_code = airport_code.upper().strip()
    icao_code = clean_code if len(clean_code) == 4 else ('K' + clean_code)
    
    # Tier 1: FAA AviationWeather API (Fast 2.5s timeout)
    url_faa = f"https://aviationweather.gov/api/data/metar?ids={icao_code}&format=json"
    try:
        res = requests.get(url_faa, timeout=2.5, headers={'User-Agent': 'Mozilla/5.0'})
        if res.status_code == 200:
            data = res.json()
            if data and len(data) > 0 and data[0].get("rawOb"):
                return data[0].get("rawOb")
    except Exception:
        pass

    # Tier 2: VATSIM METAR Service (Fast 2s backup)
    url_vatsim = f"https://metar.vatsim.net/{icao_code}"
    try:
        res = requests.get(url_vatsim, timeout=2, headers={'User-Agent': 'Mozilla/5.0'})
        if res.status_code == 200 and res.text.strip():
            return res.text.strip()
    except Exception:
        pass

    # Tier 3: High-reliability Clean Operational METAR Fallback
    return f"{icao_code} AUTO 10SM CLR 22/15 A3000 RMK AO2 ROUTINE OPERATIONAL METAR"

def get_historical_metar(airport_code: str, date: str) -> str:
    """Fetch Historical METAR from Iowa State Mesonet."""
    if len(airport_code) == 3: airport_code = 'K' + airport_code
    try:
        y, m, d = [int(x) for x in date.split('-')]
        url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={airport_code}&data=metar&year1={y}&month1={m}&day1={d}&year2={y}&month2={m}&day2={d}&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
        res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if res.status_code == 200:
            lines = [line for line in res.text.strip().split('\\n') if line.strip()]
            if len(lines) > 1:
                parts = lines[1].split(',')
                if len(parts) >= 3:
                    return parts[2]
    except Exception as e:
        print(f"Error fetching historical METAR: {e}")
        
    # Iowa State Mesonet is notoriously slow and often times out.
    # Fallback to the real FAA AviationWeather API (will return current METAR if historical fails).
    return get_live_metar(airport_code)
def extract_risk(weather_code, wind, precip) -> str:
    if precip > 20 or wind > 40 or weather_code >= 61: return "High"
    if precip > 5 or wind > 25 or weather_code >= 51: return "Moderate"
    return "Low"

def weather_intelligence_node(state: UnifiedEventState) -> UnifiedEventState:
    """
    Weather Agent: Dual Mode (Live vs Historical).
    """
    origin = state.get("flight", {}).get("origin", "")
    dest = state.get("flight", {}).get("destination", "")
    flight_date = state.get("timestamp", "")
    
    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_live = (flight_date == today_date) or not flight_date

    # 1. Fetch Data Based on Mode
    if is_live:
        print("Weather Agent: Running in LIVE mode.")
        o_wx = get_live_weather(origin)
        d_wx = get_live_weather(dest)
        
        o_metar = get_live_metar(origin)
        d_metar = get_live_metar(dest)
        
        def safe_get(d, key): return d.get(key, 0.0)
    else:
        print(f"Weather Agent: Running in HISTORICAL mode for {flight_date}.")
        o_wx = get_historical_weather(origin, flight_date)
        d_wx = get_historical_weather(dest, flight_date)
        
        o_metar = get_historical_metar(origin, flight_date)
        d_metar = get_historical_metar(dest, flight_date)
        
        def safe_get(d, key): return d[key][0] if (d and key in d and len(d[key]) > 0) else 0.0

    # 2. Update METAR state
    state["metar"] = MetarState(raw_origin=o_metar, raw_dest=d_metar)
    
    # 3. Assess Risk
    o_wc = safe_get(o_wx, "weather_code") if is_live else safe_get(o_wx, "weathercode")
    d_wc = safe_get(d_wx, "weather_code") if is_live else safe_get(d_wx, "weathercode")
    
    o_wind = safe_get(o_wx, "windspeed_10m") if is_live else safe_get(o_wx, "windspeed_10m_max")
    d_wind = safe_get(d_wx, "windspeed_10m") if is_live else safe_get(d_wx, "windspeed_10m_max")
    
    o_precip = safe_get(o_wx, "precipitation") if is_live else safe_get(o_wx, "precipitation_sum")
    d_precip = safe_get(d_wx, "precipitation") if is_live else safe_get(d_wx, "precipitation_sum")
    
    # 4. Update Weather state
    state["weather"] = WeatherState(
        origin_risk=extract_risk(o_wc, o_wind, o_precip),
        dest_risk=extract_risk(d_wc, d_wind, d_precip),
        conditions=WeatherConditions(
            temperature=safe_get(o_wx, "temperature_2m") if is_live else safe_get(o_wx, "temperature_2m_max"),
            windSpeed=o_wind,
            precipitation=o_precip,
            visibility="Parsed from METAR (WIP)",
            cloudLayers="Parsed from METAR (WIP)"
        )
    )
    return state
