import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import requests
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import build_graph
from state import UnifiedEventState
import random

import time
LIVE_FLIGHTS_CACHE = {"flights": [], "total": 0, "last_updated": 0}

async def background_monitoring_loop():
    global LIVE_FLIGHTS_CACHE
    print("Background Monitoring Agent: Started autonomous loop.")
    while True:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get("https://opensky-network.org/api/states/all?lamin=25&lomin=-125&lamax=49&lomax=-67", headers=headers, timeout=6)
            if res.status_code == 200:
                states = res.json().get("states", [])
                
                processed = []
                for s in states[:1000]:  # Up to top 1000 flights
                    if not s or len(s) < 13:
                        continue
                    callsign = (s[1] or "").strip()
                    lon, lat = s[5], s[6]
                    if lon is None or lat is None:
                        continue
                        
                    alt_m = s[7]        
                    on_ground = s[8]    
                    vel_mps = s[9]      
                    vert_rate = s[11]   
                    
                    alt_ft = round(alt_m * 3.28084) if alt_m is not None else 0
                    vel_kts = round(vel_mps * 1.94384) if vel_mps is not None else 0
                    vert_fpm = round(vert_rate * 196.85) if vert_rate is not None else 0
                    
                    is_landing = (not on_ground) and (alt_m is not None and alt_m < 3048) and (vert_rate is not None and vert_rate < -1.0)
                    
                    processed.append({
                        "id": callsign or "UNK",
                        "lon": lon,
                        "lat": lat,
                        "altitude_ft": alt_ft,
                        "velocity_kts": vel_kts,
                        "vertical_rate_fpm": vert_fpm,
                        "on_ground": bool(on_ground),
                        "is_landing": is_landing
                    })
                    
                LIVE_FLIGHTS_CACHE = {"flights": processed, "total": len(states), "last_updated": time.time()}
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Background Agent: Cached {len(processed)} active flights.")
        except Exception as e:
            print(f"Background Agent Error: {e}")
        
        # Sleep for 300 seconds (5 minutes) to completely eliminate OpenSky rate limits
        # OpenSky allows 400 anonymous requests/day (288 requests/day at 5m interval)
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_monitoring_loop())
    yield
    task.cancel()

app = FastAPI(title="Aviation Replay Engine API", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Building LangGraph Pipeline...")
pipeline = build_graph()

from pydantic import BaseModel

class ReplayRequest(BaseModel):
    mode: str = "live"
    query: str = ""

@app.get("/live-flights")
def get_live_flights():
    # Return directly from memory cache, NO API CALLS!
    return LIVE_FLIGHTS_CACHE


@app.websocket("/ws/flights")
async def websocket_flights(websocket: WebSocket):
    await websocket.accept()
    try:
        print("WebSocket client connected. Starting live stream...")
        while True:
            # Simulated Kafka stream of live updates
            stream_payload = {
                "type": "STREAM_UPDATE",
                "timestamp": datetime.now().isoformat(),
                "updates": [
                    {"flight_id": "DL400", "predicted_delay": round(random.uniform(-5, 15), 1)},
                    {"flight_id": "UA200", "predicted_delay": round(random.uniform(-2, 5), 1)},
                    {"flight_id": "AA100", "predicted_delay": round(random.uniform(10, 45), 1)}
                ]
            }
            await websocket.send_json(stream_payload)
            await asyncio.sleep(3) # Stream new prediction data every 3 seconds
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

@app.post("/analyze")
async def analyze_flight(req: ReplayRequest):
    """
    Triggers a single cycle of the Replay Engine or Live Tracker.
    """
    initial_state = UnifiedEventState(
        mode=req.mode,
        query=req.query,
        timestamp=""
    )
    
    # We invoke the LangGraph pipeline
    try:
        final_state = pipeline.invoke(initial_state)
        return final_state
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(e))

import os
import pandas as pd

@app.get("/historical-flights")
async def get_historical_flights(airport: str):
    """
    Returns 5 random historically delayed flights for a given airport.
    """
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.abspath(os.path.join(base_dir, '../data/bts_sample.csv'))
    
    airport = airport.upper().strip()
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Clean airport codes
    df['ORIGIN_AIRPORT'] = df['ORIGIN_AIRPORT'].astype(str).str.strip()
    
    # Filter for delays > 15 mins at the requested origin
    delayed = df[(df['ORIGIN_AIRPORT'] == airport) & (df['DEPARTURE_DELAY'] > 15)]
    
    if delayed.empty:
        return {"flights": []}
        
    # Sample up to 5 flights
    sample_size = min(5, len(delayed))
    sampled = delayed.sample(n=sample_size)
    
    results = []
    for _, row in sampled.iterrows():
        airline = str(row.get('AIRLINE', 'UNK')).strip()
        flight_num = str(row.get('FLIGHT_NUMBER', '000')).strip()
        year = int(row.get('YEAR', 2015))
        month = int(row.get('MONTH', 1))
        day = int(row.get('DAY', 1))
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        delay = int(row.get('DEPARTURE_DELAY', 0))
        dest = str(row.get('DESTINATION_AIRPORT', 'UNK')).strip()
        
        # ID format: AIRLINE|FLIGHT_NUMBER|YYYY-MM-DD
        unique_id = f"{airline}|{flight_num}|{date_str}"
        
        results.append({
            "id": unique_id,
            "display": f"{airline}{flight_num} to {dest} ({delay} min delay on {date_str})",
            "delay": delay
        })
        
    return {"flights": results}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
