from typing import TypedDict, Optional, List, Dict, Any
from typing_extensions import Annotated
import operator

class FlightState(TypedDict, total=False):
    flight_id: str
    carrier: str
    origin: str
    destination: str
    scheduled_departure: str
    latitude: float
    longitude: float
    origin_latitude: float
    origin_longitude: float
    dest_latitude: float
    dest_longitude: float
    altitude: int
    velocity: int
    status: str
    accumulated_delay: int
    
class WeatherConditions(TypedDict, total=False):
    temperature: float
    windSpeed: float
    precipitation: float
    visibility: str
    cloudLayers: str

class WeatherState(TypedDict, total=False):
    origin_risk: str
    dest_risk: str
    conditions: WeatherConditions

class MetarState(TypedDict, total=False):
    raw_origin: str
    raw_dest: str

class RiskState(TypedDict, total=False):
    predicted_delay_minutes: int
    confidence_score: float

class RootCauseState(TypedDict, total=False):
    primary_cause: str
    secondary_cause: str
    faa_citations: List[str]

class NotamState(TypedDict, total=False):
    active_notam: bool
    notam_type: str
    message: str
    delay_penalty_minutes: int

class RecommendationState(TypedDict, total=False):
    action: str
    reasoning: str

class GroundTruthState(TypedDict, total=False):
    actual_delay_minutes: int
    actual_action_taken: str

# The main graph state
class UnifiedEventState(TypedDict, total=False):
    mode: str          # "historical" or "live"
    query: str         # Airline or Airport code for live mode
    timestamp: str
    flight: FlightState
    weather: WeatherState
    metar: MetarState
    risk: RiskState
    root_cause: RootCauseState
    notam: NotamState
    recommendation: RecommendationState
    ground_truth: GroundTruthState
