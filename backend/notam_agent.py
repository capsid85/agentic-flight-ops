import requests
from state import UnifiedEventState, NotamState

def notam_agent_node(state: UnifiedEventState) -> UnifiedEventState:
    """
    Live Air Traffic Control (ATC) Agent.
    Queries the official FAA NAS Status API for real-time Ground Stops and Ground Delay Programs.
    """
    flight = state.get("flight", {})
    dest = flight.get("destination", "").upper()
    
    # Initialize default state (No Active Constraints)
    notam_state = NotamState(
        active_notam=False,
        notam_type="None",
        message="No active Air Traffic Control constraints at destination.",
        delay_penalty_minutes=0
    )
    
    if not dest:
        state["notam"] = notam_state
        return state

    try:
        # Hit the live FAA National Airspace System (NAS) API
        print(f"NOTAM Agent: Querying live FAA API for {dest}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get('https://nasstatus.faa.gov/api/airport-events', headers=headers, timeout=5)
        
        if res.status_code == 200:
            events = res.json()
            
            # Find the destination airport in the FAA events list
            airport_event = next((e for e in events if e.get("airportId") == dest), None)
            
            if airport_event:
                # 1. Check for Ground Stops (Worst Case)
                if airport_event.get("groundStop"):
                    reason = airport_event["groundStop"].get("reason", "Unknown")
                    notam_state = NotamState(
                        active_notam=True,
                        notam_type="FAA GROUND STOP",
                        message=f"ATC GROUND STOP IN EFFECT: {reason}. All inbound flights held at origin.",
                        delay_penalty_minutes=120
                    )
                # 2. Check for Ground Delay Programs (GDP)
                elif airport_event.get("groundDelay"):
                    reason = airport_event["groundDelay"].get("reason", "Unknown")
                    avg_delay = airport_event["groundDelay"].get("avgDelay", 45)
                    # Handle string avgDelay like "45 minutes" or "1 hours 15 minutes"
                    if isinstance(avg_delay, str):
                        import re
                        nums = re.findall(r'\d+', avg_delay)
                        avg_delay_mins = int(nums[0]) if nums else 45
                    else:
                        avg_delay_mins = avg_delay
                        
                    notam_state = NotamState(
                        active_notam=True,
                        notam_type="GROUND DELAY PROGRAM",
                        message=f"FAA GDP IN EFFECT: {reason}. Arrival capacity reduced.",
                        delay_penalty_minutes=avg_delay_mins
                    )
                # 3. Check for general Arrival Delays
                elif airport_event.get("arrivalDelay"):
                    reason = airport_event["arrivalDelay"].get("reason", "Unknown")
                    notam_state = NotamState(
                        active_notam=True,
                        notam_type="ATC ARRIVAL DELAY",
                        message=f"FAA METERING DELAY: {reason}.",
                        delay_penalty_minutes=30
                    )
                    
    except Exception as e:
        print(f"NOTAM Agent Error: Failed to fetch live FAA data - {e}")
        
    print(f"NOTAM Agent: Checked {dest} - Active NOTAM: {notam_state['active_notam']}")
    state["notam"] = notam_state
    
    return state
