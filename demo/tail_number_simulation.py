import pandas as pd
from datetime import datetime, timedelta

def simulate_tail_number_cascade():
    print("\n=======================================================")
    print("   AIRCRAFT TAIL NUMBER CASCADE DELAY SIMULATION   ")
    print("=======================================================\n")
    print("Aircraft: Boeing 737-800 | Tail Number: N123AA")
    print("Scenario: A severe morning blizzard in Chicago (ORD) causes a cascade.")
    print("-" * 55)
    
    # Define a single aircraft's schedule for the day
    schedule = [
        {"leg": 1, "route": "ORD -> JFK", "sched_dep": "08:00", "sched_arr": "11:00", "turnaround_min": 60},
        {"leg": 2, "route": "JFK -> MIA", "sched_dep": "12:00", "sched_arr": "15:00", "turnaround_min": 60},
        {"leg": 3, "route": "MIA -> LAX", "sched_dep": "16:00", "sched_arr": "19:00", "turnaround_min": 60},
    ]
    
    # Introduce the initial delay (90 mins at ORD)
    current_delay = 90
    
    for flight in schedule:
        dep_time = datetime.strptime(flight["sched_dep"], "%H:%M")
        arr_time = datetime.strptime(flight["sched_arr"], "%H:%M")
        
        # Add current delay
        actual_dep = dep_time + timedelta(minutes=current_delay)
        
        # Flight duration remains the same, so arrival is delayed by the same amount
        # (Assuming no block padding makeup for this simulation to keep it simple)
        actual_arr = arr_time + timedelta(minutes=current_delay)
        
        print(f"Leg {flight['leg']}: {flight['route']}")
        print(f"  Scheduled: {flight['sched_dep']} -> {flight['sched_arr']}")
        
        if flight['leg'] == 1:
            print(f"  Root Cause: Severe Weather at Origin (ORD)")
            print(f"  ACTUAL DEPARTURE: {actual_dep.strftime('%H:%M')} (+{current_delay} mins WEATHER DELAY)")
        else:
            print(f"  Root Cause: Late Aircraft (Cascade from previous leg)")
            print(f"  ACTUAL DEPARTURE: {actual_dep.strftime('%H:%M')} (+{current_delay} mins LATE AIRCRAFT DELAY)")
            
        print(f"  ACTUAL ARRIVAL: {actual_arr.strftime('%H:%M')}")
        print("-" * 55)
        
        # For the next flight, the delay might shrink if the scheduled turnaround time 
        # is longer than the minimum required turnaround time (e.g., 60 mins).
        # In this schedule, they have exactly 60 mins padded between flights.
        # Since they are late, they will try to turnaround in exactly 60 mins.
        # Delay carries over entirely.
        
    print("\n[SYSTEM SUMMARY]")
    print("By tracking the TAIL NUMBER (N123AA), the predictive model at JFK and MIA")
    print("knows hours in advance that their flights will be delayed, even if the")
    print("weather in JFK and MIA is perfectly clear.")
    print("=======================================================\n")

if __name__ == "__main__":
    simulate_tail_number_cascade()
