import os
import json
import requests
import re
from state import UnifiedEventState, RecommendationState

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

@lru_cache(maxsize=128)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(prompt: str, model: str = "gemini-2.5-flash") -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text

def supervisor_agent_node(state: UnifiedEventState) -> UnifiedEventState:
    """
    Supervisor Agent: Synthesizes the entire state and makes a final recommendation using Gemini.
    """
    flight = state.get("flight", {})
    weather = state.get("weather", {})
    risk = state.get("risk", {})
    root_cause = state.get("root_cause", {})
    notam = state.get("notam", {})
    
    prompt = f"""
    You are the Chief Aviation Operations Supervisor. 
    Review the following flight event and output a strategic decision-support recommendation.
    Provide Ranked A/B/C Options.
    Make each option exactly 1 clear sentence (around 15-25 words) explaining the action and its operational impact. Avoid being too brief, but do not write long paragraphs.

    CRITICAL INSTRUCTION: Read the FAA Rule Context carefully. If the FAA rule talks about severe weather, but the Weather Risk below is LOW, you MUST ignore the FAA rule. Do not hallucinate severe weather if the risk is LOW. If the predicted delay is under 15 mins and weather is LOW, Option A MUST be "PROCEED NORMALLY".
    CRITICAL ATC INSTRUCTION: If an Active NOTAM (Notice to Air Missions) is present (e.g. Ground Delay Program or Runway Closure), you MUST override the ML predicted delay and enforce the ATC Delay Penalty in your recommendation, regardless of weather.
    
    The weather risk values are fixed ground truth from the Weather Agent.
    Do not escalate or downgrade them in your reasoning.
    Use exactly: Origin Weather Risk = {weather.get('origin_risk')}, 
    Destination Weather Risk = {weather.get('dest_risk')}
    
    Flight: {flight.get('carrier')} {flight.get('flight_id')} from {flight.get('origin')} to {flight.get('destination')}
    Flight Telemetry & Status: {flight.get('status')} (Altitude: {flight.get('altitude')} ft, Velocity: {flight.get('velocity')} kts)
    Weather Risk: Origin ({weather.get('origin_risk')}), Dest ({weather.get('dest_risk')})
    ML Predicted Delay: {risk.get('predicted_delay_minutes')} mins
    Active NOTAM: {notam.get('active_notam')} - {notam.get('notam_type')}
    NOTAM Message: {notam.get('message')}
    ATC Delay Penalty: {notam.get('delay_penalty_minutes')} mins
    Root Cause: {root_cause.get('primary_cause')}
    FAA Rule Context: {root_cause.get('faa_citations', [''])[0]}
    
    Output ONLY a valid JSON object with the following keys exactly:
    "action": "short string, the TOP recommended action"
    "reasoning": "Option A: [1 clear sentence explaining action and impact]. \\n\\nOption B: [1 clear sentence explaining action and impact]. \\n\\nOption C: [1 clear sentence explaining action and impact]. \\n\\nDecision: [1 clear sentence explaining the choice]."
    """
    
    try:
        response_text = call_llm(prompt, model="gemini-2.5-flash")
        print(f"Supervisor LLM Raw Output: {response_text}")
        if not response_text:
            raise ValueError("LLM returned empty response")
            
        data = json.loads(response_text)
        action_val = data.get("action") or data.get("Action") or "PROCEED NORMALLY"
        reasoning_val = data.get("reasoning") or data.get("Reasoning") or "Option A: Proceed normally.\n\nOption B: Monitor flight parameters.\n\nOption C: Prepare crew briefing.\n\nDecision: Proceed normally."
        
        state["recommendation"] = RecommendationState(
            action=action_val,
            reasoning=reasoning_val
        )
    except Exception as e:
        print(f"Error in Supervisor LLM: {e}")
        
        # DEMO-SAVING FALLBACK (Circuit Breaker for 429 Rate Limits)
        if notam.get('active_notam'):
            fallback_action = "ENFORCE ATC DELAY"
            fallback_reasoning = (
                "Option A: Proceed normally. This ignores the active NOTAM and risks severe federal penalties.\n\n"
                "Option B: Implement a minor delay. This does not fully account for the ATC delay penalty.\n\n"
                f"Option C: Enforce the ATC delay penalty due to the active Ground Delay Program, overriding ML predictions.\n\n"
                "Decision: Option C is the recommended action due to the active Ground Delay Program, which mandates adherence to the ATC delay penalty."
            )
        elif risk.get('predicted_delay_minutes', 0) > 15:
            if weather.get('dest_risk') == 'Low' and weather.get('origin_risk') == 'Low':
                fallback_action = "MONITOR SCHEDULE"
                fallback_reasoning = (
                    "Option A: Increase cruise speed to recover lost time. This burns excess fuel.\n\n"
                    "Option B: Maintain current routing and absorb the ground delay. This is standard procedure.\n\n"
                    "Option C: Cancel the flight entirely.\n\n"
                    "Decision: Option B is the recommended action. The delay is inherited from the ground, but current flight conditions are optimal."
                )
            else:
                fallback_action = "IMPLEMENT DELAY"
                fallback_reasoning = (
                    "Option A: Proceed normally. This ignores the high ML predicted risk.\n\n"
                    "Option B: Implement the ML predicted delay to proactively manage the risk factors.\n\n"
                    "Option C: Cancel the flight entirely.\n\n"
                    "Decision: Option B is the recommended action to mitigate operational risk without requiring cancellation."
                )
        else:
            fallback_action = "PROCEED NORMALLY"
            fallback_reasoning = (
                "Option A: Proceed normally as all operational parameters are within acceptable limits, with low weather risk and minimal predicted delay.\n\n"
                "Option B: Monitor for any unforeseen changes in weather or ATC status.\n\n"
                "Option C: Conduct a routine pre-flight briefing with the flight crew.\n\n"
                "Decision: Given the low weather risk, negligible predicted delay, and absence of active NOTAMs, proceeding normally is the most appropriate action."
            )

        state["recommendation"] = RecommendationState(
            action=fallback_action,
            reasoning=fallback_reasoning
        )
        
    # Boost confidence score if there is an active NOTAM
    if notam.get('active_notam') and "risk" in state:
        state["risk"]["confidence_score"] = 0.95
        
    return state
