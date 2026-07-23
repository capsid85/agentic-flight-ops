import os
import json
import joblib
import faiss
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer
from state import UnifiedEventState, RootCauseState

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

# Global for RAG DB
RAG_DB = None
EMBEDDING_MODEL = None

def load_rag_db():
    global RAG_DB, EMBEDDING_MODEL
    if RAG_DB is None:
        base_dir = os.path.dirname(__file__)
        db_path = os.path.abspath(os.path.join(base_dir, 'vector_db/faiss_rag_db.pkl'))
        if os.path.exists(db_path):
            RAG_DB = joblib.load(db_path)
            EMBEDDING_MODEL = SentenceTransformer(RAG_DB['model_name'])
            print("Root Cause Agent: Loaded FAISS Dense Vector Database.")
        else:
            print("Root Cause Agent: FAISS DB not found. Using fallback.")

def retrieve_docs(query: str, origin_icao: str, dest_icao: str, top_k=2):
    if not RAG_DB or not EMBEDDING_MODEL:
        return ["FAA AC 00-45H - Aviation Weather Services. Severe weather requires alternate routing."]
    
    index = RAG_DB['index']
    metadata = RAG_DB['metadata']
    
    # Generate query embedding
    query_vec = EMBEDDING_MODEL.encode([query])
    query_vec = np.array(query_vec).astype('float32')
    
    # Search FAISS index
    # Search deeper to allow filtering out bad metadata
    distances, indices = index.search(query_vec, 10)
    
    results = []
    # Filter by ICAO metadata
    valid_icaos = ["ALL", origin_icao, dest_icao]
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1 and dist < 1.5:  # threshold
            doc = metadata[idx]
            if doc.get('icao', 'ALL') in valid_icaos:
                results.append(f"[{doc['id']}] {doc['title']}: {doc['content']}")
                if len(results) >= top_k:
                    break
            
    if not results:
        return ["No relevant operational guidelines found for this route."]
    return results

def root_cause_analysis_node(state: UnifiedEventState) -> UnifiedEventState:
    """
    Root Cause Agent: Uses Cloud LLM to infer the primary cause from the METAR and risk data, grounded by RAG.
    """
    flight = state.get("flight", {})
    weather = state.get("weather", {})
    metar = state.get("metar", {})
    risk = state.get("risk", {})
    
    load_rag_db()
    
    # CRITICAL FIX: Bypass LLM entirely if no delay/risk to save 50% of API quota
    if risk.get('predicted_delay_minutes', 0) < 15 and weather.get('dest_risk') == 'Low' and weather.get('origin_risk') == 'Low':
        state["root_cause"] = RootCauseState(
            primary_cause="None",
            secondary_cause="Routine Operations",
            faa_citations=["No severe operational disruptions detected. Routine operations in effect."]
        )
        return state
    
    prompt = f"""
    You are an expert Aviation Analyst. Determine the root cause of the flight delay.
    IMPORTANT: If the Predicted Delay is 0 mins, there is NO DELAY. In that case, the primary_cause MUST be "None" and secondary_cause MUST be "Routine Operations".
    
    Flight: {flight.get('carrier')} {flight.get('flight_id')} from {flight.get('origin')} to {flight.get('destination')}
    Predicted Delay: {risk.get('predicted_delay_minutes')} mins
    Origin Weather Risk: {weather.get('origin_risk')}
    Dest Weather Risk: {weather.get('dest_risk')}
    Origin METAR: {metar.get('raw_origin')}
    Dest METAR: {metar.get('raw_dest')}
    
    Output ONLY a valid JSON object with the following keys exactly:
    "primary_cause": <short string, e.g. "Severe Weather at Destination" or "None" if 0 mins delay>
    "secondary_cause": <short string>
    "search_query": <a brief search query to look up FAA guidelines regarding this specific situation>
    """
    
    try:
        response_text = call_llm(prompt, model="gemini-2.5-flash")
        if not response_text:
            raise ValueError("LLM returned empty response")
            
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        clean_json = match.group(0) if match else response_text
            
        data = json.loads(clean_json)
        
        # Perform RAG Retrieval
        primary_cause = data.get("primary_cause", "Unknown")
        search_query = data.get("search_query", primary_cause)
        
        # If there is no real delay and weather is fine, don't query RAG to prevent hallucinated rules
        if risk.get('predicted_delay_minutes', 0) < 15 and weather.get('dest_risk') == 'Low' and weather.get('origin_risk') == 'Low':
            citations = ["No severe operational disruptions detected. Routine operations in effect."]
        else:
            print(f"Root Cause Agent: Querying RAG DB for '{search_query}' (Origin: {flight.get('origin')}, Dest: {flight.get('destination')})")
            citations = retrieve_docs(search_query, flight.get('origin', ''), flight.get('destination', ''))
        
        state["root_cause"] = RootCauseState(
            primary_cause=data.get("primary_cause", "Unknown"),
            secondary_cause=data.get("secondary_cause", "None"),
            faa_citations=citations
        )
    except Exception as e:
        print(f"Error in Root Cause LLM: {e}")
        if weather.get('dest_risk') == 'Low' and weather.get('origin_risk') == 'Low' and risk.get('predicted_delay_minutes', 0) > 15:
            primary_cause = "Inherited Ground Delay"
            citation = "FAA JO 7110.65 - Air Traffic Control. Ground delay inherited from prior flight legs."
        else:
            primary_cause = "Weather/Congestion"
            citation = "FAA AC 00-45H - Aviation Weather Services. Severe weather requires alternate routing."
            
        state["root_cause"] = RootCauseState(
            primary_cause=primary_cause,
            secondary_cause="Unknown",
            faa_citations=[citation]
        )
        
    return state
