import os
import json
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Aviation Regulatory Knowledge Base (Expanded & Chunked)
DOCUMENTS = [
    {
        "id": "FAA_AC_00_45H_1",
        "title": "Aviation Weather Services - General",
        "content": "Severe weather requires immediate alternate routing or ground stops. Moderate weather allows for continued operations with increased separation and delay contingencies.",
        "icao": "ALL"
    },
    {
        "id": "FAA_ORDER_7110_65_1",
        "title": "Air Traffic Control - Low Visibility",
        "content": "During low visibility operations (RVR less than 1200), increased spacing must be applied between arriving aircraft. This typically adds 10-15 minutes of delay per aircraft.",
        "icao": "ALL"
    },
    {
        "id": "NOTAM_JFK_01",
        "title": "JFK Runway Closure",
        "content": "Runway 4L/22R closed due to snow accumulation and de-icing operations. Expect arrival delays of 45-60 minutes. Use alternate runway 4R/22L.",
        "icao": "JFK"
    },
    {
        "id": "NOTAM_ORD_01",
        "title": "ORD Ground Stop",
        "content": "Ground stop in effect for all inbound flights to Chicago O'Hare due to severe thunderstorm activity. Stop expected to lift at 0200Z.",
        "icao": "ORD"
    },
    {
        "id": "NOTAM_DEN_01",
        "title": "DEN High Winds",
        "content": "High wind warnings at Denver International. Gusts exceeding 45 knots. Expect significant spacing delays and potential diversions for light aircraft.",
        "icao": "DEN"
    },
    {
        "id": "NOTAM_LAX_01",
        "title": "LAX Coastal Fog",
        "content": "Marine layer coastal fog reducing visibility to 1/4 SM at LAX. ILS Category III approaches only. Delays averaging 25 minutes.",
        "icao": "LAX"
    },
    {
        "id": "NOTAM_ATL_01",
        "title": "ATL Severe Weather",
        "content": "Squall line moving through Atlanta terminal area. Ground operations suspended due to lightning within 5 miles. Expect cascading delays.",
        "icao": "ATL"
    },
    {
        "id": "NOTAM_DFW_01",
        "title": "DFW Convective Activity",
        "content": "Tornado watch in effect for Dallas Fort-Worth. All arrivals holding. Severe delays exceeding 90 minutes expected.",
        "icao": "DFW"
    },
    {
        "id": "NTSB_AAR_1999_01",
        "title": "NTSB Accident Report: Runway Overrun",
        "content": "Historical NTSB incident analysis: Heavy precipitation combined with tailwinds exceeding 15 knots significantly increases the risk of runway overruns. Recommend immediate diversion if braking action is reported as poor.",
        "icao": "ALL"
    },
    {
        "id": "NTSB_AAR_2005_04",
        "title": "NTSB Accident Report: Microburst and Wind Shear",
        "content": "Historical NTSB incident analysis: Convective storm cells within 10 miles of the terminal area can produce severe microbursts. Mandatory holding or diversion recommended until convective cells clear the approach path.",
        "icao": "ALL"
    }
]

def build_dense_vector_db():
    print("Initializing Dense Vector Embedding Model (all-MiniLM-L6-v2)...")
    # Using a fast, lightweight sentence transformer for local embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Embedding {len(DOCUMENTS)} FAA regulatory documents...")
    
    # Create the text corpus
    texts = [f"{doc['title']}: {doc['content']}" for doc in DOCUMENTS]
    
    # Generate embeddings
    embeddings = model.encode(texts)
    
    # Convert to float32 for FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # Initialize FAISS Index (L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings)
    
    # Prepare the DB artifact
    rag_db = {
        "index": index,
        "metadata": DOCUMENTS,
        "model_name": 'all-MiniLM-L6-v2'
    }
    
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, 'vector_db'), exist_ok=True)
    db_path = os.path.join(base_dir, 'vector_db', 'faiss_rag_db.pkl')
    
    joblib.dump(rag_db, db_path)
    print(f"FAISS Vector DB successfully built and saved to {db_path}!")

if __name__ == "__main__":
    build_dense_vector_db()
