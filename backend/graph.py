import json
from langgraph.graph import StateGraph, START, END
from state import UnifiedEventState
from flight_agent import flight_monitoring_node
from weather_agent import weather_intelligence_node
from risk_agent import risk_assessment_node
from root_cause_agent import root_cause_analysis_node
from notam_agent import notam_agent_node
from supervisor_agent import supervisor_agent_node

def build_graph():
    # 1. Initialize the StateGraph with our UnifiedEventState
    workflow = StateGraph(UnifiedEventState)

    # 2. Add all 5 agents (nodes)
    workflow.add_node("flight_agent", flight_monitoring_node)
    workflow.add_node("weather_agent", weather_intelligence_node)
    workflow.add_node("risk_agent", risk_assessment_node)
    workflow.add_node("root_cause_agent", root_cause_analysis_node)
    workflow.add_node("notam_agent", notam_agent_node)
    workflow.add_node("supervisor_agent", supervisor_agent_node)

    # 3. Define the causal chain (edges)
    workflow.add_edge(START, "flight_agent")
    workflow.add_edge("flight_agent", "weather_agent")
    workflow.add_edge("weather_agent", "risk_agent")
    workflow.add_edge("risk_agent", "root_cause_agent")
    workflow.add_edge("root_cause_agent", "notam_agent")
    workflow.add_edge("notam_agent", "supervisor_agent")
    workflow.add_edge("supervisor_agent", END)

    # 4. Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    print("Initializing LangGraph Replay Engine with Ollama...")
    app = build_graph()
    
    initial_state = UnifiedEventState()
    
    print("\\nExecuting Full Pipeline (START -> Flight -> Weather -> Risk -> Root Cause -> Supervisor -> END)...")
    print("Please wait, local Llama3 is reasoning...")
    
    final_state = app.invoke(initial_state)
    
    print("\\n--- FINAL UNIFIED EVENT STATE ---")
    print(json.dumps(final_state, indent=2))
