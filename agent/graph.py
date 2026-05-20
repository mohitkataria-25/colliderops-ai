from langgraph.graph import START, END, StateGraph
from agent.state import AgentState
from agent.nodes import (
    route_intent_node,
    rag_question_node,
    model_metadata_node,
    single_prediction_node,
    batch_prediction_node,
    batch_file_prediction_node,
    batch_export_node,
    etl_status_node,
    unknown_intent_node,
)

workflow = StateGraph(AgentState)

workflow.add_node("route_intent_node", route_intent_node)
workflow.add_node("rag_question_node", rag_question_node)
workflow.add_node("model_metadata_node", model_metadata_node)
workflow.add_node("single_prediction_node", single_prediction_node)
workflow.add_node("batch_prediction_node", batch_prediction_node)
workflow.add_node("batch_file_prediction_node", batch_file_prediction_node)
workflow.add_node("batch_export_node", batch_export_node)
workflow.add_node("etl_status_node", etl_status_node)
workflow.add_node("unknown_intent_node", unknown_intent_node)

workflow.add_edge(START, "route_intent_node")

def route_to_node(state:AgentState)->str:

    return state.get("intent", "unknown")

workflow.add_conditional_edges(
    "route_intent_node", 
    route_to_node,
    {
        "rag_question": "rag_question_node",
        "model_metadata": "model_metadata_node",
        "single_prediction": "single_prediction_node",
        "batch_prediction": "batch_prediction_node",
        "batch_file_prediction":"batch_file_prediction_node",
        "batch_export":"batch_export_node",
        "etl_status":"etl_status_node",
        "unknown":"unknown_intent_node"
    },
)

workflow.add_edge("rag_question_node", END)
workflow.add_edge("model_metadata_node", END)
workflow.add_edge("single_prediction_node", END)
workflow.add_edge("batch_prediction_node", END)
workflow.add_edge("batch_file_prediction_node", END)
workflow.add_edge("batch_export_node", END)
workflow.add_edge("etl_status_node", END)
workflow.add_edge("unknown_intent_node", END)

agent_graph = workflow.compile()

def run_agent(state:AgentState)->AgentState:
    return agent_graph.invoke(state)





