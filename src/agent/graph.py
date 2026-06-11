from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.state import TutorState
from src.agent.nodes import (
    adapt_next_node,
    evaluate_answer_node,
    generate_feedback_node,
    generate_problem_node,
    load_state_node,
    retrieve_explanation_node,
    select_topic_node,
    update_state_node,
)


memory = MemorySaver()

builder = StateGraph(TutorState)

builder.add_node("load_state_node", load_state_node)
builder.add_node("select_topic_node", select_topic_node)
builder.add_node("generate_problem_node", generate_problem_node)
builder.add_node("evaluate_answer_node", evaluate_answer_node)
builder.add_node("retrieve_explanation_node", retrieve_explanation_node)
builder.add_node("generate_feedback_node", generate_feedback_node)
builder.add_node("update_state_node", update_state_node)
builder.add_node("adapt_next_node", adapt_next_node)

builder.add_edge(START, "load_state_node")
builder.add_edge("load_state_node", "select_topic_node")
builder.add_edge("select_topic_node", "generate_problem_node")
builder.add_edge("generate_problem_node", "evaluate_answer_node")
builder.add_edge("evaluate_answer_node", "retrieve_explanation_node")
builder.add_edge("retrieve_explanation_node", "generate_feedback_node")
builder.add_edge("generate_feedback_node", "update_state_node")
builder.add_edge("update_state_node", "adapt_next_node")
builder.add_edge("adapt_next_node", "generate_problem_node")

graph = builder.compile(checkpointer=memory, interrupt_before=["evaluate_answer_node"])
