

import os
from dotenv import load_dotenv

from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langsmith = os.getenv("LANGSMITH_API_KEY")

# Set environment variables for Langchain API
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

# Initialize Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
print(f"Model Initialized: {llm}")

# Set up Arxiv and Wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki_tool.invoke("who is Einstein"))
print(arxiv_tool.invoke("Attention is all you need"))

tools = [wiki_tool, arxiv_tool]

# Define State class for Langgraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the StateGraph
graph_builder = StateGraph(State)

# Bind tools to the Langchain model
llm_with_tools = llm.bind_tools(tools=tools)

# Create the chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add the chatbot and tools nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add conditional edges to the graph
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()


# Main loop for interacting with the chatbot
user_input = "Hi there!, My name is John"
events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")

# Print the messages from the stream
for event in events:
    event["messages"][-1].pretty_print()

# Test with another input
user_input = "Attention is all you need"
events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")

# Print the messages from the stream
for event in events:
    event["messages"][-1].pretty_print()
