import os
from typing import Annotated
from dotenv import load_dotenv
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

# Testing the tools (you can remove these test calls in the final script)
print(wiki_tool.invoke("who is Einstein"))
print(arxiv_tool.invoke("Attention is all you need"))

tools = [wiki_tool]

# Define State class for Langgraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the StateGraph
graph_builder = StateGraph(State)

# Bind tools to the Langchain model
llm_with_tools = llm.bind_tools(tools=tools)

# Create Agent 1: Fetch Information 
# Create Agent 1: Fetch Information
def fetch_information(state: State):
    # The tools fetch the information (from Wikipedia or Arxiv)
    fetched_data = []
    for tool in tools:
        # Access the content of the message properly
        query = state["messages"][0].content
        response = tool.invoke(query)
        fetched_data.append(response)
    return {"messages": fetched_data}


###create Agent 2: Process Information 
def process_information(state: State):
    # Now process the fetched data with the Groq model
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# add the nodes for Agent 1 (fetching information) and Agent 2 (processing information)
graph_builder.add_node("fetch_information", fetch_information)
graph_builder.add_node("process_information", process_information)

# Add edges to connect the two agents
graph_builder.add_edge(START, "fetch_information")
graph_builder.add_edge("fetch_information", "process_information")
graph_builder.add_edge("process_information", END)

# complie the grapoh
graph = graph_builder.compile()

# main loop for interacting with the chatbot
user_input = "what is generative AI?"
events = graph.stream({"messages": [("user", user_input)]}, stream_mode="values")

# print
for event in events:
    event["messages"][-1].pretty_print()

