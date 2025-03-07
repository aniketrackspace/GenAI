import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
langsmith = os.getenv("LANGSMITH_API_KEY")

if not groq_api_key or not langsmith:
    raise ValueError("API keys not found. Please set GROQ_API_KEY and LANGSMITH_API_KEY in your .env file.")

print(langsmith)

# Set environment variables for Langchain API
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

# Initialize Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
print(f"Model Initialized: {llm}")

# Define State class for Langgraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the StateGraph
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}

# Add the "chatbot" node to the graph
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        break
    for event in graph.stream({'messages': ("user", user_input)}):
        print(event.values())
        for value in event.values():
            print(value['messages'])
            print("Assistant:", value["messages"].content)
