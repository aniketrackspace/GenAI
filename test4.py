




#Not working, need to update


import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition

# Set up API keys and other configuration
groq_api_key = "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"
langsmith = "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"
print(langsmith)

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

tools = [wiki_tool, arxiv_tool]

# Define a Memory class to store past interactions
class Memory:
    def __init__(self):
        self.memory = []

    def add_to_memory(self, message: str):
        """Store messages in memory."""
        self.memory.append(message)

    def get_memory(self):
        """Retrieve the stored memory."""
        return self.memory

# Define State class for Langgraph with memory
class State(TypedDict):
    messages: Annotated[list, add_messages]
    memory: Memory  # Adding memory to the state

# Build the StateGraph
graph_builder = StateGraph(State)

# Bind tools to the Langchain model
llm_with_tools = llm.bind_tools(tools=tools)

# Create the chatbot function
def chatbot(state: State):
    # Get previous memory and add it to the conversation context
    previous_messages = "\n".join(state["memory"].get_memory())
    user_message = state["messages"][0].content  # Accessing message content directly
    # Add the current message to memory
    state["memory"].add_to_memory(user_message)
    # Combine memory with the current message for contextual response
    context = f"{previous_messages}\nUser: {user_message}\nBot:"
    response = llm_with_tools.invoke([("user", context)])
    
    # Ensure response is a list of messages (e.g., AIMessage)
    if isinstance(response, list) and len(response) > 0:
        # Check if the first response is an AIMessage (direct access to its content)
        ai_message = response[0]  # The response is likely an object (not a dictionary)
        if hasattr(ai_message, 'content'):  # Safely check if it has the content attribute
            state["memory"].add_to_memory(ai_message.content)  # Access content from the AIMessage
        
    return {"messages": response}

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
state = {
    "messages": [("user", user_input)],
    "memory": Memory()  # Initialize memory for the session
}

# Simulate a conversation where the chatbot retains the context
events = graph.stream(state, stream_mode="values")

# Print the messages from the stream
for event in events:
    event["messages"][-1].pretty_print()

# Test with another input
user_input = "Tell me about Attention is all you need"
state["messages"] = [("user", user_input)]  # Update the user message
events = graph.stream(state, stream_mode="values")

# Print the messages from the stream
for event in events:
    event["messages"][-1].pretty_print()
