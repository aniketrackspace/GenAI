test1_updated: Langchain Chatbot with Groq and Langgraph
Description:
This Python program builds an interactive chatbot using Langchain, Groq, and Langgraph. It leverages the ChatGroq model (Gemma2-9b-It) and constructs a stateful conversation graph to handle user inputs. 

Features:
1. Environment Configuration: Loads API keys from a .env file (GROQ_API_KEY and LANGSMITH_API_KEY).
2. Graph-Based Chat Flow: Utilizes Langgraph to structure the chatbot logic with start and end nodes.
3. Streaming Responses: Streams responses in real-time, displaying assistant replies as they are generated.
4. Interactive CLI Interface: Users can chat directly in the terminal, with the option to exit by typing quit or q.
5. Responses will be saved in langsmith. 

Setup:
Install the necessary packages:


pip install langchain langchain_groq langgraph python-dotenv

Create a .env file with your API keys:
GROQ_API_KEY=your_groq_api_key  
LANGSMITH_API_KEY=your_langsmith_api_key

Run the program:

python test1_updated.py
Chat with the assistant, and type quit or q to exit.
