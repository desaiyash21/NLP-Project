from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize the language model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=1024,
)

# Define the search tool
def search_kau_website(query: str) -> str:
    """Searches the Kerala Agricultural University website for farming information."""
    search_tool = DuckDuckGoSearchRun()
    # Refine the query to target the KAU website specifically
    refined_query = f"site:kau.in {query}"
    logger.info(f" Searching KAU website with query: '{refined_query}'")
    try:
        search_results = search_tool.run(refined_query)
        if search_results:
            logger.info(f" Found results for '{refined_query}'")
            return search_results
        else:
            logger.warning(f" No specific results found on KAU website for '{query}'. Expanding search.")
            # Fallback to a general search if no results are found on the KAU website
            general_results = search_tool.run(query)
            if general_results:
                return f"I could not find specific information on the Kerala Agricultural University website, but here is some general information I found: {general_results}"
            else:
                return "I was unable to find any information regarding your query."
    except Exception as e:
        logger.error(f" An error occurred during search: {e}")
        return "Sorry, I encountered an error while trying to search for information."


# List of tools available to the agent
tools = [search_kau_website]

# Define the system prompt for the agent's persona and instructions
system_prompt = """You are Krishi Sakhi, a friendly and helpful AI assistant for farmers in Kerala.
Your name means 'Farmer's Friend'.

Your first task is to determine the user's preferred language. Start the conversation by asking in both English and Malayalam which language they would like to use. For example: 'Hello, I am Krishi Sakhi. For our conversation, would you prefer English or Malayalam? | Namaskaram, njan Krishi Sakhi. Ningalkku English-il aano atho Malayalam-il aano samsarikkan thaalparyam?'

Once the user has chosen a language, you MUST respond ONLY in that language for the rest of the conversation.

Your purpose is to answer farming-related questions about crops, pests, diseases, and modern agricultural techniques.
You will use your search tool to find the most accurate and up-to-date information from the Kerala Agricultural University (KAU) website.
If you cannot find an answer on the KAU website, you can use your general knowledge to provide a helpful response.
Your output will be converted to audio, so keep your answers clear and conversational. Avoid using complex symbols or formatting."""

# Set up memory for the agent
memory = InMemorySaver()

# Create the agent with the model, tools, prompt, and memory
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

# Configure the agent for a specific user thread
agent_config = {"configurable": {"thread_id": "kerala_farmer_assistant_v1"}}

# --- Example Usage (for testing) ---
# To test the agent, you can run the following code:
#
# if __name__ == "__main__":
#     import asyncio
#
#     async def run_agent():
#         response = agent.invoke(
#             {"messages": [("user", "നെല്ലിന് വരുന്ന പ്രധാന രോഗങ്ങൾ ഏതൊക്കെയാണ്?")]},
#             config=agent_config,
#         )
#         # The response is a dictionary, and the agent's message is in the 'messages' key
#         ai_message = response['messages'][-1].content
#         print(ai_message)
#
#     asyncio.run(run_agent())


