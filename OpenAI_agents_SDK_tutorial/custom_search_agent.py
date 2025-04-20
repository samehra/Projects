from agents import Agent, Runner, FunctionTool, function_tool
import os
from typing_extensions import TypedDict, Any, List
from exa_py import Exa
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
from agents import set_default_openai_key

load_dotenv()
set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

# Define the input and output models
class ExaSearchInput(TypedDict):
    query: str = Field(description="The search query to look for on the web")
    num_results: int = Field(default=5, description="Number of search results to return")

class ExaSearchResult(TypedDict):
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    text: str = Field(description="Extracted text content from the search result")

class ExaSearchOutput(TypedDict):
    results: List[ExaSearchResult] = Field(description="List of search results")

# Initialize the Exa client (do this outside of the function to avoid recreating it)
exa_api_key = os.environ.get("EXA_API_KEY", "")
exa_client = Exa(api_key=exa_api_key) if exa_api_key else None

# Define the search function
@function_tool
def exa_search(input_data: ExaSearchInput) -> ExaSearchOutput:

    """Fetch web data on the user query.

    Args:
        query: The query to fetch web data for.
    """

    if not exa_client:
        # Return error if API key not available
        return ExaSearchOutput(
            results=[
                ExaSearchResult(
                    title="Error",
                    url="",
                    text="EXA_API_KEY environment variable not set. Search functionality unavailable."
                )
            ]
        )
    
    try:
        print("Looking for relevant information from the web!")
        # Search using the exa-py client
        raw_results = exa_client.search_and_contents(
            query=input_data["query"],
            num_results=input_data["num_results"],
            use_autoprompt=True,
            text=True,
            type="keyword"
        )
        
        # Format the results
        formatted_results = []
        for result in raw_results.results:
            print("\n =============================== \n")
            print(f"Found result: {result.title}")
            print(f"URL: {result.url}")
            print(f"Text: {result.text[:100]}...")
            formatted_results.append(
                ExaSearchResult(
                    title=result.title,
                    url=result.url,
                    text=result.text[:1000] + "..." if len(result.text) > 1000 else result.text
                )
            )
        print("\n =============================== \n")
        print("Found some information from the web! Now summarizing...\n\n")
        
        return ExaSearchOutput(results=formatted_results)
        
    except Exception as e:
        # Handle errors gracefully
        print(f"Error: {str(e)}")
    
        return ExaSearchOutput(
            results=[
                ExaSearchResult(
                    title="Error",
                    url="",
                    text=f"Error when searching: {str(e)}"
                )
            ]
        )
        

# Create an agent with the search tool
agent = Agent(
    name="Web Researcher",
    instructions="You are a web researcher who can search for current information online. If you don't have an answer, say so. Use the exa_search tool to search for information on the web.",
    tools=[exa_search]
)

# Run the agent with a query
result = Runner.run_sync(agent, "What is the latest news on Tesla?")
print(result.final_output)