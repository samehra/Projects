from agents import Agent, FileSearchTool, Runner, WebSearchTool, FunctionTool, function_tool
import os
from dotenv import load_dotenv
from agents import set_default_openai_key
load_dotenv()
set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

"""
The example below uses `WebSearchTool`. 
Other supported tools include: 
- `FileSearchTool`(to retrieve information from OpenAI's vector stores), 
- `ComputerTool` (allows computer use tasks)
"""


# assistant agent
agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool()
    ],
)

result = Runner.run_sync(agent, "Which coffee shop should I go to, taking into the weather today in Belize?")
print(result.final_output)