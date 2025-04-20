from agents import Agent, Runner
import os
from dotenv import load_dotenv
from agents import set_default_openai_key

load_dotenv()

set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

"""
Implementation of a single agent
ref: https://openai.github.io/openai-agents-python/quickstart/
"""

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

result = Runner.run_sync(agent, "What is the square root of 16?")

print(result.final_output)


