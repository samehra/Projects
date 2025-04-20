from agents import Agent, Runner
import os
from dotenv import load_dotenv
from agents import set_default_openai_key

load_dotenv()

set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")

print(result.final_output)

# check traces here: https://platform.openai.com/traces