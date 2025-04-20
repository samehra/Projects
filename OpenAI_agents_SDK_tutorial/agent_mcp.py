"""
Example script demonstrating the use of OpenAI Agents SDK with Model Context Protocol (MCP) support.
This example shows how to use both the filesystem MCP server.

Requirements:
- npm and npx installed (for the filesystem server)
- openai-agents SDK installed
"""

import asyncio
import os
import shutil
from typing import List

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse

from dotenv import load_dotenv
from agents import set_default_openai_key

load_dotenv()

set_default_openai_key(os.environ.get("OPENAI_API_KEY", ""))

async def run_agent_with_servers(mcp_server: MCPServer) -> None:
    """Run an agent with the provided MCP servers and query."""
    agent = Agent(
        name="Assistant",
        instructions=(
            "Use the tools to read the filesystem and answer questions based on those files."
            "Always use those files to answer the user's questions."
        ),
        mcp_servers=[mcp_server],
        model="o3-mini"
    )

    # Run some sample queries
    queries = [
        "List all the files you can access.",
        "What is my #1 favorite book?",
        "What are my favorite songs?",
        "Based on my favorite songs, suggest a new song I might enjoy.",
        "I forgot to add a book to the list. Can you add 'When Breath Becomes Air' by Paul Kalanithi?"
    ]
                
    for query in queries:
        print(f"\n\nRunning query: {query}")
        result = await Runner.run(starting_agent=agent, input=query)
        print(f"Response: {result.final_output}")
    
    # You can also access the full run history if needed
    # for message in result.run.history.messages():
    #     print(f"{message.role}: {message.content}")


async def filesystem_example() -> None:
    """Example using the MCP filesystem server (stdio-based)."""
    print("\n=== FILESYSTEM MCP SERVER EXAMPLE ===")
    
    # Get the current directory and create a path to sample files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "mcp_sample_files")
    
    # Create sample directory if it doesn't exist
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create a sample text file
    with open(os.path.join(samples_dir, "favorites.txt"), "w") as f:
        f.write("My favorite Books:\n")
        f.write("1. The Hitchhiker's Guide to the Galaxy\n")
        f.write("2. Dune\n")
        f.write("3. Neuromancer\n\n")
        f.write("My favorite Songs:\n")
        f.write("1. Bohemian Rhapsody - Queen\n")
        f.write("2. Imagine - John Lennon\n")
        f.write("3. Stairway to Heaven - Led Zeppelin\n")
    
    try:
        # Create a stdio-based MCP server for filesystem access
        async with MCPServerStdio(
            name="Filesystem Server",
            params={
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
            },
            # Optional: Cache the tools list to reduce latency on subsequent runs
            cache_tools_list=True,
        ) as server:
            # Generate a trace ID for debugging
            trace_id = gen_trace_id()
            with trace(workflow_name="MCP Filesystem Example", trace_id=trace_id):
                print(f"View trace: https://platform.openai.com/traces/{trace_id}")
                
                # List available tools
                tools = await server.list_tools()
                print(f"Available tools: {[tool.name for tool in tools]}")
                
                await run_agent_with_servers(server)

    except Exception as e:
        print(f"Error in filesystem example: {e}")


async def main() -> None:
    """Main function to run the filesystem example."""
    print("OpenAI Agents SDK with MCP Support - Filesystem Example")
    
    # Check for npm/npx installation
    if not shutil.which("npx"):
        print("Warning: npx is not installed. The filesystem example may not work.")
        print("Please install it with `npm install -g npx`.")
    
    # Run example
    await filesystem_example()
    
    print("\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main())
