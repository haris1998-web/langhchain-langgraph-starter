import os
from dataclasses import dataclass

import dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

dotenv.load_dotenv()

SYSTEM_PROMPT = """
You are an expert weather forecaster who speaks in puns.

You have two tools:
- get_user_location: retrieves the user's location
- get_weather_for_location: retrieves the weather for a specific location

Always follow this process:
1. Call get_user_location to determine where the user is.
2. Then, immediately call get_weather_for_location using the output from first tool call.
3. Finally, respond with a punny weather report using both pieces of information.

Do not respond to the user until both tools have been called.
"""


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@dataclass
class ContextFormat:
    """Custom runtime context schema."""
    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[ContextFormat]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


@dataclass
class ResponseFormat(BaseModel):
    """Response schema for the agent"""
    punny_response: str
    weather_conditions: str | None = None


checkpointer = InMemorySaver()

ollama_model = ChatOllama(
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    model="llama3.2",
    temperature=0.8,
)

agent = create_agent(
    model=ollama_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=ContextFormat,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like?"}]},
    config=config,
    context=ContextFormat(user_id="1"),
)

print(response['structured_response'])
