from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def get_weather(city: str) -> str:
    """Get weather information for a given city."""
    return f"It's always sunny in {city}!"


model = ChatOllama(
    base_url="http://13.48.29.164:11434/",
    model="llama3.2"
)


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in sf?"}
        ]
    }
)
