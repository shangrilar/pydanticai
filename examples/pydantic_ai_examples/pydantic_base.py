from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

# Define a structured response model
class WeatherResponse(BaseModel):
    temperature: float = Field(..., description="Current temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions (e.g., sunny, rainy)")
    forecast: Optional[str] = Field(None, description="Brief forecast for next few hours")


# Create a basic agent with structured output
weather_agent = Agent(
    # Can use any supported model: openai:gpt-4, gemini-1.5-pro, anthropic:claude-3-opus, etc.
    'gpt-4o-mini',
    # Define the expected response type
    result_type=WeatherResponse,
    # Set a clear system prompt
    system_prompt="""
    You are a weather information assistant. When asked about weather,
    provide accurate temperature, current conditions, and optionally a forecast.
    If you don't have real-time data, provide plausible example data.
    """
)


def main():
    # Example synchronous usage
    result = weather_agent.run_sync(
        "What's the weather like in San Francisco right now?"
    )
    
    # Access structured data
    print("\nWeather Information:")
    print(f"Temperature: {result.data.temperature}°C")
    print(f"Conditions: {result.data.conditions}")
    if result.data.forecast:
        print(f"Forecast: {result.data.forecast}")


if __name__ == "__main__":
    main() 