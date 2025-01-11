import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from pydantic import ValidationError
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

# Define a structured response model for story generation
class StoryResponse(BaseModel):
    title: str = Field(..., description="Title of the story")
    content: str = Field(..., description="Main content of the story")
    moral: Optional[str] = Field(None, description="Moral or lesson of the story")


# Create a story-telling agent
story_agent = Agent(
    'gpt-4o-mini',
    result_type=StoryResponse,
    system_prompt="""
    You are a creative storyteller. When given a theme or topic,
    create a short story with a title and optional moral lesson.
    Keep stories concise but engaging.
    """
)


async def demonstrate_run():
    """Demonstrate asynchronous run()"""
    print("\n=== Async Run Example ===")
    result = await story_agent.run("Tell me a story about friendship")
    print(f"Title: {result.data.title}")
    print(f"Story: {result.data.content}")
    if result.data.moral:
        print(f"Moral: {result.data.moral}")


def demonstrate_run_sync():
    """Demonstrate synchronous run_sync()"""
    print("\n=== Sync Run Example ===")
    result = story_agent.run_sync("Tell me a story about courage")
    print(f"Title: {result.data.title}")
    print(f"Story: {result.data.content}")
    if result.data.moral:
        print(f"Moral: {result.data.moral}")


async def demonstrate_run_stream():
    """Demonstrate streaming response with run_stream()"""
    print("\n=== Streaming Run Example ===")
    print("Generating story about wisdom (streaming)...")
    
    async with story_agent.run_stream("Tell me a story about wisdom") as result:
        print("\nStreaming response:")
        
        async for message, last in result.stream_structured(debounce_by=0.1):
            try:
                story = await result.validate_structured_result(
                    message,
                    allow_partial=not last,
                )
                # Print title when it first appears
                # if story.get('title'):
                #     print(f"\nTitle: {story['title']}")
                
                # # Print content incrementally
                # if story.get('content'):
                #     print(story['content'], end='', flush=True)
                
                # # Print moral when available
                # if story.get('moral'):
                #     print(f"\n\nMoral: {story['moral']}")
                    
            except ValidationError:
                continue

        # Show final complete response and usage statistics
        print("\n\nFinal complete response:")
        final_result = await result.get_data()  # Use get_data() instead of data
        print(f"Title: {final_result.title}")
        print(f"Story: {final_result.content}")
        if final_result.moral:
            print(f"Moral: {final_result.moral}")
        
        print("\nStream Usage:", result.usage())


async def main():
    
    await demonstrate_run()
    
    await demonstrate_run_stream()


if __name__ == "__main__":
    demonstrate_run_sync()
    asyncio.run(main()) 