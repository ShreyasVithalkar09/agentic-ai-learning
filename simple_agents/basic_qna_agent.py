from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are an assistant please reply based on the question",
    show_tool_calls=True,
    tools=[DuckDuckGoTools()],
    markdown=True
)

agent.print_response("Who won the CT 2025 between India vs Newzealand finals CT")