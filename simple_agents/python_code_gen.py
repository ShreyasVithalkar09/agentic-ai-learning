from agno.agent import Agent
from agno.tools.python import PythonTools
from agno.models.groq import Groq

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[PythonTools()],
    show_tool_calls=True
)

agent.print_response("Write a python script for fibonacci series and display the result till the 10th number")