import typer
from typing import Optional
from rich.prompt import Prompt

from phi.agent import Agent
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.model.groq import Groq

"""Used this SentenceTransformerEmbedder embedder because we are not using 
OpenAI and by default it uses OpenAI embedder"""
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder


import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

vector_db = LanceDb(
    table_name="recipes",
    uri="/tmp/lancedb",
    search_type=SearchType.keyword,
    embedder=SentenceTransformerEmbedder()
)

# Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Comment out after first run
knowledge_base.load(recreate=True)

def lancedb_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        model=Groq(id="qwen-2.5-32b"),
        run_id=run_id,
        knowledge=knowledge_base,
        user_id=user,
        show_tool_calls=True,
        debug_mode=True
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(lancedb_agent)