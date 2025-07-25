import logging
import os
from typing import Generator, Optional
from .data_embedding import VectorDB
from dotenv import load_dotenv
from .llm_interaction import LLMInteraction


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME")

data_embedding= VectorDB()
llm_interaction = LLMInteraction()

logger.info("Jamb rag pipeline started.")

def get_rag_response_stream(query: str, subject: Optional[str] = None, year: Optional[int] = None) -> Generator[str, None, None]:
    logger.info(f"RAG process started for query: {query}, subject: '{subject}', year: '{year}'")

    retrieved_docs = data_embedding.retrieve_documents(query=query,k=5,subject=subject,year=year)
    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
    yield from llm_interaction.generate_response_streaming(query, retrieved_docs, [])