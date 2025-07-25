import os
import logging
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

logger = logging.getLogger(__name__)

class LLMInteraction:
    def __init__(self):
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        ollama_model = os.getenv("OLLAMA_MODEL")

        self.model = Ollama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7,
            num_ctx=4096,
            keep_alive="10m"
        )

        self.prompt_template = PromptTemplate.from_template(
            """You are a helpful and precise JAMB exam assistant. Your primary task is to retrieve and present specific JAMB chemistry past questions, including all their multiple-choice options (A, B, C, D), from the provided context.

            Present the questions as a numbered list.
            For each question that has multiple-choice options in the context, list them immediately after the question. Each option must be on a **separate line**, prefixed with A), B), C), D).

            **Crucially, ensure there is an empty line between the last option of one question and the start of the next numbered question.**

            If a question in the context is a short answer type (without options) or if the options are not present in the context, present it as a numbered question without options.

            If the provided context does not contain relevant questions to the user's query, state clearly that you don't have enough information.

            Do NOT invent questions or options. Only use information directly from the provided context.

            Here is an **EXACT EXAMPLE** of the desired format for multiple-choice questions. Replicate this spacing precisely:

            You have access to the following relevant JAMB past questions:

            Based on the above JAMB questions and the conversation history, fulfill the user's request by listing the relevant questions and their options in the specified format.

            Human: {query}
            AI:"""
        )

    def generate_response_streaming(self, query: str, context: str, chat_history: list):
        context_str = context if context else "No specific relevant questions were found in the database"

        full_context_for_ollama = f"{context_str}\n\nChat History:\n"
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                full_context_for_ollama += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_context_for_ollama += f"AI: {msg.content}\n"

        formatted_prompt = self.prompt_template.format(context=full_context_for_ollama, query=query)

        try:
            for chunk in self.model.stream(formatted_prompt):
                if isinstance(chunk, AIMessageChunk):
                    yield f"data: {chunk.content}\n\n"
                elif isinstance(chunk, str):
                    yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.error(f"Error generating response from LLM: {e}")
            yield f"data: Error: Could not generate response from LLM. Details: {e}\n\n" 