"""Creates the two models. To switch provider, only this file changes."""
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import config


def get_embed_model():
    return GoogleGenAIEmbedding(model_name=config.EMBED_MODEL, api_key=config.GOOGLE_API_KEY)


def get_llm():
    return GoogleGenAI(
        model=config.LLM_MODEL,
        api_key=config.GOOGLE_API_KEY,
        temperature=config.TEMPERATURE,
    )
