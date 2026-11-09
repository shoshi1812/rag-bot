"""Checks the API key and lists the models your account can use.

    python check_models.py

Copy a model name (without "models/") into LLM_MODEL in the .env file.
"""
import config  # first: loads .env and the certificate fix

from google import genai

client = genai.Client(api_key=config.GOOGLE_API_KEY)
for model in client.models.list():
    actions = model.supported_actions or []
    if "generateContent" in actions:
        print("answer model:   ", model.name)
    if "embedContent" in actions:
        print("embedding model:", model.name)
