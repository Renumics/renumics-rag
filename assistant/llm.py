import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


def get_embeddings_model(name: str) -> Embeddings:
    if os.getenv("OPENAI_API_TYPE") == "azure":
        return AzureOpenAIEmbeddings(azure_deployment=name)
    if "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbeddings(model=name)
    raise TypeError("Unknown API type.")
