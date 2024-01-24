import os
import sys

import dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

dotenv.load_dotenv(override=True)

if sys.platform == "linux":
    # For Linux, `sqlite3` can be outdated, so use the installed Python package.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def get_embeddings_model(name: str) -> Embeddings:
    if os.getenv("OPENAI_API_TYPE") == "azure":
        return AzureOpenAIEmbeddings(azure_deployment=name)
    if "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbeddings(model=name)
    raise TypeError("Unknown model type.")


def get_chat_model(name: str) -> BaseChatModel:
    if os.getenv("OPENAI_API_TYPE") == "azure":
        return AzureChatOpenAI(azure_deployment=name)
    if "OPENAI_API_KEY" in os.environ:
        return ChatOpenAI(model=name)
    raise TypeError("Unknown model type.")
