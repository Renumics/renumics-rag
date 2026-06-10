import dataclasses
from typing import Any, Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

ModelType = Literal["openai", "azure", "hf"]
Device = Literal["cpu", "gpu"]
PredefinedRelevanceScoreFn = Literal["l2", "ip", "cosine"]
RelevanceScoreFn = PredefinedRelevanceScoreFn | Callable[[float], float]
RetrieverSearchType = Literal["similarity", "similarity_score_threshold", "mmr"]
LLM = BaseChatModel | BaseLLM


MODEL_TYPES: dict[ModelType, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "hf": "Hugging Face",
}
PREDEFINED_RELEVANCE_SCORE_FNS: dict[PredefinedRelevanceScoreFn, str] = {
    "l2": "Squared euclidean distance",
    "ip": "Inner product",
    "cosine": "Cosine similarity",
}
RETRIEVER_SEARCH_TYPES: dict[RetrieverSearchType, str] = {
    "similarity": "Similarity",
    "similarity_score_threshold": "Similarity with score threshold",
    "mmr": "Maximal marginal relevance (MMR)",
}


# Types for Streamlit app
Role = Literal["user", "assistant", "source"]


AVATARS: dict[Role, Any] = {"user": "🧐", "assistant": "🤖", "source": "📚"}


@dataclasses.dataclass
class Message:
    role: Role
    content: str


@dataclasses.dataclass
class NestedMessage(Message):
    subcontents: list[str]
