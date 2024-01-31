from typing import Callable, Dict, Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

ModelType = Literal["openai", "azure", "hf"]
PredefinedRelevanceScoreFn = Literal["l2", "ip", "cosine"]
RelevanceScoreFn = Union[PredefinedRelevanceScoreFn, Callable[[float], float]]
RetrieverSearchType = Literal["similarity", "similarity_score_threshold", "mmr"]
LLM = Union[BaseChatModel, BaseLLM]
RAGMode = Literal["docs", "sql"]


MODEL_TYPES: Dict[ModelType, str] = {
    "openai": "OpenAI",
    "azure": "Azure OpenAI",
    "hf": "Hugging Face",
}
PREDEFINED_RELEVANCE_SCORE_FNS: Dict[PredefinedRelevanceScoreFn, str] = {
    "l2": "Squared euclidean distance",
    "ip": "Inner product",
    "cosine": "Cosine similarity",
}
RETRIEVER_SEARCH_TYPES: Dict[RetrieverSearchType, str] = {
    "similarity": "Similarity",
    "similarity_score_threshold": "Similarity with score threshold",
    "mmr": "Maximal marginal relevance (MMR)",
}
RAG_MODES: Dict[RAGMode, str] = {"docs": "Documents", "sql": "SQL"}
