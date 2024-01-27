import dataclasses
import os
from pathlib import Path
from typing import Optional

import yaml
from typing_extensions import Self

from .types import ModelType, RelevanceScoreFn, RetrieverSearchType


def guess_model_type() -> ModelType:
    if os.getenv("OPENAI_API_TYPE") == "azure":
        return "azure"
    if "OPENAI_API_KEY" in os.environ:
        return "openai"
    return "hf"


@dataclasses.dataclass
class Settings:
    llm_type: Optional[ModelType]
    llm_name: str
    relevance_score_fn: RelevanceScoreFn
    k: int
    search_type: RetrieverSearchType
    score_threshold: float
    fetch_k: int
    lambda_mult: float
    embeddings_model_type: Optional[ModelType]
    embeddings_model_name: str

    docs_db_directory: Path = Path("./db-docs")
    docs_db_collection: str = "docs_store"
    questions_db_directory: Path = Path("./db-questions")
    questions_db_collection: str = "questions_store"

    @classmethod
    def from_yaml(cls, filepath: Path) -> Self:
        assert filepath.is_file()
        assert filepath.suffix.lower() in (".yml", ".yaml")
        with filepath.open() as file:
            raw_settings = yaml.safe_load(file)
        if "llm_type" in raw_settings and raw_settings["llm_type"] is None:
            raw_settings["llm_type"] = guess_model_type()
        if (
            "embeddings_model_type" in raw_settings
            and raw_settings["embeddings_model_type"] is None
        ):
            raw_settings["embeddings_model_type"] = guess_model_type()
        return cls(**raw_settings)

    @property
    def full_llm_name(self) -> str:
        return f"{settings.llm_type}:{settings.llm_name}"

    @property
    def full_embeddings_model_name(self) -> str:
        return f"{settings.embeddings_model_type}:{settings.embeddings_model_name}"


settings_filepath = Path(os.getenv("RAG_SETTINGS", "./settings.yaml"))
settings = Settings.from_yaml(settings_filepath)
