import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    validator,
)
from typing_extensions import Self

from .types import ModelType, RelevanceScoreFn, RetrieverSearchType


def guess_model_type() -> ModelType:
    if os.getenv("OPENAI_API_TYPE") == "azure":
        return "azure"
    if "OPENAI_API_KEY" in os.environ:
        return "openai"
    return "hf"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_type: Optional[ModelType]
    llm_name: str = Field(..., min_length=1)
    relevance_score_fn: RelevanceScoreFn = "l2"
    k: PositiveInt = 4
    search_type: RetrieverSearchType = "similarity"
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    fetch_k: PositiveInt = 20
    lambda_mult: float = Field(0.5, ge=0.0, le=1.0)
    embeddings_model_type: Optional[ModelType]
    embeddings_model_name: str = Field(..., min_length=1)

    docs_db_directory: Path = Path("./db-docs")
    docs_db_collection: str = Field("docs_store", min_length=1)
    questions_db_directory: Path = Path("./db-questions")
    questions_db_collection: str = Field("questions_store", min_length=1)

    @validator("fetch_k")
    def _(cls, fetch_k: int, values: Dict[str, Any]) -> int:
        k = values["k"]
        if fetch_k < k:
            raise ValueError(
                f"`fetch_k`({fetch_k}) should be greater than or equal to `k`({k})"
            )
        return fetch_k

    @property
    def full_llm_name(self) -> str:
        return f"{self.llm_type}:{self.llm_name}"

    @property
    def full_embeddings_model_name(self) -> str:
        return f"{self.embeddings_model_type}:{self.embeddings_model_name}"

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


filepath = Path(os.getenv("RAG_SETTINGS", "./settings.yaml"))
settings = Settings.from_yaml(filepath)
