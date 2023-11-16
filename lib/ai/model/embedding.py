"""
EmbeddingAI, SearchAI에서 공통으로 사용하는 Data Model 정의
"""
from dataclasses import dataclass, field
from typing import Any, List, Literal, Union

from numpy.core._multiarray_umath import ndarray
from pydantic import BaseModel, ConfigDict


@dataclass
class Embedding:
    """
    Embedding Information
    """

    id: int = field()
    name: str = field()
    embedding: Union[str, ndarray] = field()


@dataclass
class ModelMeta:
    """
    Model Meta 정보
    """

    name: str = field()
    model_path: str = field()
    type: Literal["sentence", "lexical"] = field()


class EmbeddingResponseDto(BaseModel):
    models: List[ModelMeta]


class SearchModel(BaseModel):
    type: Literal["sentence", "lexical"]
    engine: Any
    embeddings: List[Embedding]

    model_config = ConfigDict(arbitrary_types_allowed=True)
