"""
EmbeddingAI, SearchAI에서 공통으로 사용하는 Data Model 정의
"""
from dataclasses import dataclass, field
from typing import Literal, Union

from numpy.core._multiarray_umath import ndarray


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
    type: Literal["sentence", "lexical"] = field()
    embedding_path: str = field()
