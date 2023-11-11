"""
EmbeddingAI에서만 사용하는 Data Model 정의
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelMeta:
    """
    Model Meta 정보
    """

    name: str = field()
    type: Literal["sentence", "lexical"] = field()
    embedding_path: str = field()
