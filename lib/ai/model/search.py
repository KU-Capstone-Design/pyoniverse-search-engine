"""
SearchAI에서만 사용하는 모델 정의
"""
from typing import List, Literal

from pydantic import BaseModel


class SearchResponseDto(BaseModel):
    version: str
    engine_type: Literal["ML", "ALGORITHM"]
    results: List[int]
