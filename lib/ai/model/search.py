"""
SearchAI에서만 사용하는 모델 정의
"""
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """
    검색 결과
    """

    score: float = field()
    id: int = field()
    name: str = field()
