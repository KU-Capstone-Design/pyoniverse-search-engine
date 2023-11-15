"""
DB Products 데이터에 대한 임베딩 값을 설정하는 API Endpoint 관리
"""

from fastapi import APIRouter

from lib.ai.embedding import EmbeddingAI
from lib.ai.model.common import EmbeddingResponseDto
from lib.config import get_settings
from lib.view.model.api_response import ApiResponse


settings = get_settings()

router = APIRouter()
embedding_ai = EmbeddingAI(db_uri=settings.mongo_uri, db_name=settings.mongo_db, embedding_dir=settings.embedding_dir)


@router.post("/reload", tags=["update"])
async def reload() -> ApiResponse[EmbeddingResponseDto]:
    """
    Embedding 값을 업데이트
    """
    result = embedding_ai.execute()
    return ApiResponse[EmbeddingResponseDto].ok(result)
