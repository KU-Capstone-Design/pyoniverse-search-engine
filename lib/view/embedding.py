"""
DB Products 데이터에 대한 임베딩 값을 설정하는 API Endpoint 관리
"""

from fastapi import APIRouter, Depends

from lib.ai.builder import ModelBuilder
from lib.ai.model.embedding import EmbeddingResponseDto
from lib.dependency_injector import get_model_builder
from lib.view.model.api_response import ApiResponse


router = APIRouter()


@router.post("/reload", tags=["update"])
async def reload(model_builder: ModelBuilder = Depends(get_model_builder)) -> ApiResponse[EmbeddingResponseDto]:
    """
    Embedding 값을 업데이트
    """
    result = model_builder.execute()
    return ApiResponse[EmbeddingResponseDto].ok(result)
