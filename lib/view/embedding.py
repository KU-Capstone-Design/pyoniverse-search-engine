"""
DB Products 데이터에 대한 임베딩 값을 설정하는 API Endpoint 관리
"""
import gc

from fastapi import APIRouter

from lib.ai.builder import ModelBuilder
from lib.ai.model.embedding import EmbeddingResponseDto
from lib.view.model.api_response import ApiResponse


router = APIRouter()
model_builder: ModelBuilder = None


@router.post("/reload", tags=["update"])
async def reload() -> ApiResponse[EmbeddingResponseDto]:
    """
    Embedding 값을 업데이트
    """
    global model_builder
    if model_builder is None:
        model_builder = ModelBuilder.instance()
    result = model_builder.execute()
    res = ApiResponse[EmbeddingResponseDto].ok(result)
    # Memory free
    return res
