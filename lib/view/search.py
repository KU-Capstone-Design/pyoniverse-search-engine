from fastapi import APIRouter

from lib.ai.model.search import SearchResponseDto
from lib.ai.search import SearchAI
from lib.config import get_settings
from lib.view.model.api_response import ApiResponse


settings = get_settings()

router = APIRouter()
search_ai = SearchAI()


@router.get("/{query}")
async def search(query: str) -> ApiResponse[SearchResponseDto]:
    """
    검색 결과 반환
    :param query: 검색 쿼리
    """
    result = SearchResponseDto(
        version=settings.version,
        engine_type="ML",
        results=[1, 2, 3],
    )
    return ApiResponse.ok(result)
