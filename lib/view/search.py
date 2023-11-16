from fastapi import APIRouter

from lib.ai.model.search import SearchResponseDto
from lib.ai.search import SearchAI
from lib.view.model.api_response import ApiResponse


router = APIRouter()
search_ai: SearchAI = SearchAI.instance()


@router.get("/{query}")
async def search(query: str) -> ApiResponse[SearchResponseDto]:
    """
    검색 결과 반환
    :param query: 검색 쿼리
    :param search_ai: 검색 AI
    """
    result = search_ai.search(query)
    return ApiResponse.ok(result)
