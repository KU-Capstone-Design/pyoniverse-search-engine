from fastapi import APIRouter, Depends

from lib.ai.model.search import SearchResponseDto
from lib.ai.search import SearchAI
from lib.dependency_injector import get_search_ai
from lib.view.model.api_response import ApiResponse


router = APIRouter()


@router.get("/{query}")
async def search(query: str, search_ai: SearchAI = Depends(get_search_ai)) -> ApiResponse[SearchResponseDto]:
    """
    검색 결과 반환
    :param query: 검색 쿼리
    :param search_ai: 검색 AI
    """
    result = search_ai.search(query)
    return ApiResponse.ok(result)
