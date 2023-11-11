from fastapi import APIRouter


router = APIRouter()


@router.get("/{query}")
async def search(query: str):
    return {"data": {"query": query, "results": [1, 2]}}
