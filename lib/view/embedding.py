from fastapi import APIRouter


router = APIRouter()


@router.post("/reload", tags=["update"], responses={"403": {"description": "Not Authorized"}})
async def reload():
    # TODO : Embedding 갱신
    return {"data": "ok"}
