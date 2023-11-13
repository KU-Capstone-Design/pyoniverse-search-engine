from fastapi import APIRouter, Depends
from typing_extensions import Annotated

from lib.config import Settings, get_settings


router = APIRouter()


@router.post("/reload", tags=["update"], responses={"403": {"description": "Not Authorized"}})
async def reload(settings: Annotated[Settings, Depends(get_settings)]):
    return {"ok": 1}
