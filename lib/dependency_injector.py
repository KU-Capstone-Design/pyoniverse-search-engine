from functools import lru_cache

from fastapi import Depends
from sentence_transformers import CrossEncoder

from lib.ai.builder import ModelBuilder
from lib.ai.loader import ModelLoader
from lib.ai.search import SearchAI
from lib.config import Settings, get_settings


async def get_model_builder(settings: Settings = Depends(get_settings)):
    return ModelBuilder(
        db_uri=settings.mongo_uri,
        db_name=settings.mongo_db,
        model_dir=settings.model_dir,
    )


async def get_model_loader(settings: Settings = Depends(get_settings)):
    return ModelLoader(model_dir=settings.model_dir)


async def get_search_ai(loader: ModelLoader = Depends(get_model_loader), settings: Settings = Depends(get_settings)):
    return SearchAI(
        models=loader.load(), cross_encoder=CrossEncoder("bongsoo/kpf-cross-encoder-v1"), version=settings.version
    )
