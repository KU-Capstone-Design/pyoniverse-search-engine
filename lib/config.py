from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "pyoniverse-search-engine api"
    stage: str = ""
    version: Literal["v1"] = "v1"
    summary: str = "Search Engine's API"
    mongo_uri: str = ""
    mongo_db: str = ""
    model_dir: str = "resource/model"
    # .env에서 파일을 읽어 Settings의 값을 바꾼다. - Pydantic Configuration이기 때문에 Field 이름 역시 model_config로 고정
    model_config = SettingsConfigDict(env_file=".env")


def get_settings():
    return Settings()
