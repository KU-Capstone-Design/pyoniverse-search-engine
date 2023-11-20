import logging
import os
import pickle
from pathlib import Path
from typing import List

import boto3
from boto3_type_annotations.s3 import ServiceResource

from lib.ai.model.embedding import SearchModel
from lib.config import get_settings


class ModelLoader:
    """
    model_dir에 있는 검색 모델을 로드
    """

    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            settings = get_settings()
            cls.__instance = cls(model_dir=settings.model_dir)
        return cls.__instance

    def __init__(self, model_dir: str = "resource/model"):
        assert model_dir
        self.__model_dir = Path(model_dir)
        if not self.__model_dir.exists():
            self.__model_dir.mkdir(parents=True, exist_ok=True)
        if not self.__model_dir.is_dir():
            raise RuntimeError(f"{model_dir} isn't directory")
        self.logger = logging.getLogger(__name__)

    def load(self) -> List[SearchModel]:
        models: List[SearchModel] = []
        resource: ServiceResource = boto3.resource("s3")
        for file_obj in resource.Bucket(os.getenv("BUCKET")).objects.filter(Prefix=os.getenv("BUCKET_KEY")):
            if file_obj.key.endswith(".pickle"):
                saved_path = f"{self.__model_dir}/{file_obj.key.split('/')[-1]}"
                self.logger.info(f"Download s3://{os.getenv('BUCKET')}/{file_obj.key} to {saved_path}")
                resource.meta.client.download_file(Bucket=os.getenv("BUCKET"), Key=file_obj.key, Filename=saved_path)
        for model_path in self.__model_dir.glob("*.pickle"):
            try:
                with open(model_path, "rb") as fd:
                    model = pickle.load(fd)
            except Exception as e:
                self.logger.error(e)
                raise RuntimeError(f"Cannot load {model_path}")
            else:
                if not isinstance(model, SearchModel):
                    raise RuntimeError(f"{model_path} isn't SearchModel")
                models.append(model)
        if not models:
            raise RuntimeError(f"Empty: {self.__model_dir}")
        return models
