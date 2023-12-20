import logging
import pickle
from pathlib import Path
from typing import List

import boto3
from boto3_type_annotations.s3 import ServiceResource

from lib.ai.model.embedding import SearchModel
from lib.config import Settings, get_settings


class ModelLoader:
    """
    model_dir에 있는 검색 모델을 로드
    """

    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            settings: Settings = get_settings()
            cls.__instance = cls(model_dir=settings.model_dir, bucket=settings.bucket, bucket_key=settings.bucket_key)
        return cls.__instance

    def __init__(self, bucket: str, bucket_key: str, model_dir: str = "resource/model"):
        assert model_dir and bucket and bucket_key
        self.__model_dir = Path(model_dir)
        if not self.__model_dir.exists():
            self.__model_dir.mkdir(parents=True, exist_ok=True)
        if not self.__model_dir.is_dir():
            raise RuntimeError(f"{model_dir} isn't directory")

        self.__bucket = bucket
        self.__bucket_key = bucket_key

        self.logger = logging.getLogger(__name__)

    def load(self) -> List[SearchModel]:
        # lexical model 로드
        models: List[SearchModel] = []
        for model_path in self.__model_dir.glob("bm250k.pickle"):
            models.append(self.__unpickle(str(model_path)))
            break
        if not models:
            resource: ServiceResource = boto3.resource("s3")
            for file_obj in resource.Bucket(self.__bucket).objects.filter(Prefix=self.__bucket_key):
                if file_obj.key.endswith("bm250k.pickle"):
                    saved_path = f"{self.__model_dir}/{file_obj.key.split('/')[-1]}"
                    self.logger.info(f"Download s3://{self.__bucket}/{file_obj.key} to {saved_path}")
                    resource.meta.client.download_file(Bucket=self.__bucket, Key=file_obj.key, Filename=saved_path)
                    models.append(self.__unpickle(saved_path))
                    break
            if not models:
                raise RuntimeError(f"Empty: {self.__model_dir}")
        return models

    def __unpickle(self, model_path: str) -> SearchModel:
        try:
            with open(model_path, "rb") as fd:
                model = pickle.load(fd)
        except Exception as e:
            self.logger.error(e)
            raise RuntimeError(f"Cannot load {model_path}")
        else:
            if not isinstance(model, SearchModel):
                raise RuntimeError(f"{model_path} isn't SearchModel")
            return model
