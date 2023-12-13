import logging
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import boto3
from boto3_type_annotations.s3 import Client
from pykospacing import Spacing
from pymongo import MongoClient, ReadPreference, UpdateOne
from pymongo.errors import ConfigurationError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from lib.ai.model.embedding import Embedding, EmbeddingResponseDto, ModelMeta, SearchModel
from lib.config import get_settings


class ModelBuilder:
    """
    Lexical, Sentence Search Model과 이에 대응하는 Embedding을 만들고, 이것을 Pickling하여 저장하는 클래스
    """

    Data = namedtuple("Data", ["id", "company", "name"])
    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            settings = get_settings()
            cls.__instance = cls(
                db_uri=settings.mongo_uri,
                db_name=settings.mongo_db,
                model_dir=settings.model_dir,
                bucket=settings.bucket,
                bucket_key=settings.bucket_key,
            )
        return cls.__instance

    def __init__(
        self,
        db_uri: str = None,
        db_name: str = None,
        model_dir: str = "resource/model",
        bucket: str = None,
        bucket_key: str = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.__lexical_models = {}
        self.__sentence_models = {}
        # Model을 저장할 경로 저장
        self.__model_dir = Path(model_dir)
        if not self.__model_dir.exists():
            # 새로운 폴더 생성
            self.__model_dir.mkdir(parents=True, exist_ok=True)
        if not self.__model_dir.is_dir():
            raise RuntimeError(f"{self.__model_dir} Not Directory")
        self.__db_uri = db_uri
        if not self.__db_uri:
            raise RuntimeError(f"{self.__db_uri} is empty")
        self.__db_name = db_name
        if not self.__db_name:
            raise RuntimeError(f"{self.__db_name} is empty")

        assert bucket and bucket_key
        self.__bucket = bucket
        self.__bucket_key = bucket_key

    def execute(self, clean: bool = False) -> EmbeddingResponseDto:
        """
        사용 가능한 모든 모델의 임베딩을 만들어 저장
        @params
        1. clean: 이전 임베딩 데이터를 버릴지 판단하기(True: 버리고 재생성)
        @step
        1. Data 전처리
        2. lexical, sentence model & embedding 생성
        3. embedding 저장
        4. embedding 저장 경로 반환
        :return: [ModelMeta]
        """
        lexical_model_names = []
        sentence_model_names = []
        self.logger.info("Preprocess Data")
        data = self.preprocess_data(clean=clean)
        self.logger.info("Build lexical model")
        model_name = self.make_bm250k_model(data)
        lexical_model_names.append(model_name)
        # self.logger.info("Build sentence model")
        # model_name = self.make_sroberta_sts_model(data, clean=clean)
        # sentence_model_names.append(model_name)
        # model_name = self.make_sroberta_multitask_model(data, clean=clean)
        # sentence_model_names.append(model_name)
        result: List[ModelMeta] = []
        result += [
            ModelMeta(name=name, type="lexical", model_path=self.get_model_path(name)) for name in lexical_model_names
        ]
        # result += [
        #     ModelMeta(name=name, type="sentence", model_path=self.get_model_path(name))
        #     for name in sentence_model_names
        # ]
        return EmbeddingResponseDto(models=result)

    @classmethod
    def clean(cls):
        cls.__instance = None

    def is_lexical_model(self, model_name: str) -> bool:
        """
        :param model_name: get_...로 반환되는 key
        """
        return model_name in self.__lexical_models

    def is_sentence_model(self, model_name: str) -> bool:
        """
        :param model_name: get_...로 반환되는 key
        """
        return model_name in self.__sentence_models

    def preprocess_data(self, clean: bool = False) -> List["ModelBuilder.Data"]:
        """
        @fields
        self.__data_path: Json 형식의 임베딩될 데이터 파일 위치
        clean: 이전 데이터(spaced name)을 버릴지 여부 판단(True: 버리고 재생성)
        @steps
        1. 상품 제조 회사 찾기
        2. 상품 임베딩을 위해 단어 공백 띄우기(맥락을 고려하기 위함)
        @returns
        [{"id": d["id"], "name": d["name"], "company": ...} for d in data]
        """
        try:
            client = MongoClient(self.__db_uri)
            client.admin.command("ping")
        except ConfigurationError as e:
            self.logger.error("Cannot connect db")
            raise RuntimeError(e)
        db = client.get_database(self.__db_name, read_preference=ReadPreference.SECONDARY_PREFERRED)
        data = list(db["products"].find(projection={"_id": False, "id": True, "name": True, "spaced_name": True}))
        # Kospacing
        updated = []
        spacing = Spacing()
        response: List[ModelBuilder.Data] = []  # dictionary list {id, company, name}
        for datum in data:
            p = datum["name"].find(")")
            if p != -1:
                company = datum["name"][:p]
            else:
                company = None
            try:
                # spaced_name이 없다면 새롭게 생성 - profile 결과로 spacing 연산의 비용이 크다.
                if clean or "spaced_name" not in datum:
                    datum["spaced_name"] = spacing(datum["name"], ignore="none")
                    updated.append({"id": datum["id"], "spaced_name": datum["spaced_name"]})
            except Exception as e:
                raise RuntimeError(e)
            response.append(ModelBuilder.Data(id=datum["id"], company=company, name=datum["spaced_name"]))
        # update db
        modified_count = 0
        for p in range(0, len(updated), 100):
            buffer = [
                UpdateOne(filter={"id": d["id"]}, update={"$set": {"spaced_name": d["spaced_name"]}})
                for d in updated[p : p + 100]
            ]
            res = client.get_database(self.__db_name)["products"].bulk_write(buffer)
            modified_count += res.modified_count
        self.logger.info(f"Update {self.__db_name}.products: {modified_count}")
        return response

    def make_bm250k_model(self, data: List["ModelBuilder.Data"]) -> str:
        """
        Lexical Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: model_name
        """
        embeddings = [Embedding(embedding=d.name, id=d.id, name=d.name) for d in data]
        model = BM25Okapi([doc.embedding.split(" ") for doc in embeddings])
        self.__lexical_models["bm250k"] = (model, embeddings)
        self.save(model_name="bm250k")
        return "bm250k"

    def make_sroberta_multitask_model(self, data: List["ModelBuilder.Data"], clean: bool = False) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :param clean: 이전 임베딩 데이터를 지울지 여부(True: 지우고 재생성)
        :return: model_name
        """
        model_name = "sroberta_multitask"
        model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        embedding = self.__make_embedding(model=model, data=data, model_name=model_name, clean=clean)
        self.__sentence_models[model_name] = (model, embedding)
        self.logger.info(f"Save embedding for {model}")
        self.save(model_name=model_name)
        return model_name

    def make_sroberta_sts_model(self, data: List["ModelBuilder.Data"], clean: bool = False) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :param clean: 이전 임베딩 데이터를 지울지 여부(True: 지우고 재생성)
        :return: model_name
        """
        model_name = "sroberta_sts"
        model = SentenceTransformer("jhgan/ko-sroberta-sts")
        embedding = self.__make_embedding(model=model, data=data, model_name=model_name, clean=clean)
        self.__sentence_models[model_name] = (model, embedding)
        self.logger.info(f"Save embedding for {model}")
        self.save(model_name=model_name)
        return model_name

    def __make_embedding(
        self, model: SentenceTransformer, data: List["ModelBuilder.Data"], model_name: str, clean: bool = False
    ) -> List[Embedding]:
        """
        :param model: SentenceTransformer
        :param data: [{"id": .., "company": .., "name": ...}]
        :param clean: 이전 임베딩 데이터를 지울지 여부(True: 지우고 재생성)
        :return: [{"embedding": ..., "id": ..., "name": ..., "company": ...}]
        """
        self.logger.info(f"Load embedding for {model}")
        try:
            # embeddings = self.get_embedding(model_name=model_name)
            embeddings = []
        except Exception as e:
            embeddings = []
        self.logger.info(f"Make embedding for {model}")
        already_embedded = set(e.id for e in embeddings)
        for datum in data:
            if clean or datum.id not in already_embedded:
                encoded = model.encode(
                    datum.name, show_progress_bar=False, batch_size=1, convert_to_tensor=True, normalize_embeddings=True
                )
                embeddings.append(
                    Embedding(
                        embedding=encoded,
                        id=datum.id,
                        name=datum.name,
                    )
                )
        return embeddings

    def save(self, model_name: str) -> str:
        """
        SearchModel 형식으로 Model과 Embeddings를 저장한다.
        :param model_name: 저장된 모델 이름
        :return: 저장 경로
        """
        saved_path = self.get_model_path(model_name)
        if self.is_lexical_model(model_name):
            engine, embeddings = self.__lexical_models[model_name]
            type_ = "lexical"
        elif self.is_sentence_model(model_name):
            engine, embeddings = self.__sentence_models[model_name]
            type_ = "sentence"
        else:
            raise RuntimeError(f"{model_name} not in lexical or sentence models")

        search_model = SearchModel(type=type_, engine=engine, embeddings=embeddings)
        with open(saved_path, "wb") as fd:
            pickle.dump(search_model, fd)
        self.logger.info(f"{model_name} is saved at {saved_path}")
        client: Client = boto3.client("s3")
        try:
            client.upload_file(saved_path, self.__bucket, self.get_s3_model_key(model_name))
            self.logger.info(
                f"{model_name} is uploaded at " f"s3://{self.__bucket}/{self.get_s3_model_key(model_name)}"
            )
        except Exception as e:
            raise RuntimeError(e)
        return str(saved_path)

    def get_embedding(self, model_name: str) -> List[Embedding]:
        """
        save_embedding으로 저장된 embedding 가져오기
        :return: embedded data
        """
        path = self.get_model_path(model_name)
        if not Path(path).exists():
            self.logger.info(f"Download model from s3://{self.__bucket}/{self.get_s3_model_key(model_name)}")
            client: Client = boto3.client("s3")
            try:
                client.download_file(self.__bucket, self.get_s3_model_key(model_name), self.get_model_path(model_name))
            except Exception as e:
                raise RuntimeError(e)
        try:
            with open(path, "rb") as fd:
                model: SearchModel = pickle.load(fd)
                self.logger.info(f"Load {model_name} from {path}")
                data = model.embeddings
        except Exception as e:
            self.logger.info(f"{model_name} wasn't saved")
            data = []
        return data

    def get_model(
        self, model_name: str
    ) -> Dict[Literal["type", "model"], Union[Literal["lexical", "sentence"], Tuple[Any, List[Embedding]]]]:
        """
        model_name에 해당하는 모델의 타입과 모델 반환
        :returns: {"type": lexical|sentence, "model": ...}
        """
        result: Dict[Literal["type", "model"], Any] = {}
        if self.is_lexical_model(model_name):
            result["type"] = "lexical"
            result["model"] = self.__lexical_models[model_name]
        elif self.is_sentence_model(model_name):
            result["type"] = "sentence"
            result["model"] = self.__sentence_models[model_name]
        else:
            raise RuntimeError(f"{model_name} is not lexical or sentence model")
        return result

    def get_model_path(self, model_name: str) -> str:
        return str(self.__model_dir / f"{model_name}.pickle")

    def get_s3_model_key(self, model_name: str) -> str:
        return f"{self.__bucket_key}/{model_name}.pickle"
