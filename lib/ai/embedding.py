import json
import logging
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from pykospacing import Spacing
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from lib.ai.model.common import Embedding, ModelMeta


class EmbeddingAI:
    Data = namedtuple("Data", ["id", "company", "name"])

    def __init__(self, data_path: str, embedding_dir: str = "resource/embedding"):
        self.logger = logging.getLogger(__name__)
        self.__data_path = Path(data_path)
        if not self.__data_path.exists():
            raise RuntimeError(f"{self.__data_path} Not Found")
        if not self.__data_path.is_file():
            raise RuntimeError(f"{self.__data_path} Not File")
        if self.__data_path.suffix != ".json":
            raise RuntimeError(f"{self.__data_path} is not json format")
        # model과 corpus/embedding 함께 저장
        self.__lexical_models = {}
        self.__sentence_models = {}
        # embedding을 저장할 경로 저장
        self.__embedding_dir = Path(embedding_dir)
        if not self.__embedding_dir.exists():
            # 새로운 폴더 생성
            self.__embedding_dir.mkdir(parents=True, exist_ok=True)
        if not self.__embedding_dir.is_dir():
            raise RuntimeError(f"{self.__embedding_dir} Not Directory")

    def execute(self) -> List[ModelMeta]:
        """
        사용 가능한 모든 모델의 임베딩을 만들어 저장
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
        data = self.preprocess_data()
        self.logger.info("Build lexical model")
        model_name = self.get_bm250k_model(data)
        lexical_model_names.append(model_name)
        self.logger.info("Build sentence model")
        model_name = self.get_sroberta_sts_model(data)
        sentence_model_names.append(model_name)
        model_name = self.get_sroberta_multitask_model(data)
        sentence_model_names.append(model_name)
        self.logger.info("Save Embedding")
        result: List[ModelMeta] = []
        for model_name in lexical_model_names:
            saved_path = self.save_embedding(model_name)
            result.append(ModelMeta(name=model_name, embedding_path=saved_path, type="lexical"))
        for model_name in sentence_model_names:
            saved_path = self.save_embedding(model_name)
            result.append(ModelMeta(name=model_name, embedding_path=saved_path, type="sentence"))
        return result

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

    def preprocess_data(self) -> List["EmbeddingAI.Data"]:
        """
        @fields
        self.__data_path: Json 형식의 임베딩될 데이터 파일 위치
        @steps
        1. 상품 제조 회사 찾기
        2. 상품 임베딩을 위해 단어 공백 띄우기(맥락을 고려하기 위함)
        @returns
        [{"id": d["id"], "name": d["name"], "company": ...} for d in data]
        """
        try:
            with open(self.__data_path, "r") as fd:
                data = json.load(fd)
        except Exception:
            raise RuntimeError(f"{self.__data_path} is invalid json")
        if not data:
            raise RuntimeError(f"{self.__data_path} is empty")
        # Kospacing
        spacing = Spacing()
        response: List[EmbeddingAI.Data] = []  # dictionary list {id, company, name}
        for datum in data:
            p = datum["name"].find(")")
            if p != -1:
                company = datum["name"][:p]
            else:
                company = None
            try:
                # spaced_name이 없다면 새롭게 생성 - profile 결과로 spacing 연산의 비용이 크다.
                if "spaced_name" not in datum:
                    datum["spaced_name"] = spacing(datum["name"], ignore="none")
            except Exception as e:
                raise RuntimeError(e)
            response.append(EmbeddingAI.Data(id=datum["id"], company=company, name=datum["spaced_name"]))
        # update saved file
        with open(self.__data_path, "w") as fd:
            json.dump(data, fd, ensure_ascii=False)
        return response

    def get_bm250k_model(self, data: List["EmbeddingAI.Data"]) -> str:
        """
        Lexical Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: model_name
        """
        if "bm250k" not in self.__lexical_models:
            embeddings = [Embedding(embedding=d.name, id=d.id, name=d.name) for d in data]
            model = BM25Okapi([doc.embedding.split(" ") for doc in embeddings])
            self.__lexical_models["bm250k"] = (model, embeddings)
        return "bm250k"

    def get_sroberta_multitask_model(self, data: List["EmbeddingAI.Data"]) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: model_name
        """
        model_name = "sroberta_multitask"
        if model_name not in self.__sentence_models:
            model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            embedding = self.__make_embedding(model=model, data=data, model_name=model_name)
            self.__sentence_models[model_name] = (model, embedding)
            self.logger.info(f"Save embedding for {model}")
            self.save_embedding(model_name=model_name)
        return model_name

    def get_sroberta_sts_model(self, data: List["EmbeddingAI.Data"]) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: model_name
        """
        model_name = "sroberta_sts"
        if model_name not in self.__sentence_models:
            model = SentenceTransformer("jhgan/ko-sroberta-sts")
            embedding = self.__make_embedding(model=model, data=data, model_name=model_name)
            self.__sentence_models[model_name] = (model, embedding)
            self.logger.info(f"Save embedding for {model}")
            self.save_embedding(model_name=model_name)
        return model_name

    def __make_embedding(
        self, model: SentenceTransformer, data: List["EmbeddingAI.Data"], model_name: str
    ) -> List[Embedding]:
        """
        :param model: SentenceTransformer
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: [{"embedding": ..., "id": ..., "name": ..., "company": ...}]
        """
        self.logger.info(f"Load embedding for {model}")
        try:
            embeddings = self.get_embedding(model_name=model_name)
        except Exception as e:
            embeddings = []
        self.logger.info(f"Make embedding for {model}")
        already_embedded = set(e.id for e in embeddings)
        for datum in data:
            if datum.id not in already_embedded:
                embeddings.append(
                    Embedding(
                        embedding=model.encode(datum.name),
                        id=datum.id,
                        name=datum.name,
                    )
                )
        return embeddings

    def save_embedding(self, model_name: str) -> str:
        """
        :param model_name: 저장된 모델 이름
        :return: 저장 경로
        """
        saved_path = self.get_embedding_path(model_name)
        if self.is_lexical_model(model_name):
            embeddings: List[Embedding] = self.__lexical_models[model_name][1]
        elif self.is_sentence_model(model_name):
            embeddings: List[Embedding] = self.__sentence_models[model_name][1]
        else:
            raise RuntimeError(f"{model_name} not in lexical or sentence models")
        with open(saved_path, "wb") as fd:
            pickle.dump(embeddings, fd)
        self.logger.info(f"embedding for {model_name} is saved at {saved_path}")
        return str(saved_path)

    def get_embedding(self, model_name: str) -> List[Embedding]:
        """
        save_embedding으로 저장된 embedding 가져오기
        :return: embedded data
        """
        path = self.get_embedding_path(model_name)
        try:
            with open(path, "rb") as fd:
                data = pickle.load(fd)
        except Exception as e:
            raise RuntimeError(e)
        else:
            self.logger.info(f"Load embedding for {model_name} from {path}")
            return data

    def get_model(self, model_name: str) -> Dict[Literal["type", "model"], Tuple[Any, List[Embedding]]]:
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

    def get_embedding_path(self, model_name: str) -> str:
        return str(self.__embedding_dir / f"{model_name}.pickle")
