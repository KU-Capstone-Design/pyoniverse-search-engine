import json
import logging
import traceback
from pathlib import Path
from typing import List

import numpy as np
from pykospacing import Spacing
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class EmbeddingAI:
    def __init__(self, data_path: str = "resource/data", embedding_dir: str = "resource/embedding"):
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
            raise RuntimeError(f"{self.__embedding_dir} Not Found")
        if not self.__embedding_dir.is_dir():
            raise RuntimeError(f"{self.__embedding_dir} Not Directory")

    @property
    def lexical_models(self):
        return self.__lexical_models

    @property
    def sentence_models(self):
        return self.__sentence_models

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

    def preprocess_data(self) -> List[dict]:
        """
        @fields
        self.__data_path: Json 형식의 임베딩될 데이터 파일 위치
        @steps
        1. 상품 제조 회사 찾기
        2. 상품 임베딩을 위해 단어 공백 띄우기(맥락을 고려하기 위함)
        @returns
        [{"id": d["id"], "name": d["name"], "company": ...} for d in data]
        """
        self.logger.info("Load Data")
        try:
            with open(self.__data_path, "r") as fd:
                data = json.load(fd)
        except Exception:
            raise RuntimeError(f"{self.__data_path} is invalid json")
        if not data:
            raise RuntimeError(f"{self.__data_path} is empty")
        self.logger.info("Preprocess Data")
        # Kospacing
        spacing = Spacing()
        response: List[dict] = []  # dictionary list {id, company, product}
        for datum in tqdm(data):
            p = datum["name"].find(")")
            if p != -1:
                company = datum["name"][:p]
            else:
                company = None
            try:
                spaced_name = spacing(datum["name"], ignore="none")
            except Exception as e:
                raise RuntimeError(e)
            response.append({"id": datum["id"], "company": company, "product": spaced_name})
        return response

    def get_bm250k_model(self, data: List[dict]) -> str:
        """
        Lexical Model
        :param data: [{"id": .., "company": .., "product": ...}]
        :return: model_name
        """
        if "bm250k" not in self.__lexical_models:
            corpus = [{"embedding": d["product"], "id": d["id"]} for d in data]
            model = BM25Okapi([doc["embedding"].split(" ") for doc in corpus])
            self.__lexical_models["bm250k"] = (model, corpus)
        return "bm250k"

    def get_sroberta_multitask_model(self, data: List[dict]) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "product": ...}]
        :return: model_name
        """
        if "sroberta_multitask" not in self.__sentence_models:
            model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            embedding = self.__make_embedding(model, data)
            self.__sentence_models["sroberta_multitask"] = (model, embedding)
        return "sroberta_multitask"

    def get_sroberta_sts_model(self, data: List[dict]) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "product": ...}]
        :return: model_name
        """
        if "sroberta_sts" not in self.__sentence_models:
            model = SentenceTransformer("jhgan/ko-sroberta-sts")
            embedding = self.__make_embedding(model, data)
            self.__sentence_models["sroberta_sts"] = (model, embedding)
        return "sroberta_sts"

    def __make_embedding(self, model: SentenceTransformer, data: List[dict]) -> List[dict]:
        """
        :param model: SentenceTransformer
        :param data: [{"id": .., "company": .., "product": ...}]
        :return: [{"embedding": ..., "id": ...}]
        """
        self.logger.info(f"Make embedding for {model}")
        embeddings = []
        for datum in tqdm(data):
            embeddings.append({"embedding": model.encode(datum["product"]), "id": datum["id"]})
        return embeddings

    def save_embedding(self, model_name: str) -> str:
        """
        :param model_name: 저장된 모델 이름
        :return: 저장 경로
        """
        if self.is_lexical_model(model_name):
            corpus = self.__lexical_models[model_name][1]
            saved_path = self.__embedding_dir / f"{model_name}.json"
            with open(saved_path, "w") as fd:
                json.dump(corpus, fd, ensure_ascii=False)
        elif self.is_sentence_model(model_name):
            embedding = self.__sentence_models[model_name][1]
            saved_path = self.__embedding_dir / f"{model_name}.npy"
            try:
                np.save(str(saved_path), embedding)
            except Exception as e:
                self.logger.error(traceback.format_exc())
                raise RuntimeError(f"{model_name} Not Saved")
        else:
            raise RuntimeError(f"{model_name} not in lexical or sentence models")
        return str(saved_path)
