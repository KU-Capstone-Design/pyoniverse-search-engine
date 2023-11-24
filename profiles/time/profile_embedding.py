"""
[Sentence Model Embedding 속도 측정]
이유: Lexical Model의 경우, 단순히 Split연산만 하지만 Sentence Model은 그렇지 않다.
테스트 내용: 각 Sentence Model에 대해 100개의 무작위 데이터에 대한 Embedding 속도 측정
테스트 실행: kernprof -l -v -u 1e-3 profiles/time/profile_embedding.py > profiles/time/report/profile_embedding.txt(밀리초)
테스트 결과:
1. sroberta_multitask: 430ms
2. sroberta_sts: 580ms
결론: 임베딩 결과를 저장하자
"""
import json
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from random import sample
from typing import Any, Dict, List, Literal

import numpy as np
from line_profiler import profile
from pykospacing import Spacing
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class EmbeddingAI:
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

    def execute(self) -> Dict[Literal["lexical", "sentence"], Dict[str, str]]:
        """
        사용 가능한 모든 모델의 임베딩을 만들어 저장
        @step
        1. Data 전처리
        2. lexical, sentence model & embedding 생성
        3. embedding 저장
        4. embedding 저장 경로 반환
        :return: {"lexical": {model_name: embedding_path}, "sentence": {model_name: embedding_path}}
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
        result = defaultdict(dict)
        for model_name in lexical_model_names:
            saved_path = self.save_embedding(model_name)
            result["lexical"][model_name] = saved_path
        for model_name in sentence_model_names:
            saved_path = self.save_embedding(model_name)
            result["sentence"][model_name] = saved_path
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
        try:
            with open(self.__data_path, "r") as fd:
                data = json.load(fd)
        except Exception:
            raise RuntimeError(f"{self.__data_path} is invalid json")
        if not data:
            raise RuntimeError(f"{self.__data_path} is empty")
        # Kospacing
        spacing = Spacing()
        response: List[dict] = []  # dictionary list {id, company, name}
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
            response.append({"id": datum["id"], "company": company, "name": datum["spaced_name"]})
        # update saved file
        with open(self.__data_path, "w") as fd:
            json.dump(data, fd, ensure_ascii=False)
        return response

    def get_bm250k_model(self, data: List[dict]) -> str:
        """
        Lexical Model
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: model_name
        """
        if "bm250k" not in self.__lexical_models:
            corpus = [{"embedding": d["name"], "id": d["id"], "name": d["name"], "company": d["company"]} for d in data]
            model = BM25Okapi([doc["embedding"].split(" ") for doc in corpus])
            self.__lexical_models["bm250k"] = (model, corpus)
        return "bm250k"

    def get_sroberta_multitask_model(self, data: List[dict]) -> str:
        """
        Sentence Model
        :param data: [{"id": .., "company": .., "name": ...}]
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
        :param data: [{"id": .., "company": .., "name": ...}]
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
        :param data: [{"id": .., "company": .., "name": ...}]
        :return: [{"embedding": ..., "id": ..., "name": ..., "company": ...}]
        """
        self.logger.info(f"Make embedding for {model}")
        embeddings = []
        for datum in data:
            embeddings.append(
                {
                    "embedding": model.encode(datum["name"]),
                    "id": datum["id"],
                    "name": datum["name"],
                    "company": datum["company"],
                }
            )
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

    def get_model(self, model_name: str) -> Dict[Literal["type", "model"], Any]:
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


@profile
def execute_sroberta_multitask_model(data: list):
    """
    embedding 수행
    :param data: products 데이터
    """
    embedding_ai.get_sroberta_multitask_model(data)


@profile
def execute_sroberta_sts_model(data: list):
    """
    embedding 수행
    :param data: products 데이터
    """
    embedding_ai.get_sroberta_sts_model(data)


def profile_sentence_model(count: int = 10):
    """
    @step
    1. model.preprocess 수행 -> data 얻기
    2. 랜덤 100개 선택 x count
    3. execute_... 실행
    """
    data = embedding_ai.preprocess_data()
    for _ in range(count):
        samples = sample(data, k=100)
        execute_sroberta_sts_model(samples)
        execute_sroberta_multitask_model(samples)


if __name__ == "__main__":
    embedding_ai = EmbeddingAI(data_path="profiles/resource/products.json", embedding_dir="profiles/resource/embedding")
    profile_sentence_model()
