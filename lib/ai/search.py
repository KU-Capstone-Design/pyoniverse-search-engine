import logging
from collections import namedtuple
from typing import List, Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from lib.ai.model.embedding import SearchModel
from lib.ai.model.search import SearchResponseDto


class SearchAI:
    Data = namedtuple("Data", ["id", "name", "score"])

    def __init__(self, version: Literal["v1"], models: List[SearchModel], cross_encoder: CrossEncoder):
        """
        :param models 사용 가능한 검색 모델 리스트
        """
        self.logger = logging.getLogger(__name__)
        self.__models = models
        self.__cross_encoder = cross_encoder
        assert isinstance(self.__cross_encoder, CrossEncoder)
        self.__version = version
        assert self.__version in {"v1"}

    def search(self, query: str, limit: float = 0.5) -> SearchResponseDto:
        """
        :param query: 검색어
        :param limit: 반환할 결과의 최소 정확도
        :return: SearchResponseDto
        """
        data: List[SearchAI.Data] = []
        for model in self.__models:
            if model.type == "lexical":
                data += self.__search_with_lexical(model=model, query=query, limit=limit)
            elif model.type == "sentence":
                data += self.__search_with_sentence(model=model, query=query, limit=limit)
            else:
                self.logger.info(f"{model.type} isn't supported")
        results = self.__ansible(query=query, data=data)
        response = SearchResponseDto(version=self.__version, engine_type="ML", results=[r.id for r in results])
        return response

    def __search_with_sentence(self, model: SearchModel, query: str, limit: float) -> List["SearchAI.Data"]:
        engine: SentenceTransformer = model.engine
        encoded_query = engine.encode(query)
        if norm := np.linalg.norm(encoded_query):
            encoded_query = encoded_query / norm
        else:
            encoded_query = np.zeros_like(encoded_query)
        result: List[SearchAI.Data] = []
        for e in model.embeddings:
            if norm := np.linalg.norm(e.embedding):
                normalized_embedding = e.embedding / norm
            else:
                normalized_embedding = np.zeros_like(e.embedding)
            score = np.dot(encoded_query, normalized_embedding)
            result.append(SearchAI.Data(score=score, id=e.id, name=e.name))

        result = sorted(result, key=(lambda x: x.score), reverse=True)
        result = list(filter(lambda x: x.score >= limit, result))
        return result

    def __search_with_lexical(self, model: SearchModel, query: str, limit: float) -> List["SearchAI.Data"]:
        engine: BM25Okapi = model.engine
        tokenized_query = query.split(" ")
        # corpus 순서대로 스코어 반환
        scores = engine.get_scores(tokenized_query)
        result: List[SearchAI.Data] = []
        for score, e in zip(scores, model.embeddings):
            if score >= limit:
                result.append(SearchAI.Data(score=score, id=e.id, name=e.name))
        return sorted(result, key=lambda x: x.score, reverse=True)

    def __ansible(self, query: str, data: List["SearchAI.Data"], limit: float = 0.5) -> List["SearchAI.Data"]:
        """
        여러 모델의 검색 결과를 하나로 합쳐 가장 높은 유사도를 찾는다.
        :param results: [{"score": ..., "id": ..., "name": ...}]
        :param limit: 최하 유사도
        :return: [{"score": ..., "id": ..., "name": ...}]
        """
        scores = self.__cross_encoder.predict([[query, d.name] for d in data])
        result: List[SearchAI.Data] = []
        for d, score in zip(data, scores):
            if score >= limit:
                result.append(SearchAI.Data(score=score, id=d.id, name=d.name))
        return sorted(result, key=lambda x: x.score, reverse=True)
