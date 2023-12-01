import logging
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
from fastapi import HTTPException
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import dot_score

from lib.ai.loader import ModelLoader
from lib.ai.model.embedding import SearchModel
from lib.ai.model.search import SearchResponseDto
from lib.config import get_settings


@dataclass(frozen=True)
class SearchData:
    id: int
    name: str
    score: float


class SearchAI:
    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            settings = get_settings()
            loader = ModelLoader.instance()
            # cross_encoder = CrossEncoder("bongsoo/kpf-cross-encoder-v1")
            cls.__instance = cls(version=settings.version, loader=loader)
        return cls.__instance

    def __init__(self, version: Literal["v1"], loader: ModelLoader, cross_encoder: CrossEncoder = None):
        """
        :param models 사용 가능한 검색 모델 리스트
        """
        self.logger = logging.getLogger(__name__)
        self.__models: List[SearchModel] = loader.load()
        # self.__cross_encoder = cross_encoder
        # assert isinstance(self.__cross_encoder, CrossEncoder)
        self.__version = version
        assert self.__version in {"v1"}

        # TODO : 병렬 성능 테스트

        # self.__pipes: Dict[str, Connection] = {}
        # self.__processes: Dict[str, Process] = {}
        # self.__setup_parallel()

    def search(self, query: str, limit: float = 0.5) -> SearchResponseDto:
        """
        :param query: 검색어
        :param limit: 반환할 결과의 최소 정확도
        :return: SearchResponseDto
        """
        data: List[SearchData] = []
        for model in self.__models:
            if model.type == "lexical":
                data += self._search_with_lexical(model=model, query=query, limit=limit)
                break
        for model in self.__models:
            if model.type == "sentence":
                data += self._search_with_sentence(model=model, query=query, limit=limit)
                break
        # results = self.__ansible(query=query, data=data)
        results = data
        if not results:
            raise HTTPException(status_code=404, detail="Empty Result")
        response = SearchResponseDto(version=self.__version, engine_type="ML", results=[r.id for r in results])
        return response

    def _search_with_sentence(self, model: SearchModel, query: str, limit: float) -> List[SearchData]:
        engine: SentenceTransformer = model.engine
        encoded_query = engine.encode(
            query, show_progress_bar=False, normalize_embeddings=True, batch_size=1, convert_to_tensor=True
        )
        result: List[SearchData] = []
        for e in model.embeddings:
            score = dot_score(encoded_query, e.embedding)
            result.append(SearchData(score=score, id=e.id, name=e.name))

        result = sorted(result, key=(lambda x: x.score), reverse=True)
        result = list(filter(lambda x: x.score >= limit, result))
        return result

    def _search_with_lexical(self, model: SearchModel, query: str, limit: float) -> List[SearchData]:
        engine: BM25Okapi = model.engine
        tokenized_query = query.split(" ")
        # corpus 순서대로 스코어 반환
        scores = engine.get_scores(tokenized_query)
        result: List[SearchData] = []
        for score, e in zip(scores, model.embeddings):
            if score >= limit:
                result.append(SearchData(score=score, id=e.id, name=e.name))
        return sorted(result, key=lambda x: x.score, reverse=True)

    def __ansible(self, query: str, data: List[SearchData], limit: float = 0.5) -> List[SearchData]:
        """
        여러 모델의 검색 결과를 하나로 합쳐 가장 높은 유사도를 찾는다.
        :param results: [{"score": ..., "id": ..., "name": ...}]
        :param limit: 최하 유사도
        :return: [{"score": ..., "id": ..., "name": ...}]
        """
        scores = self.__cross_encoder.predict(
            [[query, d.name] for d in data],
            show_progress_bar=False,
        )
        result: List[SearchData] = []
        for d, score in zip(data, scores):
            if score >= limit:
                result.append(SearchData(score=score, id=d.id, name=d.name))
        return sorted(result, key=lambda x: x.score, reverse=True)

    # def __setup_parallel(self):
    #     self.__processes: Dict[str, Process] = {}
    #     for model in self.__models:
    #         model_name = str(model.engine)
    #         p_conn, c_conn = Pipe()
    #         if model.type == "lexical":
    #             self.__processes[model_name] = Process(target=self._search_with_lexical_parallel,
    #                                                    kwargs={"model": model_name, "conn": c_conn})
    #         elif model.type == "sentence":
    #             self.__processes[model_name] = Process(target=self._search_with_sentence_parallel,
    #                                                    kwargs={"model": model_name, "conn": c_conn})
    #         else:
    #             raise RuntimeError(f"Don't support {model.type}")
    #         self.__pipes[model_name] = p_conn
    #         self.__processes[model_name].start()

    # def search_parallel(self, query: str, limit: float = 0.5) -> SearchResponseDto:
    #     """
    #     :param query: 검색어
    #     :param limit: 반환할 결과의 최소 정확도
    #     :return: SearchResponseDto
    #     """
    #     data = []
    #     for model in self.__models:
    #         self.__pipes[str(model.engine)].send((query, limit))
    #
    #     for model in self.__models:
    #         data += self.__pipes[str(model.engine)].recv()
    #     results = self.__ansible(query=query, data=data)
    #     if not results:
    #         raise HTTPException(status_code=404, detail="Empty Result")
    #     response = SearchResponseDto(version=self.__version, engine_type="ML", results=[r.id for r in results])
    #     return response
    #
    # def _search_with_sentence_parallel(self, model: str, conn: Connection) -> NoReturn:
    #     for m in self.__models:
    #         if str(m.engine) == model:
    #             model = m
    #             break
    #     engine: SentenceTransformer = model.engine
    #
    #     while True:
    #         try:
    #             query, limit = conn.recv()
    #         except EOFError:
    #             self.logger.info("Close connection")
    #             return
    #         encoded_query = engine.encode(query)
    #         if norm := np.linalg.norm(encoded_query):
    #             encoded_query = encoded_query / norm
    #         else:
    #             encoded_query = np.zeros_like(encoded_query)
    #         result: List[SearchData] = []
    #         for e in model.embeddings:
    #             if norm := np.linalg.norm(e.embedding):
    #                 normalized_embedding = e.embedding / norm
    #             else:
    #                 normalized_embedding = np.zeros_like(e.embedding)
    #             score = np.dot(encoded_query, normalized_embedding)
    #             result.append(SearchData(score=score, id=e.id, name=e.name))
    #
    #         result = sorted(result, key=(lambda x: x.score), reverse=True)
    #         result = list(filter(lambda x: x.score >= limit, result))
    #         conn.send(result)
    #
    # def _search_with_lexical_parallel(self, model: str, conn: Connection) -> NoReturn:
    #     for m in self.__models:
    #         if str(m.engine) == model:
    #             model = m
    #             break
    #     engine: BM25Okapi = model.engine
    #     while True:
    #         try:
    #             query, limit = conn.recv()
    #         except EOFError:
    #             self.logger.info("Close connection")
    #             return
    #         tokenized_query = query.split(" ")
    #         # corpus 순서대로 스코어 반환
    #         scores = engine.get_scores(tokenized_query)
    #         result: List[SearchData] = []
    #         for score, e in zip(scores, model.embeddings):
    #             if score >= limit:
    #                 result.append(SearchData(score=score, id=e.id, name=e.name))
    #         result = sorted(result, key=lambda x: x.score, reverse=True)
    #         conn.send(result)
    #
    # def exit(self):
    #     for conn in self.__pipes.values():
    #         conn.close()
    #     for p in self.__processes.values():
    #         p.terminate()
    #         p.close()
