import logging
from dataclasses import dataclass
from typing import List, Literal

from fastapi import HTTPException
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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
            cross_encoder = CrossEncoder("bongsoo/kpf-cross-encoder-v1")
            cls.__instance = cls(version=settings.version, loader=loader, cross_encoder=cross_encoder)
        return cls.__instance

    def __init__(self, version: Literal["v1"], loader: ModelLoader, cross_encoder: CrossEncoder = None):
        """
        :param models 사용 가능한 검색 모델 리스트
        """
        self.logger = logging.getLogger(__name__)
        self.__models: List[SearchModel] = loader.load()
        self.__cross_encoder = cross_encoder
        self.__version = version
        assert self.__version in {"v1"}

    def search(self, query: str, limit: float = 0.5) -> SearchResponseDto:
        """
        :param query: 검색어
        :param limit: 반환할 결과의 최소 정확도
        :return: SearchResponseDto
        """
        results: List[SearchData] = []
        for model in self.__models:
            if model.type == "lexical":
                results += self._search_with_lexical(model=model, query=query, limit=limit)
                break

        if not results:
            raise HTTPException(status_code=404, detail="Empty Result")

        search_pairs = []
        for instance in results:
            search_pairs.append([query, instance.name])
            search_pairs.append([instance.name, query])
            search_pairs.append([self.make_template(query), self.make_template(instance.name)])
            search_pairs.append([self.make_template(instance.name), self.make_template(query)])
        scores = self.__cross_encoder.predict(search_pairs)

        final_score = []
        for i, instance in enumerate(results):
            score = min(scores[i * 4], scores[i * 4 + 1], scores[i * 4 + 2], scores[i * 4 + 3])
            instance.score = score
            final_score.append({"score": score, "searchData": instance})
        score_name = sorted(final_score, key=(lambda x: x["score"]), reverse=True)

        final_result = []
        for i, instance in enumerate(score_name):
            score = instance["score"]
            if score < 0.15:  # threshold
                break
            final_result.append(instance.id)

        if not final_result:
            raise HTTPException(status_code=404, detail="Empty Result")
        response = SearchResponseDto(version=self.__version, engine_type="ML", results=final_result)
        return response

    def make_template(self, word):
        return f'"{word}" 을(를) 구매한다.\n\n'

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
