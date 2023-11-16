from sentence_transformers import CrossEncoder


class SearchAI:
    __cross_encoder = CrossEncoder("bongsoo/kpf-cross-encoder-v1")

    # def __init__(
    #         self
    # ):
    #     """
    #     :param model_type: lexical or sentence model(검색 방식이 달라진다)
    #     :param model: EmbeddingAI로 생성된 모델
    #     :param model_name: 모델 이름
    #     """
    #     self.logger = logging.getLogger(__name__)
    #     self.__model_name = model_name
    #     self.__model_type = model_type
    #
    # def __load_corpus(self, path: str):
    #     path = Path(path)
    #     if not path.exists():
    #         raise RuntimeError(f"{path} Not Found")
    #     if not path.is_file():
    #         raise RuntimeError(f"{path} Not File")
    #     if path.suffix != ".npy":
    #         raise RuntimeError(f"{path} should be npy format")
    #     try:
    #         corpus = np.load(str(path), allow_pickle=True)
    #     except Exception as e:
    #         self.logger.error(traceback.format_exc())
    #         raise RuntimeError(f"{path} Load Failed")
    #     return corpus
    #
    # def search(self, query: str, limit: float = 0.5) -> List[dict]:
    #     """
    #     :param query: 검색어
    #     :param limit: 반환할 결과의 최소 정확도
    #     :return: [{"name": ..., "id": ..., "accuracy": ...}]
    #     """
    #     if self.__model_type == "lexical":
    #         results = self.search_with_lexical(query=query, limit=limit)
    #     elif self.__model_type == "sentence":
    #         results = self.search_with_sentence(query=query, limit=limit)
    #     else:
    #         raise RuntimeError(f"{self.__model_type} Not Supported")
    #     return results
    #
    # def search_with_sentence(self, query: str, limit: float) -> List[dict]:
    #     query_embedding = self.__model.encode(query)
    #     result = []
    #     for instance in self.__corpus:
    #         score = np.dot(query_embedding, instance["embedding"])
    #         result.append({"score": score, "id": instance["id"], "name": instance["name"]})
    #
    #     result = sorted(result, key=(lambda x: x["score"]), reverse=True)
    #     result = list(filter(lambda x: x["score"] >= limit, result))
    #     return result
    #
    # def search_with_lexical(self, query: str, limit: float) -> List[dict]:
    #     tokenized_query = query.split(" ")
    #     # corpus 순서대로 스코어 반환
    #     scores = self.__model.get_scores(tokenized_query)
    #     result = []
    #     for score, datum in zip(scores, self.__corpus):
    #         if score >= limit:
    #             result.append({"score": score, "id": datum["id"], "name": datum["name"]})
    #     return sorted(result, key=lambda x: x["score"], reverse=True)
    #
    # @classmethod
    # def ansible(cls, query: str, data: List[dict], limit: float = 0.5) -> List[dict]:
    #     """
    #     여러 모델의 검색 결과를 하나로 합쳐 가장 높은 유사도를 찾는다.
    #     :param results: [{"score": ..., "id": ..., "name": ...}]
    #     :param limit: 최하 유사도
    #     :return: [{"score": ..., "id": ..., "name": ...}]
    #     """
    #     scores = cls.__cross_encoder.predict([[query, d["name"]] for d in data])
    #     for d, s in zip(data, scores):
    #         d["score"] = s
    #     result = sorted(filter(lambda x: x["score"] >= limit, data), key=lambda x: x["score"], reverse=True)
    #     return result
