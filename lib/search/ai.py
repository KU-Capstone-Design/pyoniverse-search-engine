import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pykospacing import Spacing
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class ItemSearchAI:
    def __init__(self, data_path: str, embedding_paths=None, root_save_path="./data"):
        self.logger = logging.getLogger(__name__)
        self.__data_path = Path(data_path)
        if not self.__data_path.exists():
            raise RuntimeError(f"{self.__data_path} Not Found")
        if not self.__data_path.is_file():
            raise RuntimeError(f"{self.__data_path} Not File")
        if self.__data_path.suffix != ".json":
            raise RuntimeError(f"{self.__data_path} is not json format")
        # model 저장
        self.__lexical_models = {}
        self.__sentence_models = {}

        # self.sentence_models = {
        #     'sroberta_multitask': SentenceTransformer('jhgan/ko-sroberta-multitask'),
        #     # 'sbert_nli': SentenceTransformer('jhgan/ko-sbert-nli'),
        #     # 'sroberta_nli': SentenceTransformer('jhgan/ko-sroberta-nli'),
        #     'sroberta_sts': SentenceTransformer('jhgan/ko-sroberta-sts')
        # }
        #
        # if embedding_paths is None:
        #     self.embedding_dict = self.make_embeddings()
        #     np.save(os.path.join(root_save_path, 'sroberta_multitask.npy'), self.embedding_dict['sroberta_multitask'])
        #     # np.save('./data/sbert_nli.npy', sbert_nli_embedding)
        #     # np.save('./data/sroberta_nli.npy', sroberta_nli_embedding)
        #     np.save(os.path.join(root_save_path, 'sroberta_sts.npy'), self.embedding_dict['sroberta_sts'])
        # else:
        #     self.embedding_dict = {
        #         'sroberta_multitask': np.load('./data/sroberta_multitask.npy', allow_pickle=True),
        #         # 'sbert_nli': np.load('./data/sbert_nli.npy', allow_pickle=True),
        #         # 'sroberta_nli': np.load('./data/sroberta_nli.npy', allow_pickle=True),
        #         'sroberta_sts': np.load('./data/sroberta_sts.npy', allow_pickle=True)
        #     }
        #
        # self.cross_encoder = CrossEncoder('bongsoo/kpf-cross-encoder-v1')
        #
        # self.name2id_dict = {}
        # for instance in self.item_list:
        #     self.name2id_dict[instance['product']] = instance['id']
        #
        # self.id2name_dict = {}
        # for instance in self.item_list:
        #     self.id2name_dict[instance['id']] = instance['product']

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

    def get_lexical_model(self, data: List[dict]) -> Tuple[BM25Okapi, List[str]]:
        corpus = [d["product"] for d in data]
        if "bm250k" in self.__lexical_models:
            model = self.__lexical_models["bm250k"]
        else:
            model = BM25Okapi([doc.split(" ") for doc in corpus])
        return model, corpus

    def make_embedding_one_model(self, model):
        embeddings = []
        print("bulid embedding")
        for instance in tqdm(self.item_list):
            embeddings.append({"embedding": model.encode(instance["product"]), "id": instance["id"]})
        return embeddings

    def make_embeddings(self):
        embedding_dict = {}
        for model_name, model in self.sentence_models.items():
            embedding_dict[model_name] = self.make_embedding_one_model(model)
        return embedding_dict

    def find_item_one_model(self, model, model_name, query, num_result=10):
        query_embedding = model.encode(query)
        score_id = []
        for instance in self.embedding_dict[model_name]:
            score = np.dot(query_embedding, instance["embedding"])
            score_id.append({"score": score, "id": instance["id"]})

        score_id = sorted(score_id, key=(lambda x: x["score"]), reverse=True)

        return score_id[:num_result]

    def find_item_with_lexical_model(self, query, num_result=10):
        tokenized_query = self._lexical_tokenizer(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=10)
        return results

        # selected_id = []
        # for name in results[:num_result] :
        #    selected_id.append(self.name2id_dict[name])
        # return selected_id

    def search(self, query, num_results=10):
        results = []
        for model_name, model in self.sentence_models.items():
            score_id = self.find_item_one_model(model=model, model_name=model_name, query=query, num_result=num_results)
            items = [self.id2name_dict[_["id"]] for _ in score_id]
            results.append({"model": model_name, "items": items})

        results.append(
            {"model": "bm25", "items": self.find_item_with_lexical_model(query=query, num_result=num_results)}
        )

        # HANDLE DUPLICATE CASE
        ensembled_result = []
        for model_result in results:
            items = model_result["items"]
            scores = self.cross_encoder.predict([(query, _) for _ in items])

            lst = []
            for i, name in enumerate(items):
                lst.append({"score": scores[i], "name": name})
            ensembled_result += lst

        score_name = sorted(ensembled_result, key=(lambda x: x["score"]), reverse=True)

        return score_name
