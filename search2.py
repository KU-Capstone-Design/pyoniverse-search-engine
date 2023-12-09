import json
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np

from pykospacing import Spacing
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import json
import os

class Item_Search_Module() :
    def __init__(self, data_path, embedding_paths=None, root_save_path='./data') :
        with open(data_path, 'r', encoding='utf8') as f :
            self.data = json.load(f)
        self.item_list = self._preproces_data() ## [{id, company, product}]  
        self.corpus = [instance['product'] for instance in self.item_list]
        self.bm25 = BM25Okapi([self._lexical_tokenizer(doc) for doc in self.corpus])

        self.cross_encoder = CrossEncoder('bongsoo/kpf-cross-encoder-v1')

        self.name2id_dict = {}
        for instance in self.item_list :
            self.name2id_dict[instance['product']] = instance['id']

        self.id2name_dict = {}
        for instance in self.item_list :
            self.id2name_dict[instance['id']] = instance['product']
        

    def _lexical_tokenizer(self, sent):
        return sent.split(" ")
        
    def _preproces_data(self) :
        spacing = Spacing()
        product_list = [] # dictionary list {id, company, product}
        print('data preprocessing')
        for id, instance in enumerate(tqdm(self.data)) :
            name = instance['name']
            p = name.find(')')
            company = name[:p]
            product = spacing(name[p+1:], ignore='none')

            product_list.append({'id': id,
                                'company': company,
                                'product': product})
        return product_list        
    
    def find_item_with_lexical_model(self, query, num_result=10) :
        tokenized_query = self._lexical_tokenizer(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=num_result)
        return results

        #selected_id = []
        #for name in results[:num_result] :
        #    selected_id.append(self.name2id_dict[name])
        #return selected_id
    
    def search(self, query, num_results=10) :
        results = self.find_item_with_lexical_model(query=query, num_result=num_results * 5)

        search_pairs = []
        for name in results :
            search_pairs.append((query, name))
            search_pairs.append((name, query))
            search_pairs.append((self.make_template(query), self.make_template(name)))
            search_pairs.append((self.make_template(name), self.make_template(query)))
        scores = self.cross_encoder.predict(search_pairs)
        #scores = self.cross_encoder.predict([(self.make_template(query), self.make_template(_) ) for _ in results])

        final_score = []
        for i, name in enumerate(results):
            score = min(scores[i*4], scores[i*4+1], scores[i*4+2], scores[i*4+3])
            final_score.append({'score': score, 'name': name})
        score_name = sorted(final_score, key=(lambda x: x['score']), reverse=True)
        
        for i, intance in enumerate(score_name) :
            score = intance['score']
            if score < 0.15 : ## threshold
                return score_name[:i-1] if i > 0 else None
        return score_name
    
    def make_template(self, word) :
        return f'"{word}" 을(를) 구매한다.\n\n'

def make_embeddings(data_path, root_save_path='./data') :
    item_search_module = Item_Search_Module(data_path=data_path, root_save_path=root_save_path)
    return


if __name__ == '__main__' :
    #sroberta_multitask_embedding = np.load('./data/sroberta_multitask.npy', allow_pickle=True)
    #sbert_nli_embedding = np.load('./data/sbert_nli.npy', allow_pickle=True)
    #sroberta_nli_embedding = np.load('./data/sroberta_nli.npy', allow_pickle=True)
    #sroberta_sts_embedding = np.load('./data/sroberta_sts_embedding.npy', allow_pickle=True)

    #embeddings = {
    #    'sroberta_multitask_embedding': sroberta_multitask_embedding,
    #    'sbert_nli_embedding': sbert_nli_embedding,
    #    'sroberta_nli_embedding': sroberta_nli_embedding,
    #    'sroberta_sts_embedding': sroberta_sts_embedding,
    #}
    #print("make embedding")
    #make_embeddings(data_path='./data/products.json', root_save_path='./data')

    print("start test")
    item_search_module = Item_Search_Module(data_path='./data/products.json', embedding_paths=True)

    print(item_search_module.search('치즈 김밥'))


    while True :
        query = input()
        if query == 'x' :
            break
        print(item_search_module.search(query))

    