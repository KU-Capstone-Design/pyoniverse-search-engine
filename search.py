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

        self.sentence_models = {
            'sroberta_multitask': SentenceTransformer('jhgan/ko-sroberta-multitask'),
            #'sbert_nli': SentenceTransformer('jhgan/ko-sbert-nli'),
            #'sroberta_nli': SentenceTransformer('jhgan/ko-sroberta-nli'),
            'sroberta_sts': SentenceTransformer('jhgan/ko-sroberta-sts')
        }
 
        if embedding_paths is None :
            self.embedding_dict = self.make_embeddings()
            np.save(os.path.join(root_save_path, 'sroberta_multitask.npy'), self.embedding_dict['sroberta_multitask'])
            #np.save('./data/sbert_nli.npy', sbert_nli_embedding)
            #np.save('./data/sroberta_nli.npy', sroberta_nli_embedding)
            np.save(os.path.join(root_save_path, 'sroberta_sts.npy'), self.embedding_dict['sroberta_sts'])
        else : 
            self.embedding_dict = {
                'sroberta_multitask': np.load('./data/sroberta_multitask.npy', allow_pickle=True),
                #'sbert_nli': np.load('./data/sbert_nli.npy', allow_pickle=True),
                #'sroberta_nli': np.load('./data/sroberta_nli.npy', allow_pickle=True),
                'sroberta_sts': np.load('./data/sroberta_sts.npy', allow_pickle=True)  
            }  
        
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

    def make_embedding_one_model(self, model) :
        embeddings = []
        print('bulid embedding')
        for instance in tqdm(self.item_list) :
            embeddings.append({'embedding': model.encode(instance['product']), 'id': instance['id']})
        return embeddings
    
    def make_embeddings(self) :
        embedding_dict = {}
        for model_name, model in self.sentence_models.items() :
            embedding_dict[model_name] = self.make_embedding_one_model(model)
        return embedding_dict
    
    def find_item_one_model(self, model, model_name, query, num_result=10) :
        query_embedding = model.encode(query)
        score_id = []
        for instance in self.embedding_dict[model_name] :
            score = np.dot(query_embedding, instance['embedding'])
            score_id.append({'score': score, 'id': instance['id']})

        score_id = sorted(score_id, key=(lambda x: x['score']), reverse=True)

        return score_id[:num_result]
    
    def find_item_with_lexical_model(self, query, num_result=10) :
        tokenized_query = self._lexical_tokenizer(query)
        results = self.bm25.get_top_n(tokenized_query, self.corpus, n=10)
        return results

        #selected_id = []
        #for name in results[:num_result] :
        #    selected_id.append(self.name2id_dict[name])
        #return selected_id
    
    def search(self, query, num_results=10) :
        results = []
        for model_name, model in self.sentence_models.items() :
            score_id = self.find_item_one_model(model=model, model_name=model_name, query=query, num_result=num_results)
            items = [self.id2name_dict[_['id']] for _ in score_id]
            results.append({'model': model_name, 'items': items})
        
        results.append({
            'model': 'bm25',
            'items': self.find_item_with_lexical_model(query=query, num_result=num_results)
        })

        ## HANDLE DUPLICATE CASE
        ensembled_result = []
        for model_result in results :
            items = model_result['items']
            scores = self.cross_encoder.predict([(query, _ ) for _ in items])

            l = []
            for i, name in enumerate(items):
                l.append({'score': scores[i], 'name': name})
            ensembled_result += l

        score_name = sorted(ensembled_result, key=(lambda x: x['score']), reverse=True)

        return score_name


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
    print("make embedding")
    make_embeddings(data_path='./data/products.json', root_save_path='./data')

    print("start test")
    item_search_module = Item_Search_Module(data_path='./data/products.json', embedding_paths=True)

    print(item_search_module.search('치즈 김밥'))


    while True :
        query = input()
        if query == 'x' :
            break
        print(item_search_module.search(query))

    