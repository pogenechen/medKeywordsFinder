import json, os
import datetime as dt
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
import faiss

from model import ContrastiveEmbedder

class medKeywordsFinder:
    def __init__(self,checkpoint='default/contrastive_embedder_finetuned.pt'):
        self.model = ContrastiveEmbedder()
        self.model.load_state_dict(torch.load(checkpoint))
        
        
    def _normalize(self,x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def _biobert_embedding(self,text):
        tokenizer = AutoTokenizer.from_pretrained("models/biobert-base-cased-v1.1",model_max_length=512)
        model = AutoModel.from_pretrained("models/biobert-base-cased-v1.1")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs,)
        embedding = torch.mean(outputs.last_hidden_state, dim=1)
        embedding = embedding.cpu().numpy()
        embedding = self._normalize(embedding)
        return embedding
    
    def _embed_query(self,query):
        self.model.eval()
        q_vec = self._biobert_embedding(query)
        q_vec = torch.tensor(q_vec, dtype=torch.float32)
        q_proj = self.model(q_vec).cpu().detach().numpy()
        faiss.normalize_L2(q_proj)
        return q_proj
    
    def _generate_kw_idx_map(self,kw_vec_pairs_path):
        kw_vec_pairs = pd.read_pickle(kw_vec_pairs_path)
        self.kw_map, self.vec_map = {},{}
        kw_lst = []
        for idx, pair in enumerate(kw_vec_pairs):
            if pair[0] in kw_lst:
                continue
            else:
                kw_lst.append(pair[0])
            self.kw_map[idx] = pair[0]
            t_vec = torch.tensor(pair[1], dtype=torch.float32)
            vec = self.model(t_vec).squeeze(0).cpu().detach().numpy()
            self.vec_map[idx] = vec
    
    def build_db(self,kw_vec_pairs_path,prefix=f"{dt.datetime.today():%Y%m%d}"):
        self.model.eval()
        self._generate_kw_idx_map(kw_vec_pairs_path)
        d = 256
        base_index = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIDMap(base_index)
        ids = np.array(list(self.kw_map.keys())).astype('int64')

        vecs = np.vstack(list(self.vec_map.values())).astype('float32')
        faiss.normalize_L2(vecs)
        self.index.add_with_ids(vecs, list(self.vec_map.keys()))
        with open(os.path.join('query2kw','keywords map',f"{prefix}.json"),'w') as f:
            json.dump(self.kw_map,f)
        faiss.write_index(self.index, os.path.join('query2kw','index',f'{prefix}.index'))

    def load(self, kw_map_path='default/default_kw_map.json', index_path='default/default.index'):
        with open(kw_map_path,'r') as f:
            self.kw_map = json.load(f)
        self.kw_map = {int(i):j for i,j in self.kw_map.items()}
        self.index = faiss.read_index(index_path)
                
    def search(self,question,topK=5):
        q_proj = self._embed_query(question)
        D, I = self.index.search(q_proj,topK)
        I = I[0]
        keywords = [self.kw_map[i] for i in I]
        return keywords

