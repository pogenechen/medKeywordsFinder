from keybert import KeyBERT
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import warnings
from functools import reduce
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("models/biobert-base-cased-v1.1",model_max_length=512)
model = AutoModel.from_pretrained("models/biobert-base-cased-v1.1")

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def biobert_embedding(texts):
    embedding_list = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = torch.mean(outputs.last_hidden_state, dim=1)
        embedding = embedding.cpu().numpy()
        embedding = normalize(embedding)
        embedding_list.append(embedding)
    return embedding_list

def embed_question_and_kws(data):
    k = KeyBERT(model=model)
    question_list = []
    keyword_lists = []
    for i in data['questions']:
        keybert_extract = [[i for i,_ in k.extract_keywords(text)] for text in i['ideal_answer']]
        keybert_extract = list(set(reduce(lambda x,y:x+y, keybert_extract)))
        if keybert_extract:
            question_list.append(i['body'])
            keyword_lists.append(keybert_extract)
    print("start embedding...")
    embedded_questions = biobert_embedding(question_list)
    embedded_kws = []
    cnt = 0

    all_kws_vec_pairs = []

    for _kws in keyword_lists:
        cnt+=1
        print(f"{cnt/len(keyword_lists):.1%} ({cnt}/{len(keyword_lists)})",end='\r')
        kws_vecs = biobert_embedding(_kws)
        embedded_kws.append(kws_vecs)
        all_kws_vec_pairs+=tuple(zip(_kws,kws_vecs))

    dataset = list(zip(embedded_questions,embedded_kws))
    return dataset, all_kws_vec_pairs
