import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

df = pd.read_csv('sdfe.csv')

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

df['embedding'] = df['embedding'].apply(json.loads)
while True:
    print("=" * 50)
    text = input("입력: ")
    embedding = model.encode(text)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    print(answer['구분'])
    print(answer['유저'])
    print(answer['챗봇'])