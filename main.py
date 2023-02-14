from collections import Counter
import json
import os
from fastapi import FastAPI
import sklearn
from fastapi.middleware.cors import CORSMiddleware
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)
from pydantic import BaseModel
import pickle

app = FastAPI()

origins = ["http://localhost",
"http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

df = joblib.load('scraper/article_df.joblib','r')
similarity = joblib.load('scraper/content_based.joblib','r')
xgb = pickle.load(open('scraper/xgb.pkl','rb'))
model = pickle.load(open('scraper/tv.pkl','rb'))

# ----------------- API ROUTES -----------------

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend/{uuid}")
def recommend(uuid):
    idx = df[df['uuid'] == uuid].index[0]
    distances = similarity[idx]
    rec = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:11]
    L = []
    for i in rec:
        L.append(df.iloc[i[0]])
    return L

class WordCloudModel(BaseModel):
    content: str

class PropagandaModel(BaseModel):
    content: str

import base64

propagandas = ['appeal_to_authority', 'appeal_to_fear',
       'prejudice', 'bandwagon', 'reductio_ad_hitlerum',
       'black-and-white_fallacy', 'causal_oversimplification', 'doubt',
       'exaggeration', 'minimisation', 'flag-waving', 'loaded_language',
       'name_calling', 'labeling', 'repetition', 'slogans',
       'thought-terminating_cliches', 'whataboutism', 'straw_man',
       'red_herring', 'no_class']

@app.post("/propaganda")
def propaganda(body: PropagandaModel):
    vector = model.transform([body.content])
    p = xgb.predict(vector)[0]
    L = []
    for idx, i in enumerate(p):
        if i == 1:
            L.append(propagandas[idx])
    return {"propaganda": L}

@app.post("/wordcloud")
def wordcloud(body: WordCloudModel):
    WC = WordCloud(stopwords=stopwords,max_words=25,background_color="white").generate(body.content)
    plt.imshow(WC)
    plt.axis("off")
    # plt.show()
    plt.savefig('test.png')
    with open("test.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return {"wordcloud": encoded_string.decode('utf-8')}

# ----------------- MODEL METHODS -----------------

import pandas as pd
ndf = pd.read_csv('scraper/ndf.csv')
d = Counter()
for x in ndf.groupby('uuid'):
    d[x[0]] = (list(x[1]['author_id']), list(x[1]['created_at']), list(x[1]['text']), list(x[1]['public_metrics']))

@app.get("/graph/{uuid}")
def check(uuid):
    edges = []
    s = set()
    for x in zip(d[uuid][0], d[uuid][1]):
        date_x = x[1].split('T')[0]
        time_x = x[1].split('T')[1].split('.')[0]
        for y in zip(d[uuid][0], d[uuid][1]):
            if x[1] == y[1]:
                continue
            date_y = y[1].split('T')[0]
            time_y = y[1].split('T')[1].split('.')[0]
            # if date_x == date_y:
            hour_x, hour_y = int(time_x.split(':')[0]), int(time_y.split(':')[0])
            min_x, min_y = int(time_x.split(':')[1]), int(time_y.split(':')[1])
            sec_x, sec_y = int(time_x.split(':')[2]), int(time_y.split(':')[2])
            # index_x = author_ids.index(x[0])
            # index_y = author_ids.index(y[0])
            # a, b = min(index_x, index_y), max(index_x, index_y)
            if (
                abs(hour_x - hour_y) <= 1
            ):
                edges.append([x[0], y[0]]) 
            s.add(x[0])
            s.add(y[0])
    retweet_count = Counter()
    for i in range(len(d[uuid][3])):
        if(type(d[uuid][3][i]) == str):
            d[uuid][3][i] = eval(d[uuid][3][i])
        retweet_count[d[uuid][0][i]] = d[uuid][3][i]['retweet_count']
    print(retweet_count)
    print(s)
    print(edges)
    return {"nodes": list(s), "edges": edges, "retweet_count": (retweet_count)}

class FakeModel(BaseModel):
    content: str

vectorizer = pickle.load(open('scraper/vectorizer.pkl','rb'))
fakeModel = pickle.load(open('scraper/fake_news.pkl','rb'))

@app.post("/fake")
def fake(body: FakeModel):
    vector = vectorizer.transform([body.content])
    p = fakeModel.predict_proba(vector)[0][1]*100
    return {"fake": p}
