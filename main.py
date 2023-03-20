from collections import Counter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pickle
from transformers import BertTokenizerFast
from model import BertForTokenAndSequenceJointClassification
import torch
import pandas as pd


app = FastAPI()

origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = joblib.load("scraper/article_df.joblib", "r")
similarity = joblib.load("scraper/content_based.joblib", "r")
vectorizer = pickle.load(open("scraper/vectorizer.pkl", "rb"))
fakeModel = pickle.load(open("scraper/fake_news.pkl", "rb"))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/recommend/{uuid}")
def recommend(uuid):
    idx = df[df["uuid"] == uuid].index[0]
    distances = similarity[idx]
    rec = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    L = [df.iloc[i[0]] for i in rec]
    return L


class PropagandaModel(BaseModel):
    content: str


@app.post("/propaganda")
def propaganda(body: PropagandaModel):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    model = BertForTokenAndSequenceJointClassification.from_pretrained(
        "QCRI/PropagandaTechniquesAnalysis-en-BERT",
        revision="v0.1.0",
    )
    inputs = tokenizer.encode_plus(body.content, return_tensors="pt")
    outputs = model(**inputs)
    sequence_class_index = torch.argmax(outputs.sequence_logits, dim=-1)
    sequence_class = model.sequence_tags[sequence_class_index[0]]
    token_class_index = torch.argmax(outputs.token_logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][1:-1])
    tags = [model.token_tags[i] for i in token_class_index[0].tolist()[1:-1]]
    return {"sequence_class": sequence_class, "tokens": tokens, "tags": tags}


ndf = pd.read_csv("scraper/ndf.csv")
d = Counter()
for x in ndf.groupby("uuid"):
    d[x[0]] = (
        list(x[1]["author_id"]),
        list(x[1]["created_at"]),
        list(x[1]["text"]),
        list(x[1]["public_metrics"]),
    )


@app.get("/graph/{uuid}")
def check(uuid):
    edges = []
    s = set()
    for x in zip(d[uuid][0], d[uuid][1]):
        date_x = x[1].split("T")[0]
        time_x = x[1].split("T")[1].split(".")[0]
        for y in zip(d[uuid][0], d[uuid][1]):
            if x[1] == y[1]:
                continue
            date_y = y[1].split("T")[0]
            time_y = y[1].split("T")[1].split(".")[0]
            hour_x, hour_y = int(time_x.split(":")[0]), int(time_y.split(":")[0])
            min_x, min_y = int(time_x.split(":")[1]), int(time_y.split(":")[1])
            sec_x, sec_y = int(time_x.split(":")[2]), int(time_y.split(":")[2])
            if abs(hour_x - hour_y) <= 1:
                edges.append([x[0], y[0]])
            s.add(x[0])
            s.add(y[0])
    retweet_count = Counter()
    for i in range(len(d[uuid][3])):
        if type(d[uuid][3][i]) == str:
            d[uuid][3][i] = eval(d[uuid][3][i])
        retweet_count[d[uuid][0][i]] = d[uuid][3][i]["retweet_count"]
    return {"nodes": list(s), "edges": edges, "retweet_count": (retweet_count)}


class FakeModel(BaseModel):
    content: str


@app.post("/fake")
def fake(body: FakeModel):
    vector = vectorizer.transform([body.content])
    p = fakeModel.predict_proba(vector)[0][1] * 100
    return {"fake": p}
