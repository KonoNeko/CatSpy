from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import PhishingDetector

app = FastAPI()
detector = PhishingDetector()

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(req: PredictRequest):
    label = detector.predict(req.text)
    return {"label": label}

class TrainRequest(BaseModel):
    texts: list[str]
    labels: list[int]

@app.post("/train")
def train(req: TrainRequest):
    loss = detector.train(req.texts, req.labels)
    return {"loss": loss}
