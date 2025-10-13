from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
try:
    # When running as a package: `uvicorn backend.api:app`
    from .chat_model import LocalChatModel
    from .model import PhishingDetector
    from .utils import map_model_to_frontend
except ImportError:
    # When running from backend directory: `uvicorn api:app`
    from chat_model import LocalChatModel
    from model import PhishingDetector
    from utils import map_model_to_frontend

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
detector = PhishingDetector()
chatbot = LocalChatModel()

# Serve the frontend index.html at root to avoid file:// CORS issues
import os
FRONTEND_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")))
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    return {"detail": "index.html not found"}


@app.get("/index.html", include_in_schema=False)
def serve_index_alias():
    return serve_index()


class PredictRequest(BaseModel):
    text: str = Field(..., example="Congratulations! You've won a prize.")


class PredictResponse(BaseModel):
    # kept for backwards compatibility; actual predict returns richer schema
    label: int
    scores: List[float]
    model: str = Field(default="local", description="Which model produced this prediction")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system | user | assistant")
    content: str = Field(..., description="Message body")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history in chronological order")


class ChatResponse(BaseModel):
    response: str
    prompt_tokens: int
    generated_tokens: int
    model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model_out = detector.predict(req.text)
        frontend = map_model_to_frontend(model_out, req.text, model_name="distilbert-base-uncased")
        return frontend
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.options("/predict", include_in_schema=False)
def predict_options():
    return Response(status_code=200)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = chatbot.chat([msg.dict() for msg in req.messages])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    texts: List[str]
    labels: List[int]


class TrainResponse(BaseModel):
    loss: float


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    try:
        loss = detector.train(req.texts, req.labels)
        return {"loss": float(loss)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
