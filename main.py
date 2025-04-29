from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag_system import ask_question, prepare_vector_store
from speech import generate_speech
import threading
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

threading.Thread(target=prepare_vector_store, daemon=True).start()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(request: QuestionRequest):
    result = ask_question(request.question)
    return {
        "answer": result["answer"],
        "context": [doc.page_content for doc in result["context"]]
    }

class SpeechRequest(BaseModel):
    text: str

@app.post("/speech")
async def speech(request: SpeechRequest):
    path = generate_speech(request.text)
    return {"audio_path": path}

@app.get("/speech.mp3")
async def get_speech():
    return FileResponse("speech.mp3")

@app.get("/")
def read_root():
    return {"message": "API da IA rodando! 🚀"}