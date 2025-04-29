from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_system import ask_question, prepare_vector_store
from speech import generate_speech
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
def read_root():
    return {"message": "API da IA rodando! 🚀"}