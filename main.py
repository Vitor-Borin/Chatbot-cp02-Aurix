from fastapi import FastAPI
from pydantic import BaseModel
from rag_system import ask_question
from speech import generate_speech

app = FastAPI()

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
