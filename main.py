import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_system import ask_question
from speech import generate_speech

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)