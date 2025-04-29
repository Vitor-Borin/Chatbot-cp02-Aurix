import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from youtube_transcript_api import YouTubeTranscriptApi
from langchain import hub
from typing_extensions import TypedDict, List
import re

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = InMemoryVectorStore(embeddings)

urls = [
    "https://blogdaengenharia.com/especiais/tecnologia/inteligencia-artificial/ia-no-desenvolvimento-de-materiais/",
    "https://ufop.br/noticias/em-discussao/os-avancos-da-inteligencia-artificial-e-o-embate-humano-x-maquina?",
    "https://www.ufmg.br/espacodoconhecimento/inteligencia-artificial-e-arte/",
    "https://revistathebard.com/materia-de-capa-a-arte-e-a-inteligencia-artificial/",
    "https://brasilescola.uol.com.br/informatica/inteligencia-artificial.htm",
    "https://dittomusic.com/pt/blog/ai-for-music-production-tools-for-musicians",
    "https://clubedoaudio.com.br/edicao-296/opiniao-inteligencia-artificial-na-musica-e-no-audio/",
    "https://www.totvs.com/blog/inovacoes/o-que-e-inteligencia-artificial/",
    "https://exame.com/inteligencia-artificial/como-surgiu-a-inteligencia-artificial/",
    "https://industriall.ai/blog/historia-inteligencia-artificial"
]

def load_documents():
    docs = []
    youtube_links = [
        "https://youtu.be/lZqSUSewvKU",
        "https://www.youtube.com/watch?v=uK6ZAvS8JAo",
        "https://www.youtube.com/watch?v=zxCY01vGXLM",
        "https://www.youtube.com/watch?v=lZqSUSewvKU",
    ]
    for l in youtube_links:
        loader = YoutubeLoader.from_youtube_url(l, add_video_info=False)
        transcript_list = YouTubeTranscriptApi.get_transcript(loader.video_id, languages=['pt'])
        transcript = " ".join([entry['text'] for entry in transcript_list])
        docs.append(Document(page_content=transcript, metadata=loader._metadata))
    loader = UnstructuredURLLoader(urls=urls)
    docs += loader.load()
    return docs

def prepare_vector_store():
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")
prompt.messages[0].prompt.template = '''You are Aurix an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say:

"Desculpe, não encontrei uma resposta com base nas informações disponíveis no momento. Estou sempre aprendendo e buscando me aprimorar. Você pode tentar reformular a pergunta ou consultar outra fonte confiável."

Use one sentence.
Question: {question}
Context: {context}
Answer:'''

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

prepare_vector_store()

def is_identity_question(question: str) -> bool:
    question_lower = question.lower()

    identity_patterns = [
        r"quem (é|e|eh|seria) voc(ê|e)",
        r"qual (é|e|eh) (o )?seu nome",
        r"como voc(ê|e) se chama",
        r"quem (é|e|eh) aurix",
        r"o que (é|e|eh) voc(ê|e)",
        r"me fale sobre voc(ê|e)",
        r"se apresente",
        r"qual sua identidade",
        r"quem te criou",
        r"quem te desenvolveu",
        r"who are you",
        r"what are you",
        r"what is your name",
        r"tell me about yourself"
    ]
    for pattern in identity_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

IDENTITY_RESPONSE = "Eu sou Aurix, um assistente virtual criado pelo Portal Synthetica. Fui projetado para ajudar você a responder perguntas, criar conteúdo, resolver problemas, programar, estudar ou trabalhar em projetos, sempre buscando ser claro, direto e o mais útil possível."

def ask_question(question: str):
    if is_identity_question(question):
        dummy_doc = Document(page_content="Identity information", metadata={})
        return {
            "question": question,
            "context": [dummy_doc],
            "answer": IDENTITY_RESPONSE
        }
    return graph.invoke({"question": question})