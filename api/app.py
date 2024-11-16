from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langserve import add_routes
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import os
import uvicorn

load_dotenv()

app = FastAPI(
    title="Langchain Server", version="1.0", description="A simple API server"
)

add_routes(app, ChatOllama(model="phi"), path="/openai")

llm1 = OllamaLLM(model="phi")
llm2 = OllamaLLM(model="phi")

prompt1 = ChatPromptTemplate.from_template(
    "Write an essay about {topic} with 100 words"
)
prompt2 = ChatPromptTemplate.from_template("Write an poem about {topic} with 100 words")

add_routes(app, prompt1 | llm1, path="/essay")

add_routes(app, prompt2 | llm2, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
