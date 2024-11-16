from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}"),
    ]
)

st.title("Langchain demo with Ollama")
input_text = st.text_input("Search the topic you want")

llm = OllamaLLM(model="phi")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
