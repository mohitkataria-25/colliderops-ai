from __future__ import annotations
from langchain_ollama import ChatOllama
import os


DEFAULT_ANSWER_MODEL = "llama3.1:8b"
DEFAULT_JUDGE_MODEL = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def create_answer_llm(
        model_name = DEFAULT_ANSWER_MODEL,
        temperature = DEFAULT_TEMPERATURE,
)->ChatOllama:
    


    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=DEFAULT_OLLAMA_BASE_URL
    )

def create_judge_llm(
        model_name= DEFAULT_JUDGE_MODEL,
        tempereature= DEFAULT_TEMPERATURE,
)->ChatOllama:
    


    return ChatOllama(
        model=model_name,
        temperature=tempereature,
        base_url=DEFAULT_OLLAMA_BASE_URL
    )





