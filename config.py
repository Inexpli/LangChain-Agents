from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("API_KEY"),
)

# model = ChatOllama(
#     model="qwen3:30b",
#     temperature=0,
# )