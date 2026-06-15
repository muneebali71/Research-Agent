from config import GROQ_API_KEY
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()  # Load environment variables from .env file


def get_llm(model="llama-3.3-70b-versatile"):
    llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=0)

    return llm