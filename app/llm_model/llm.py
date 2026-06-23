from config import GROQ_API_KEY
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()  


# def get_llm(model="llama-3.3-70b-versatile"):
def get_llm(model="meta-llama/llama-4-scout-17b-16e-instruct"):
    llm = ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=0)

    return llm