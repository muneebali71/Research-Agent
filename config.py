from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Configuration variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")