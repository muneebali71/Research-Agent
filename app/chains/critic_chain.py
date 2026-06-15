
# from langchain.agents import create_agent
# from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


# from app.mcp_servers.websearch_server import web_search,scrap_url
# from dotenv import load_dotenv
import os
from app.llm_model.llm import get_llm

# from config import GROQ_API_KEY


# load_dotenv()  # Load environment variables from .env file

llm = get_llm()






# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a sharp and Constructive research critic. Be honest and specific"),
    ("human","""Review the research report below and evaluate it strictly.
    Report:
    {report}
     
    Respond in this exact format:
    
    Score: X/10
     
    Strengths:
    -...
    -...
    
    Areas to Improve:
    -...
    -...
    
    One line verdict:
    ... """)
])


critic_chain = critic_prompt | llm | StrOutputParser()





