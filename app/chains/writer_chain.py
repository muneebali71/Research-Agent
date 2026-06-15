from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_model.llm import get_llm

llm = get_llm()


# ── Original writer chain ──────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a comprehensive research report on the topic below.
    Topic: {topic}

    Research Gathered:
    {research}

    Structure of the report should be as follows:
    1. Introduction
    2. Key Findings (minimum 4 well explained points)
    3. Conclusion
    4. References (List all URLs used as research with proper formatting)

    Be detailed, factual and professional.
    """)
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ── Revision chain (used when critic score < 8) ────────────────────────────────
revision_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Revise and improve reports based on critic feedback."),
    ("human", """You previously wrote a research report that received critical feedback.
    Revise the report to address ALL the areas of improvement mentioned.

    Topic: {topic}

    Original Report:
    {report}

    Critic Feedback:
    {feedback}

    Research Gathered (for reference):
    {research}

    Write a fully revised version of the report using the same structure:
    1. Introduction
    2. Key Findings (minimum 4 well explained points)
    3. Conclusion
    4. References (List all URLs used as research with proper formatting)

    Fix every weakness the critic identified. Be detailed, factual and professional.
    """)
])

revision_chain = revision_prompt | llm | StrOutputParser()