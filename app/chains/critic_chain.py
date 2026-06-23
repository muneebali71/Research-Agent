from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_model.llm import get_llm

llm = get_llm()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a balanced research report critic. Evaluate based on what the topic "
     "actually needed — not every response should be a formal report.\n\n"

     "STEP 1 — Identify the format used:\n"
     "- Direct Answer: short factual questions answered in paragraphs\n"
     "- Structured Briefing: news/updates with dated sections and [FACT]/[ANALYSIS] labels\n"
     "- Comparison: side-by-side structure\n"
     "- List: numbered/bulleted list\n"
     "Do NOT penalize a Direct Answer for lacking 'Introduction/Key Findings/Conclusion' "
     "headers — that format is intentionally prose-based.\n\n"

     "STEP 2 — Score on FOUR criteria:\n\n"

     "1. FORMAT FIT (2 points): Did the writer choose the right format for the topic? "
     "A news update using a structured briefing = good. A simple factual question "
     "getting a full formal report = bad.\n\n"

     "2. SOURCE GROUNDING (3 points): Do facts in the response match the provided "
     "source material? Award full marks if claims use proper attribution and sources "
     "back them up. Only deduct if a claim is clearly absent from ALL sources.\n\n"

     "3. CLARITY & CORRECTNESS (3 points): Is the response clear, well-written, "
     "and logically structured for its chosen format? Are [FACT]/[ANALYSIS] labels "
     "used correctly where applicable?\n\n"

     "4. SOURCE DISCIPLINE (2 points): Are references real and only from provided "
     "sources? No fabricated URLs.\n\n"

     "SCORING GUIDE:\n"
     "8-10: Right format, well-grounded, clear, proper citations\n"
     "6-7:  Mostly good, minor attribution or format issues\n"
     "4-5:  Wrong format for the query OR several unsupported claims\n"
     "1-3:  Mostly fabricated, completely wrong format, or off-topic\n\n"
     "A well-attributed response in the correct format SHOULD score 8 or above."
    ),
    ("human", """Evaluate this research response.

Topic the response was answering: {topic}

Source Material it was based on:
{sources}

Response to evaluate:
{report}

Respond in EXACTLY this format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
""")
])

critic_chain = critic_prompt | llm | StrOutputParser()