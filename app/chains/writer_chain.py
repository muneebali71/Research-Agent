from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm_model.llm import get_llm

llm = get_llm()

# ── Writer chain ───────────────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent research writer. Your first job is to READ the topic "
     "and choose the RIGHT format — do not default to a formal report for everything.\n\n"

     "FORMAT SELECTION RULES:\n\n"

     "Use FORMAT A (Direct Answer) when the topic is:\n"
     "- A simple factual question ('What is X?', 'Who is Y?', 'When did Z happen?')\n"
     "- A definition or explanation request\n"
     "- A how-to or process question\n"
     "Format A: Answer directly in 2-4 clear paragraphs. No headers. No bullet lists.\n"
     "Cite sources inline: 'According to Reuters...'\n\n"

     "Use FORMAT B (Structured Briefing) when the topic is:\n"
     "- A request for recent news or updates on an ongoing situation\n"
     "- A multi-event topic spanning different dates or actors\n"
     "- Anything where timeline or chronology matters\n"
     "Format B:\n"
     "## [Topic Title]\n"
     "### [Date Period e.g. April 2026]\n"
     "- [FACT] One sentence. (Source, date)\n"
     "### [Next Date Period]\n"
     "- [FACT/ANALYSIS] One sentence. (Source, date)\n"
     "## Conclusion\n"
     "## References\n\n"

     "Use FORMAT C (Comparison / Analysis) when the topic is:\n"
     "- 'Compare X and Y', 'What are the pros and cons of...', 'Difference between...'\n"
     "Format C: Use a clear side-by-side structure with headers for each side.\n"
     "Cite sources inline.\n\n"

     "Use FORMAT D (List / Ranked) when the topic is:\n"
     "- 'What are the best...', 'Top X ways to...', 'List of...'\n"
     "Format D: Numbered or bulleted list with a one-line intro and brief conclusion.\n"
     "Cite sources inline.\n\n"

     "UNIVERSAL RULES (apply to ALL formats):\n"
     "1. Every factual claim must come from Research Gathered below — no exceptions.\n"
     "2. No background knowledge from training unless it appears in the sources.\n"
     "3. Label interpretations/assessments as [ANALYSIS], direct reported events as [FACT].\n"
     "4. One claim per sentence, one source per sentence.\n"
     "5. References: only list URLs from Source References Available.\n"
     "6. Never invent, infer, or expand beyond what sources explicitly state."
    ),
    ("human", """Write a response for the topic below. First decide which format fits best, then write.

Topic: {topic}

Research Gathered (your ONLY allowed source of facts):
{research}

Source References Available (cite ONLY these):
{sources}
""")
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ── Revision chain ─────────────────────────────────────────────────────────────
revision_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent research writer revising a response based on critic feedback.\n\n"
     "Keep the same format the original used (Direct Answer / Structured Briefing / "
     "Comparison / List) unless the critic explicitly says the format was wrong.\n\n"
     "UNIVERSAL RULES:\n"
     "1. Every factual claim from Research Gathered only — no exceptions.\n"
     "2. No background knowledge not present in sources.\n"
     "3. Label [FACT] or [ANALYSIS] on every finding.\n"
     "4. One claim per sentence, one source per sentence.\n"
     "5. References: only URLs from Source References Available."
    ),
    ("human", """Revise the response below. Fix every issue the critic identified.

Topic: {topic}

Original Response:
{report}

Critic Feedback:
{feedback}

Research Gathered (ONLY allowed source of facts):
{research}

Source References Available (cite ONLY these):
{sources}
""")
])

revision_chain = revision_prompt | llm | StrOutputParser()