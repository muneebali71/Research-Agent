from typing import TypedDict, Annotated, Literal, Optional
import operator
from pydantic import BaseModel

Route = Literal["pdf", "web", "both", "chat"]


class ResearchState(TypedDict, total=False):
    topic:             str
    thread_id:         str
    has_pdf:           bool
    route:             Route
    pdf_context:       str
    pdf_chunks_meta:   list[dict]   
    search_results:    str
    scraped_content:   str
    research_combined: str
    sources:           str          
    report:            str
    feedback:          str
    final_score:       Optional[float]
    revision_count:    int
    revision_history:  Annotated[list, operator.add]



class SessionOut(BaseModel):
    id: str
    title: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    session_id: str
    route: str | None
    answer: str
    score: float | None

class MessageOut(BaseModel):
    role: str
    content: str
    route: str | None = None
    score: float | None = None