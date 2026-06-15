from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    topic:             str
    search_results:    str
    scraped_content:   str
    research_combined: str
    report:            str
    feedback:          str
    final_score:       float
    revision_count:    int
    revision_history:  Annotated[list, operator.add]