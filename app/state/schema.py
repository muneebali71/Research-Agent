from pydantic import BaseModel


class request_query(BaseModel):
    query: str
    tavily_api_key: str
