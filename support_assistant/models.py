from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(min_length=1)

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)
