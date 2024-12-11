from pydantic import BaseModel

class AskQuery(BaseModel):
    query: str
