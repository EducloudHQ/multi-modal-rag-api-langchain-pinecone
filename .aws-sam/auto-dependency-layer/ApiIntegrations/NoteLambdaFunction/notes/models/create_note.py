from pydantic import BaseModel


class CreateNoteInput(BaseModel):
    id:str
    note: str
    title: str
    username: str
    status: bool
    createdOn:int
