from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str
    class Config:
        orm_mode = True

class ProfileUpdate(BaseModel):
    dietary_preferences: str
