from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models.user import User

security = HTTPBearer()

def get_current_user(token: str = Depends(security)):
    db = SessionLocal()
    user = db.query(User).filter(User.id == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user
