from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models.user import User, UserProfile
from ..schemas.user import UserCreate, UserLogin

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    new_user = User(username=user.username, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    profile = UserProfile(user_id=new_user.id, dietary_preferences="")
    db.add(profile)
    db.commit()
    return {"message": "Registered successfully"}

@router.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or db_user.password != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": str(db_user.id), "username": db_user.username}
