from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models.user import UserProfile
from ..schemas.user import ProfileUpdate
from ..services.auth_service import get_current_user

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_profile(current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if profile:
        return {"dietary_preferences": profile.dietary_preferences}
    return {"dietary_preferences": ""}

@router.post("/")
def update_profile(update: ProfileUpdate, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        profile = UserProfile(user_id=current_user.id, dietary_preferences=update.dietary_preferences)
        db.add(profile)
    else:
        profile.dietary_preferences = update.dietary_preferences
    db.commit()
    return {"message": "Profile updated"}
