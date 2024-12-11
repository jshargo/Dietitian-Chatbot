from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

    profile = relationship("UserProfile", back_populates="user", uselist=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    dietary_preferences = Column(Text)
    user = relationship("User", back_populates="profile")
