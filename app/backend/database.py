import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up from backend/ to app/
DATABASE_PATH = os.path.join(BASE_DIR, "data", "dietbot.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

Base = declarative_base()

def get_engine():

    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    return create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

def get_db_session():
    return SessionLocal()

class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    password = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=False)
    height = Column(Integer, nullable=False)
    weight = Column(Integer, nullable=False)
    activity_level = Column(String(50), nullable=False)
    allergies = Column(String(200)) 
    daily_nutrient_intake = relationship('DailyNutrientIntake', backref='user', lazy=True)

class DailyNutrientIntake(Base):
    __tablename__ = "daily_nutrient_intake"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user_profile.id'), nullable=False)
    date = Column(Date, nullable=False)
    dish_name = Column(String(100), nullable=False)
    calories = Column(Float, default=0)
    protein = Column(Float, default=0)
    fat = Column(Float, default=0)
    carbs = Column(Float, default=0)
    fiber = Column(Float, default=0)
    sodium = Column(Float, default=0)