from fastapi import FastAPI
from database import Base, engine
from api import auth, profile, ask
from models.user import User, UserProfile

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(profile.router, prefix="/profile", tags=["profile"])
app.include_router(ask.router, prefix="/ask", tags=["ask"])

@app.get("/")
def root():
    return {"message": "API is running"}
