from fastapi import APIRouter, Depends, HTTPException
from ..services.auth_service import get_current_user
from ..schemas.ask import AskQuery
from ..services.embeddings import load_embedding_model, get_embedding
from ..services.llm_service import load_llm_model, generate_answer
from ..services.chromadb import get_chroma_client, get_or_create_collection
from ..utils.config import CHROMA_PATH, EMBEDDING_MODEL_PATH, LLM_MODEL_PATH
from ..utils.prompts import decision_system_prompt, user_decision_prompt, system_prompt, user_query_prompt
from sqlalchemy.orm import Session
from ..database import SessionLocal

router = APIRouter()

# Global loading (could also be done in main.py startup event)
embedding_model = load_embedding_model(EMBEDDING_MODEL_PATH)
llm_model, llm_tokenizer = load_llm_model(LLM_MODEL_PATH)
chroma_client = get_chroma_client(CHROMA_PATH)
collection = get_or_create_collection(chroma_client, "pdf_chunks")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/")
def ask_question(query: AskQuery, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    user_prefs = current_user.profile.dietary_preferences if current_user.profile else ""

    query_embedding = get_embedding(embedding_model, query.query)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_chunks = [doc for doc in results["documents"][0]]

    context = "\n\n".join(retrieved_chunks)

    # Decide if context can answer the question
    decision_prompt = decision_system_prompt.format(context=context) + user_decision_prompt.format(question=query.query)
    decision = generate_answer(llm_model, llm_tokenizer, decision_prompt, max_length=50).strip()

    if decision == "1":
        # Use context
        final_prompt = system_prompt.format(context=f"{user_prefs}\n\n{context}") + user_query_prompt.format(question=query.query)
        answer = generate_answer(llm_model, llm_tokenizer, final_prompt, max_length=500)
        return {"answer": answer}
    else:
        # No helpful context found
        return {"answer": "I don't know."}
