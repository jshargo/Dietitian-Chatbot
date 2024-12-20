import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from model import query_chroma, generate_answer
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
collection_name = "guidlines_embeddings"
embedding_model_name = "dunzhang/stella_en_1.5B_v5"
chroma_db_path = "../data/chromadb"
llm_model_name = "meta-llama/Llama-3.2-1B-Instruct"

@app.get("/stream")
async def stream(query: str):
    logger.info(f"Received query: {query}")
    
    try:
        # Query ChromaDB
        results = query_chroma(
            query=query,
            collection_name=collection_name,
            chroma_db_path=chroma_db_path,
            embedding_model_name=embedding_model_name
        )
        logger.info("Retrieved results from ChromaDB")

        # Generate answer
        answer = generate_answer(
            query=query,
            retrieved_results=results,
            llm_model_name=llm_model_name
        )
        logger.info("Generated answer from LLM")

        async def event_generator():
            for line in answer.split('\n'):
                yield f"data: {line}\n\n"
                await asyncio.sleep(0.1)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return StreamingResponse(
            iter([f"data: Error: {str(e)}\n\n"]), 
            media_type="text/event-stream"
        )