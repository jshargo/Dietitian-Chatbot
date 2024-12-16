from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from model import query_chroma, generate_answer
import asyncio
import io

app = FastAPI()

collection_name = "pdf_embeddings"
embedding_model_name = "all-mpnet-base-v2"
chroma_db_path = "./data/chroma_db"
llm_model_name = "meta-llama/Llama-3.2-1B"


@app.get("/stream")
async def stream(query: str):
    # Run the RAG pipeline
    results = query_chroma(
        query=query, 
        collection_name=collection_name, 
        chroma_db_path=chroma_db_path, 
        embedding_model_name=embedding_model_name
    )
    final_answer = generate_answer(
        query=query,
        retrieved_results=results,
        llm_model_name=llm_model_name
    )

    # Here we simulate streaming by chunking the final_answer
    async def event_generator():
        for line in final_answer.split('\n'):
            await asyncio.sleep(0.1)  # simulate a delay
            yield f"data: {line}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
