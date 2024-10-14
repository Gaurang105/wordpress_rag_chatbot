import os
import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import subprocess
import sys
from utils import (
    fetch_wordpress_posts,
    chunk_posts,
    embed_query,
    create_pinecone_index,
    similarity_search,
    get_context,
    augment_query,
    query_claude,
    save_data,
    load_data,
    batch_upsert,
    model  
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store our data
posts = []
chunked_posts = []
pinecone_index = None

# Cache file paths
CACHE_DIR = "cache"
POSTS_CACHE = os.path.join(CACHE_DIR, "posts.pkl")
CHUNKED_POSTS_CACHE = os.path.join(CACHE_DIR, "chunked_posts.pkl")

# Pinecone configuration
EMBEDDING_DIMENSION = 384 
PINECONE_INDEX_NAME = "rajesh-jain-posts"

@app.on_event("startup")
async def startup_event():
    global posts, chunked_posts, pinecone_index
    
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    try:
        # Document Collection
        if os.path.exists(POSTS_CACHE):
            posts = load_data(POSTS_CACHE)
        else:
            wordpress_url = "https://rajeshjain.com/wp-json/wp/v2/posts"
            posts = fetch_wordpress_posts(wordpress_url)
            if not posts:
                logger.error("No posts were fetched. Cannot proceed with startup.")
                return
            save_data(posts, POSTS_CACHE)

        # Document Chunking
        if os.path.exists(CHUNKED_POSTS_CACHE):
            chunked_posts = load_data(CHUNKED_POSTS_CACHE)
        else:
            chunked_posts = chunk_posts(posts)
            save_data(chunked_posts, CHUNKED_POSTS_CACHE)

        # Create or connect to Pinecone index
        pinecone_index = create_pinecone_index(PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)

        # Check if the index needs to be populated
        total_chunks = sum(len(post['chunks']) for post in chunked_posts)
        index_stats = pinecone_index.describe_index_stats()
        vectors_in_index = index_stats['total_vector_count']

        if vectors_in_index < total_chunks:
            vectors_to_upsert = []
            for post in chunked_posts:
                for i, chunk in enumerate(post['chunks']):
                    vector_id = f"{post['id']}_{i}"
                    if not pinecone_index.fetch(ids=[vector_id])['vectors']:
                        vector = model.encode(chunk).tolist()
                        vectors_to_upsert.append((vector_id, vector, {"text": chunk}))
            
            if vectors_to_upsert:
                batch_upsert(pinecone_index, vectors_to_upsert)

        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

class Query(BaseModel):
    query: str

@app.post("/query")
async def process_query(query: Query):
    if not pinecone_index:
        return JSONResponse(status_code=500, content={"error": "Server is not ready. Please try again later."})
    try:
        query_vector = embed_query(query.query)
        similar_ids = similarity_search(query_vector, pinecone_index)
        context = get_context(similar_ids, chunked_posts)
        augmented_query = augment_query(query.query, context)
        
        try:
            response = query_claude(augmented_query)
        except ValueError as e:
            logger.error(f"Error querying Claude API: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Error querying AI model: {str(e)}"})
        
        return JSONResponse(content={"response": response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

def restart_server():
    os.execv(sys.executable, ['python'] + sys.argv)

@app.post("/restart")
async def restart(background_tasks: BackgroundTasks):
    background_tasks.add_task(restart_server)
    return {"message": "Server restart initiated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)