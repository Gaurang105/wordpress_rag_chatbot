import os
import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
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
    model,
    posts_are_equal
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
        # Always fetch the latest posts
        wordpress_url = "https://rajeshjain.com/wp-json/wp/v2/posts"
        latest_posts = fetch_wordpress_posts(wordpress_url)
        
        if not latest_posts:
            logger.error("No posts were fetched. Cannot proceed with startup.")
            return

        # Compare with cached posts and update if necessary
        if os.path.exists(POSTS_CACHE):
            cached_posts = load_data(POSTS_CACHE)
            new_posts = [post for post in latest_posts if not any(posts_are_equal(post, cached_post) for cached_post in cached_posts)]
            if new_posts:
                posts = cached_posts + new_posts
                save_data(posts, POSTS_CACHE)
                logger.info(f"Added {len(new_posts)} new posts to the cache.")
            else:
                posts = cached_posts
        else:
            posts = latest_posts
            save_data(posts, POSTS_CACHE)

        # Update chunked posts
        chunked_posts = chunk_posts(posts)
        save_data(chunked_posts, CHUNKED_POSTS_CACHE)

        # Create or connect to Pinecone index
        pinecone_index = create_pinecone_index(PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)

        # Update Pinecone index with new posts
        update_pinecone_index(pinecone_index, chunked_posts)

        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)

def update_pinecone_index(index, chunked_posts):
    vectors_to_upsert = []
    for post in chunked_posts:
        for i, chunk in enumerate(post['chunks']):
            vector_id = f"{post['id']}_{i}"
            if not index.fetch(ids=[vector_id])['vectors']:
                vector = model.encode(chunk).tolist()
                vectors_to_upsert.append((vector_id, vector, {"text": chunk}))
    
    if vectors_to_upsert:
        batch_upsert(index, vectors_to_upsert)
        logger.info(f"Upserted {len(vectors_to_upsert)} new vectors to Pinecone index")

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

async def update_server():
    global posts, chunked_posts, pinecone_index
    
    try:
        # Fetch the latest posts
        wordpress_url = "https://rajeshjain.com/wp-json/wp/v2/posts"
        latest_posts = fetch_wordpress_posts(wordpress_url)
        
        if not latest_posts:
            logger.error("No posts were fetched. Cannot proceed with update.")
            return

        # Compare with cached posts and update if necessary
        if os.path.exists(POSTS_CACHE):
            cached_posts = load_data(POSTS_CACHE)
            new_posts = [post for post in latest_posts if not any(posts_are_equal(post, cached_post) for cached_post in cached_posts)]
            if new_posts:
                posts = cached_posts + new_posts
                save_data(posts, POSTS_CACHE)
                logger.info(f"Added {len(new_posts)} new posts to the cache.")
            else:
                logger.info("No new posts found.")
                return  # Exit if no new posts
        else:
            new_posts = latest_posts
            posts = new_posts
            save_data(posts, POSTS_CACHE)

        # Only chunk and process new posts
        new_chunked_posts = chunk_posts(new_posts)
        
        # Update chunked_posts in memory and cache
        if os.path.exists(CHUNKED_POSTS_CACHE):
            existing_chunked_posts = load_data(CHUNKED_POSTS_CACHE)
            chunked_posts = existing_chunked_posts + new_chunked_posts
        else:
            chunked_posts = new_chunked_posts
        save_data(chunked_posts, CHUNKED_POSTS_CACHE)

        # Update Pinecone index with only the new posts
        update_pinecone_index(pinecone_index, new_chunked_posts)

        logger.info(f"Server update completed successfully. Processed {len(new_posts)} new posts.")
    except Exception as e:
        logger.error(f"Error during server update: {str(e)}", exc_info=True)

@app.post("/update")
async def update(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_server)
    return {"message": "Server update initiated. New posts will be fetched and added to the index."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)