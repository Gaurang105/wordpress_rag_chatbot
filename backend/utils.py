import os
import requests
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
import pickle
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('punkt', quiet=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_wordpress_posts(url, per_page=100):
    all_posts = []
    page = 1
    while True:
        try:
            response = requests.get(f"{url}?per_page={per_page}&page={page}")
            if response.status_code == 400:
                break
            response.raise_for_status()
            posts = response.json()
            if not posts:
                break
            all_posts.extend(posts)
            page += 1
        except requests.RequestException as e:
            logger.error(f"Error fetching posts: {str(e)}")
            break
    logger.info(f"Total posts fetched: {len(all_posts)}")
    return all_posts

def chunk_posts(posts, max_chunk_size=1000):
    chunked_posts = []
    for post in posts:
        content = post['content']['rendered']
        sentences = sent_tokenize(content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        chunked_posts.append({"id": post["id"], "chunks": chunks})
    return chunked_posts

def generate_embeddings_batch(chunks, batch_size=32):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            yield model.encode(batch, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            yield []

def create_pinecone_index(index_name, dimension):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" 
            )
        )
    return pc.Index(index_name)

def batch_upsert(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

def embed_query(query):
    return model.encode([query])[0].tolist()

def similarity_search(query_vector, index, top_k=5):
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [match['id'] for match in results['matches']]

def get_context(similar_ids, chunked_posts):
    context = []
    for id_str in similar_ids:
        post_id, chunk_id = map(int, id_str.split('_'))
        for post in chunked_posts:
            if post['id'] == post_id:
                context.append(post['chunks'][chunk_id])
                break
    return context

def augment_query(query, context):
    return f"Context: {' '.join(context)}\n\nQuery: {query}"

def query_claude(augmented_query):
    API_URL = 'https://api.anthropic.com/v1/messages'
    API_KEY = os.getenv('CLAUDE_API_KEY')

    if not API_KEY:
        raise ValueError("Claude API key is missing")

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY,
        'anthropic-version': '2023-06-01'
    }
    
    system_message = (
        "You are an AI assistant for Rajesh Jain's blog. Answer questions based solely on the provided context from his blog posts. "
        "If the answer cannot be found in the context, respond with 'I don't have enough information to answer that question.' "
        "Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context from Rajesh Jain's blog posts. "
        "IMPORTANT: Structure your response with clear formatting:\n"
        "1. Use bullet points or numbered lists for key points.\n"
        "2. Separate distinct ideas into different paragraphs.\n"
        "3. Use headings or subheadings if appropriate.\n"
        "4. Ensure your response is well-organized and easy to read.\n"
        "5. If providing a summary, clearly label it as such.\n"
        "6. Do not start your response with any introductory phrases like 'Based on the context provided'. Start directly with the answer."
    )
    
    data = {
        'model': 'claude-3-sonnet-20240229',
        'system': system_message,
        'messages': [
            {'role': 'user', 'content': augmented_query}
        ],
        'max_tokens': 1000
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        
        if 'content' in response_data and len(response_data['content']) > 0:
            return response_data['content'][0]['text']
        else:
            raise ValueError("Unexpected response structure from Claude API")
    except Exception as e:
        logger.error(f"Error querying Claude API: {str(e)}")
        raise
    
def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)