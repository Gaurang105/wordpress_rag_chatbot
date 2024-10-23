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
from datetime import datetime
import re

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
            response = requests.get(f"{url}?per_page={per_page}&page={page}&_fields=id,content,modified")
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

def posts_are_equal(post1, post2):
    """Compare two posts to check if they are the same."""
    return post1['id'] == post2['id'] and post1['modified'] == post2['modified']

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

def augment_query(query, context, conversation_history):
    # Format previous conversation
    conversation_context = ""
    if conversation_history:
        conversation_context = "\nPrevious conversation:\n" + "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in conversation_history[-4:]  # Include last 2 exchanges (4 messages)
        ])
    
    return f"Context from blog posts: {' '.join(context)}{conversation_context}\n\nCurrent query: {query}"

def query_claude(augmented_query, conversation_history):
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
        "You are a knowledgeable and friendly AI assistant having a natural conversation about Rajesh Jain based on his blog posts. "
        "Your goal is to make the conversation feel human and engaging.\n\n"
        "### Core Guidelines:\n"
        "1. Be concise by default. Only provide detailed information when specifically asked.\n"
        "2. Never start responses with phrases like 'Based on the context' or 'According to'. Jump straight into the answer.\n"
        "3. Use a warm, conversational tone while maintaining accuracy.\n"
        "4. If you don't have enough information, simply say 'I don't have enough information about that.'\n\n"
        "### Response Style:\n"
        "- Keep initial responses brief (1-2 sentences) unless asked for more detail\n"
        "- Use natural language rather than bullet points unless specifically requested\n"
        "- Make smooth references to previous conversation points when relevant\n"
        "- Avoid formal or academic tones - think friendly conversation\n\n"
        "### Examples:\n"
        "❌ 'Based on the context, Rajesh Jain is the founder of...'\n"
        "✅ 'Rajesh Jain is the founder of...'\n\n"
        "❌ 'The available information indicates that...'\n"
        "✅ 'He started the company in...'\n\n"
        "❌ Long detailed response for a simple question\n"
        "✅ Short, direct answer unless more detail is requested"
    )

    messages = []
    if conversation_history:
        for msg in conversation_history[-4:]:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
    
    messages.append({'role': 'user', 'content': augmented_query})
    
    data = {
        'model': 'claude-3-sonnet-20240229',
        'system': system_message,
        'messages': messages,
        'max_tokens': 1000,
        'temperature': 0.3
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
    """Save data to a pickle file with error handling."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Successfully saved data to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        raise

def load_data(filename):
    """Load data from a pickle file with error handling."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully loaded data from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        raise

def clean_old_conversations(conversations, max_age_hours=24):
    """Clean up old conversations to prevent memory bloat."""
    current_time = datetime.now()
    return {
        conv_id: conv_data 
        for conv_id, conv_data in conversations.items() 
        if (current_time - conv_data['timestamp']).total_seconds() < max_age_hours * 3600
    }

def format_conversation_for_context(conversation_history, max_turns=2):
    """Format conversation history for context inclusion."""
    if not conversation_history:
        return ""
    relevant_history = conversation_history[-max_turns*2:]
    formatted_history = []
    for msg in relevant_history:
        role_prefix = "User: " if msg['role'] == 'user' else "Assistant: "
        formatted_history.append(f"{role_prefix}{msg['content']}")
    return "\n".join(formatted_history)

def sanitize_text(text):
    """Clean and sanitize text content."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    text = ' '.join(text.split())
    
    return text.strip()

def truncate_context(context, max_length=8000):
    """Truncate context to prevent exceeding API limits."""
    if len(context) <= max_length:
        return context
    sentences = sent_tokenize(context)
    
    truncated = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) + 1 <= max_length:
            truncated.append(sentence)
            current_length += len(sentence) + 1
        else:
            break
    
    return ' '.join(truncated)

def handle_api_error(response):
    """Handle API error responses."""
    try:
        error_data = response.json()
        error_message = error_data.get('error', {}).get('message', 'Unknown error')
        error_type = error_data.get('error', {}).get('type', 'UnknownError')
        
        logger.error(f"API Error: {error_type} - {error_message}")
        return f"Error: {error_message}"
    except Exception as e:
        logger.error(f"Error parsing API error response: {str(e)}")
        return "An unexpected error occurred"