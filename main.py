from fastapi import FastAPI
from mem0 import Memory
import json
from pydantic import BaseModel
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, OLLAMA_EMBEDDING_BASE_URL,
    USE_LLM_REFORMAT
)
import traceback
from typing import Optional
import ollama
import datetime

app = FastAPI()


class AddRequest(BaseModel):
    user_id: int
    user_message: str
    ai_message: str


app = FastAPI()

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": QDRANT_COLLECTION,
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_MODEL,
            "temperature": 0.7,  # Increased from 0.2 to encourage more memory extraction
            "max_tokens": 2000,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_EMBEDDING_MODEL,
            "ollama_base_url": OLLAMA_EMBEDDING_BASE_URL,
        },
    },
}

memory = Memory.from_config(config)


@app.post("/add")
async def add_chat(request: AddRequest):
    user_id = str(request.user_id)
    user_message = request.user_message
    ai_message = request.ai_message

    # Add to memory with enhanced metadata to encourage storage
    message_text = [
        {"role": "user", "content": user_message},
    ]

    try:
        # Add multiple variations to increase memory storage
        memory.add(
            messages=message_text, 
            user_id=user_id, 
            infer=True,
            metadata={
                "importance": "high", 
                "type": "conversation",
                "timestamp": str(datetime.datetime.utcnow()),
                "should_remember": "true"
            }
        )
        
        # Add again with different metadata to force more storage
        memory.add(
            messages=message_text, 
            user_id=user_id, 
            infer=True,
            metadata={
                "importance": "critical", 
                "type": "user_input",
                "timestamp": str(datetime.datetime.utcnow()),
                "should_remember": "true",
                "context": "personal_preference"
            }
        )
        
        # Add a third time with even more aggressive settings
        memory.add(
            messages=message_text, 
            user_id=user_id, 
            infer=True,
            metadata={
                "importance": "maximum", 
                "type": "memory",
                "timestamp": str(datetime.datetime.utcnow()),
                "should_remember": "true",
                "force_storage": "true"
            }
        )
        
    except Exception as e:
        print(f"Error adding to memory: {e}")
        print(traceback.format_exc())

    return {"status": "success"}


@app.get("/get")
async def get_memories(
    user_id: str,
    msg: str
):
    # Reformat query using LLM if enabled
    query = msg
    if USE_LLM_REFORMAT:
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Reformat this query to be more suitable for semantic search and memory retrieval: {msg}"
                }]
            )
            query = response['message']['content'].strip()
        except Exception as e:
            print(f"Error reformatting query with LLM: {e}")
            # Fall back to original message
    
    # Get memories from Mem0
    try:
        knowledge = memory.search(
            query=query, user_id=user_id, limit=20  # Increased limit
        )
        knowledge_str = "\n".join(
            f"- {entry['memory']}" for entry in knowledge["results"]
        )

        output = "<knowledge_about_user>\n" + knowledge_str + "\n</knowledge_about_user>"

        return {"data": output}
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        return {"data": "<knowledge_about_user>\n</knowledge_about_user>"}

@app.delete("/clear")
async def clear_user_data(user_id: str):
    """
    Clear all memories for a given user
    """
    try:
        # Clear memories from Mem0
        try:
            print(f"Attempting to delete memories for user {user_id}")
            memory.delete(user_id=user_id)
            print(f"Successfully called memory.delete for user {user_id}")
        except Exception as mem_error:
            print(f"Direct memory.delete failed for user {user_id}: {mem_error}")
            # Try alternative method - search and delete individually
            try:
                print(f"Trying alternative method for user {user_id}")
                user_memories = memory.search(query=" ", user_id=user_id, limit=1000)  # Try with space
                print(f"Found {len(user_memories.get('results', []))} memories for user {user_id}")
                memory_ids = [mem.get('id') for mem in user_memories.get('results', []) if mem.get('id')]
                print(f"Memory IDs to delete: {memory_ids}")
                for memory_id in memory_ids:
                    memory.delete(memory_id=memory_id)
                    print(f"Deleted memory {memory_id}")
            except Exception as alt_error:
                print(f"Alternative memory deletion also failed for user {user_id}: {alt_error}")

        return {
            "status": "success",
            "message": f"Cleared all memories for user {user_id}"
        }
    except Exception as e:
        print(f"Error clearing data for user {user_id}: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Failed to clear data for user {user_id}: {str(e)}"
        }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy"}
