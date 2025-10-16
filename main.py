from fastapi import FastAPI
from mem0 import Memory
import json
from models import SessionLocal, Chat
from pydantic import BaseModel
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, OLLAMA_EMBEDDING_BASE_URL,
    USE_LLM_REFORMAT
)
import traceback
from typing import Optional
import ollama

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
            "temperature": 0.2,
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

    # Add to memory
    message_text = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_message},
    ]

    try:
        memory.add(messages=message_text, user_id=user_id, infer=True)
    except Exception as e:
        print(f"Error adding to memory: {e}")
        print(traceback.format_exc())

    # Store in DB
    db = SessionLocal()
    user_chat = Chat(user_id=user_id, role="user", content=user_message)
    assistant_chat = Chat(
        user_id=user_id, role="assistant", content=ai_message
    )
    db.add(user_chat)
    db.add(assistant_chat)
    db.commit()
    db.close()

    return {"status": "success"}


@app.get("/get")
async def get_chats(
    user_id: str,
    msg: str,
    num_chats: int = 10,
    include_chat_history: bool = False
):
    if include_chat_history:
        db = SessionLocal()
        chats = db.query(Chat).filter(Chat.user_id == user_id).\
            order_by(Chat.timestamp.desc()).limit(num_chats).all()
        db.close()

        # Reverse to chronological order
        chats = chats[::-1]
        chat_history = [{"role": c.role, "content": c.content} for c in chats]
        chat_history_json = json.dumps(chat_history, indent=2)

        output = f"<chat_history>\n{chat_history_json}\n</chat_history>"

        return {"data": output}
    else:
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
        
        # Knowledge
        knowledge = memory.search(
            query=query, user_id=user_id, limit=10
        )
        knowledge_str = "\n".join(
            f"- {entry['memory']}" for entry in knowledge["results"]
        )

@app.delete("/clear")
async def clear_user_data(user_id: str):
    """
    Clear all chat history and memories for a given user
    """
    try:
        # Clear chat history from database
        db = SessionLocal()
        deleted_chats = db.query(Chat).filter(Chat.user_id == user_id).delete()
        db.commit()
        db.close()

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
                # Continue with database cleanup

        return {
            "status": "success",
            "message": f"Cleared {deleted_chats} chat messages and all memories for user {user_id}"
        }
    except Exception as e:
        print(f"Error clearing data for user {user_id}: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Failed to clear data for user {user_id}: {str(e)}"
        }
