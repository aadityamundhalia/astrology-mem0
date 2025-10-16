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

        output = "<knowledge_about_user>\n" + knowledge_str + "\n</knowledge_about_user>"

        return {"data": output}
