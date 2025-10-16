# Mem0 Chat Memory API

A FastAPI-based chat memory system that leverages Mem0 for persistent, user-specific conversation memory. Store and retrieve chat histories while maintaining contextual knowledge across sessions using vector embeddings and a relational database.

## Features

- **Persistent Chat Storage**: Store user and AI messages in PostgreSQL
- **Memory Management**: Use Mem0 for intelligent memory retrieval and context
- **Vector Search**: Qdrant-powered semantic search for relevant memories
- **LLM Integration**: Ollama-based embeddings and language models
- **RESTful API**: Simple endpoints for adding and retrieving chats
- **Docker Support**: Easy containerized deployment

## Prerequisites

### For Local Development
- Python 3.11+
- PostgreSQL database
- Qdrant vector database
- Ollama with required models

### For Docker Deployment
- Docker and Docker Compose
- External services (PostgreSQL, Qdrant, Ollama) or use included Docker setup

## Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aadityamundhalia/astrology-mem0.git
   cd mem0
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file or set environment variables:
   ```bash
   DATABASE_URL=postgresql://user:password@localhost:5432/mem0_db
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=granite4:latest
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   OLLAMA_EMBEDDING_BASE_URL=http://localhost:11434
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION=mem0_collection
   USE_LLM_REFORMAT=false  # Set to true to enable LLM-based query reformatting
   ```

5. **Start external services**:
   - PostgreSQL
   - Qdrant
   - Ollama (with models `granite4:latest` and `nomic-embed-text`)

6. **Run database migrations**:
   ```bash
   alembic upgrade head
   ```

7. **Start the application**:
   ```bash
   uvicorn main:app --reload
   ```

### Docker Deployment

1. **Ensure external services are running** (PostgreSQL, Qdrant, Ollama)

2. **Build and start**:
   ```bash
   docker-compose up --build -d
   ```

3. **Run migrations** (if needed):
   ```bash
   docker-compose exec app alembic upgrade head
   ```

## API Documentation

### POST /add
Add a new conversation to memory and database. Only user messages are stored in memory for context retrieval.

**Request Body**:
```json
{
  "user_id": 123456789,
  "user_message": "What movie should I watch?",
  "ai_message": "I recommend Inception!"
}
```

**Notes**:
- Only the `user_message` is stored in Mem0 memory for future context retrieval
- Both user and AI messages are stored in the database for chat history
- The `ai_message` is not added to memory to avoid storing generated responses as contextual memory

**Response**:
```json
{
  "status": "success"
}
```

### GET /get
Retrieve chat history and relevant memories.

**Query Parameters**:
- `user_id` (required): User identifier
- `msg` (required): Current message/query
- `num_chats` (optional): Number of recent chats to include (default: 10)
- `include_chat_history` (optional): Include chat history in response (default: false)

**Notes**:
- If `USE_LLM_REFORMAT=true`, the `msg` parameter will be reformatted using the LLM for better search results before querying memories.

### DELETE /clear
Clear all chat history and memories for a specific user.

**Query Parameters**:
- `user_id` (required): User identifier to clear data for

**Response**:
```json
{
  "status": "success",
  "message": "Cleared X chat messages and all memories for user Y"
}
```

**Error Response**:
```json
{
  "status": "error",
  "message": "Failed to clear data for user Y: error details"
}
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Development

### Code Structure
- `main.py`: FastAPI application and endpoints
- `models.py`: SQLAlchemy database models
- `config.py`: Configuration settings
- `alembic/`: Database migrations
- `tests/`: Unit tests

### Adding New Features
1. Update models in `models.py`
2. Create migration: `alembic revision --autogenerate -m "description"`
3. Apply migration: `alembic upgrade head`
4. Add tests in `tests/`
5. Update API endpoints in `main.py`

### Rebuilding Docker
After code changes:
```bash
docker-compose build
docker-compose up -d
```

## Logs

Check application logs:
```bash
# Local
# Logs appear in terminal when running uvicorn

# Docker
docker-compose logs -f app
```

## Stopping

```bash
# Local
# Ctrl+C to stop uvicorn

# Docker
docker-compose down
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT