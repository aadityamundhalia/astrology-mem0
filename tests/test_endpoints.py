import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import json
from main import app

client = TestClient(app)

@pytest.fixture
def mock_memory():
    with patch('main.memory') as mock_mem:
        mock_mem.search.return_value = {"results": [{"memory": "User likes sci-fi movies"}]}
        mock_mem.add.return_value = None
        yield mock_mem


@pytest.fixture
def mock_db():
    with patch('main.SessionLocal') as mock_session:
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        yield mock_db


def test_add_chat_success(mock_memory, mock_db):
    """Test successful chat addition"""
    request_data = {
        "user_id": 123456789,
        "user_message": "What movie should I watch?",
        "ai_message": "I recommend Inception!"
    }

    response = client.post("/add", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "success"}

    # Verify memory add was called
    mock_memory.add.assert_called_once()
    add_call_args = mock_memory.add.call_args
    messages = add_call_args[1]['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == 'What movie should I watch?'
    assert messages[1]['role'] == 'assistant'
    assert messages[1]['content'] == 'I recommend Inception!'

    # Verify DB operations
    assert mock_db.add.call_count == 2  # user and assistant messages
    mock_db.commit.assert_called_once()


def test_get_chats_success(mock_memory, mock_db):
    """Test successful chat retrieval"""
    # Mock DB query results
    mock_chat1 = MagicMock()
    mock_chat1.role = "user"
    mock_chat1.content = "Hello"
    mock_chat2 = MagicMock()
    mock_chat2.role = "assistant"
    mock_chat2.content = "Hi there!"

    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        mock_chat2, mock_chat1  # reversed order from DB
    ]

    response = client.get("/get?user_id=test_user&num_chats=10&msg=test&include_chat_history=true")

    assert response.status_code == 200
    data = response.json()
    assert "data" in data

    # Parse the response
    response_text = data["data"]
    assert "<chat_history>" in response_text
    # Note: with include_chat_history=true, only chat history is returned

def test_get_chats_empty(mock_memory, mock_db):
    """Test getting chats when no chats exist"""
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    response = client.get("/get?user_id=test_user&num_chats=5&msg=test&include_chat_history=true")

    assert response.status_code == 200
    data = response.json()
    response_text = data["data"]

    # Should contain empty chat history
    assert "[]" in response_text

def test_add_chat_missing_fields():
    """Test add chat with missing required fields"""
    # Missing user_id
    response = client.post(
        "/add", json={"user_message": "Hello", "ai_message": "Hi"}
    )
    assert response.status_code == 422

    # Missing user_message
    response = client.post("/add", json={"user_id": 123, "ai_message": "Hi"})
    assert response.status_code == 422

    # Missing ai_message
    response = client.post(
        "/add", json={"user_id": 123, "user_message": "Hello"}
    )
    assert response.status_code == 422

def test_get_chats_missing_user_id():
    """Test get chats without user_id"""
    response = client.get("/get")
    assert response.status_code == 422


def test_clear_user_data_success(mock_memory, mock_db):
    """Test successful user data clearing"""
    # Mock DB delete to return number of deleted rows
    mock_db.query.return_value.filter.return_value.delete.return_value = 5

    response = client.delete("/clear?user_id=test_user")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Cleared 5 chat messages" in data["message"]
    assert "test_user" in data["message"]

    # Verify DB operations
    mock_db.query.assert_called()
    mock_db.commit.assert_called_once()

    # Verify memory delete was called with user_id
    mock_memory.delete.assert_called_once_with(user_id="test_user")


def test_clear_user_data_error(mock_memory, mock_db):
    """Test clearing user data with error"""
    # Mock DB to raise exception
    mock_db.query.side_effect = Exception("Database error")

    response = client.delete("/clear?user_id=test_user")

    assert response.status_code == 200  # API returns 200 with error status
    data = response.json()
    assert data["status"] == "error"
    assert "Failed to clear data" in data["message"]