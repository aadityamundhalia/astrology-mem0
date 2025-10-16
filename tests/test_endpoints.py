import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@patch('main.memory')
def test_add_chat_success(mock_memory):
    mock_memory.add = MagicMock()
    
    response = client.post("/add", json={
        "user_id": 1,
        "user_message": "Hello",
        "ai_message": "Hi there"
    })
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    assert mock_memory.add.call_count == 3  # Called three times as per code

@patch('main.memory')
def test_add_chat_memory_error(mock_memory):
    mock_memory.add = MagicMock(side_effect=Exception("Memory error"))
    
    response = client.post("/add", json={
        "user_id": 1,
        "user_message": "Hello",
        "ai_message": "Hi there"
    })
    
    assert response.status_code == 200
    assert response.json() == {"status": "success"}  # Still returns success even on error

@patch('main.memory')
@patch('main.ollama.chat')
def test_get_memories_with_reformat(mock_ollama_chat, mock_memory):
    # Mock USE_LLM_REFORMAT to True, but since it's imported, patch it
    with patch('main.USE_LLM_REFORMAT', True):
        mock_ollama_chat.return_value = {'message': {'content': 'Reformatted query'}}
        mock_memory.search.return_value = {"results": [{"memory": "Test memory"}]}
        
        response = client.get("/get?user_id=1&msg=Hello")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert "<knowledge_about_user>" in data
        assert "Test memory" in data

@patch('main.memory')
@patch('main.ollama.chat')
def test_get_memories_without_reformat(mock_ollama_chat, mock_memory):
    with patch('main.USE_LLM_REFORMAT', False):
        mock_memory.search.return_value = {"results": [{"memory": "Test memory"}]}
        
        response = client.get("/get?user_id=1&msg=Hello")
        
        assert response.status_code == 200
        data = response.json()["data"]
        assert "<knowledge_about_user>" in data
        assert "Test memory" in data
        mock_ollama_chat.assert_not_called()

@patch('main.memory')
def test_get_memories_error(mock_memory):
    mock_memory.search.side_effect = Exception("Search error")
    
    response = client.get("/get?user_id=1&msg=Hello")
    
    assert response.status_code == 200
    assert response.json()["data"] == "<knowledge_about_user>\n</knowledge_about_user>"

@patch('main.memory')
def test_clear_memories_success(mock_memory):
    mock_memory.delete = MagicMock()
    
    response = client.delete("/clear?user_id=1")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_memory.delete.assert_called_with(user_id="1")

@patch('main.memory')
def test_clear_memories_error(mock_memory):
    # Note: The code always returns success even on errors, so testing that it handles errors gracefully
    mock_memory.delete.side_effect = Exception("Delete error")
    mock_memory.search.return_value = {"results": [{"id": "mem1"}]}
    
    response = client.delete("/clear?user_id=1")
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"  # Code always returns success

def test_health_check():
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}