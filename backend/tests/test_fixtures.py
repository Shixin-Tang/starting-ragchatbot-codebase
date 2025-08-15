"""
Test fixtures and utilities for RAG system testing
"""
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Add the backend directory to the Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem

# Test data constants
SAMPLE_COURSE_TITLES = [
    "Building Towards Computer Use with Anthropic",
    "MCP: Build Rich-Context AI Apps with Anthropic", 
    "Advanced Retrieval for AI with Chroma"
]

SAMPLE_MCP_LESSONS = [
    {"lesson_number": 0, "lesson_title": "Introduction"},
    {"lesson_number": 1, "lesson_title": "Why MCP"},
    {"lesson_number": 2, "lesson_title": "MCP Architecture"},
    {"lesson_number": 3, "lesson_title": "Chatbot Example"},
    {"lesson_number": 4, "lesson_title": "Creating An MCP Server"},
    {"lesson_number": 5, "lesson_title": "Creating An MCP Client"},
    {"lesson_number": 6, "lesson_title": "Connecting The MCP Chatbot To Reference Servers"}
]

class TestVectorStore:
    """Test utilities for vector store operations"""
    
    @staticmethod
    def create_test_vector_store():
        """Create a vector store instance for testing"""
        return VectorStore(
            chroma_path=config.CHROMA_PATH,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=config.MAX_RESULTS
        )
    
    @staticmethod
    def get_test_search_results(documents=None, metadata=None, distances=None):
        """Create mock SearchResults for testing"""
        return SearchResults(
            documents=documents or ["Sample document content"],
            metadata=metadata or [{"course_title": "Test Course", "lesson_number": 1}],
            distances=distances or [0.5]
        )

class TestAIGenerator:
    """Test utilities for AI generator operations"""
    
    @staticmethod
    def create_mock_ai_generator():
        """Create a mock AI generator for testing"""
        mock_generator = Mock(spec=AIGenerator)
        mock_generator.generate_response = Mock(return_value="Test response")
        return mock_generator
    
    @staticmethod
    def create_mock_openai_response(tool_calls=None, content="Test response"):
        """Create a mock OpenAI API response"""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        mock_message.content = content
        mock_message.tool_calls = tool_calls
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        return mock_response

class TestRAGSystem:
    """Test utilities for RAG system operations"""
    
    @staticmethod
    def create_test_rag_system():
        """Create a RAG system instance for testing"""
        return RAGSystem(config)

def print_test_header(test_name: str):
    """Print a formatted test header"""
    print(f"\n{'='*60}")
    print(f"TESTING: {test_name}")
    print(f"{'='*60}")

def print_test_result(test_name: str, passed: bool, details: str = ""):
    """Print formatted test results"""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {test_name}")
    if details:
        print(f"  Details: {details}")

def print_section_header(section_name: str):
    """Print a formatted section header"""
    print(f"\n{'-'*40}")
    print(f"{section_name}")
    print(f"{'-'*40}")