#!/usr/bin/env python3
"""
Verification script to test the RAG system fix for lesson 5 MCP content
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from config import config

def test_fix():
    """Test that the fix resolves the original issue"""
    print("🔧 TESTING RAG SYSTEM FIX")
    print("=" * 50)
    
    # Initialize RAG system
    print("1. Initializing RAG system...")
    rag_system = RAGSystem(config)
    
    # Check initial state
    analytics = rag_system.get_course_analytics()
    print(f"   Initial courses: {analytics['total_courses']}")
    
    # Load data if not already loaded
    if analytics['total_courses'] == 0:
        print("2. Loading course data...")
        docs_path = "../docs"
        courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)
        print(f"   ✅ Loaded {courses} courses with {chunks} chunks")
        
        analytics = rag_system.get_course_analytics()
        print(f"   📊 Final course count: {analytics['total_courses']}")
    else:
        print("2. Course data already loaded ✅")
    
    # Test the original failing query
    print("\n3. Testing original failing query...")
    print("   Query: 'What's in lesson 5 of the MCP course?'")
    
    try:
        # Mock the OpenAI API call since we're testing the search functionality
        with MockOpenAI():
            response, sources = rag_system.query(
                "What's in lesson 5 of the MCP course?", 
                session_id="test_session"
            )
            
            print(f"   Response length: {len(response)}")
            print(f"   Sources count: {len(sources)}")
            print(f"   Response preview: {response[:100]}...")
            
            if len(sources) > 0:
                print(f"   Source: {sources[0]}")
            
            # Test specific lesson 5 content search
            print("\n4. Testing direct CourseSearchTool...")
            search_result = rag_system.search_tool.execute(
                query="creating MCP client",
                course_name="MCP", 
                lesson_number=5
            )
            
            print(f"   Search result length: {len(search_result)}")
            print(f"   Search preview: {search_result[:200]}...")
            
            success = (
                len(search_result) > 100 and 
                "No relevant content found" not in search_result and
                "lesson 5" in search_result.lower()
            )
            
            if success:
                print("\n✅ SUCCESS: Original issue is resolved!")
                print("   - Course data is loaded")
                print("   - Lesson 5 MCP content is searchable")
                print("   - Search tool returns relevant results")
            else:
                print("\n❌ FAILURE: Issue not fully resolved")
                return False
                
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test course outline tool as well
    print("\n5. Testing CourseOutlineTool...")
    try:
        outline_result = rag_system.outline_tool.execute(course_title="MCP")
        print(f"   Outline length: {len(outline_result)}")
        
        if "Lesson 5: Creating An MCP Client" in outline_result:
            print("   ✅ Lesson 5 found in course outline")
        else:
            print("   ❌ Lesson 5 not found in course outline")
            
    except Exception as e:
        print(f"   ❌ Error testing outline tool: {e}")
    
    return True

class MockOpenAI:
    """Context manager to mock OpenAI for testing"""
    def __enter__(self):
        # Mock the generate_response method to avoid actual API calls
        import unittest.mock
        from ai_generator import AIGenerator
        
        def mock_generate_response(self, query, conversation_history=None, tools=None, tool_manager=None):
            if tools and tool_manager:
                # Simulate tool calling
                tool_result = tool_manager.execute_tool(
                    "search_course_content",
                    query="creating MCP client", 
                    course_name="MCP",
                    lesson_number=5
                )
                return f"Based on the search results: {tool_result[:100]}..."
            else:
                return "I don't have access to course content without tools."
        
        self.patcher = unittest.mock.patch.object(
            AIGenerator, 'generate_response', mock_generate_response
        )
        self.patcher.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()

if __name__ == "__main__":
    test_fix()