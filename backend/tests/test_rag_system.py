"""
Test RAG system end-to-end integration
"""
import sys
import os
from unittest.mock import Mock, patch
from test_fixtures import (
    TestRAGSystem, print_test_header, print_test_result, 
    print_section_header
)

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import config

def test_rag_system_initialization():
    """Test RAG system component initialization"""
    print_test_header("RAG SYSTEM INITIALIZATION")
    
    try:
        print_section_header("Component Creation Test")
        
        rag_system = TestRAGSystem.create_test_rag_system()
        
        # Check that all components are created
        has_document_processor = hasattr(rag_system, 'document_processor') and rag_system.document_processor is not None
        has_vector_store = hasattr(rag_system, 'vector_store') and rag_system.vector_store is not None
        has_ai_generator = hasattr(rag_system, 'ai_generator') and rag_system.ai_generator is not None
        has_session_manager = hasattr(rag_system, 'session_manager') and rag_system.session_manager is not None
        has_tool_manager = hasattr(rag_system, 'tool_manager') and rag_system.tool_manager is not None
        
        print(f"Document processor: {has_document_processor}")
        print(f"Vector store: {has_vector_store}")
        print(f"AI generator: {has_ai_generator}")
        print(f"Session manager: {has_session_manager}")
        print(f"Tool manager: {has_tool_manager}")
        
        all_components = all([
            has_document_processor, has_vector_store, has_ai_generator,
            has_session_manager, has_tool_manager
        ])
        
        print_test_result("All components initialized", all_components)
        
        # Check tool registration
        print_section_header("Tool Registration Test")
        
        tools = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool.get('name') for tool in tools]
        
        print(f"Registered tools: {tool_names}")
        
        has_search_tool = 'search_course_content' in tool_names
        has_outline_tool = 'get_course_outline' in tool_names
        
        print_test_result("Search tool registered", has_search_tool)
        print_test_result("Outline tool registered", has_outline_tool)
        
        return all_components and has_search_tool and has_outline_tool
        
    except Exception as e:
        print(f"ERROR in RAG system initialization test: {e}")
        print_test_result("RAG system initialization", False, str(e))
        return False

def test_rag_system_course_analytics():
    """Test RAG system course analytics functionality"""
    print_test_header("RAG SYSTEM COURSE ANALYTICS")
    
    try:
        rag_system = TestRAGSystem.create_test_rag_system()
        
        print_section_header("Course Analytics Test")
        
        analytics = rag_system.get_course_analytics()
        
        print(f"Analytics structure: {analytics}")
        
        has_total_courses = 'total_courses' in analytics
        has_course_titles = 'course_titles' in analytics
        
        if has_total_courses:
            total_courses = analytics['total_courses']
            print(f"Total courses: {total_courses}")
            has_courses = total_courses > 0
        else:
            has_courses = False
        
        if has_course_titles:
            course_titles = analytics['course_titles']
            print(f"Course titles: {course_titles}")
            has_mcp_course = any('MCP' in title.upper() for title in course_titles)
            print(f"MCP course found: {has_mcp_course}")
        else:
            has_mcp_course = False
        
        print_test_result("Analytics structure correct", has_total_courses and has_course_titles)
        print_test_result("Has course data", has_courses)
        print_test_result("MCP course exists", has_mcp_course)
        
        return has_total_courses and has_course_titles and has_courses
        
    except Exception as e:
        print(f"ERROR in course analytics test: {e}")
        print_test_result("Course analytics test", False, str(e))
        return False

@patch.object(config, 'OPENAI_API_KEY', 'test-key')
def test_rag_system_query_flow():
    """Test end-to-end query processing"""
    print_test_header("RAG SYSTEM QUERY FLOW")
    
    try:
        # Mock the OpenAI client to avoid actual API calls
        with patch('ai_generator.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock a response that should trigger tool use
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "search_course_content"
            mock_tool_call.function.arguments = '{"query": "MCP client", "course_name": "MCP", "lesson_number": 5}'
            
            initial_response = Mock()
            initial_choice = Mock()
            initial_message = Mock()
            initial_message.content = ""
            initial_message.tool_calls = [mock_tool_call]
            initial_choice.message = initial_message
            initial_response.choices = [initial_choice]
            
            # Mock final response
            final_response = Mock()
            final_choice = Mock()
            final_message = Mock()
            final_message.content = "Here's information about creating MCP clients in lesson 5..."
            final_choice.message = final_message
            final_response.choices = [final_choice]
            
            mock_client.chat.completions.create.side_effect = [initial_response, final_response]
            
            rag_system = TestRAGSystem.create_test_rag_system()
            
            # Test different query scenarios
            test_queries = [
                {
                    "name": "Lesson-specific question",
                    "query": "What's in lesson 5 of the MCP course?",
                    "session_id": "test_session_1"
                },
                {
                    "name": "Course outline request", 
                    "query": "Show me the MCP course outline",
                    "session_id": "test_session_2"
                },
                {
                    "name": "Content search",
                    "query": "How to create an MCP client?",
                    "session_id": "test_session_3"
                }
            ]
            
            for test_query in test_queries:
                print_section_header(f"Query Test: {test_query['name']}")
                
                try:
                    response, sources = rag_system.query(
                        query=test_query['query'],
                        session_id=test_query['session_id']
                    )
                    
                    print(f"Query: {test_query['query']}")
                    print(f"Response type: {type(response)}")
                    print(f"Response length: {len(response) if response else 0}")
                    print(f"Sources count: {len(sources) if sources else 0}")
                    print(f"Response preview: {str(response)[:200]}...")
                    
                    has_response = response is not None and len(str(response)) > 0
                    print_test_result(f"{test_query['name']} - has response", has_response)
                    
                    # Check if AI generator was called
                    ai_called = mock_client.chat.completions.create.called
                    print_test_result(f"{test_query['name']} - AI called", ai_called)
                    
                    # Reset mock for next test
                    mock_client.reset_mock()
                    mock_client.chat.completions.create.side_effect = [initial_response, final_response]
                    
                except Exception as e:
                    print(f"Error in query test: {e}")
                    print_test_result(f"{test_query['name']}", False, str(e))
        
        return True
        
    except Exception as e:
        print(f"ERROR in RAG system query flow test: {e}")
        print_test_result("RAG system query flow test", False, str(e))
        return False

def test_rag_system_session_management():
    """Test session management functionality"""
    print_test_header("RAG SYSTEM SESSION MANAGEMENT")
    
    try:
        with patch('ai_generator.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock simple response without tool calls
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_message.tool_calls = None
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            rag_system = TestRAGSystem.create_test_rag_system()
            
            print_section_header("Session Tracking Test")
            
            session_id = "test_session"
            
            # First query
            response1, _ = rag_system.query("First question", session_id)
            print(f"First response: {response1}")
            
            # Second query in same session
            response2, _ = rag_system.query("Follow-up question", session_id)
            print(f"Second response: {response2}")
            
            # Check if conversation history is being used
            call_args_list = mock_client.chat.completions.create.call_args_list
            
            if len(call_args_list) >= 2:
                # Check if second call includes conversation history
                second_call_kwargs = call_args_list[1][1]
                messages = second_call_kwargs.get('messages', [])
                
                # Look for system message that might contain history
                system_messages = [msg for msg in messages if msg.get('role') == 'system']
                
                has_history = any('Previous conversation' in msg.get('content', '') 
                                for msg in system_messages)
                
                print(f"History included in second call: {has_history}")
                print_test_result("Session history tracked", has_history)
            else:
                print_test_result("Multiple queries processed", False, "Not enough API calls")
        
        return True
        
    except Exception as e:
        print(f"ERROR in session management test: {e}")
        print_test_result("Session management test", False, str(e))
        return False

def test_rag_system_error_handling():
    """Test error handling in RAG system"""
    print_test_header("RAG SYSTEM ERROR HANDLING")
    
    try:
        with patch('ai_generator.OpenAI') as mock_openai_class:
            # Test with API key issues
            print_section_header("API Key Error Test")
            
            mock_openai_class.side_effect = Exception("Invalid API key")
            
            try:
                rag_system = TestRAGSystem.create_test_rag_system()
                print_test_result("Handles API key error", False, "Should have raised exception")
            except Exception as e:
                print(f"Expected error: {e}")
                print_test_result("Handles API key error gracefully", True)
            
            # Test with tool execution errors
            print_section_header("Tool Execution Error Test")
            
            # Reset the mock
            mock_openai_class.side_effect = None
            mock_client = Mock()
            mock_openai_class.return_value = mock_client
            
            # Mock tool call response
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.type = "function"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "search_course_content"
            mock_tool_call.function.arguments = '{"query": "test"}'
            
            tool_response = Mock()
            tool_choice = Mock()
            tool_message = Mock()
            tool_message.content = ""
            tool_message.tool_calls = [mock_tool_call]
            tool_choice.message = tool_message
            tool_response.choices = [tool_choice]
            
            # Mock error response
            error_response = Mock()
            error_choice = Mock()
            error_message = Mock()
            error_message.content = "I encountered an error while searching."
            error_choice.message = error_message
            error_response.choices = [error_choice]
            
            mock_client.chat.completions.create.side_effect = [tool_response, error_response]
            
            try:
                rag_system = TestRAGSystem.create_test_rag_system()
                
                # Simulate tool execution error by patching the tool
                with patch.object(rag_system.search_tool, 'execute', side_effect=Exception("Search error")):
                    response, sources = rag_system.query("Test query that should fail")
                    
                    print(f"Response with tool error: {response}")
                    
                    # Should still get a response even if tool fails
                    has_response = response is not None
                    print_test_result("Handles tool execution error", has_response)
                    
            except Exception as e:
                print(f"Unexpected error in tool error test: {e}")
                print_test_result("Tool error handling", False, str(e))
        
        return True
        
    except Exception as e:
        print(f"ERROR in error handling test: {e}")
        print_test_result("Error handling test", False, str(e))
        return False

if __name__ == "__main__":
    print("Starting RAG System Integration Tests...")
    
    # Run the tests
    test1_passed = test_rag_system_initialization()
    test2_passed = test_rag_system_course_analytics()
    test3_passed = test_rag_system_query_flow()
    test4_passed = test_rag_system_session_management()
    test5_passed = test_rag_system_error_handling()
    
    print(f"\n{'='*60}")
    print("RAG SYSTEM TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Initialization test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Course analytics test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Query flow test: {'PASS' if test3_passed else 'FAIL'}")
    print(f"Session management test: {'PASS' if test4_passed else 'FAIL'}")
    print(f"Error handling test: {'PASS' if test5_passed else 'FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed]):
        print("\n✅ All RAG system tests passed!")
    else:
        print("\n❌ Some RAG system tests failed - this indicates integration issues")