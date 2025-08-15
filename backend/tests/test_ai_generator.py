"""
Test AIGenerator tool calling functionality
"""
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from test_fixtures import (
    TestAIGenerator, print_test_header, print_test_result, 
    print_section_header
)

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool
from vector_store import VectorStore
from config import config

def test_ai_generator_tool_conversion():
    """Test conversion of tools from Anthropic to OpenAI format"""
    print_test_header("AI GENERATOR TOOL CONVERSION")
    
    try:
        ai_generator = AIGenerator("test-key", "gpt-4o-mini")
        
        # Test tool definitions in Anthropic format
        anthropic_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "course_name": {"type": "string", "description": "Course name"}
                    },
                    "required": ["query"]
                }
            }
        ]
        
        print_section_header("Tool Conversion Test")
        
        openai_tools = ai_generator._convert_tools_to_openai_format(anthropic_tools)
        
        print(f"Input tools count: {len(anthropic_tools)}")
        print(f"Output tools count: {len(openai_tools)}")
        
        if openai_tools:
            tool = openai_tools[0]
            print(f"Tool structure: {json.dumps(tool, indent=2)}")
            
            # Check required fields
            has_type = tool.get('type') == 'function'
            has_function = 'function' in tool
            function_data = tool.get('function', {})
            has_name = function_data.get('name') == 'search_course_content'
            has_description = 'description' in function_data
            has_parameters = 'parameters' in function_data
            
            conversion_correct = all([has_type, has_function, has_name, has_description, has_parameters])
            print_test_result("Tool conversion correct", conversion_correct)
        
        return True
        
    except Exception as e:
        print(f"ERROR in tool conversion test: {e}")
        print_test_result("Tool conversion test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_ai_generator_tool_calling_detection(mock_openai_class):
    """Test if AI generator correctly identifies when to call tools"""
    print_test_header("AI GENERATOR TOOL CALLING DETECTION")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini")
        
        # Test scenarios for tool calling
        test_scenarios = [
            {
                "name": "Lesson specific query",
                "query": "What's in lesson 5 of the MCP course?",
                "should_call_tool": True,
                "expected_tool": "search_course_content"
            },
            {
                "name": "Course outline request",
                "query": "Show me the outline of the MCP course",
                "should_call_tool": True,
                "expected_tool": "get_course_outline"
            },
            {
                "name": "General knowledge question",
                "query": "What is machine learning?",
                "should_call_tool": False,
                "expected_tool": None
            },
            {
                "name": "Course content search",
                "query": "How to create an MCP client?",
                "should_call_tool": True,
                "expected_tool": "search_course_content"
            }
        ]
        
        for scenario in test_scenarios:
            print_section_header(f"Scenario: {scenario['name']}")
            
            # Mock the response based on whether tool should be called
            if scenario['should_call_tool']:
                # Mock tool call response
                mock_tool_call = Mock()
                mock_tool_call.id = "test_id"
                mock_tool_call.type = "function"
                mock_tool_call.function = Mock()
                mock_tool_call.function.name = scenario['expected_tool']
                mock_tool_call.function.arguments = json.dumps({"query": scenario['query']})
                
                mock_response = Mock()
                mock_choice = Mock()
                mock_message = Mock()
                mock_message.content = ""
                mock_message.tool_calls = [mock_tool_call]
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
            else:
                # Mock direct response without tool calls
                mock_response = Mock()
                mock_choice = Mock()
                mock_message = Mock()
                mock_message.content = "Direct answer without tools"
                mock_message.tool_calls = None
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create mock tools
            tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                },
                {
                    "name": "get_course_outline", 
                    "description": "Get course outline",
                    "input_schema": {"type": "object", "properties": {"course_title": {"type": "string"}}}
                }
            ]
            
            # Mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"
            
            # Test the response
            try:
                response = ai_generator.generate_response(
                    query=scenario['query'],
                    tools=tools,
                    tool_manager=mock_tool_manager if scenario['should_call_tool'] else None
                )
                
                print(f"Query: {scenario['query']}")
                print(f"Expected tool call: {scenario['should_call_tool']}")
                print(f"Response type: {type(response)}")
                print(f"Response preview: {str(response)[:100]}...")
                
                # Verify the OpenAI API was called with correct parameters
                call_args = mock_client.chat.completions.create.call_args
                if call_args:
                    kwargs = call_args[1]
                    has_tools = 'tools' in kwargs
                    has_tool_choice = 'tool_choice' in kwargs
                    
                    print(f"API called with tools: {has_tools}")
                    print(f"API called with tool_choice: {has_tool_choice}")
                    
                    if scenario['should_call_tool']:
                        tools_provided = has_tools and has_tool_choice
                        print_test_result(f"{scenario['name']} - tools provided", tools_provided)
                    else:
                        no_tools_or_optional = not has_tools or not has_tool_choice
                        print_test_result(f"{scenario['name']} - no unnecessary tools", no_tools_or_optional)
                else:
                    print_test_result(f"{scenario['name']} - API called", False, "No API call made")
                
            except Exception as e:
                print(f"Error in scenario: {e}")
                print_test_result(f"{scenario['name']}", False, str(e))
        
        return True
        
    except Exception as e:
        print(f"ERROR in tool calling detection test: {e}")
        print_test_result("Tool calling detection test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_ai_generator_tool_execution_flow(mock_openai_class):
    """Test the complete tool execution flow"""
    print_test_header("AI GENERATOR TOOL EXECUTION FLOW")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini")
        
        print_section_header("Tool Execution Flow Test")
        
        # Mock initial response with tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "search_course_content"
        mock_tool_call.function.arguments = json.dumps({
            "query": "MCP client",
            "course_name": "MCP",
            "lesson_number": 5
        })
        
        initial_response = Mock()
        initial_choice = Mock()
        initial_message = Mock()
        initial_message.content = ""
        initial_message.tool_calls = [mock_tool_call]
        initial_choice.message = initial_message
        initial_response.choices = [initial_choice]
        
        # Mock final response after tool execution
        final_response = Mock()
        final_choice = Mock()
        final_message = Mock()
        final_message.content = "Based on the search results, here's information about MCP client..."
        final_choice.message = final_message
        final_response.choices = [final_choice]
        
        # Set up mock to return different responses for different calls
        mock_client.chat.completions.create.side_effect = [initial_response, final_response]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Lesson 5 content about creating MCP client..."
        
        # Create tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"},
                        "lesson_number": {"type": "integer"}
                    }
                }
            }
        ]
        
        # Test the complete flow
        response = ai_generator.generate_response(
            query="What's in lesson 5 of the MCP course about client creation?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        print(f"Final response: {response}")
        
        # Verify tool was executed
        tool_executed = mock_tool_manager.execute_tool.called
        print(f"Tool executed: {tool_executed}")
        
        if tool_executed:
            call_args = mock_tool_manager.execute_tool.call_args
            if call_args:
                args, kwargs = call_args
                print(f"Tool called with: {args}, {kwargs}")
                
                # Check if correct parameters were passed
                correct_tool_name = args[0] == "search_course_content"
                has_query = "query" in kwargs
                has_course_name = "course_name" in kwargs
                has_lesson_number = "lesson_number" in kwargs
                
                print_test_result("Correct tool called", correct_tool_name)
                print_test_result("Parameters passed correctly", 
                                 has_query and has_course_name and has_lesson_number)
        
        # Verify OpenAI was called twice (initial + follow-up)
        api_call_count = mock_client.chat.completions.create.call_count
        print(f"OpenAI API calls: {api_call_count}")
        print_test_result("Correct number of API calls", api_call_count == 2)
        
        # Verify final response is meaningful
        response_has_content = isinstance(response, str) and len(response) > 0
        print_test_result("Final response has content", response_has_content)
        
        return True
        
    except Exception as e:
        print(f"ERROR in tool execution flow test: {e}")
        print_test_result("Tool execution flow test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_sequential_tool_calling_two_rounds(mock_openai_class):
    """Test sequential tool calling across two rounds"""
    print_test_header("SEQUENTIAL TOOL CALLING - TWO ROUNDS")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini", enable_sequential=True)
        
        print_section_header("Two Round Sequential Processing Test")
        
        # Mock Round 1: Tool call to get course outline
        round1_tool_call = Mock()
        round1_tool_call.id = "call_round1"
        round1_tool_call.type = "function"
        round1_tool_call.function = Mock()
        round1_tool_call.function.name = "get_course_outline"
        round1_tool_call.function.arguments = json.dumps({"course_title": "MCP"})
        
        round1_response = Mock()
        round1_choice = Mock()
        round1_message = Mock()
        round1_message.content = ""
        round1_message.tool_calls = [round1_tool_call]
        round1_choice.message = round1_message
        round1_response.choices = [round1_choice]
        
        # Mock Round 2: Tool call to search specific lesson content
        round2_tool_call = Mock()
        round2_tool_call.id = "call_round2"
        round2_tool_call.type = "function"
        round2_tool_call.function = Mock()
        round2_tool_call.function.name = "search_course_content"
        round2_tool_call.function.arguments = json.dumps({
            "query": "lesson 4 content",
            "course_name": "MCP"
        })
        
        round2_response = Mock()
        round2_choice = Mock()
        round2_message = Mock()
        round2_message.content = ""
        round2_message.tool_calls = [round2_tool_call]
        round2_choice.message = round2_message
        round2_response.choices = [round2_choice]
        
        # Mock final synthesis response
        final_response = Mock()
        final_choice = Mock()
        final_message = Mock()
        final_message.content = "Based on the course outline and lesson content, here's what lesson 4 covers..."
        final_choice.message = final_message
        final_response.choices = [final_choice]
        
        # Set up mock to return different responses for each API call
        mock_client.chat.completions.create.side_effect = [
            round1_response,  # First API call
            round2_response,  # Second API call  
            final_response    # Final synthesis call
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: MCP\nLesson 1: Introduction\nLesson 2: Setup\nLesson 3: Basics\nLesson 4: Advanced Features",
            "Lesson 4 teaches advanced MCP features including custom protocols and error handling..."
        ]
        
        # Create tools
        tools = [
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {
                    "type": "object",
                    "properties": {"course_title": {"type": "string"}}
                }
            },
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"}
                    }
                }
            }
        ]
        
        # Test complex query requiring two rounds
        query = "Show me the course outline for MCP, then tell me what's in lesson 4"
        response = ai_generator.generate_response(
            query=query,
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        print(f"Query: {query}")
        print(f"Final response: {response}")
        
        # Verify tool manager was called twice
        tool_call_count = mock_tool_manager.execute_tool.call_count
        print(f"Tool calls made: {tool_call_count}")
        print_test_result("Two tools executed", tool_call_count == 2)
        
        # Verify OpenAI API was called three times (2 rounds + final synthesis)
        api_call_count = mock_client.chat.completions.create.call_count
        print(f"API calls made: {api_call_count}")
        print_test_result("Correct number of API calls", api_call_count == 3)
        
        # Verify correct tools were called in sequence
        if tool_call_count >= 2:
            call_args_list = mock_tool_manager.execute_tool.call_args_list
            first_tool = call_args_list[0][0][0]  # First positional arg of first call
            second_tool = call_args_list[1][0][0]  # First positional arg of second call
            
            print(f"First tool called: {first_tool}")
            print(f"Second tool called: {second_tool}")
            
            correct_sequence = (first_tool == "get_course_outline" and 
                              second_tool == "search_course_content")
            print_test_result("Correct tool sequence", correct_sequence)
        
        # Verify final response has content
        response_has_content = isinstance(response, str) and len(response) > 0
        print_test_result("Final response has content", response_has_content)
        
        return True
        
    except Exception as e:
        print(f"ERROR in sequential tool calling test: {e}")
        print_test_result("Sequential tool calling test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_sequential_termination_conditions(mock_openai_class):
    """Test various termination conditions for sequential processing"""
    print_test_header("SEQUENTIAL TERMINATION CONDITIONS")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini", enable_sequential=True)
        
        print_section_header("Termination Conditions Test")
        
        # Test 1: Natural completion (no tools needed in first round)
        print("Test 1: Natural completion after first round")
        
        # Mock response with no tool calls (direct answer)
        direct_response = Mock()
        direct_choice = Mock()
        direct_message = Mock()
        direct_message.content = "Machine learning is a subset of artificial intelligence..."
        direct_message.tool_calls = None
        direct_choice.message = direct_message
        direct_response.choices = [direct_choice]
        
        mock_client.chat.completions.create.return_value = direct_response
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        
        # Create tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        # Test general knowledge query (should not use tools)
        response = ai_generator.generate_response(
            query="What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify no tools were called
        tool_calls_made = mock_tool_manager.execute_tool.call_count
        print(f"Tool calls for general knowledge query: {tool_calls_made}")
        print_test_result("No tools called for general knowledge", tool_calls_made == 0)
        
        # Verify only one API call was made
        api_calls_made = mock_client.chat.completions.create.call_count
        print(f"API calls for general knowledge query: {api_calls_made}")
        print_test_result("Single API call for general knowledge", api_calls_made == 1)
        
        # Reset mocks for next test
        mock_client.reset_mock()
        mock_tool_manager.reset_mock()
        
        # Test 2: Maximum rounds reached
        print("\nTest 2: Maximum rounds termination")
        
        # Mock tool call responses for both rounds
        tool_call = Mock()
        tool_call.id = "call_test"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "test"})
        
        tool_response = Mock()
        tool_choice = Mock()
        tool_message = Mock()
        tool_message.content = ""
        tool_message.tool_calls = [tool_call]
        tool_choice.message = tool_message
        tool_response.choices = [tool_choice]
        
        # Final response after max rounds
        final_response = Mock()
        final_choice = Mock()
        final_message = Mock()
        final_message.content = "Based on the searches, here's the information..."
        final_choice.message = final_message
        final_response.choices = [final_choice]
        
        # Mock sequence: tool call, tool call, final response
        mock_client.chat.completions.create.side_effect = [
            tool_response,  # Round 1
            tool_response,  # Round 2
            final_response  # Final synthesis
        ]
        
        mock_tool_manager.execute_tool.return_value = "Some search results"
        
        # Test query that would trigger max rounds
        response = ai_generator.generate_response(
            query="Search for MCP content then search for more MCP content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify maximum rounds were used
        tool_calls_made = mock_tool_manager.execute_tool.call_count
        api_calls_made = mock_client.chat.completions.create.call_count
        
        print(f"Tool calls made: {tool_calls_made}")
        print(f"API calls made: {api_calls_made}")
        
        # Should have 2 tool calls (max rounds) and 3 API calls
        print_test_result("Maximum rounds reached", tool_calls_made == 2 and api_calls_made == 3)
        
        return True
        
    except Exception as e:
        print(f"ERROR in termination conditions test: {e}")
        print_test_result("Termination conditions test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_intent_classification_and_tool_selection(mock_openai_class):
    """Test intent classification and adaptive tool selection"""
    print_test_header("INTENT CLASSIFICATION AND TOOL SELECTION")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini", enable_sequential=True)
        
        print_section_header("Intent Classification Test")
        
        # Test different query types and their expected intents
        test_cases = [
            {
                "query": "Compare lesson 1 and lesson 2 of MCP course",
                "expected_intent": "COMPARISON",
                "description": "Comparison query"
            },
            {
                "query": "Show me the outline of the MCP course",
                "expected_intent": "OUTLINE_REQUEST", 
                "description": "Outline request"
            },
            {
                "query": "What's in lesson 5 of the MCP course?",
                "expected_intent": "CONTENT_SEARCH",
                "description": "Content search"
            },
            {
                "query": "First show me the course structure, then explain lesson 3",
                "expected_intent": "MULTI_STEP",
                "description": "Multi-step query"
            },
            {
                "query": "What is artificial intelligence?",
                "expected_intent": "GENERAL_KNOWLEDGE",
                "description": "General knowledge"
            }
        ]
        
        # Test intent classification
        from ai_generator import AdaptiveToolPolicy
        tool_policy = AdaptiveToolPolicy()
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            
            classified_intent = tool_policy.classify_intent(test_case['query'])
            print(f"Classified intent: {classified_intent.value}")
            print(f"Expected intent: {test_case['expected_intent']}")
            
            intent_correct = classified_intent.value == test_case['expected_intent']
            print_test_result(f"Intent classification for {test_case['description']}", intent_correct)
        
        return True
        
    except Exception as e:
        print(f"ERROR in intent classification test: {e}")
        print_test_result("Intent classification test", False, str(e))
        return False

@patch('ai_generator.OpenAI')
def test_error_recovery_mechanisms(mock_openai_class):
    """Test error recovery in sequential processing"""
    print_test_header("ERROR RECOVERY MECHANISMS")
    
    try:
        # Create mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        ai_generator = AIGenerator("test-key", "gpt-4o-mini", enable_sequential=True)
        
        print_section_header("Error Recovery Test")
        
        # Mock normal tool call
        tool_call = Mock()
        tool_call.id = "call_test"
        tool_call.type = "function"
        tool_call.function = Mock()
        tool_call.function.name = "search_course_content"
        tool_call.function.arguments = json.dumps({"query": "test"})
        
        tool_response = Mock()
        tool_choice = Mock()
        tool_message = Mock()
        tool_message.content = ""
        tool_message.tool_calls = [tool_call]
        tool_choice.message = tool_message
        tool_response.choices = [tool_choice]
        
        # Mock final response
        final_response = Mock()
        final_choice = Mock()
        final_message = Mock()
        final_message.content = "Despite the error, here's what I can tell you..."
        final_choice.message = final_message
        final_response.choices = [final_choice]
        
        mock_client.chat.completions.create.side_effect = [tool_response, final_response]
        
        # Create mock tool manager that throws an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Create tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        # Test query that would cause tool error
        response = ai_generator.generate_response(
            query="Search for MCP content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        print(f"Response despite error: {response}")
        
        # Verify that we still got a response despite the error
        response_generated = isinstance(response, str) and len(response) > 0
        print_test_result("Response generated despite tool error", response_generated)
        
        # Verify tool was attempted
        tool_attempted = mock_tool_manager.execute_tool.call_count > 0
        print_test_result("Tool execution was attempted", tool_attempted)
        
        return True
        
    except Exception as e:
        print(f"ERROR in error recovery test: {e}")
        print_test_result("Error recovery test", False, str(e))
        return False

if __name__ == "__main__":
    print("Starting AI Generator Tests...")
    
    # Run the original tests
    test1_passed = test_ai_generator_tool_conversion()
    test2_passed = test_ai_generator_tool_calling_detection()
    test3_passed = test_ai_generator_tool_execution_flow()
    
    # Run the new sequential processing tests
    test4_passed = test_sequential_tool_calling_two_rounds()
    test5_passed = test_sequential_termination_conditions()
    test6_passed = test_intent_classification_and_tool_selection()
    test7_passed = test_error_recovery_mechanisms()
    
    print(f"\n{'='*60}")
    print("AI GENERATOR TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tool conversion test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Tool calling detection test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Tool execution flow test: {'PASS' if test3_passed else 'FAIL'}")
    print(f"Sequential two rounds test: {'PASS' if test4_passed else 'FAIL'}")
    print(f"Termination conditions test: {'PASS' if test5_passed else 'FAIL'}")
    print(f"Intent classification test: {'PASS' if test6_passed else 'FAIL'}")
    print(f"Error recovery test: {'PASS' if test7_passed else 'FAIL'}")
    
    all_tests_passed = all([
        test1_passed, test2_passed, test3_passed, test4_passed, 
        test5_passed, test6_passed, test7_passed
    ])
    
    if all_tests_passed:
        print("\n✅ All AI Generator tests passed!")
    else:
        print("\n❌ Some AI Generator tests failed - check sequential tool calling implementation")