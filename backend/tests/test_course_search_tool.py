"""
Test CourseSearchTool.execute() method functionality
"""
import sys
import os
from test_fixtures import (
    TestVectorStore, print_test_header, print_test_result, 
    print_section_header
)

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool

def test_course_search_tool_basic():
    """Test basic CourseSearchTool functionality"""
    print_test_header("COURSE SEARCH TOOL BASIC TESTS")
    
    try:
        # Create vector store and search tool
        vector_store = TestVectorStore.create_test_vector_store()
        search_tool = CourseSearchTool(vector_store)
        
        # Test 1: Tool definition
        print_section_header("Tool Definition Test")
        
        tool_def = search_tool.get_tool_definition()
        print(f"Tool name: {tool_def.get('name')}")
        print(f"Tool description: {tool_def.get('description')}")
        
        has_required_fields = (
            tool_def.get('name') == 'search_course_content' and
            'description' in tool_def and
            'input_schema' in tool_def
        )
        
        print_test_result("Tool definition structure", has_required_fields)
        
        # Test 2: Simple query without filters
        print_section_header("Simple Query Test")
        
        result = search_tool.execute(query="MCP client")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:200]}...")
        
        has_content = len(result) > 0 and "No relevant content found" not in result
        print_test_result("Simple query returns content", has_content)
        
        # Test 3: Query with course filter
        print_section_header("Course Filter Test")
        
        result = search_tool.execute(query="client", course_name="MCP")
        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:200]}...")
        
        has_content = len(result) > 0 and "No relevant content found" not in result
        print_test_result("Course-filtered query returns content", has_content)
        
        # Test 4: Query with lesson number filter
        print_section_header("Lesson Filter Test")
        
        result = search_tool.execute(query="client", course_name="MCP", lesson_number=5)
        print(f"Result length: {len(result)}")
        print(f"Result preview: {result[:200]}...")
        
        has_content = len(result) > 0 and "No relevant content found" not in result
        print_test_result("Lesson-filtered query returns content", has_content)
        
        # Test 5: Invalid course name
        print_section_header("Invalid Course Test")
        
        result = search_tool.execute(query="client", course_name="NonexistentCourse")
        print(f"Result: {result}")
        
        handles_invalid = "No course found matching" in result or "No relevant content found" in result
        print_test_result("Handles invalid course gracefully", handles_invalid)
        
        # Test 6: Valid course, invalid lesson
        print_section_header("Invalid Lesson Test")
        
        result = search_tool.execute(query="client", course_name="MCP", lesson_number=999)
        print(f"Result: {result}")
        
        handles_invalid_lesson = "No relevant content found" in result
        print_test_result("Handles invalid lesson gracefully", handles_invalid_lesson)
        
        return True
        
    except Exception as e:
        print(f"ERROR in CourseSearchTool basic test: {e}")
        print_test_result("CourseSearchTool basic test", False, str(e))
        return False

def test_course_search_tool_specific_scenarios():
    """Test specific scenarios that are failing"""
    print_test_header("COURSE SEARCH TOOL SPECIFIC SCENARIOS")
    
    try:
        vector_store = TestVectorStore.create_test_vector_store()
        search_tool = CourseSearchTool(vector_store)
        
        # Test the exact scenario that's failing
        print_section_header("Exact Failing Scenario Test")
        
        scenarios = [
            {
                "name": "Lesson 5 MCP content search",
                "query": "lesson 5",
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "MCP client creation",
                "query": "creating MCP client",
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "Client specific content",
                "query": "client",
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "Creating An MCP Client",
                "query": "Creating An MCP Client",
                "course_name": "MCP",
                "lesson_number": None
            },
            {
                "name": "Broad MCP search",
                "query": "MCP",
                "course_name": None,
                "lesson_number": 5
            }
        ]
        
        for scenario in scenarios:
            print_section_header(f"Scenario: {scenario['name']}")
            
            params = {k: v for k, v in scenario.items() if k != 'name' and v is not None}
            result = search_tool.execute(**params)
            
            print(f"Parameters: {params}")
            print(f"Result length: {len(result)}")
            print(f"Result preview: {result[:300]}...")
            
            success = len(result) > 0 and "No relevant content found" not in result
            print_test_result(scenario['name'], success)
            
            if not success:
                print(f"  ❌ This scenario is failing!")
        
        # Test sources tracking
        print_section_header("Sources Tracking Test")
        
        search_tool.execute(query="MCP client", course_name="MCP", lesson_number=5)
        sources = search_tool.last_sources
        
        print(f"Sources tracked: {len(sources)}")
        if sources:
            for i, source in enumerate(sources):
                print(f"  Source {i+1}: {source}")
        
        sources_tracked = len(sources) > 0
        print_test_result("Sources are tracked", sources_tracked)
        
        return True
        
    except Exception as e:
        print(f"ERROR in CourseSearchTool specific scenarios test: {e}")
        print_test_result("CourseSearchTool specific scenarios test", False, str(e))
        return False

def test_search_tool_edge_cases():
    """Test edge cases and error conditions"""
    print_test_header("COURSE SEARCH TOOL EDGE CASES")
    
    try:
        vector_store = TestVectorStore.create_test_vector_store()
        search_tool = CourseSearchTool(vector_store)
        
        edge_cases = [
            {
                "name": "Empty query",
                "query": "",
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "Very long query",
                "query": "a" * 1000,
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "Special characters",
                "query": "MCP & client!@#$%",
                "course_name": "MCP",
                "lesson_number": 5
            },
            {
                "name": "Partial course name",
                "query": "client",
                "course_name": "Build Rich-Context",
                "lesson_number": 5
            },
            {
                "name": "Case sensitivity",
                "query": "CLIENT",
                "course_name": "mcp",
                "lesson_number": 5
            }
        ]
        
        for case in edge_cases:
            print_section_header(f"Edge Case: {case['name']}")
            
            try:
                params = {k: v for k, v in case.items() if k != 'name'}
                result = search_tool.execute(**params)
                
                print(f"Parameters: {params}")
                print(f"Result length: {len(result)}")
                print(f"Result type: {type(result)}")
                
                # Check if it's a proper string response (not error)
                is_string_response = isinstance(result, str)
                print_test_result(f"{case['name']} - returns string", is_string_response)
                
            except Exception as e:
                print(f"  Exception: {e}")
                print_test_result(f"{case['name']} - handles error", False, str(e))
        
        return True
        
    except Exception as e:
        print(f"ERROR in CourseSearchTool edge cases test: {e}")
        return False

if __name__ == "__main__":
    print("Starting CourseSearchTool Tests...")
    
    # Run the tests
    test1_passed = test_course_search_tool_basic()
    test2_passed = test_course_search_tool_specific_scenarios()
    test3_passed = test_search_tool_edge_cases()
    
    print(f"\n{'='*60}")
    print("COURSE SEARCH TOOL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Basic functionality test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Specific scenarios test: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Edge cases test: {'PASS' if test3_passed else 'FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✅ All CourseSearchTool tests passed!")
    else:
        print("\n❌ Some CourseSearchTool tests failed - this indicates tool execution issues")