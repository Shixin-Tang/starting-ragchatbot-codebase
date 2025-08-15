"""
Test vector store data and search functionality
"""
import sys
import os
import json
from test_fixtures import (
    TestVectorStore, print_test_header, print_test_result, 
    print_section_header, SAMPLE_COURSE_TITLES
)

def test_vector_store_data_inspection():
    """Comprehensive test to inspect what's actually stored in the vector store"""
    print_test_header("VECTOR STORE DATA INSPECTION")
    
    try:
        vector_store = TestVectorStore.create_test_vector_store()
        
        # Test 1: Check if collections exist and have data
        print_section_header("Collection Data Overview")
        
        # Get course count and titles
        course_count = vector_store.get_course_count()
        course_titles = vector_store.get_existing_course_titles()
        
        print(f"Total courses in vector store: {course_count}")
        print(f"Course titles: {course_titles}")
        
        passed = course_count > 0 and len(course_titles) > 0
        print_test_result("Collection has data", passed, 
                         f"Found {course_count} courses")
        
        # Test 2: Check for MCP course specifically
        print_section_header("MCP Course Verification")
        
        mcp_course_found = False
        mcp_course_title = None
        
        for title in course_titles:
            if "MCP" in title.upper():
                mcp_course_found = True
                mcp_course_title = title
                break
        
        print(f"MCP course found: {mcp_course_found}")
        if mcp_course_title:
            print(f"MCP course title: {mcp_course_title}")
        
        print_test_result("MCP course exists", mcp_course_found, 
                         f"Title: {mcp_course_title}")
        
        # Test 3: Verify course name resolution
        print_section_header("Course Name Resolution Test")
        
        resolved_title = vector_store._resolve_course_name("MCP")
        print(f"'MCP' resolves to: {resolved_title}")
        
        resolved_title_full = vector_store._resolve_course_name("MCP: Build Rich-Context AI Apps")
        print(f"'MCP: Build Rich-Context AI Apps' resolves to: {resolved_title_full}")
        
        resolution_works = resolved_title is not None
        print_test_result("Course name resolution works", resolution_works,
                         f"'MCP' -> '{resolved_title}'")
        
        # Test 4: Get detailed course metadata
        if mcp_course_title:
            print_section_header("MCP Course Metadata Analysis")
            
            try:
                metadata_results = vector_store.course_catalog.get(ids=[mcp_course_title])
                if metadata_results and metadata_results['metadatas']:
                    metadata = metadata_results['metadatas'][0]
                    lessons_json = metadata.get('lessons_json', '[]')
                    lessons = json.loads(lessons_json)
                    
                    print(f"Course link: {metadata.get('course_link', 'N/A')}")
                    print(f"Instructor: {metadata.get('instructor', 'N/A')}")
                    print(f"Total lessons: {len(lessons)}")
                    
                    # Check for lesson 5 specifically
                    lesson_5_found = False
                    for lesson in lessons:
                        if lesson.get('lesson_number') == 5:
                            lesson_5_found = True
                            print(f"Lesson 5 title: {lesson.get('lesson_title', 'N/A')}")
                            break
                    
                    print_test_result("Lesson 5 metadata exists", lesson_5_found)
                    
                    # Print all lessons for verification
                    print("\nAll lessons in MCP course:")
                    for lesson in lessons:
                        print(f"  Lesson {lesson.get('lesson_number', '?')}: {lesson.get('lesson_title', 'No title')}")
                
            except Exception as e:
                print(f"Error getting metadata: {e}")
                print_test_result("Metadata retrieval", False, str(e))
        
        # Test 5: Check content chunks for lesson 5
        print_section_header("Content Chunks Analysis for Lesson 5")
        
        if mcp_course_title:
            # Search for content chunks with lesson_number = 5
            try:
                # Use ChromaDB directly to check what's stored
                content_results = vector_store.course_content.get(
                    where={"$and": [
                        {"course_title": mcp_course_title},
                        {"lesson_number": 5}
                    ]}
                )
                
                chunk_count = len(content_results['documents']) if content_results['documents'] else 0
                print(f"Content chunks for lesson 5: {chunk_count}")
                
                if chunk_count > 0:
                    print("Sample chunk content:")
                    for i, doc in enumerate(content_results['documents'][:2]):  # Show first 2 chunks
                        print(f"  Chunk {i+1}: {doc[:100]}...")
                
                print_test_result("Lesson 5 content chunks exist", chunk_count > 0,
                                 f"{chunk_count} chunks found")
                
            except Exception as e:
                print(f"Error checking content chunks: {e}")
                print_test_result("Content chunk check", False, str(e))
        
        # Test 6: Test direct search for lesson 5 content
        print_section_header("Direct Search Test for Lesson 5")
        
        if mcp_course_title:
            search_results = vector_store.search(
                query="creating MCP client",
                course_name="MCP",
                lesson_number=5
            )
            
            print(f"Search results found: {len(search_results.documents)}")
            print(f"Search error: {search_results.error}")
            
            if search_results.documents:
                print("Sample search result:")
                print(f"  {search_results.documents[0][:200]}...")
            
            search_successful = len(search_results.documents) > 0 and not search_results.error
            print_test_result("Direct search for lesson 5 content", search_successful,
                             f"{len(search_results.documents)} results")
        
        return True
        
    except Exception as e:
        print(f"ERROR in vector store test: {e}")
        print_test_result("Vector store test", False, str(e))
        return False

def test_search_variations():
    """Test various search parameter combinations"""
    print_test_header("SEARCH VARIATIONS TEST")
    
    try:
        vector_store = TestVectorStore.create_test_vector_store()
        
        test_cases = [
            {"query": "MCP client", "course_name": None, "lesson_number": None},
            {"query": "creating client", "course_name": "MCP", "lesson_number": None},
            {"query": "client", "course_name": "MCP", "lesson_number": 5},
            {"query": "MCP client creation", "course_name": "MCP: Build Rich-Context AI Apps", "lesson_number": 5},
            {"query": "lesson 5", "course_name": "MCP", "lesson_number": None},
        ]
        
        for i, test_case in enumerate(test_cases):
            print_section_header(f"Test Case {i+1}")
            print(f"Query: '{test_case['query']}'")
            print(f"Course: {test_case['course_name']}")
            print(f"Lesson: {test_case['lesson_number']}")
            
            results = vector_store.search(**test_case)
            
            print(f"Results found: {len(results.documents)}")
            print(f"Error: {results.error}")
            
            if results.documents:
                print(f"First result preview: {results.documents[0][:100]}...")
            
            success = len(results.documents) > 0 and not results.error
            print_test_result(f"Search case {i+1}", success)
        
        return True
        
    except Exception as e:
        print(f"ERROR in search variations test: {e}")
        return False

if __name__ == "__main__":
    print("Starting Vector Store Tests...")
    
    # Run the tests
    test1_passed = test_vector_store_data_inspection()
    test2_passed = test_search_variations()
    
    print(f"\n{'='*60}")
    print("VECTOR STORE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Data inspection test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Search variations test: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All vector store tests passed!")
    else:
        print("\n❌ Some vector store tests failed - this indicates data or search issues")