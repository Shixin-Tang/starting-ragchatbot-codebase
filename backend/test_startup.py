#!/usr/bin/env python3
"""
Test the startup event to ensure it loads data correctly
"""
import sys
import os
import asyncio

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from config import config

async def test_startup():
    """Test the startup event functionality"""
    print("🚀 TESTING STARTUP EVENT LOGIC")
    print("=" * 40)
    
    # Initialize RAG system
    rag_system = RAGSystem(config)
    
    # Clear existing data first
    print("1. Clearing existing data...")
    rag_system.vector_store.clear_all_data()
    
    analytics_before = rag_system.get_course_analytics()
    print(f"   Courses before startup: {analytics_before['total_courses']}")
    
    # Simulate the startup event logic
    print("2. Running startup logic...")
    
    # First try the relative path from project root
    docs_path = "../docs"
    if not os.path.exists(docs_path):
        # Fallback to absolute path construction
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs_path = os.path.join(current_dir, "docs")
    
    if os.path.exists(docs_path):
        print(f"   📁 Loading documents from: {docs_path}")
        try:
            # Use clear_existing=True to ensure fresh data load
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)
            print(f"   ✅ Loaded {courses} courses with {chunks} chunks")
            
            # Verify the load was successful
            analytics = rag_system.get_course_analytics()
            print(f"   📊 Final analytics: {analytics['total_courses']} courses loaded")
            
        except Exception as e:
            print(f"   ❌ Error loading documents: {e}")
            return False
    else:
        print(f"   ❌ Docs directory not found at: {docs_path}")
        return False
    
    # Check the results
    analytics_after = rag_system.get_course_analytics()
    
    if analytics_after['total_courses'] > 0:
        print("   🎯 Startup logic successfully loaded course data")
        print(f"   📚 Loaded courses: {analytics_after['course_titles']}")
        
        # Test a quick search to make sure it works
        print("\n3. Testing search functionality...")
        from search_tools import CourseSearchTool
        search_tool = CourseSearchTool(rag_system.vector_store)
        
        result = search_tool.execute(
            query="MCP client",
            course_name="MCP",
            lesson_number=5
        )
        
        if len(result) > 100 and "No relevant content found" not in result:
            print("   ✅ Search functionality working correctly")
            print(f"   📄 Search result length: {len(result)} characters")
        else:
            print("   ❌ Search functionality not working")
            print(f"   📄 Search result: {result}")
    else:
        print("   ❌ Startup logic failed to load course data")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_startup())
    if success:
        print("\n🎉 All startup tests passed!")
    else:
        print("\n💥 Startup tests failed!")
        sys.exit(1)