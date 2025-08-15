import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os

from config import config
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Union[str, Dict[str, Any]]]  # Support both old string format and new object format
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    # First try the relative path from project root
    docs_path = "../docs"
    if not os.path.exists(docs_path):
        # Fallback to absolute path construction
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        docs_path = os.path.join(current_dir, "docs")
    
    if os.path.exists(docs_path):
        print(f"Loading initial documents from: {docs_path}")
        try:
            # Use clear_existing=True to ensure fresh data load
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)
            print(f"✅ Loaded {courses} courses with {chunks} chunks")
            
            # Verify the load was successful
            analytics = rag_system.get_course_analytics()
            print(f"📊 Final analytics: {analytics['total_courses']} courses loaded")
            
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Docs directory not found at: {docs_path}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Files in current directory: {os.listdir('.')}")
        # Try to find docs directory
        for root, dirs, files in os.walk(".."):
            if "docs" in dirs:
                found_docs = os.path.join(root, "docs")
                print(f"🔍 Found docs directory at: {found_docs}")
                break

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")