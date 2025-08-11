# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup:**
```bash
uv sync                           # Install dependencies
echo "ANTHROPIC_API_KEY=key" > .env  # Configure API key
```

**Running:**
```bash
./run.sh                         # Quick start (recommended)
cd backend && uv run uvicorn app:app --reload --port 8000  # Manual start
```

**Dependency Management:**
```bash
uv add package_name               # Add new dependency
uv remove package_name            # Remove dependency
uv sync                           # Sync dependencies after changes
```

**Important:** Always use `uv` for all dependency management. Do not use `pip` directly in this project.

Access at: http://localhost:8000

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system for querying course materials using a **tool-calling architecture** where Claude AI decides when to search.

**Core Flow:**
```
User Query → FastAPI → RAG System → AI Generator (Claude) → CourseSearchTool → Vector Store → ChromaDB
```

**Key Design Pattern:** AI uses function calling to decide when course content search is needed, rather than always searching. This enables both general knowledge and specific course content responses.

## Critical Components

**RAG System** (`backend/rag_system.py`) - Main orchestrator that coordinates all components

**AI Generator** (`backend/ai_generator.py`) - Anthropic Claude integration with tool calling:
- Model: `claude-sonnet-4-20250514`
- Uses system prompt to guide tool usage behavior
- Handles tool execution and follow-up responses

**Vector Store** (`backend/vector_store.py`) - ChromaDB wrapper with **dual collections**:
- `course_catalog`: Course metadata for semantic course name resolution
- `course_content`: Actual content chunks for retrieval
- Smart course name matching (e.g., "MCP" finds "MCP: Build Rich-Context AI Apps")

**Document Processor** (`backend/document_processor.py`) - Parses structured course documents:
- Expected format: Course metadata (title/link/instructor) + lesson markers
- Sentence-based chunking (800 chars, 100 overlap)
- Adds context prefixes: "Course X Lesson Y content: [chunk]"

**Search Tools** (`backend/search_tools.py`) - Tool calling framework:
- `CourseSearchTool` implements function calling interface
- Tracks sources for frontend display
- Supports course name and lesson number filtering

## Configuration

**Main Config** (`backend/config.py`):
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100
- `CHROMA_PATH`: "./chroma_db"

**Environment**: Requires `.env` with `ANTHROPIC_API_KEY`

## Document Processing

**Expected Document Structure:**
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content...]
```

Documents in `/docs/` are auto-loaded on startup. Processing creates searchable chunks with course/lesson context preservation.

## Key Patterns

**Tool-Based Search:** AI autonomously decides when to use `search_course_content` tool vs. general knowledge

**Session Management:** Automatic session creation with configurable conversation history (default: 2 exchanges)

**Source Tracking:** Search results include course/lesson references displayed in frontend

**Dual Collection Strategy:** Separate vector collections for course metadata vs. content enables both name resolution and semantic search

**Smart Chunking:** Sentence-based chunking preserves context while maintaining searchable units

## API Endpoints

- `POST /api/query` - Main chat interface with session management
- `GET /api/courses` - Course statistics and titles
- Frontend served from root path

This architecture is specifically optimized for educational content with structured lessons and requires course documents to follow the expected format for proper processing.