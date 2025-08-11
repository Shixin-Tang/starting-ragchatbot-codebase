# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend (script.js)
    participant API as FastAPI (app.py)
    participant RAG as RAG System
    participant SM as Session Manager
    participant AI as AI Generator
    participant TM as Tool Manager
    participant ST as Search Tool
    participant VS as Vector Store
    participant DB as ChromaDB

    %% User initiates query
    U->>F: Types query & clicks send
    F->>F: Disable input, show loading
    F->>API: POST /api/query {query, session_id}

    %% API processes request
    API->>RAG: query(request.query, session_id)
    
    %% RAG orchestrates the flow
    RAG->>SM: get_conversation_history(session_id)
    SM-->>RAG: Previous chat history
    
    RAG->>TM: get_tool_definitions()
    TM-->>RAG: Available tool schemas
    
    RAG->>AI: generate_response(query, history, tools, tool_manager)
    
    %% AI decides whether to use tools
    AI->>AI: Call Claude API with tools
    
    Note over AI: Claude decides: "This needs course content search"
    
    AI->>TM: execute_tool("search_course_content", query="...", course_name="...")
    TM->>ST: execute(query, course_name, lesson_number)
    
    %% Search tool performs vector search
    ST->>VS: search(query, course_name, lesson_number)
    VS->>VS: _resolve_course_name() if needed
    VS->>VS: _build_filter() for ChromaDB
    VS->>DB: query(query_texts, n_results, where=filter)
    
    %% Results flow back
    DB-->>VS: Raw vector search results
    VS-->>ST: SearchResults object
    ST->>ST: _format_results() + track sources
    ST-->>TM: Formatted search results
    TM-->>AI: Tool execution results
    
    %% AI generates final response
    AI->>AI: Call Claude API again with tool results
    AI-->>RAG: Generated response text
    
    %% Complete the response chain
    RAG->>TM: get_last_sources()
    TM-->>RAG: Sources from search
    RAG->>TM: reset_sources()
    RAG->>SM: add_exchange(session_id, query, response)
    RAG-->>API: (response, sources)
    
    %% API returns to frontend
    API-->>F: {answer, sources, session_id}
    F->>F: Remove loading, display response
    F->>F: Show sources, enable input
    F-->>U: Display AI response with sources

    %% Visual styling
    rect rgb(240, 248, 255)
        Note over U, F: Frontend Layer
    end
    
    rect rgb(255, 248, 240)
        Note over API, RAG: API Layer
    end
    
    rect rgb(248, 255, 240)
        Note over SM, AI: Processing Layer
    end
    
    rect rgb(255, 240, 248)
        Note over TM, VS: Search Layer
    end
    
    rect rgb(240, 240, 255)
        Note over DB: Storage Layer
    end
```

## Key Components & Flow

### 1. **Frontend (script.js)**
- User input handling
- Loading states
- API communication
- Response display with sources

### 2. **API Layer (app.py)**  
- FastAPI endpoint `/api/query`
- Request validation
- RAG system orchestration

### 3. **Processing Layer**
- **RAG System**: Main orchestrator
- **Session Manager**: Conversation history
- **AI Generator**: Claude API integration

### 4. **Search Layer**
- **Tool Manager**: Tool registration & execution
- **Search Tool**: Course content search logic
- **Vector Store**: ChromaDB interface

### 5. **Storage Layer**
- **ChromaDB**: Vector embeddings & metadata

## Flow Highlights

1. **Tool-Based Architecture**: Claude decides when to search using function calling
2. **Smart Course Resolution**: Semantic matching for course names  
3. **Context Preservation**: Session history maintained throughout
4. **Source Tracking**: UI shows which courses/lessons were referenced
5. **Error Handling**: Graceful fallbacks at each layer