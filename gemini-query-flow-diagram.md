sequenceDiagram
participant User
participant Frontend (script.js)
participant Backend API (app.py)
participant RAG System
participant AI Generator
participant Tool Manager
participant Vector Store
participant Claude API

    User->>+Frontend (script.js): 1. Enters query and sends
    Frontend (script.js)->>User: Displays user message & loading indicator
    Frontend (script.js)->>+Backend API (app.py): 2. POST /api/query with query & session_id

    Backend API (app.py)->>+RAG System: 3. Calls rag_system.query()
    RAG System->>+AI Generator: 4. Calls generate_response() with prompt, history, and tools
    AI Generator->>+Claude API: 5. Sends 1st request (with tool definitions)
    Claude API-->>-AI Generator: 6. Responds with request to use search tool

    AI Generator->>+Tool Manager: 7. Calls execute_tool("search_course_content")
    Tool Manager->>+Vector Store: 8. Calls search()
    Vector Store-->>-Tool Manager: 9. Returns relevant document chunks
    Tool Manager-->>-AI Generator: 10. Returns formatted search results

    AI Generator->>+Claude API: 11. Sends 2nd request (with search results as context)
    Claude API-->>-AI Generator: 12. Generates final text answer
    AI Generator-->>-RAG System: 13. Returns final answer

    RAG System->>Tool Manager: 14. Gets sources used in the search
    RAG System->>RAG System: 15. Updates conversation history
    RAG System-->>-Backend API (app.py): 16. Returns final answer and sources

    Backend API (app.py)-->>-Frontend (script.js): 17. Sends JSON response
    Frontend (script.js)->>-User: 18. Displays final answer and sources
