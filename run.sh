#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Check for .env file and OPENAI_API_KEY
if [ ! -f ".env" ] || ! grep -q "OPENAI_API_KEY=" ".env"; then
    echo "Warning: .env file not found or OPENAI_API_KEY is not set."
    echo "Please create a .env file with your key. See .env.example for reference."
    # Optionally, exit if the key is mandatory for startup
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your OPENAI_API_KEY in .env"

# Change to backend directory and start the server
cd backend

# Try uv first, fallback to direct python if uv not available
if command -v uv >/dev/null 2>&1; then
    uv run uvicorn app:app --reload --port 8000
else
    echo "uv not found, using activated virtual environment..."
    python -m uvicorn app:app --reload --port 8000
fi