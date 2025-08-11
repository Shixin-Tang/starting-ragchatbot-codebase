#!/bin/bash

# Create necessary directories
mkdir -p docs 

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

# Check for .env file and ANTHROPIC_API_KEY
if [ ! -f ".env" ] || ! grep -q "ANTHROPIC_API_KEY=" ".env"; then
    echo "Warning: .env file not found or ANTHROPIC_API_KEY is not set."
    echo "Please create a .env file with your key. See .env.example for reference."
    # Optionally, exit if the key is mandatory for startup
fi

echo "Starting Course Materials RAG System..."
echo "Make sure you have set your ANTHROPIC_API_KEY in .env"

# Change to backend directory and start the server
cd backend && uv run uvicorn app:app --reload --port 8000