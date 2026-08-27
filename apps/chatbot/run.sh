#!/bin/bash

# PuppyGraph RAG Chatbot Launcher Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🐶 PuppyGraph RAG Chatbot Demo"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "🔧 Please edit .env with your configuration before running!"
    echo "   Especially set your ANTHROPIC_API_KEY"
fi

# Run integration tests
echo "🧪 Running integration tests..."
if python test_integration.py; then
    echo "✅ Integration tests passed!"
    echo ""
    echo "🚀 Starting PuppyGraph RAG Chatbot..."
    echo "   Access the UI at: http://localhost:7860"
    echo "   Press Ctrl+C to stop"
    echo ""
    
    # Start the application
    python gradio_app.py
else
    echo "❌ Integration tests failed. Check .env, PuppyGraph connectivity, and the logs above."
    exit 1
fi
