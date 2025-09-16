#!/bin/bash

# PuppyGraph RAG Chatbot Launcher Script

set -e

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
python test_integration.py

if [ $? -eq 0 ]; then
    echo "✅ Integration tests passed!"
    echo ""
    echo "🚀 Starting PuppyGraph RAG Chatbot..."
    echo "   Access the UI at: http://localhost:7860"
    echo "   Press Ctrl+C to stop"
    echo ""
    
    # Start the application
    python gradio_app.py
else
    echo "❌ Integration tests failed. Please check the configuration."
    exit 1
fi