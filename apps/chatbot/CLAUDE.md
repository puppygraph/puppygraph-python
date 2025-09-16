# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Quick Start
```bash
# Run the complete application with setup
./run.sh

# Or run individual components:
python gradio_app.py          # Start web UI at http://localhost:7860
python mcp_server.py          # Run MCP server standalone
python test_integration.py    # Run integration tests
```

### Setup Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Environment setup
cp .env.example .env  # Edit with your API keys
```

### Testing
```bash
# Integration tests (checks all components)
python test_integration.py

# Test individual components
python -c "from backend import get_chatbot; chatbot = get_chatbot(); print(chatbot.get_graph_stats())"
python -c "from rag_system import TextToCypherRAG; rag = TextToCypherRAG(); print('RAG system OK')"
```

## Architecture Overview

### Core Components
This is a RAG-powered chatbot that converts natural language queries to Cypher queries for PuppyGraph:

1. **gradio_app.py** - Main web interface (Gradio UI)
2. **backend.py** - Central coordinator (`PuppyGraphChatbot` class)
3. **rag_system.py** - Text-to-Cypher conversion using embeddings + Claude Sonnet 4.0
4. **mcp_server.py** - Model Context Protocol server for PuppyGraph operations

### Data Flow
```
User Question → Gradio UI → Backend → RAG System → Claude Sonnet 4.0 → Cypher Query → MCP Server → PuppyGraph → Results
```

### Key Classes
- `PuppyGraphChatbot` (backend.py) - Main orchestrator
- `TextToCypherRAG` (rag_system.py) - Handles NL→Cypher conversion using ChromaDB + embeddings
- `PuppyGraphMCPServer` (mcp_server.py) - MCP tools for schema, query execution, validation

## Configuration

### Required Environment Variables
- `ANTHROPIC_API_KEY` - Required for Claude Sonnet 4.0 integration
- `PUPPYGRAPH_BOLT_URI` - Default: `bolt://localhost:7687`
- `PUPPYGRAPH_HTTP_URI` - Default: `http://localhost:8081`
- `PUPPYGRAPH_USERNAME` - Default: `puppygraph`
- `PUPPYGRAPH_PASSWORD` - Default: `puppygraph123`

### Dependencies
- **Core**: gradio, anthropic, mcp, neo4j, requests
- **RAG**: sentence-transformers, chromadb, langchain
- **Server**: uvicorn, fastapi, python-dotenv

## Development Notes

### Multi-Round Query Execution
The system automatically breaks complex questions into multiple Cypher queries. Each round uses results from previous queries to generate more specific follow-ups.

### RAG System Details
- Uses `all-MiniLM-L6-v2` for embeddings by default
- ChromaDB stores question/Cypher examples
- Claude Sonnet 4.0 generates queries using retrieved examples as context
- Examples can be added via UI or programmatically

### MCP Integration
The MCP server provides these tools:
- `execute_cypher` - Run Cypher queries with result formatting
- `get_schema_info` - Retrieve graph schema with optional node samples
- `validate_cypher` - Validate query syntax before execution

### Error Handling
- Connection failures to PuppyGraph are handled gracefully
- RAG system falls back to basic query generation if examples are unavailable
- All components have comprehensive logging