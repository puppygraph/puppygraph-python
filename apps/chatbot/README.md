# PuppyGraph RAG Chatbot Demo

A conversational AI interface for PuppyGraph that converts natural language questions into Cypher queries using Retrieval-Augmented Generation (RAG).

## Features

- 🤖 **Natural Language to Cypher**: Ask questions in plain English, get intelligent query execution
- 🔄 **Multi-Round Execution**: Automatically generates and executes multiple queries as needed
- ⚡ **Real-time Streaming**: Watch each query step execute live as it happens
- 🧠 **RAG-Powered**: Uses embeddings and similar examples to improve query generation
- 🧭 **Claude Sonnet 5**: Powered by Anthropic's current Sonnet model with intelligent stopping
- 📊 **Graph Exploration**: Built-in schema viewer and statistics
- 🎯 **Interactive UI**: Clean Gradio interface with real-time updates
- 📚 **Learning System**: Add your own examples to improve performance

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   Gradio UI     │◄──►│  Python Backend  │◄──►│  PuppyGraph     │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │                  │
                       │   RAG System     │
                       │ (ChromaDB +      │
                       │  Embeddings +    │
                       │  Claude Sonnet)  │
                       └──────────────────┘
```

The chatbot backend connects directly to PuppyGraph over Bolt and HTTP.

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your Anthropic API key and PuppyGraph settings
   ```

4. **Ensure PuppyGraph is running:**
   - Bolt protocol on port 7687
   - HTTP API on port 8081
   - Default credentials: puppygraph/puppygraph123

## Usage

### Quick Start

```bash
# From the repository root:
./apps/chatbot/run.sh
```

Then open http://localhost:7860 in your browser.

### Components

#### 1. RAG System (`rag_system.py`)
Handles text-to-Cypher conversion using:
- Sentence embeddings for question similarity
- ChromaDB for example storage
- Claude Sonnet 5 for query generation
- Confidence scoring

#### 2. Backend (`backend.py`)
Coordinates the application components:
- Integrates RAG system
- Handles conversation history
- Connects directly to PuppyGraph over Bolt and HTTP
- Provides unified API

#### 3. Gradio UI (`gradio_app.py`)
Interactive web interface with:
- Chat interface for questions
- Schema and statistics viewer
- Example management system
- Help documentation

## Example Queries

### Simple Queries (typically 1 round):
- "Show me all nodes in the graph"
- "Count the total number of relationships"
- "What types of nodes exist?"
- "Show me the graph schema"

### Complex Queries (typically 2-3 rounds):
- "Which users have the most connections and what do they connect to?"
- "What percentage of nodes have more than 5 relationships?"
- "Find nodes that are connected to both X and Y type nodes"
- "Show me the top 5 most connected entities and their relationship types"
- "How many different paths exist between node A and node B?"

## Adding Custom Examples

Use the "Add Examples" tab to teach the system new patterns:

1. **Question**: "Find users who bought expensive products"
2. **Cypher**: `MATCH (u:User)-[:BOUGHT]->(p:Product) WHERE p.price > 100 RETURN u, p`
3. **Description**: "Finds users who purchased products over $100"

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Required for Anthropic Claude integration
- `ANTHROPIC_MODEL`: Anthropic model ID (default: claude-sonnet-5)
- `PUPPYGRAPH_BOLT_URI`: PuppyGraph Bolt endpoint (default: bolt://localhost:7687)
- `PUPPYGRAPH_HTTP_URI`: PuppyGraph HTTP API (default: http://localhost:8081)
- `PUPPYGRAPH_USERNAME`: Database username (default: puppygraph)
- `PUPPYGRAPH_PASSWORD`: Database password (default: puppygraph123)

### Customization

#### RAG System
- **Embedding Model**: Change in `rag_system.py` (default: all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB configuration
- **LLM Model**: Claude model selection (default: claude-sonnet-5)

#### UI Customization
- **Port**: Modify in `gradio_app.py` (default: 7860)
- **Styling**: Update CSS in the interface creation
- **Tabs**: Add/remove functionality tabs

## API Reference

### Backend Methods

```python
from backend import PuppyGraphChatbot

chatbot = PuppyGraphChatbot()

# Process natural language query
result = chatbot.process_natural_language_query("Show all nodes")

# Add custom example
chatbot.add_query_example(
    question="Find connected users",
    cypher="MATCH (u1:User)-[:FRIEND]->(u2:User) RETURN u1, u2",
    description="Shows friendship connections"
)

# Get graph statistics
stats = chatbot.get_graph_stats()
```

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure PuppyGraph is running and accessible
2. **Anthropic API Error**: Check your API key and credits
3. **Import Errors**: Install all requirements with `pip install -r requirements.txt`

### Logs

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Connection

Test PuppyGraph connectivity:
```python
from backend import PuppyGraphChatbot
chatbot = PuppyGraphChatbot()
print(chatbot.get_graph_stats())
```

## Development

### Project Structure

```
rag-demo/
├── gradio_app.py        # Main Gradio UI application
├── backend.py           # Backend coordinator
├── rag_system.py        # RAG/text-to-cypher system
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

### Adding Features

1. **New Query Types**: Add examples to `rag_system.py`
2. **UI Components**: Extend tabs in `gradio_app.py`
3. **Backend Logic**: Extend `backend.py`

### Testing

```bash
# Test RAG system
python -c "from rag_system import TextToCypherRAG; rag = TextToCypherRAG(); print('RAG system OK')"

# Test backend
python -c "from backend import get_chatbot; chatbot = get_chatbot(); print(chatbot.get_graph_stats())"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the PuppyGraph ecosystem. See the main repository for license information.

## Support

For issues and questions:
- Check the troubleshooting section
- Review PuppyGraph documentation
- Open an issue in the main repository
