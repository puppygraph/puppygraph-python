#!/usr/bin/env python3

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
import mcp.types as types
from neo4j import GraphDatabase, Driver
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("puppygraph-mcp")

class PuppyGraphMCPServer:
    def __init__(self, 
                 bolt_uri: str = "bolt://localhost:7687", 
                 http_uri: str = "http://localhost:8081",
                 username: str = "puppygraph", 
                 password: str = "puppygraph123"):
        self.bolt_uri = bolt_uri
        self.http_uri = http_uri  
        self.username = username
        self.password = password
        self.driver: Optional[Driver] = None
        self.schema_cache: Optional[Dict[str, Any]] = None
        
        # Initialize Neo4j driver
        try:
            self.driver = GraphDatabase.driver(bolt_uri, auth=(username, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to PuppyGraph at {bolt_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to PuppyGraph: {e}")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def get_schema(self) -> Dict[str, Any]:
        """Fetch schema from PuppyGraph HTTP API"""
        if self.schema_cache:
            return self.schema_cache
        
        try:
            response = requests.get(
                f"{self.http_uri}/schemajson",
                auth=(self.username, self.password),
                timeout=10
            )
            response.raise_for_status()
            self.schema_cache = response.json()
            return self.schema_cache
        except Exception as e:
            logger.error(f"Failed to fetch schema: {e}")
            return {"vertices": [], "edges": []}
    
    def execute_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query against PuppyGraph"""
        if not self.driver:
            raise RuntimeError("Not connected to PuppyGraph")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            raise

server = Server("puppygraph-mcp")
puppygraph = PuppyGraphMCPServer()

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="puppygraph://schema",
            name="PuppyGraph Schema",
            description="Current graph schema with vertex and edge definitions",
            mimeType="application/json",
        ),
        Resource(
            uri="puppygraph://stats", 
            name="Graph Statistics",
            description="Basic statistics about the graph (node/edge counts)",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a resource by URI"""
    if uri == "puppygraph://schema":
        schema = puppygraph.get_schema()
        return json.dumps(schema, indent=2)
    
    elif uri == "puppygraph://stats":
        try:
            stats = puppygraph.execute_cypher("""
                MATCH (n) 
                WITH count(n) as node_count
                MATCH ()-[r]->()  
                WITH node_count, count(r) as edge_count
                RETURN node_count, edge_count
            """)
            return json.dumps(stats[0] if stats else {"node_count": 0, "edge_count": 0}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="execute_cypher",
            description="Execute a Cypher query against PuppyGraph and return results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to execute",
                    },
                    "parameters": {
                        "type": "object", 
                        "description": "Optional parameters for the query",
                        "additionalProperties": True,
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_schema_info",
            description="Get detailed schema information about vertices and edges",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_samples": {
                        "type": "boolean",
                        "description": "Whether to include sample data",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="validate_cypher",
            description="Validate a Cypher query without executing it",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to validate",
                    }
                },
                "required": ["query"],
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "execute_cypher":
        query = arguments.get("query")
        parameters = arguments.get("parameters", {})
        
        if not query:
            return [types.TextContent(type="text", text="Error: No query provided")]
        
        try:
            results = puppygraph.execute_cypher(query, parameters)
            return [types.TextContent(
                type="text", 
                text=json.dumps(results, indent=2, default=str)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text", 
                text=f"Error executing query: {str(e)}"
            )]
    
    elif name == "get_schema_info":
        include_samples = arguments.get("include_samples", False)
        
        try:
            schema = puppygraph.get_schema()
            
            if include_samples:
                # Get sample data for each vertex and edge type
                samples = {}
                for vertex in schema.get("vertices", []):
                    label = vertex["label"]
                    try:
                        sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
                        samples[f"vertex_{label}"] = puppygraph.execute_cypher(sample_query)
                    except:
                        samples[f"vertex_{label}"] = []
                
                for edge in schema.get("edges", []):
                    label = edge["label"]
                    try:
                        sample_query = f"MATCH ()-[r:{label}]->() RETURN r LIMIT 3"
                        samples[f"edge_{label}"] = puppygraph.execute_cypher(sample_query)
                    except:
                        samples[f"edge_{label}"] = []
                
                result = {"schema": schema, "samples": samples}
            else:
                result = {"schema": schema}
            
            return [types.TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error getting schema: {str(e)}"
            )]
    
    elif name == "validate_cypher":
        query = arguments.get("query")
        
        if not query:
            return [types.TextContent(type="text", text="Error: No query provided")]
        
        try:
            # Try to explain the query to validate syntax
            explain_query = f"EXPLAIN {query}"
            puppygraph.execute_cypher(explain_query)
            return [types.TextContent(
                type="text",
                text="Query syntax is valid"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Query validation failed: {str(e)}"
            )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    # Register cleanup handler
    import atexit
    atexit.register(puppygraph.close)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="puppygraph-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())