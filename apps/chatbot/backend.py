import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import requests
from rag_system import TextToCypherRAG, QueryExample, PromptConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


class PuppyGraphChatbot:
    """Main chatbot backend that coordinates MCP server, RAG system, and PuppyGraph"""
    
    def __init__(self, 
                 puppygraph_bolt_uri: str = "bolt://localhost:7687",
                 puppygraph_http_uri: str = "http://localhost:8081", 
                 puppygraph_username: str = "puppygraph",
                 puppygraph_password: str = "puppygraph123",
                 prompt_config: Optional[PromptConfig] = None):
        
        self.puppygraph_bolt_uri = puppygraph_bolt_uri
        self.puppygraph_http_uri = puppygraph_http_uri
        self.puppygraph_username = puppygraph_username
        self.puppygraph_password = puppygraph_password
        
        # Initialize RAG system with optional prompt configuration
        self.rag_system = TextToCypherRAG(prompt_config=prompt_config)
        
        # MCP server process
        self.mcp_process = None
        
        # Cache for schema and frequently used data
        self.schema_cache = None
        self.schema_cache_time = 0
        self.cache_duration = 300  # 5 minutes
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
    
    async def start_mcp_server(self):
        """Start the MCP server process"""
        try:
            # Set environment variables for MCP server
            env = {
                "PUPPYGRAPH_BOLT_URI": self.puppygraph_bolt_uri,
                "PUPPYGRAPH_HTTP_URI": self.puppygraph_http_uri,
                "PUPPYGRAPH_USERNAME": self.puppygraph_username,
                "PUPPYGRAPH_PASSWORD": self.puppygraph_password
            }
            
            self.mcp_process = subprocess.Popen(
                ["python", "mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            if self.mcp_process.poll() is None:
                logger.info("MCP server started successfully")
                return True
            else:
                logger.error("MCP server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return False
    
    def stop_mcp_server(self):
        """Stop the MCP server process"""
        if self.mcp_process and self.mcp_process.poll() is None:
            self.mcp_process.terminate()
            self.mcp_process.wait()
            logger.info("MCP server stopped")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema from PuppyGraph with caching"""
        current_time = time.time()
        
        # Return cached schema if still valid
        if (self.schema_cache and 
            current_time - self.schema_cache_time < self.cache_duration):
            return self.schema_cache
        
        try:
            response = requests.get(
                f"{self.puppygraph_http_uri}/schemajson",
                auth=(self.puppygraph_username, self.puppygraph_password),
                timeout=10
            )
            response.raise_for_status()
            
            raw_schema = response.json()
            
            # Convert PuppyGraph schema format to our expected format
            converted_schema = self._convert_puppygraph_schema(raw_schema)
            
            self.schema_cache = converted_schema
            self.schema_cache_time = current_time
            
            return self.schema_cache
            
        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            # Return cached schema if available, otherwise empty schema
            return self.schema_cache or {"vertices": [], "edges": []}
    
    def _convert_puppygraph_schema(self, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PuppyGraph schema format to our expected format based on graph_schema.proto"""
        
        try:
            # Extract graph definition (non-deprecated format)
            graph_def = raw_schema.get("graph", {})
            
            # Convert vertices from Graph.VertexSchema format
            vertices = []
            for vertex in graph_def.get("vertices", []):
                converted_vertex = {
                    "label": vertex.get("label", "Unknown"),
                    "attributes": [],
                    "description": vertex.get("description", "")
                }
                
                # Handle OneToOne mapping (most common)
                one_to_one = vertex.get("oneToOne", {})
                if one_to_one:
                    # Extract attributes from MappedField format
                    attributes = one_to_one.get("attributes", [])
                    for attr in attributes:
                        converted_vertex["attributes"].append({
                            "name": attr.get("alias", attr.get("field", "unknown")),
                            "type": self._map_puppygraph_type(attr.get("type", "String"))
                        })
                
                # Handle ManyToOne mapping if present
                many_to_one = vertex.get("manyToOne", {})
                if many_to_one:
                    # For ManyToOne, we'll just show it has complex mapping
                    converted_vertex["attributes"].append({
                        "name": "complex_mapping",
                        "type": "ManyToOne"
                    })
                
                vertices.append(converted_vertex)
            
            # Convert edges from Graph.EdgeSchema format
            edges = []
            for edge in graph_def.get("edges", []):
                converted_edge = {
                    "label": edge.get("label", "Unknown"),
                    "from": edge.get("fromVertex", "Unknown"),
                    "to": edge.get("toVertex", "Unknown"),
                    "attributes": [],
                    "description": edge.get("description", "")
                }
                
                # Extract attributes from MappedField format
                attributes = edge.get("attributes", [])
                for attr in attributes:
                    converted_edge["attributes"].append({
                        "name": attr.get("alias", attr.get("field", "unknown")),
                        "type": self._map_puppygraph_type(attr.get("type", "String"))
                    })
                
                edges.append(converted_edge)
            
            return {
                "vertices": vertices,
                "edges": edges
            }
            
        except Exception as e:
            logger.error(f"Error converting PuppyGraph schema: {e}")
            return {"vertices": [], "edges": []}
    
    def _map_puppygraph_type(self, puppygraph_type: str) -> str:
        """Map PuppyGraph types to standard types"""
        type_mapping = {
            "String": "String",
            "Int": "Integer", 
            "Double": "Double",
            "Boolean": "Boolean",
            "Long": "Long",
            "Float": "Float"
        }
        return type_mapping.get(puppygraph_type, "String")
    
    def execute_cypher_direct(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute Cypher query directly against PuppyGraph"""
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable, AuthError
        
        try:
            driver = GraphDatabase.driver(
                self.puppygraph_bolt_uri,
                auth=(self.puppygraph_username, self.puppygraph_password)
            )
            
            with driver.session() as session:
                result = session.run(query, params or {})
                records = [record.data() for record in result]
                
            driver.close()
            
            return {
                "success": True,
                "data": records,
                "query": query,
                "record_count": len(records)
            }
            
        except ServiceUnavailable as e:
            error_msg = f"PuppyGraph server not available at {self.puppygraph_bolt_uri}. Please ensure PuppyGraph is running."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
        except AuthError as e:
            error_msg = f"Authentication failed. Check PuppyGraph credentials."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def process_natural_language_query_streaming(self, question: str):
        """Process a natural language question with streaming progress updates"""
        
        try:
            # Get current schema
            schema = self.get_schema()
            
            # Initialize progress
            executed_steps = []
            max_rounds = 5
            
            # Yield initial status
            yield self._format_streaming_update("🤖 Starting multi-round query execution...", executed_steps, None, question)
            
            for round_num in range(1, max_rounds + 1):
                # Generate next query or decision to stop
                yield self._format_streaming_update(f"🔄 Round {round_num}: Analyzing question and generating query...", executed_steps, None, question)
                
                cypher, explanation, should_stop, prompt, llm_response = self.rag_system.generate_next_query(
                    question, schema, executed_steps, max_rounds
                )
                
                if should_stop:
                    # Final answer ready
                    final_result = {
                        "question": question,
                        "executed_steps": executed_steps,
                        "final_answer": explanation,
                        "total_rounds": round_num - 1,
                        "success": True,
                        "stopped_reason": "LLM determined sufficient information gathered"
                    }
                    self.conversation_history.append(final_result)
                    yield self._format_streaming_update("✅ Analysis complete!", executed_steps, explanation, question, final=True)
                    return
                
                # Show generated query
                yield self._format_streaming_update(f"📝 Generated query for round {round_num}", executed_steps, None, question, current_query=cypher, current_description=explanation, current_prompt=prompt, current_llm_response=llm_response)
                
                # Create query step
                from rag_system import QueryStep
                query_step = QueryStep(
                    step_number=round_num,
                    description=explanation,
                    cypher=cypher,
                    prompt=prompt,
                    llm_response=llm_response
                )
                
                # Execute the query
                yield self._format_streaming_update(f"⚡ Executing query {round_num}...", executed_steps, None, question, current_query=cypher)
                
                try:
                    execution_result = self.execute_cypher_direct(cypher)
                    if execution_result.get("success", False):
                        query_step.result = execution_result.get("data", [])
                        result_summary = f"✅ Query {round_num} completed: {len(query_step.result)} records returned"
                    else:
                        query_step.error = execution_result.get("error", "Unknown error")
                        result_summary = f"❌ Query {round_num} failed: {query_step.error}"
                except Exception as e:
                    query_step.error = str(e)
                    result_summary = f"❌ Query {round_num} error: {str(e)}"
                
                executed_steps.append(query_step)
                
                # Add query result to RAG system conversation history
                if query_step.result is not None:
                    total_records = len(query_step.result)
                    rag_result_summary = f"Query executed successfully. Found {total_records} records."
                    if query_step.result:
                        sample = query_step.result[:5]  # Show first 5 records as sample
                        rag_result_summary += f" Sample data (showing {len(sample)} of {total_records}): {json.dumps(sample, default=str)}"
                        if total_records > 5:
                            rag_result_summary += f" (Note: {total_records - 5} additional records omitted)"
                elif query_step.error:
                    rag_result_summary = f"Query failed with error: {query_step.error}"
                else:
                    rag_result_summary = "Query executed with no result data."
                
                self.rag_system.conversation_messages.append({
                    "role": "assistant",
                    "content": f"I executed this query: {query_step.cypher}\nResult: {rag_result_summary}"
                })
                
                # Show execution result
                yield self._format_streaming_update(result_summary, executed_steps, None, question)
            
            # If we reach max rounds, force stop
            final_answer = self.rag_system.generate_final_answer_from_steps(question, executed_steps)
            final_result = {
                "question": question,
                "executed_steps": executed_steps,
                "final_answer": final_answer,
                "total_rounds": max_rounds,
                "success": True,
                "stopped_reason": f"Reached maximum rounds ({max_rounds})"
            }
            self.conversation_history.append(final_result)
            yield self._format_streaming_update("🛑 Maximum rounds reached", executed_steps, final_answer, question, final=True)
            
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            yield f"❌ Error: {str(e)}"
    
    def _format_streaming_update(self, status: str, executed_steps, final_answer: str = None, question: str = "", final: bool = False, current_query: str = None, current_description: str = None, current_prompt: str = None, current_llm_response: str = None) -> str:
        """Format a streaming progress update in chronological order"""
        
        response = f"**Question:** {question}\n\n"
        
        # If we have a final answer, show it prominently first and collapse the details
        if final_answer and final:
            response += f"**🎯 Final Answer:**\n{final_answer}\n\n"
            response += f"---\n\n"
            
            # Put all processing details in a collapsible section
            details_content = f"**Status:** {status}\n\n"
            
            # System configuration in details
            system_prompt = self.rag_system.get_system_prompt()
            if system_prompt:
                details_content += f"**🔧 System Configuration:**\n"
                details_content += f"<details><summary>View System Prompt (Schema, Rules, Examples)</summary>\n\n```\n{system_prompt}\n```\n</details>\n\n"
            
            # Processing steps in details
            if executed_steps:
                details_content += f"**Processing Steps ({len(executed_steps)}):**\n\n"
                for step in executed_steps:
                    details_content += f"---\n**Step {step.step_number}:** {step.description}\n\n"
                    
                    # LLM request for this step
                    if hasattr(step, 'prompt') and step.prompt:
                        details_content += f"**🤖 LLM Request:**\n```\n{step.prompt}\n```\n\n"
                    
                    # LLM response (tool use)
                    if hasattr(step, 'llm_response') and step.llm_response:
                        details_content += f"**🤖 LLM Response:**\n```json\n{step.llm_response}\n```\n\n"
                    
                    # Generated Cypher query
                    details_content += f"**🔗 Cypher Query:**\n```cypher\n{step.cypher}\n```\n\n"
                    
                    # Query execution results
                    if step.result is not None:
                        details_content += f"**📊 Query Results:** ✅ {len(step.result)} records returned\n"
                        if step.result:
                            sample_size = min(5, len(step.result))
                            details_content += f"**Sample Data (showing {sample_size} of {len(step.result)} records):**\n"
                            details_content += f"```json\n{json.dumps(step.result[:sample_size], indent=2, default=str)}\n```\n"
                            if len(step.result) > sample_size:
                                details_content += f"... and {len(step.result) - sample_size} more records\n"
                        details_content += "\n"
                    elif step.error:
                        details_content += f"**Query Error:** ❌ {step.error}\n\n"
                    
                    details_content += "\n"
            
            # Wrap all details in a collapsible section
            response += f"<details><summary>📋 Click to view detailed processing steps ({len(executed_steps)} queries executed)</summary>\n\n{details_content}\n</details>\n"
            
            return response
        
        # If not final, show the regular streaming format
        response += f"**Status:** {status}\n\n"
        
        # 1. System prompt (always first)
        system_prompt = self.rag_system.get_system_prompt()
        if system_prompt:
            response += f"**🔧 System Configuration:**\n"
            response += f"<details><summary>View System Prompt (Schema, Rules, Examples)</summary>\n\n```\n{system_prompt}\n```\n</details>\n\n"
        
        # 2. Initial user query (what started everything)
        response += f"**🧑 User Query:** {question}\n\n"
        
        # 3. Show completed steps in chronological order
        if executed_steps:
            response += f"**Processing Steps ({len(executed_steps)}):**\n\n"
            for step in executed_steps:
                response += f"---\n**Step {step.step_number}:** {step.description}\n\n"
                
                # 3a. LLM request for this step
                if hasattr(step, 'prompt') and step.prompt:
                    response += f"**🤖 LLM Request:**\n```\n{step.prompt}\n```\n\n"
                
                # 3b. LLM response (tool use)
                if hasattr(step, 'llm_response') and step.llm_response:
                    response += f"**🤖 LLM Response:**\n```json\n{step.llm_response}\n```\n\n"
                
                # 3c. Generated Cypher query
                response += f"**🔗 Cypher Query:**\n```cypher\n{step.cypher}\n```\n\n"
                
                # 3d. Query execution results
                if step.result is not None:
                    response += f"**📊 Query Results:** ✅ {len(step.result)} records returned\n"
                    if step.result:
                        sample_size = min(5, len(step.result))
                        response += f"**Sample Data (showing {sample_size} of {len(step.result)} records):**\n"
                        response += f"```json\n{json.dumps(step.result[:sample_size], indent=2, default=str)}\n```\n"
                        if len(step.result) > sample_size:
                            response += f"... and {len(step.result) - sample_size} more records\n"
                    response += "\n"
                elif step.error:
                    response += f"**Query Error:** ❌ {step.error}\n\n"
                
                response += "\n"
        
        # 4. Show current step in progress (if any)
        if current_query and not final:
            step_num = len(executed_steps) + 1
            response += f"---\n**Step {step_num} (In Progress):** {current_description or 'Processing...'}\n\n"
            
            # 4a. LLM request for current step
            if current_prompt:
                response += f"**🤖 LLM Request:**\n```\n{current_prompt}\n```\n\n"
            
            # 4b. LLM response for current step
            if current_llm_response:
                response += f"**🤖 LLM Response:**\n```json\n{current_llm_response}\n```\n\n"
            
            # 4c. Generated Cypher query
            response += f"**🔗 Cypher Query:**\n```cypher\n{current_query}\n```\n\n"
            response += f"**📊 Status:** Executing query...\n\n"
        
        return response
    
    def add_query_example(self, question: str, cypher: str, description: str) -> bool:
        """Add a new query example to the RAG system"""
        try:
            example = QueryExample(
                question=question,
                cypher=cypher,
                description=description
            )
            self.rag_system.add_example(example)
            return True
        except Exception as e:
            logger.error(f"Error adding query example: {e}")
            return False
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        # Also clear the RAG system's conversation history
        self.rag_system.clear_conversation()
    
    def update_prompt_config(self, prompt_config: PromptConfig):
        """Update the prompt configuration for the RAG system"""
        self.rag_system.update_prompt_config(prompt_config)
        logger.info("Chatbot prompt configuration updated")
    
    def get_prompt_config(self) -> PromptConfig:
        """Get the current prompt configuration"""
        return self.rag_system.get_prompt_config()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic graph statistics"""
        try:
            stats_query = """
            MATCH (n) 
            WITH count(n) as node_count
            MATCH ()-[r]->()  
            RETURN node_count, count(r) as edge_count
            """
            
            result = self.execute_cypher_direct(stats_query)
            
            if result["success"] and result["data"]:
                stats = result["data"][0]
                
                # Get node labels and relationship types separately
                node_labels = []
                relationship_types = []
                
                try:
                    # Try to get distinct node labels
                    labels_result = self.execute_cypher_direct("MATCH (n) RETURN DISTINCT labels(n) as node_labels LIMIT 20")
                    if labels_result["success"]:
                        node_labels = [item["node_labels"][0] for item in labels_result["data"] if item.get("node_labels")]
                except:
                    pass
                
                try:
                    # Try to get distinct relationship types
                    types_result = self.execute_cypher_direct("MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_type LIMIT 20")
                    if types_result["success"]:
                        relationship_types = [item["relationship_type"] for item in types_result["data"]]
                except:
                    pass
                
                return {
                    "node_count": stats.get("node_count", 0),
                    "edge_count": stats.get("edge_count", 0),
                    "node_labels": node_labels,
                    "relationship_types": relationship_types
                }
            else:
                return {"node_count": 0, "edge_count": 0, "node_labels": [], "relationship_types": []}
                
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_mcp_server()


# Global chatbot instance
chatbot = None

def get_chatbot() -> PuppyGraphChatbot:
    """Get or create the global chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = PuppyGraphChatbot()
    return chatbot

def shutdown_chatbot():
    """Shutdown the global chatbot instance"""
    global chatbot
    if chatbot:
        chatbot.cleanup()
        chatbot = None