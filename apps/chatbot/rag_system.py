import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_system")


@dataclass
class QueryExample:
    """Example query with natural language question and corresponding Cypher"""
    question: str
    cypher: str
    description: str
    schema_context: Optional[str] = None


@dataclass
class PromptConfig:
    """Configuration for customizable prompt components"""
    role_definition: str = """You are an expert at converting natural language questions to Cypher queries for graph databases."""
    
    plan_generation_instruction: str = """First, analyze the question and create a step-by-step plan for generating the appropriate Cypher query."""
    
    puppygraph_differences: str = """PUPPYGRAPH DIFFERENCES FROM STANDARD CYPHER:
- PuppyGraph supports standard Cypher syntax
- Use proper node and relationship patterns: (n)-[r]->(m)
- Always list required properties in the query, do not return nodes or relationships without the required properties
- Only return one aggregate value per query, including collect(), count(), size(), type(), etc."""
    
    output_format_instruction: str = """OUTPUT FORMAT:
Use the generate_cypher_query tool to create a Cypher query that answers this question.
Provide:
1. A complete, valid Cypher query
2. A clear explanation of what the query does
3. Step-by-step reasoning (optional but helpful)"""


@dataclass
class QueryStep:
    """A single step in a multi-query execution plan"""
    step_number: int
    description: str
    cypher: str
    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    prompt: Optional[str] = None
    llm_response: Optional[str] = None


class TextToCypherRAG:
    """RAG system for converting natural language to Cypher queries"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "cypher_examples",
                 anthropic_api_key: Optional[str] = None,
                 prompt_config: Optional[PromptConfig] = None):
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Anthropic client
        self.anthropic_client = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize prompt configuration
        self.prompt_config = prompt_config or PromptConfig()
        
        # Initialize conversation history - maintain context across questions
        self.conversation_messages = []
        self.current_schema = None
        
        # Initialize with some default examples
        self._initialize_examples()
    
    def _initialize_examples(self):
        """Initialize the RAG system with some common query examples"""
        examples = [
            QueryExample(
                question="Show all nodes in the graph",
                cypher="MATCH (n) RETURN n LIMIT 100",
                description="Returns all nodes with a limit"
            ),
            QueryExample(
                question="Count all nodes",
                cypher="MATCH (n) RETURN count(n) as node_count",
                description="Counts total number of nodes"
            ),
            QueryExample(
                question="Count all relationships",
                cypher="MATCH ()-[r]->() RETURN count(r) as relationship_count",
                description="Counts total number of relationships"
            ),
            QueryExample(
                question="Show graph statistics",
                cypher="MATCH (n) WITH count(n) as node_count MATCH ()-[r]->() RETURN node_count, count(r) as edge_count",
                description="Shows basic graph statistics"
            ),
            QueryExample(
                question="Find nodes with specific property",
                cypher="MATCH (n) WHERE n.name IS NOT NULL RETURN n.name LIMIT 10",
                description="Finds nodes that have a name property"
            ),
            QueryExample(
                question="Show all relationship types",
                cypher="MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_type",
                description="Returns all unique relationship types in the graph"
            ),
            QueryExample(
                question="Show all node labels",
                cypher="MATCH (n) RETURN DISTINCT labels(n) as node_labels",
                description="Returns all unique node labels in the graph"
            ),
            QueryExample(
                question="Find connected nodes",
                cypher="MATCH (n)-[r]-(m) RETURN n, type(r) as relationship, m LIMIT 20",
                description="Shows connected nodes with their relationships"
            ),
            QueryExample(
                question="Find shortest path between nodes",
                cypher="MATCH p = shortestPath((start)-[*]-(end)) WHERE id(start) = $start_id AND id(end) = $end_id RETURN p",
                description="Finds shortest path between two specific nodes"
            ),
            QueryExample(
                question="Find nodes by degree",
                cypher="MATCH (n) WITH n, size((n)--()) as degree WHERE degree > 5 RETURN n, degree ORDER BY degree DESC",
                description="Finds nodes with high connectivity (degree > 5)"
            )
        ]
        
        self._add_examples_to_collection(examples)
    
    def _add_examples_to_collection(self, examples: List[QueryExample]):
        """Add examples to the ChromaDB collection"""
        if not examples:
            return
        
        # Check if examples already exist
        existing_count = self.collection.count()
        if existing_count >= len(examples):
            logger.info(f"Examples already exist in collection ({existing_count} items)")
            return
        
        # Prepare data for ChromaDB
        questions = [ex.question for ex in examples]
        embeddings = self.embedding_model.encode(questions).tolist()
        
        ids = [f"example_{i}" for i in range(len(examples))]
        documents = questions
        metadatas = [
            {
                "cypher": ex.cypher,
                "description": ex.description,
                "schema_context": ex.schema_context or ""
            }
            for ex in examples
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(examples)} examples to the collection")
    
    def add_example(self, example: QueryExample):
        """Add a single example to the RAG system"""
        self._add_examples_to_collection([example])
    
    def update_prompt_config(self, prompt_config: PromptConfig):
        """Update the prompt configuration"""
        self.prompt_config = prompt_config
        logger.info("Prompt configuration updated")
    
    def get_prompt_config(self) -> PromptConfig:
        """Get the current prompt configuration"""
        return self.prompt_config
    
    def _build_system_prompt(self, schema: Dict[str, Any]) -> str:
        """Build the system prompt with static information that doesn't change during conversation"""
        
        # Format schema information
        schema_info = self._format_schema_for_prompt(schema)
        
        # Get similar examples for general context
        examples_text = "\n".join([
            f"Q: {ex['question']}\nCypher: {ex['cypher']}\nDescription: {ex['description']}\n"
            for ex in self.find_similar_examples("graph query examples", k=5)
        ])
        
        system_prompt = f"""{self.prompt_config.role_definition}

{self.prompt_config.plan_generation_instruction}

GRAPH SCHEMA:
{schema_info}

EXAMPLE QUERY PATTERNS:
{examples_text}

{self.prompt_config.puppygraph_differences}

RULES:
1. Always use proper Cypher syntax
2. Include appropriate LIMIT clauses for large result sets  
3. Use parameterized queries when possible
4. Consider the graph schema when writing queries
5. Return only valid, executable Cypher
6. Be conservative with result sizes (use LIMIT 100 or less by default)
7. IMPORTANT: You have a limited number of query rounds. When approaching the limit, prioritize gathering the most essential information.
8. CRITICAL: When you reach the maximum number of rounds, you MUST stop and provide a comprehensive summary based on all gathered data.

{self.prompt_config.output_format_instruction}

You will be asked to help answer questions by generating Cypher queries step by step. Use the tools provided to generate queries or make decisions about when to stop and provide a final answer. Consider previous conversation context when making decisions."""

        return system_prompt
    
    def _update_schema_if_needed(self, schema: Dict[str, Any]):
        """Update the schema and rebuild system prompt if schema has changed"""
        if self.current_schema != schema:
            self.current_schema = schema
            # Only reset conversation if this is truly a new schema (not just the first time)
            if hasattr(self, 'conversation_messages') and self.conversation_messages:
                logger.warning("Schema changed mid-conversation, resetting conversation history")
                self.conversation_messages = []
            logger.info("Schema updated")
    
    def clear_conversation(self):
        """Clear the conversation history (but keep the current schema/system prompt)"""
        self.conversation_messages = []
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt"""
        if self.current_schema:
            return self._build_system_prompt(self.current_schema)
        return ""
    
    def find_similar_examples(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find similar examples for a given question"""
        # Generate embedding for the question
        question_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        similar_examples = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                similar_examples.append({
                    "question": results["documents"][0][i],
                    "cypher": results["metadatas"][0][i]["cypher"],
                    "description": results["metadatas"][0][i]["description"],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return similar_examples
    
    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema information for the LLM prompt"""
        schema_text = "VERTICES:\n"
        
        for vertex in schema.get("vertices", []):
            label = vertex.get("label", "Unknown")
            attributes = vertex.get("attributes", [])
            description = vertex.get("description", "").strip()
            
            # Label on its own line
            schema_text += f"- {label}\n"
            
            # Description section (if available)
            if description:
                schema_text += f"  Description: {description}\n"
            
            # Attributes
            if attributes:
                attr_text = ", ".join([f"{attr['name']}:{attr['type']}" for attr in attributes])
                schema_text += f"  Attributes: {attr_text}\n"
            else:
                schema_text += f"  Attributes: (none)\n"
            
            schema_text += "\n"
        
        schema_text += "EDGES:\n"
        for edge in schema.get("edges", []):
            label = edge.get("label", "Unknown")
            from_vertex = edge.get("from", "Unknown")
            to_vertex = edge.get("to", "Unknown")
            attributes = edge.get("attributes", [])
            description = edge.get("description", "").strip()
            
            # Edge pattern on its own line
            schema_text += f"- (:{from_vertex})-[:{label}]->(:{to_vertex})\n"
            
            # Description section (if available)
            if description:
                schema_text += f"  Description: {description}\n"
            
            # Attributes
            if attributes:
                attr_text = ", ".join([f"{attr['name']}:{attr['type']}" for attr in attributes])
                schema_text += f"  Attributes: {attr_text}\n"
            else:
                schema_text += f"  Attributes: (none)\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def generate_next_query(self, 
                           question: str,
                           schema: Dict[str, Any],
                           previous_steps: List[QueryStep],
                           max_rounds: int = 5) -> Tuple[str, str, bool, str, str]:
        """Generate the next query in a multi-round execution, or decide to stop"""
        
        # Ensure we have the schema set up in our conversation
        self._update_schema_if_needed(schema)
        
        # If this is the start of a new question, add it to conversation
        if not previous_steps:
            self.conversation_messages.append({
                "role": "user", 
                "content": f"Please help me answer this question: {question}"
            })
        
        # Debug: Print conversation history at each round
        logger.info(f"=== Round {len(previous_steps) + 1} - Conversation History ({len(self.conversation_messages)} messages) ===")
        for i, msg in enumerate(self.conversation_messages):
            logger.info(f"  {i+1}. {msg['role']}: {msg['content'][:100]}...")
        logger.info("=== End Conversation History ===")
        
        
        # Define the tool for multi-round query generation
        multi_round_tool = {
            "name": "multi_round_query_decision",
            "description": "Decide whether to generate another query or provide final answer in multi-round execution",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["CONTINUE", "STOP"],
                        "description": "Whether to continue with another query or stop and provide final answer. You MUST choose STOP if you're at the maximum number of rounds."
                    },
                    "cypher_query": {
                        "type": "string",
                        "description": "The next Cypher query to execute (only if action is CONTINUE)"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation of what this query will do (if CONTINUE) or the final answer to the original question (if STOP)"
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "The final answer to the original question (if STOP). This should be a comprehensive summary based on all gathered data. Do not include query details or reasoning - focus on answering the user's original question directly."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for why you chose to continue or stop"
                    }
                },
                "required": ["action", "explanation"]
            }
        }
        
        # Add the current request to ask what to do next
        current_round = len(previous_steps) + 1
        remaining_rounds = max_rounds - len(previous_steps)
        
        if previous_steps:
            if current_round >= max_rounds:
                # Force summarization on last round
                current_request = f"I have executed {len(previous_steps)} queries and this is my final round (round {max_rounds}). I must now STOP and provide a comprehensive final answer to the original question '{question}' based on all the information I've gathered."
            elif current_round == max_rounds - 1:
                # Warn about approaching limit
                current_request = f"I have executed {len(previous_steps)} queries so far and only have {remaining_rounds} round left. Based on the results from our conversation, should I continue with one final query or do I have enough information to answer the original question '{question}'? If I continue, the next round will be my last."
            else:
                current_request = f"I have executed {len(previous_steps)} queries so far ({remaining_rounds} rounds remaining). Based on the results from our conversation, should I continue with another query or do I have enough information to answer the original question '{question}'?"
        else:
            current_request = f"I need to answer the question: '{question}'. Should I execute a query to gather information or do I already have what I need? I have up to {max_rounds} rounds available."
        
        # Create a copy of messages for this request (don't permanently add the request)
        current_messages = self.conversation_messages + [{"role": "user", "content": current_request}]
        
        # Build simple prompt that will be shown in UI (the real context is in conversation history)
        display_prompt = f"Question: {question}\nCurrent request: {current_request}"
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0,
                system=self.get_system_prompt(),
                tools=[multi_round_tool],
                tool_choice={"type": "tool", "name": "multi_round_query_decision"},
                messages=current_messages
            )
            
            # Extract tool use result
            if response.content and response.content[0].type == "tool_use":
                tool_input = response.content[0].input
                action = tool_input.get("action", "STOP")
                explanation = tool_input.get("explanation", "")
                reasoning = tool_input.get("reasoning", "")
                
                # Format the raw response for display
                raw_response = f"Multi-round tool response:\n{json.dumps(tool_input, indent=2)}"
                
                if action == "STOP" or current_round >= max_rounds:
                    # Force stop if we've reached max rounds, regardless of LLM decision
                    if current_round >= max_rounds and action == "CONTINUE":
                        logger.warning(f"LLM tried to continue past max_rounds ({max_rounds}), forcing stop")
                        action = "STOP"
                        explanation = "Reached maximum number of query rounds - providing summary based on gathered data."
                    
                    # Include reasoning in the final answer if provided
                    final_answer = tool_input.get("final_answer", "")
                    if not final_answer:
                        final_answer = explanation
                        if reasoning:
                            final_answer = f"{explanation}\n\nReasoning: {reasoning}"
                    
                    # Add the final response to conversation history
                    self.conversation_messages.append({
                        "role": "assistant",
                        "content": f"I have enough information to answer. Final answer: {final_answer}"
                    })
                    
                    return "", final_answer, True, display_prompt, raw_response
                else:
                    # Continue with next query
                    cypher_query = tool_input.get("cypher_query", "")
                    if not cypher_query:
                        logger.warning("No cypher query provided for CONTINUE action")
                        return "MATCH (n) RETURN count(n) as count", "No query provided", False, display_prompt, raw_response
                    
                    # Include reasoning in explanation if provided
                    full_explanation = explanation
                    if reasoning:
                        full_explanation = f"{explanation} (Reasoning: {reasoning})"
                    
                    # Add the decision to conversation history so Claude can see its own reasoning
                    self.conversation_messages.append({
                        "role": "assistant",
                        "content": f"I decided to continue with another query: {cypher_query}\nExplanation: {full_explanation}"
                    })
                    
                    return cypher_query, full_explanation, False, display_prompt, raw_response
            else:
                logger.warning("No tool use found in multi-round response")
                return "MATCH (n) RETURN count(n) as count", "Error: No valid tool response received", True, display_prompt, "No tool use found"
                
        except Exception as e:
            logger.error(f"Error generating next query: {e}")
            return "MATCH (n) RETURN count(n) as count", f"Error: {str(e)}", True, "", f"Error: {str(e)}"
    
    def generate_final_answer_from_steps(self, question: str, executed_steps: List[QueryStep]) -> str:
        """Generate final answer using LLM to analyze conversation history"""
        
        if not executed_steps:
            return "I wasn't able to execute any queries to answer your question."
        
        # Ask Claude to summarize based on the conversation history
        summary_request = f"Based on our conversation and the {len(executed_steps)} queries I've executed, please provide a comprehensive answer to the original question: '{question}'. Summarize what we learned and provide the best answer you can based on the data we gathered."
        
        current_messages = self.conversation_messages + [{"role": "user", "content": summary_request}]
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=1,
                thinking={
                  "type": "enabled",
                  "budget_tokens": 1600
                },
                system=self.get_system_prompt(),
                messages=current_messages
            )
            
            if response.content and response.content[0].type == "text":
                return response.content[0].text
            else:
                # Fallback to basic summary
                return f"I executed {len(executed_steps)} queries to gather information, but couldn't generate a proper summary."
                
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return f"I executed {len(executed_steps)} queries but encountered an error when summarizing: {str(e)}"