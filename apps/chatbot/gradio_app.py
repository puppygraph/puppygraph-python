#!/usr/bin/env python3

import gradio as gr
import json
import time
from typing import List, Tuple, Dict, Any
import logging

from backend import get_chatbot, shutdown_chatbot
from rag_system import PromptConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_app")

# Global variables for maintaining state
chatbot_instance = None


def initialize_chatbot():
    """Initialize the chatbot backend"""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = get_chatbot()
    return chatbot_instance


def process_message_streaming(message: str, history: List[Dict[str, str]]):
    """Process user message with streaming updates"""
    if not message.strip():
        return
    
    try:
        chatbot = initialize_chatbot()
        
        # Add user message to history with empty response initially
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history
        
        # Stream the processing
        full_response = ""
        for update in chatbot.process_natural_language_query_streaming(message):
            full_response = update
            # Update the last assistant message in history
            history[-1] = {"role": "assistant", "content": full_response}
            yield history
            
    except Exception as e:
        error_response = f"❌ Error processing your message: {str(e)}"
        history[-1] = {"role": "assistant", "content": error_response}
        yield history




def get_graph_stats() -> str:
    """Get and format graph statistics"""
    try:
        chatbot = initialize_chatbot()
        stats = chatbot.get_graph_stats()
        
        if "error" in stats:
            return f"❌ Error getting stats: {stats['error']}"
        
        stats_text = f"""
📊 **Graph Statistics**

🔢 **Nodes:** {stats.get('node_count', 0)}
🔗 **Edges:** {stats.get('edge_count', 0)}

🏷️ **Node Labels:** {', '.join(stats.get('node_labels', []))}
⚡ **Relationship Types:** {', '.join(stats.get('relationship_types', []))}
        """
        
        return stats_text.strip()
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_schema_info() -> str:
    """Get and format schema information"""
    try:
        chatbot = initialize_chatbot()
        schema = chatbot.get_schema()
        
        if not schema:
            return "📋 **Graph Schema**\n\n⚠️ No schema information available. Please ensure PuppyGraph is running and has a configured schema."
        
        schema_text = "📋 **Graph Schema**\n\n"
        
        # Format vertices
        vertices = schema.get("vertices", [])
        edges = schema.get("edges", [])
        
        if not vertices and not edges:
            schema_text += "⚠️ **No schema found**\n\n"
            schema_text += "This could mean:\n"
            schema_text += "• PuppyGraph is not running\n"
            schema_text += "• No schema has been configured in PuppyGraph\n"
            schema_text += "• Connection to PuppyGraph failed\n\n"
            schema_text += "Please check your PuppyGraph server status and configuration."
            return schema_text
        
        if vertices:
            schema_text += "🟢 **Vertices:**\n"
            for vertex in vertices:
                label = vertex.get("label", "Unknown")
                attributes = vertex.get("attributes", [])
                if attributes:
                    attr_text = ", ".join([f"{attr['name']}:{attr['type']}" for attr in attributes])
                    schema_text += f"  • **{label}**: {attr_text}\n"
                else:
                    schema_text += f"  • **{label}**: (no attributes)\n"
            schema_text += "\n"
        
        if edges:
            schema_text += "🔗 **Edges:**\n"
            for edge in edges:
                label = edge.get("label", "Unknown")
                from_vertex = edge.get("from", "Unknown")
                to_vertex = edge.get("to", "Unknown")
                attributes = edge.get("attributes", [])
                if attributes:
                    attr_text = ", ".join([f"{attr['name']}:{attr['type']}" for attr in attributes])
                    schema_text += f"  • **{from_vertex}** -[{label}]-> **{to_vertex}**: {attr_text}\n"
                else:
                    schema_text += f"  • **{from_vertex}** -[{label}]-> **{to_vertex}**: (no attributes)\n"
        
        if not vertices and edges:
            schema_text += "\n⚠️ **Note**: Found edge definitions but no vertex definitions."
        elif vertices and not edges:
            schema_text += "\n⚠️ **Note**: Found vertex definitions but no edge definitions."
        
        return schema_text
        
    except Exception as e:
        return f"📋 **Graph Schema**\n\n❌ Error getting schema: {str(e)}\n\nPlease check that PuppyGraph is running and accessible."


def start_new_session():
    """Start a new session by clearing chat and conversation history"""
    try:
        chatbot = initialize_chatbot()
        chatbot.clear_conversation_history()
        logger.info("New session started - conversation history cleared")
        return []
    except Exception as e:
        logger.error(f"Error starting new session: {e}")
        return []


def add_example_query(question: str, cypher: str, description: str) -> str:
    """Add a new example query to the RAG system"""
    if not question.strip() or not cypher.strip():
        return "❌ Question and Cypher query are required"
    
    try:
        chatbot = initialize_chatbot()
        success = chatbot.add_query_example(question, cypher, description or "User-added example")
        
        if success:
            return "✅ Example added successfully to the knowledge base"
        else:
            return "❌ Failed to add example"
            
    except Exception as e:
        return f"❌ Error adding example: {str(e)}"


def get_current_prompt_config() -> Tuple[str, str, str, str]:
    """Get the current prompt configuration components"""
    try:
        chatbot = initialize_chatbot()
        config = chatbot.get_prompt_config()
        return (
            config.role_definition,
            config.plan_generation_instruction,
            config.puppygraph_differences,
            config.output_format_instruction
        )
    except Exception as e:
        error_msg = f"Error getting config: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg


def update_prompt_config(role_def: str, plan_gen: str, puppygraph_diff: str, output_format: str) -> str:
    """Update the prompt configuration"""
    try:
        chatbot = initialize_chatbot()
        
        # Create new config with updated values
        new_config = PromptConfig(
            role_definition=role_def.strip() or """You are a helpful assistant to help answer user questions about assets in a mining site.
You will need to use the information stored in the graph database to answer the user's questions.
Here is some information about the graph database schema.""",
            plan_generation_instruction=plan_gen.strip() or """You must first output a PLAN, then you can use the PLAN to call the tools.
Each STEP of the PLAN should be corresponding to one or more function calls (but not less), either simple or complex.
Minimize the number of steps in the PLAN, but make sure the PLAN is workable.
Remember, each step can be converted to a Cypher query, since Cypher query can handle quite complex queries,
each step can be complex as well as long as it can be converted to a Cypher query.

IMPORTANT RESULT HANDLING STRATEGY:
- If your query results are truncated (you see "[Results truncated...]"), you have several options:
  1. Use a smaller LIMIT size to get a sample of results first for exploration
  2. Add COUNT(*) queries to understand total result sizes before fetching data
  3. For final comprehensive results, remove LIMIT clauses entirely to provide complete downloadable data
- When providing final conclusions to users, ensure the last query retrieves complete data (no LIMIT) for download
- Structure your approach: exploration -> understanding -> comprehensive final result""",
            puppygraph_differences=puppygraph_diff.strip() or """PUPPYGRAPH DIFFERENCES FROM STANDARD CYPHER:
When calculating failures for a particular asset, also first find out the work orders that are related to the asset, 
then count the work orders that are related to the failure using related_to_failure. 
DO NOT USE can_have_failure for counting total number of failures, USE related_to_failure instead.""",
            output_format_instruction=output_format.strip() or """OUTPUT FORMAT:
Always use the format {
'THINKING': <the thought process in PLAIN TEXT>,
'PLAN': <the plan contains multiple steps in PLAIN TEXT, Your Original plan or Update plan after seeing some executed results>,
'CONCLUSION': <Keep your conclusion simple and clear if you decide to conclude>,
'FINAL_DATA_AVAILABLE': <true/false - whether comprehensive downloadable data is available>,
'QUERY_EXECUTION_SUMMARY': <summary of queries executed and their purpose>}

RESULT MANAGEMENT GUIDELINES:
- For exploratory queries, use appropriate LIMIT clauses (10-50 records)
- For final results intended for user download, use NO LIMIT to provide complete data
- Always inform users when data is available for download
- Include query execution summary to help users understand what was analyzed"""
        )
        
        chatbot.update_prompt_config(new_config)
        return "✅ Prompt configuration updated successfully! The new settings will be used for future queries."
        
    except Exception as e:
        return f"❌ Error updating prompt configuration: {str(e)}"


def reset_prompt_config() -> Tuple[str, str, str, str, str]:
    """Reset prompt configuration to defaults"""
    try:
        chatbot = initialize_chatbot()
        default_config = PromptConfig()  # Create with defaults
        chatbot.update_prompt_config(default_config)
        
        return (
            default_config.role_definition,
            default_config.plan_generation_instruction,
            default_config.puppygraph_differences,
            default_config.output_format_instruction,
            "✅ Prompt configuration reset to defaults!"
        )
    except Exception as e:
        error_msg = f"❌ Error resetting config: {str(e)}"
        return "", "", "", "", error_msg


def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-message {
        font-family: 'Courier New', monospace;
    }
    .stats-display {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    """
    
    with gr.Blocks(css=css, title="PuppyGraph RAG Chat") as interface:
        
        gr.Markdown("""
        # 🐶 PuppyGraph RAG Chatbot
        
        Ask questions about your graph in natural language! Watch in **real-time** as I analyze your question, generate and execute multiple Cypher queries, and build a comprehensive answer step by step.
        
        **🆕 Optimized Conversation System:** Now uses efficient context management:
        - **System prompt**: Schema, rules, and examples defined once per session
        - **Conversation history**: Maintains context across questions without repetition
        - **Full transparency**: Complete prompts, responses, queries, and results
        - **Efficient prompting**: Only dynamic content sent to LLM, reducing costs
        - **Multi-round execution**: Context-aware query generation
        
        **Examples to try:**
        - "Show me all nodes in the graph"
        - "Count all relationships" 
        - "What are the different types of nodes?"
        - "Find highly connected nodes"
        - "Which users have the most connections and what do they connect to?"
        - "What percentage of nodes have more than 5 relationships?"
        """)
        
        with gr.Tab("💬 Chat"):
            gr.Markdown("""
            ### Chat with PuppyGraph 🔄
            **Full conversation transparency enabled** - See every prompt, response, query, and result in detail
            """)
            
            chatbot_ui = gr.Chatbot(
                value=[],
                height=600,
                label="PuppyGraph Assistant (Full Conversation Details)",
                show_label=True,
                elem_classes=["chat-message"],
                type="messages",
                render_markdown=True,
                sanitize_html=False
            )
            
            msg = gr.Textbox(
                placeholder="Ask me anything about your graph...",
                label="Your Question",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", size="sm")
                clear_btn = gr.Button("🔄 New Session", size="sm", variant="stop")
            
            # Event handlers for chat (always streaming)
            msg.submit(
                process_message_streaming,
                inputs=[msg, chatbot_ui],
                outputs=[chatbot_ui]
            ).then(
                lambda: "", outputs=[msg]  # Clear input after submission
            )
            
            submit_btn.click(
                process_message_streaming,
                inputs=[msg, chatbot_ui], 
                outputs=[chatbot_ui]
            ).then(
                lambda: "", outputs=[msg]  # Clear input after submission
            )
            
            clear_btn.click(
                start_new_session,
                outputs=[chatbot_ui]
            )
        
        with gr.Tab("📊 Graph Info"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Graph Statistics")
                    stats_display = gr.Textbox(
                        label="Current Stats",
                        lines=8,
                        interactive=False,
                        elem_classes=["stats-display"]
                    )
                    stats_btn = gr.Button("Refresh Stats", variant="secondary")
                
                with gr.Column():
                    gr.Markdown("### Schema Information")
                    schema_display = gr.Textbox(
                        label="Graph Schema",
                        lines=8,
                        interactive=False,
                        elem_classes=["stats-display"]
                    )
                    schema_btn = gr.Button("Refresh Schema", variant="secondary")
            
            # Event handlers for info tab
            stats_btn.click(
                get_graph_stats,
                outputs=[stats_display]
            )
            
            schema_btn.click(
                get_schema_info,
                outputs=[schema_display]
            )
        
        with gr.Tab("➕ Add Examples"):
            gr.Markdown("""
            ### Add Query Examples
            Help improve the chatbot by adding your own question-to-Cypher examples!
            """)
            
            example_question = gr.Textbox(
                label="Natural Language Question",
                placeholder="e.g., 'Find all users who like movies'",
                lines=2
            )
            
            example_cypher = gr.Textbox(
                label="Corresponding Cypher Query",
                placeholder="e.g., 'MATCH (u:User)-[:LIKES]->(m:Movie) RETURN u, m LIMIT 20'",
                lines=3
            )
            
            example_description = gr.Textbox(
                label="Description (optional)",
                placeholder="Brief description of what this query does",
                lines=1
            )
            
            add_example_btn = gr.Button("Add Example", variant="primary")
            example_result = gr.Textbox(
                label="Result",
                interactive=False,
                lines=2
            )
            
            # Event handler for adding examples
            add_example_btn.click(
                add_example_query,
                inputs=[example_question, example_cypher, example_description],
                outputs=[example_result]
            )
        
        with gr.Tab("⚙️ Prompt Config"):
            gr.Markdown("""
            ### Configure System Prompts
            Customize how the AI assistant behaves by configuring the four key components of the system prompt.
            Changes take effect immediately for new queries.
            """)
            
            # Role Definition
            with gr.Group():
                gr.Markdown("#### 1️⃣ Role Definition")
                gr.Markdown("Define the AI's role and expertise level.")
                role_definition = gr.Textbox(
                    label="Role Definition",
                    lines=2,
                    placeholder="You are an expert at converting natural language questions to Cypher queries for graph databases."
                )
            
            # Plan Generation Instruction  
            with gr.Group():
                gr.Markdown("#### 2️⃣ Plan Generation Instruction")
                gr.Markdown("Tell the AI to create a plan before generating queries.")
                plan_generation = gr.Textbox(
                    label="Plan Generation Instruction",
                    lines=2,
                    placeholder="First, analyze the question and create a step-by-step plan for generating the appropriate Cypher query."
                )
            
            # PuppyGraph Differences
            with gr.Group():
                gr.Markdown("#### 3️⃣ PuppyGraph vs Standard Cypher Differences")
                gr.Markdown("Explain how PuppyGraph differs from standard Cypher syntax.")
                puppygraph_differences = gr.Textbox(
                    label="PuppyGraph Differences",
                    lines=4,
                    placeholder="""PUPPYGRAPH DIFFERENCES FROM STANDARD CYPHER:
- PuppyGraph supports standard Cypher syntax
- Use proper node and relationship patterns: (n)-[r]->(m)
- Labels and properties follow Neo4j conventions
- Functions like count(), size(), type() work as expected"""
                )
            
            # Output Format Instruction
            with gr.Group():
                gr.Markdown("#### 4️⃣ Output Format Instruction")
                gr.Markdown("Define the expected output format and structure.")
                output_format = gr.Textbox(
                    label="Output Format Instruction",
                    lines=4,
                    placeholder="""OUTPUT FORMAT:
Use the generate_cypher_query tool to create a Cypher query that answers this question.
Provide:
1. A complete, valid Cypher query
2. A clear explanation of what the query does
3. Step-by-step reasoning (optional but helpful)"""
                )
            
            # Action buttons
            with gr.Row():
                load_current_btn = gr.Button("Load Current Config", variant="secondary")
                update_config_btn = gr.Button("Update Configuration", variant="primary")
                reset_config_btn = gr.Button("Reset to Defaults", variant="stop")
            
            # Result display
            config_result = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
                placeholder="Click 'Load Current Config' to see the current settings."
            )
            
            # Event handlers for prompt config
            load_current_btn.click(
                get_current_prompt_config,
                outputs=[role_definition, plan_generation, puppygraph_differences, output_format]
            ).then(
                lambda: "✅ Current configuration loaded into form fields above.",
                outputs=[config_result]
            )
            
            update_config_btn.click(
                update_prompt_config,
                inputs=[role_definition, plan_generation, puppygraph_differences, output_format],
                outputs=[config_result]
            )
            
            reset_config_btn.click(
                reset_prompt_config,
                outputs=[role_definition, plan_generation, puppygraph_differences, output_format, config_result]
            )
        
        with gr.Tab("📝 Debug/Prompts"):
            gr.Markdown("### Prompt Debugging")
            gr.Markdown("""
            This tab shows the latest prompt used to generate Cypher queries. This is useful for understanding
            how the RAG system constructs prompts and for debugging query generation issues.
            """)
            
            with gr.Row():
                with gr.Column():
                    latest_prompt_display = gr.Textbox(
                        label="Latest Prompt Used",
                        lines=12,
                        interactive=False,
                        placeholder="No prompts captured yet. Run a query to see the prompt used."
                    )
                    
                with gr.Column():
                    latest_response_display = gr.Textbox(
                        label="Latest LLM Response",
                        lines=12,
                        interactive=False,
                        placeholder="No LLM responses captured yet. Run a query to see the response."
                    )
            
            refresh_debug_btn = gr.Button("Refresh Latest Prompt & Response", variant="secondary")
            
            # Function to get the latest prompt and response from conversation history
            def get_latest_debug_info():
                try:
                    chatbot = initialize_chatbot()
                    history = chatbot.get_conversation_history(1)
                    if history and "executed_steps" in history[0]:
                        steps = history[0]["executed_steps"]
                        if steps:
                            latest_step = steps[-1]
                            prompt = getattr(latest_step, 'prompt', None) or "No prompt available"
                            response = getattr(latest_step, 'llm_response', None) or "No LLM response available"
                            return prompt, response
                    return "No debug information available in recent conversation history.", "No debug information available in recent conversation history."
                except Exception as e:
                    error_msg = f"Error retrieving debug info: {str(e)}"
                    return error_msg, error_msg
            
            refresh_debug_btn.click(
                get_latest_debug_info,
                outputs=[latest_prompt_display, latest_response_display]
            )
        
        with gr.Tab("ℹ️ Help"):
            gr.Markdown("""
            ## How to Use PuppyGraph RAG Chatbot
            
            ### 🗣️ Chat Tab
            - Type natural language questions about your graph
            - **Real-time streaming**: Watch each query step execute live as it happens
            - **Multi-round execution**: generates and runs multiple Cypher queries as needed
            - Claude Sonnet 4.0 decides when it has enough information to provide a complete answer
            - **🆕 Full conversation transparency**: See complete details including:
              - Full prompts sent to Claude Sonnet 4.0 with schema and context
              - Complete LLM responses showing reasoning and decision-making
              - All generated Cypher queries with explanations
              - Full query results with detailed data samples
              - Step-by-step progression through the entire conversation
            
            ### 📊 Graph Info Tab
            - View basic statistics about your graph (node count, edge count, etc.)
            - Explore the schema to understand available node types and relationships
            
            ### ➕ Add Examples Tab
            - Teach the chatbot new patterns by adding question-query pairs
            - Your examples will be used to improve future query generation
            
            ### ⚙️ Prompt Config Tab
            - **Customize AI behavior**: Configure the 4 key components of system prompts
            - **Role Definition**: Set the AI's expertise level and domain knowledge
            - **Plan Generation**: Control whether the AI creates execution plans first
            - **PuppyGraph Differences**: Define how PuppyGraph differs from standard Cypher
            - **Output Format**: Specify the expected response structure
            - **Real-time updates**: Changes take effect immediately for new queries
            - **Reset to defaults**: Easily restore original prompt settings
            
            ### 📝 Debug/Prompts Tab
            - View the exact prompts sent to Claude Sonnet 4.0 for query generation
            - View the raw LLM responses received from Claude
            - Useful for understanding how the RAG system works and debugging issues
            - Prompts include schema info, similar examples, and conversation context
            
            ### 🔧 Technical Details
            - **Backend**: Python with FastAPI
            - **Graph DB**: PuppyGraph (Cypher queries via Bolt protocol)
            - **RAG System**: ChromaDB + SentenceTransformers + Claude Sonnet 4.0
            - **MCP Integration**: Custom Model Context Protocol server
            
            ### 💡 Tips
            - Be specific in your questions for better results
            - Check the confidence score - higher scores indicate more reliable queries
            - Use the schema information to understand what data is available
            - Add your own examples to improve performance for your specific use case
            """)
        
        # Load initial data when interface starts
        interface.load(
            fn=lambda: (get_graph_stats(), get_schema_info()),
            outputs=[stats_display, schema_display]
        )
    
    return interface


def main():
    """Main function to run the application"""
    try:
        # Initialize the chatbot
        logger.info("Initializing PuppyGraph RAG Chatbot...")
        initialize_chatbot()
        
        # Create and launch the interface
        interface = create_interface()
        
        logger.info("Starting Gradio interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            debug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running application: {e}")
    finally:
        # Cleanup
        shutdown_chatbot()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()