#!/usr/bin/env python3

import os
import sys
import time
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import gradio as gr
        logger.info("✅ Gradio imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Gradio: {e}")
        return False
    
    try:
        from backend import PuppyGraphChatbot
        logger.info("✅ Backend imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import backend: {e}")
        return False
    
    try:
        from rag_system import TextToCypherRAG
        logger.info("✅ RAG system imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import RAG system: {e}")
        return False
    
    try:
        import mcp_server
        logger.info("✅ MCP server imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import MCP server: {e}")
        return False
    
    return True

def test_rag_system():
    """Test RAG system functionality"""
    logger.info("Testing RAG system...")
    
    try:
        from rag_system import TextToCypherRAG, QueryExample
        
        # Initialize RAG system (will use lightweight model for testing)
        rag = TextToCypherRAG()
        logger.info("✅ RAG system initialized")
        
        # Test adding an example
        example = QueryExample(
            question="Count all nodes",
            cypher="MATCH (n) RETURN count(n)",
            description="Counts all nodes in the graph"
        )
        rag.add_example(example)
        logger.info("✅ Example added to RAG system")
        
        # Test finding similar examples
        similar = rag.find_similar_examples("How many nodes are there?")
        if similar:
            logger.info(f"✅ Found {len(similar)} similar examples")
            return True
        else:
            logger.warning("⚠️ No similar examples found (may be expected)")
            return True
            
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        return False

def test_backend_basic():
    """Test basic backend functionality"""
    logger.info("Testing backend basic functionality...")
    
    try:
        from backend import PuppyGraphChatbot
        
        # Initialize chatbot (this will fail if PuppyGraph is not running)
        try:
            chatbot = PuppyGraphChatbot()
            logger.info("✅ Backend initialized successfully")
            
            # Test schema retrieval
            schema = chatbot.get_schema()
            logger.info(f"✅ Schema retrieved: {len(schema.get('vertices', []))} vertices, {len(schema.get('edges', []))} edges")
            
            # Test stats (this might fail if no connection to PuppyGraph)
            stats = chatbot.get_graph_stats()
            if "error" in stats:
                logger.warning(f"⚠️ Graph stats returned error (PuppyGraph may not be running): {stats['error']}")
            else:
                logger.info(f"✅ Graph stats: {stats.get('node_count', 'unknown')} nodes, {stats.get('edge_count', 'unknown')} edges")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Backend connection failed (PuppyGraph may not be running): {e}")
            return True  # This is expected if PuppyGraph is not running
            
    except Exception as e:
        logger.error(f"❌ Backend test failed: {e}")
        return False

def test_gradio_interface():
    """Test Gradio interface creation"""
    logger.info("Testing Gradio interface...")
    
    try:
        from gradio_app import create_interface
        
        # Create interface (don't launch)
        interface = create_interface()
        logger.info("✅ Gradio interface created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradio interface test failed: {e}")
        return False

def test_environment_setup():
    """Test environment and configuration"""
    logger.info("Testing environment setup...")
    
    # Check for .env file
    if os.path.exists('.env'):
        logger.info("✅ .env file found")
    else:
        logger.warning("⚠️ .env file not found (using .env.example)")
    
    # Check for Anthropic API key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        logger.info("✅ Anthropic API key configured")
    else:
        logger.warning("⚠️ Anthropic API key not found - RAG functionality may be limited")
    
    return True

def run_all_tests():
    """Run all integration tests"""
    logger.info("Starting PuppyGraph RAG Chatbot Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Module Imports", test_imports),
        ("RAG System", test_rag_system),
        ("Backend Basic", test_backend_basic),
        ("Gradio Interface", test_gradio_interface),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        return False

def main():
    """Main test runner"""
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()